import os
import sys
import time

import numpy as np
import torch

import config
from safety_shield import RuntimeSafetyShield


MODEL1_DIR = os.path.join(config.NETWORK_DIR, "TD3_base")
if config.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, config.PROJECT_ROOT)
if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)


def resolve_model_path(model_path=None):
    candidates = []
    if model_path:
        candidates.append(model_path)
        if not model_path.endswith(".zip"):
            candidates.append(model_path + ".zip")
    candidates.extend([config.DEFAULT_TD3_MODEL, config.FALLBACK_TD3_MODEL])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No TD3 checkpoint found. Tried: {candidates}")


class TD3SegmentExecutor:
    """Lower action layer: use TD3_base TD3 to fly from local target n to n+1."""

    def __init__(self, model_path=None):
        from reinforcement_network import AirSimUAVEnv
        from stable_baselines3 import TD3

        self.env = AirSimUAVEnv()
        self.model_path = resolve_model_path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Astar_planner] Loading TD3 lower policy: {self.model_path}")
        self.model = TD3.load(self.model_path, env=self.env, device=self.device)
        self.safety_shield = (
            RuntimeSafetyShield(self.env)
            if bool(getattr(config, "SAFETY_SHIELD_ENABLED", False))
            else None
        )

    def draw_global_path(self, points):
        if not config.VISUALIZE_GLOBAL_PATH:
            return
        try:
            import airsim

            airsim_points = [
                airsim.Vector3r(float(point[0]), float(point[1]), float(point[2]) - config.VISUAL_Z_OFFSET)
                for point in points
            ]
            self.env.client.simPlotLineStrip(
                airsim_points,
                color_rgba=[0.0, 0.4, 1.0, 1.0],
                thickness=6.0,
                duration=-1.0,
                is_persistent=True,
            )
            self.env.client.simPlotPoints(
                airsim_points,
                color_rgba=[1.0, 0.6, 0.0, 1.0],
                size=12.0,
                duration=-1.0,
                is_persistent=True,
            )
        except Exception as exc:
            print(f"[Astar_planner] Path visualization skipped: {exc}")

    def execute_path(self, points):
        summaries = []
        planned_points = [np.array(point, dtype=np.float32).copy() for point in points]
        points = [point.copy() for point in planned_points]
        self.draw_global_path(points)

        for segment_index in range(len(points) - 1):
            target = np.array(points[segment_index + 1], dtype=np.float32)
            is_final_segment = segment_index == len(points) - 2
            teleport_source_start = None

            if segment_index == 0:
                # First segment: place at the original planned start, climb to the
                # requested start altitude, then start TD3 navigation.
                start = np.array(points[segment_index], dtype=np.float32)
                teleport_source_start = np.array(planned_points[segment_index], dtype=np.float32)
            else:
                # Subsequent segments: use actual drone position, NOT the planned waypoint.
                # The drone may still be 1-3m away from the waypoint when the intermediate
                # tolerance is satisfied.  Teleporting to the waypoint would cause an
                # unrealistic jump and the next segment would start from the wrong place.
                start = self._get_actual_position()
                print(f"[Astar_planner] Segment {segment_index} starts from actual pos {start} "
                      f"(planned waypoint was {points[segment_index]})")

            summary = self.execute_segment(
                segment_index,
                start,
                target,
                is_final_segment=is_final_segment,
                teleport_to_start=segment_index == 0,
                teleport_source_start=teleport_source_start,
            )
            summaries.append(summary)
            if not summary["arrived"] and config.STOP_ON_SEGMENT_FAILURE:
                print(f"[Astar_planner] Stop after failed segment {segment_index}.")
                break

        return summaries

    def _get_actual_position(self):
        """Read the drone's current position from AirSim."""
        try:
            state = self.env.client.getMultirotorState(vehicle_name=self.env.vehicle_name)
            pos = state.kinematics_estimated.position
            return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        except Exception:
            # Fallback: use the environment's cached relative position
            if hasattr(self.env, "current_start_pos"):
                return self.env.current_start_pos.copy()
            return np.array([0.0, 0.0, -2.0], dtype=np.float32)

    @staticmethod
    def _altitude_to_ned_z(altitude):
        return float(config.OCCUPANCY_GROUND_Z - float(altitude))

    def execute_segment(
        self,
        segment_index,
        start,
        target,
        is_final_segment=False,
        teleport_to_start=False,
        teleport_source_start=None,
    ):
        print(f"[Astar_planner] Segment {segment_index}: {start} -> {target}")
        start = np.array(start, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        if teleport_to_start:
            start = self._teleport_to_segment_start(
                start,
                pose_start=teleport_source_start,
            )

        reset_options = {
            "start_pos": start.tolist(),
            "target": target.tolist(),
            "region": f"model6_segment_{segment_index}",
            "reward_mode": "hierarchical_test",
            "disable_crossed_border_termination": True,
        }
        if teleport_to_start:
            reset_options["skip_stabilization"] = True

        obs, _ = self.env.reset(options=reset_options)
        if self.safety_shield is not None:
            self.safety_shield.reset()

        initial_segment_distance = float(
            getattr(
                self.env,
                "start_dist",
                np.linalg.norm(target - start),
            )
        )
        if not np.isfinite(initial_segment_distance) or initial_segment_distance <= 1e-6:
            initial_segment_distance = float(np.linalg.norm(target - start))

        total_reward = 0.0
        last_info = {}
        stuck_near_target_steps = 0
        stalled_progress_steps = 0
        step = 0
        while step < self._segment_step_budget():
            step += 1
            action, _ = self.model.predict(obs, deterministic=config.DETERMINISTIC_POLICY)
            if self.safety_shield is not None:
                action, shield_diag = self.safety_shield.filter_action(action, obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.safety_shield is not None:
                info["safety_shield_reason"] = self.safety_shield.last_reason
                info["safety_shield_interventions"] = self.safety_shield.interventions
                info["safety_shield_recovery_interventions"] = self.safety_shield.recovery_interventions
                info["safety_shield_recovery_steps"] = self.safety_shield.recovery_steps
            total_reward += float(reward)
            last_info = info

            arrived = bool(info.get("arrived", False))
            distance = float(info.get("dis2goal", np.inf))
            axis_error = self._axis_error_to_target(target)
            progress = float(info.get("progress", 0.0))
            speed = float(info.get("v_magnitude", np.inf))
            stuck_near_target_steps = self._update_stuck_near_target_steps(
                stuck_near_target_steps,
                distance,
                progress,
                speed,
                is_final_segment,
            )
            stalled_progress_steps = self._update_intermediate_stall_steps(
                stalled_progress_steps,
                progress,
                speed,
                step,
                is_final_segment,
            )
            segment_reached = self._segment_reached(arrived, distance, axis_error, is_final_segment)
            if (
                not segment_reached
                and stuck_near_target_steps >= config.INTERMEDIATE_STUCK_WINDOW
                and self._intermediate_has_enough_progress(
                    distance,
                    initial_segment_distance,
                    is_final_segment,
                )
            ):
                segment_reached = True
                info["end_reason"] = "stuck_near_intermediate"
            if not segment_reached:
                advance_reason = self._intermediate_progress_advance_reason(
                    distance,
                    initial_segment_distance,
                    stalled_progress_steps,
                    is_final_segment,
                    timed_out=False,
                )
                if advance_reason is not None:
                    segment_reached = True
                    info["end_reason"] = advance_reason
            if segment_reached:
                print(
                    f"\n[Astar_planner] Segment {segment_index} reached in {step} steps, "
                    f"dist={distance:.2f}, axis_error={axis_error}, reason={info.get('end_reason', 'arrived')}"
                )
                return {
                    "segment": segment_index,
                    "arrived": True,
                    "steps": step,
                    "reward": total_reward,
                    "end_reason": info.get("end_reason", "arrived"),
                    "distance": distance,
                    "axis_error": axis_error.tolist(),
                    "safety_shield_interventions": self._shield_interventions(),
                    "safety_shield_emergency_interventions": self._shield_emergencies(),
                    "safety_shield_recovery_interventions": self._shield_recoveries(),
                    "safety_shield_recovery_steps": self._shield_recovery_steps(),
                }

            if terminated or truncated:
                reason = self.env._resolve_end_reason(info, terminated=terminated, truncated=truncated)
                print(f"\n[Astar_planner] Segment {segment_index} ended: {reason}")
                return {
                    "segment": segment_index,
                    "arrived": False,
                    "steps": step,
                    "reward": total_reward,
                    "end_reason": reason,
                    "distance": distance,
                    "axis_error": axis_error.tolist(),
                    "safety_shield_interventions": self._shield_interventions(),
                    "safety_shield_emergency_interventions": self._shield_emergencies(),
                    "safety_shield_recovery_interventions": self._shield_recoveries(),
                    "safety_shield_recovery_steps": self._shield_recovery_steps(),
                }

        distance = float(last_info.get("dis2goal", np.inf))
        axis_error = self._axis_error_to_target(target)
        advance_reason = self._intermediate_progress_advance_reason(
            distance,
            initial_segment_distance,
            stalled_progress_steps,
            is_final_segment,
            timed_out=True,
        )
        if advance_reason is not None:
            print(
                f"\n[Astar_planner] Segment {segment_index} advanced after timeout "
                f"in {step} steps, dist={distance:.2f}, axis_error={axis_error}, "
                f"reason={advance_reason}"
            )
            return {
                "segment": segment_index,
                "arrived": True,
                "steps": step,
                "reward": total_reward,
                "end_reason": advance_reason,
                "distance": distance,
                "axis_error": axis_error.tolist(),
                "safety_shield_interventions": self._shield_interventions(),
                "safety_shield_emergency_interventions": self._shield_emergencies(),
                "safety_shield_recovery_interventions": self._shield_recoveries(),
                "safety_shield_recovery_steps": self._shield_recovery_steps(),
            }

        print(
            f"\n[Astar_planner] Segment {segment_index} timeout, "
            f"dist={distance:.2f}, budget={step}"
        )
        return {
            "segment": segment_index,
            "arrived": False,
            "steps": step,
            "reward": total_reward,
            "end_reason": "segment_timeout",
            "distance": distance,
            "axis_error": axis_error.tolist(),
            "safety_shield_interventions": self._shield_interventions(),
            "safety_shield_emergency_interventions": self._shield_emergencies(),
            "safety_shield_recovery_interventions": self._shield_recoveries(),
            "safety_shield_recovery_steps": self._shield_recovery_steps(),
        }

    def _segment_step_budget(self):
        base_steps = int(config.SEGMENT_MAX_STEPS)
        if self.safety_shield is None:
            return base_steps

        step_extension = int(getattr(config, "SEGMENT_SHIELD_STEP_EXTENSION", 0))
        max_extra_steps = int(getattr(config, "SEGMENT_SHIELD_MAX_EXTRA_STEPS", 0))
        if step_extension <= 0 or max_extra_steps <= 0:
            return base_steps

        extra_steps = min(
            max_extra_steps,
            self._shield_recovery_steps() * step_extension,
        )
        return base_steps + int(extra_steps)

    def _shield_interventions(self):
        if self.safety_shield is None:
            return 0
        return int(self.safety_shield.interventions)

    def _shield_emergencies(self):
        if self.safety_shield is None:
            return 0
        return int(self.safety_shield.emergency_interventions)

    def _shield_recoveries(self):
        if self.safety_shield is None:
            return 0
        return int(self.safety_shield.recovery_interventions)

    def _shield_recovery_steps(self):
        if self.safety_shield is None:
            return 0
        return int(self.safety_shield.recovery_steps)

    def _teleport_to_segment_start(self, start, pose_start=None):
        """Place the first segment at the planned start, then climb before navigation."""
        import airsim

        execution_start = np.array(start, dtype=np.float32).copy()
        pose_start = np.array(
            execution_start if pose_start is None else pose_start,
            dtype=np.float32,
        ).copy()
        if float(pose_start[2]) >= 0.0:
            pose_start[2] = -1.0

        yaw_rad = self._current_yaw_rad()
        pose = airsim.Pose(
            airsim.Vector3r(float(pose_start[0]), float(pose_start[1]), float(pose_start[2])),
            airsim.to_quaternion(0.0, 0.0, yaw_rad),
        )

        print(f"[Astar_planner] Placing first segment start at {pose_start}")
        try:
            self.env.client.reset()
            self.env.client.enableApiControl(True, vehicle_name=self.env.vehicle_name)
            self.env.client.armDisarm(True, vehicle_name=self.env.vehicle_name)
            self.env.client.simSetVehiclePose(
                pose,
                True,
                vehicle_name=self.env.vehicle_name,
            )
            time.sleep(0.5)
            self._climb_to_start_altitude()
            execution_start = self._get_actual_position()
            self.env.client.moveByVelocityAsync(
                0.0,
                0.0,
                0.0,
                0.5,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.env.vehicle_name,
            ).join()
        except Exception as exc:
            print(f"[Astar_planner] Start teleport failed: {exc}")
        return execution_start

    def _climb_to_start_altitude(self):
        """Climb once near the global start before TD3 takes over."""
        import airsim

        target_z = self._altitude_to_ned_z(config.NAVIGATION_START_ALTITUDE)
        try:
            current = self._get_actual_position()
            if float(current[2]) <= target_z:
                return

            print(f"[Astar_planner] Initial climb to {float(config.NAVIGATION_START_ALTITUDE):.1f}m.")
            self.env.client.moveToZAsync(
                float(target_z),
                float(config.NAVIGATION_START_ASCENT_VELOCITY),
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.env.vehicle_name,
            ).join()
            time.sleep(float(config.NAVIGATION_START_HOVER_SECONDS))
            self.env.client.hoverAsync(vehicle_name=self.env.vehicle_name).join()
        except Exception as exc:
            print(f"[Astar_planner] Initial climb skipped: {exc}")

    def _current_yaw_rad(self):
        try:
            import airsim

            state = self.env.client.getMultirotorState(
                vehicle_name=self.env.vehicle_name,
            )
            _, _, yaw_rad = airsim.to_eularian_angles(
                state.kinematics_estimated.orientation,
            )
            return float(yaw_rad)
        except Exception:
            return 0.0

    def _axis_error_to_target(self, target):
        if hasattr(self.env, "current_rel_pos"):
            return np.abs(np.asarray(self.env.current_rel_pos, dtype=np.float32))
        try:
            state = self.env.client.getMultirotorState(vehicle_name=self.env.vehicle_name)
            pos = state.kinematics_estimated.position
            current = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
            return np.abs(np.asarray(target, dtype=np.float32) - current)
        except Exception:
            return np.ones(3, dtype=np.float32) * np.inf

    def _segment_reached(self, model1_arrived, distance, axis_error, is_final_segment):
        if model1_arrived:
            return True
        if is_final_segment:
            return bool(np.all(axis_error <= config.FINAL_AXIS_TOLERANCE))
        return bool(np.all(axis_error <= config.INTERMEDIATE_AXIS_TOLERANCE))

    def _update_stuck_near_target_steps(self, current_count, distance, progress, speed, is_final_segment):
        if is_final_segment:
            return 0
        close_enough = float(distance) <= float(config.INTERMEDIATE_STUCK_ACCEPT_DISTANCE)
        low_progress = abs(float(progress)) <= float(config.INTERMEDIATE_STUCK_PROGRESS_EPS)
        slow = float(speed) <= float(config.INTERMEDIATE_STUCK_SPEED_MAX)
        if close_enough and low_progress and slow:
            return int(current_count) + 1
        return 0

    def _update_intermediate_stall_steps(self, current_count, progress, speed, step, is_final_segment):
        if is_final_segment:
            return 0
        if int(step) < int(getattr(config, "INTERMEDIATE_STALL_MIN_STEPS", 0)):
            return 0

        low_progress = abs(float(progress)) <= float(config.INTERMEDIATE_STALL_PROGRESS_EPS)
        slow = float(speed) <= float(config.INTERMEDIATE_STALL_SPEED_MAX)
        if low_progress and slow:
            return int(current_count) + 1
        return 0

    def _intermediate_progress_advance_reason(
        self,
        distance,
        initial_distance,
        stalled_progress_steps,
        is_final_segment,
        timed_out=False,
    ):
        if is_final_segment:
            return None

        if not self._intermediate_has_enough_progress(
            distance,
            initial_distance,
            is_final_segment,
        ):
            return None

        if timed_out:
            return "intermediate_timeout_progressed"

        if int(stalled_progress_steps) >= int(config.INTERMEDIATE_STALL_WINDOW):
            return "intermediate_stalled_progressed"

        return None

    def _intermediate_has_enough_progress(self, distance, initial_distance, is_final_segment):
        if is_final_segment:
            return False

        distance = float(distance)
        initial_distance = float(initial_distance)
        if not np.isfinite(distance) or not np.isfinite(initial_distance) or initial_distance <= 1e-6:
            return False

        distance_threshold = min(
            float(config.INTERMEDIATE_ADVANCE_ACCEPT_DISTANCE),
            initial_distance * float(config.INTERMEDIATE_ADVANCE_PROGRESS_RATIO),
        )
        min_progress = min(
            float(config.INTERMEDIATE_ADVANCE_MIN_PROGRESS),
            initial_distance * 0.25,
        )
        reduced_distance = initial_distance - distance
        return bool(distance <= distance_threshold and reduced_distance >= min_progress)

    def close(self):
        try:
            self.env.client.armDisarm(False, vehicle_name=self.env.vehicle_name)
            self.env.client.enableApiControl(False, vehicle_name=self.env.vehicle_name)
        except Exception:
            pass
