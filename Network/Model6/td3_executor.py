import os
import sys
import time

import numpy as np
import torch

import config


MODEL1_DIR = os.path.join(config.NETWORK_DIR, "Model1")
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
    """Lower action layer: use Model1 TD3 to fly from local target n to n+1."""

    def __init__(self, model_path=None):
        from reinforcement_network import AirSimUAVEnv
        from stable_baselines3 import TD3

        self.env = AirSimUAVEnv()
        self.model_path = resolve_model_path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model6] Loading TD3 lower policy: {self.model_path}")
        self.model = TD3.load(self.model_path, env=self.env, device=self.device)

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
            print(f"[Model6] Path visualization skipped: {exc}")

    def execute_path(self, points):
        summaries = []
        self.draw_global_path(points)

        for segment_index in range(len(points) - 1):
            target = np.array(points[segment_index + 1], dtype=np.float32)
            is_final_segment = segment_index == len(points) - 2

            if segment_index == 0:
                # First segment: start from the planned waypoint
                start = np.array(points[segment_index], dtype=np.float32)
            else:
                # Subsequent segments: use actual drone position, NOT the planned waypoint.
                # The drone may still be 1-3m away from the waypoint when the intermediate
                # tolerance is satisfied.  Teleporting to the waypoint would cause an
                # unrealistic jump and the next segment would start from the wrong place.
                start = self._get_actual_position()
                print(f"[Model6] Segment {segment_index} starts from actual pos {start} "
                      f"(planned waypoint was {points[segment_index]})")

            summary = self.execute_segment(
                segment_index,
                start,
                target,
                is_final_segment=is_final_segment,
                teleport_to_start=segment_index == 0,
            )
            summaries.append(summary)
            if not summary["arrived"] and config.STOP_ON_SEGMENT_FAILURE:
                print(f"[Model6] Stop after failed segment {segment_index}.")
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

    def execute_segment(
        self,
        segment_index,
        start,
        target,
        is_final_segment=False,
        teleport_to_start=False,
    ):
        print(f"[Model6] Segment {segment_index}: {start} -> {target}")
        start = np.array(start, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        if teleport_to_start:
            start = self._teleport_to_segment_start(start)

        reset_options = {
            "start_pos": start.tolist(),
            "target": target.tolist(),
            "region": f"model6_segment_{segment_index}",
        }
        if teleport_to_start:
            reset_options["skip_stabilization"] = True

        obs, _ = self.env.reset(options=reset_options)

        total_reward = 0.0
        last_info = {}
        for step in range(config.SEGMENT_MAX_STEPS):
            action, _ = self.model.predict(obs, deterministic=config.DETERMINISTIC_POLICY)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            last_info = info

            arrived = bool(info.get("arrived", False))
            distance = float(info.get("dis2goal", np.inf))
            axis_error = self._axis_error_to_target(target)
            segment_reached = self._segment_reached(arrived, distance, axis_error, is_final_segment)
            if segment_reached:
                print(
                    f"\n[Model6] Segment {segment_index} reached in {step + 1} steps, "
                    f"dist={distance:.2f}, axis_error={axis_error}"
                )
                return {
                    "segment": segment_index,
                    "arrived": True,
                    "steps": step + 1,
                    "reward": total_reward,
                    "end_reason": "arrived",
                    "distance": distance,
                    "axis_error": axis_error.tolist(),
                }

            if terminated or truncated:
                reason = self.env._resolve_end_reason(info, terminated=terminated, truncated=truncated)
                print(f"\n[Model6] Segment {segment_index} ended: {reason}")
                return {
                    "segment": segment_index,
                    "arrived": False,
                    "steps": step + 1,
                    "reward": total_reward,
                    "end_reason": reason,
                    "distance": distance,
                    "axis_error": axis_error.tolist(),
                }

        distance = float(last_info.get("dis2goal", np.inf))
        print(f"\n[Model6] Segment {segment_index} timeout, dist={distance:.2f}")
        return {
            "segment": segment_index,
            "arrived": False,
            "steps": config.SEGMENT_MAX_STEPS,
            "reward": total_reward,
            "end_reason": "segment_timeout",
            "distance": distance,
            "axis_error": self._axis_error_to_target(target).tolist(),
        }

    def _teleport_to_segment_start(self, start):
        """Teleport the first Model6 segment to the planned global start."""
        import airsim

        start = np.array(start, dtype=np.float32).copy()
        if float(start[2]) >= 0.0:
            start[2] = -10.0

        yaw_rad = self._current_yaw_rad()
        pose = airsim.Pose(
            airsim.Vector3r(float(start[0]), float(start[1]), float(start[2])),
            airsim.to_quaternion(0.0, 0.0, yaw_rad),
        )

        print(f"[Model6] Teleporting first segment start to {start}")
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
            print(f"[Model6] Start teleport failed: {exc}")
        return start

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

    def close(self):
        try:
            self.env.client.armDisarm(False, vehicle_name=self.env.vehicle_name)
            self.env.client.enableApiControl(False, vehicle_name=self.env.vehicle_name)
        except Exception:
            pass
