import os
import sys

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
            start = np.array(points[segment_index], dtype=np.float32)
            target = np.array(points[segment_index + 1], dtype=np.float32)
            is_final_segment = segment_index == len(points) - 2
            summary = self.execute_segment(segment_index, start, target, is_final_segment=is_final_segment)
            summaries.append(summary)
            if not summary["arrived"] and config.STOP_ON_SEGMENT_FAILURE:
                print(f"[Model6] Stop after failed segment {segment_index}.")
                break

        return summaries

    def execute_segment(self, segment_index, start, target, is_final_segment=False):
        print(f"[Model6] Segment {segment_index}: {start} -> {target}")
        obs, _ = self.env.reset(options={
            "start_pos": start.tolist(),
            "target": target.tolist(),
            "region": f"model6_segment_{segment_index}",
        })

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
