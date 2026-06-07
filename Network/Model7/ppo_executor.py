import math
import os
import sys

import numpy as np

import config


def resolve_model_path(model_path=None):
    candidates = []
    if model_path:
        candidates.append(model_path)
        if not model_path.endswith(".zip"):
            candidates.append(model_path + ".zip")
    candidates.append(config.DEFAULT_PPO_MODEL)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No PPO checkpoint found. Tried: {candidates}")


class PPOSegmentExecutor:
    """Lower action layer: use the external pretrained PPO image policy per segment.

    The referenced PPO policy is not goal-conditioned.  For each local segment,
    Model7 aligns the UAV yaw toward the next local target, then lets PPO's
    body-frame forward/lateral/vertical actions execute until the segment is
    reached, collides, or times out.
    """

    def __init__(self, model_path=None, ip_address="127.0.0.1"):
        import airsim
        from stable_baselines3 import PPO

        self.airsim = airsim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.vehicle_name = self._detect_vehicle_name()
        self.model_path = resolve_model_path(model_path)
        print(f"[Model7] Loading PPO lower policy: {self.model_path}")
        self.model = PPO.load(self.model_path)
        self.collision_time = 0

    def _detect_vehicle_name(self):
        try:
            vehicles = self.client.listVehicles()
            return vehicles[0] if vehicles else "Drone1"
        except Exception:
            return "Drone1"

    def draw_global_path(self, points):
        if not config.VISUALIZE_GLOBAL_PATH:
            return
        try:
            airsim_points = [
                self.airsim.Vector3r(float(point[0]), float(point[1]), float(point[2]) - config.VISUAL_Z_OFFSET)
                for point in points
            ]
            self.client.simPlotLineStrip(
                airsim_points,
                color_rgba=[0.55, 0.0, 1.0, 1.0],
                thickness=6.0,
                duration=-1.0,
                is_persistent=True,
            )
            self.client.simPlotPoints(
                airsim_points,
                color_rgba=[0.0, 1.0, 0.9, 1.0],
                size=12.0,
                duration=-1.0,
                is_persistent=True,
            )
        except Exception as exc:
            print(f"[Model7] Path visualization skipped: {exc}")

    def execute_path(self, points):
        summaries = []
        self.draw_global_path(points)

        for segment_index in range(len(points) - 1):
            target = np.array(points[segment_index + 1], dtype=np.float32)
            is_final_segment = segment_index == len(points) - 2
            start = np.array(points[segment_index], dtype=np.float32) if segment_index == 0 else self._get_actual_position()
            if segment_index > 0:
                print(f"[Model7] Segment {segment_index} starts from actual pos {start} "
                      f"(planned waypoint was {points[segment_index]})")

            summary = self.execute_segment(segment_index, start, target, is_final_segment=is_final_segment)
            summaries.append(summary)
            if not summary["arrived"] and config.STOP_ON_SEGMENT_FAILURE:
                print(f"[Model7] Stop after failed segment {segment_index}.")
                break

        return summaries

    def execute_segment(self, segment_index, start, target, is_final_segment=False):
        print(f"[Model7] Segment {segment_index}: {start} -> {target}")
        self._reset_to_segment_start(start, target)

        for step in range(config.PPO_SEGMENT_MAX_STEPS):
            obs = self._get_rgb_obs()
            action, _ = self.model.predict(obs, deterministic=config.PPO_DETERMINISTIC_POLICY)
            self._apply_action(int(np.asarray(action).squeeze()))

            current = self._get_actual_position()
            distance = float(np.linalg.norm(target - current))
            axis_error = np.abs(target - current)
            collision = self._has_collision()
            if collision:
                print(f"\n[Model7] Segment {segment_index} collision, dist={distance:.2f}")
                return self._summary(segment_index, False, step + 1, "collision", distance, axis_error)

            if self._segment_reached(distance, axis_error, is_final_segment):
                print(f"\n[Model7] Segment {segment_index} reached in {step + 1} steps, "
                      f"dist={distance:.2f}, axis_error={axis_error}")
                return self._summary(segment_index, True, step + 1, "arrived", distance, axis_error)

        current = self._get_actual_position()
        distance = float(np.linalg.norm(target - current))
        axis_error = np.abs(target - current)
        print(f"\n[Model7] Segment {segment_index} timeout, dist={distance:.2f}")
        return self._summary(segment_index, False, config.PPO_SEGMENT_MAX_STEPS, "segment_timeout", distance, axis_error)

    def _reset_to_segment_start(self, start, target):
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        yaw = self._yaw_to_target(start, target)
        pose = self.airsim.Pose(
            self.airsim.Vector3r(float(start[0]), float(start[1]), float(start[2])),
            self.airsim.to_quaternion(0.0, 0.0, yaw),
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.vehicle_name)
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.2, vehicle_name=self.vehicle_name).join()
        self.collision_time = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

    @staticmethod
    def _yaw_to_target(start, target):
        delta = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
        if float(np.linalg.norm(delta[:2])) < 1e-6:
            return 0.0
        return float(math.atan2(float(delta[1]), float(delta[0])))

    def _get_rgb_obs(self):
        request = self.airsim.ImageRequest("0", self.airsim.ImageType.Scene, False, False)
        responses = self.client.simGetImages([request], vehicle_name=self.vehicle_name)
        if not responses or responses[0].width <= 0 or responses[0].height <= 0:
            image = np.zeros(config.PPO_IMAGE_SHAPE, dtype=np.uint8)
        else:
            image = self._decode_scene_image(responses[0])
            image = self._resize_nearest(image, config.PPO_IMAGE_SHAPE[:2])
        return np.transpose(image, (2, 0, 1))

    @staticmethod
    def _decode_scene_image(response):
        data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        width = int(response.width)
        height = int(response.height)

        for channels in (3, 4):
            expected = width * height * channels
            if data.size == expected:
                image = data.reshape((height, width, channels))
                return image[:, :, :3]

        inferred = PPOSegmentExecutor._infer_image_shape(data.size)
        if inferred is not None:
            inferred_height, inferred_width, channels = inferred
            image = data.reshape((inferred_height, inferred_width, channels))
            return image[:, :, :3]

        print(
            f"[Model7] WARNING: unexpected RGB image buffer size={data.size}, "
            f"reported={width}x{height}; using blank observation."
        )
        return np.zeros(config.PPO_IMAGE_SHAPE, dtype=np.uint8)

    @staticmethod
    def _infer_image_shape(buffer_size):
        for channels in (3, 4):
            if buffer_size % channels != 0:
                continue
            pixels = buffer_size // channels
            side = int(math.sqrt(pixels))
            if side * side == pixels:
                return side, side, channels
        return None

    @staticmethod
    def _resize_nearest(image, target_hw):
        target_h, target_w = target_hw
        if image.shape[0] == target_h and image.shape[1] == target_w:
            return image
        row_idx = np.linspace(0, image.shape[0] - 1, target_h).astype(np.int32)
        col_idx = np.linspace(0, image.shape[1] - 1, target_w).astype(np.int32)
        return image[row_idx][:, col_idx]

    def _apply_action(self, action):
        speed_yz = config.PPO_LATERAL_SPEED
        if action == 0:
            vy, vz = (-speed_yz, -config.PPO_VERTICAL_SPEED)
        elif action == 1:
            vy, vz = (0.0, -config.PPO_VERTICAL_SPEED)
        elif action == 2:
            vy, vz = (speed_yz, -config.PPO_VERTICAL_SPEED)
        elif action == 3:
            vy, vz = (-speed_yz, 0.0)
        elif action == 4:
            vy, vz = (0.0, 0.0)
        elif action == 5:
            vy, vz = (speed_yz, 0.0)
        elif action == 6:
            vy, vz = (-speed_yz, config.PPO_VERTICAL_SPEED)
        elif action == 7:
            vy, vz = (0.0, config.PPO_VERTICAL_SPEED)
        else:
            vy, vz = (speed_yz, config.PPO_VERTICAL_SPEED)

        self.client.moveByVelocityBodyFrameAsync(
            float(config.PPO_FORWARD_SPEED),
            float(vy),
            float(vz),
            float(config.PPO_ACTION_DURATION),
            vehicle_name=self.vehicle_name,
        ).join()
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.05, vehicle_name=self.vehicle_name)

    def _get_actual_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _has_collision(self):
        try:
            collision_time = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp
            return bool(collision_time != self.collision_time)
        except Exception:
            return False

    def _segment_reached(self, distance, axis_error, is_final_segment):
        if distance <= config.SEGMENT_GOAL_TOLERANCE:
            return True
        if is_final_segment:
            return bool(np.all(axis_error <= config.FINAL_AXIS_TOLERANCE))
        return bool(np.all(axis_error <= config.INTERMEDIATE_AXIS_TOLERANCE))

    @staticmethod
    def _summary(segment_index, arrived, steps, end_reason, distance, axis_error):
        return {
            "segment": segment_index,
            "arrived": bool(arrived),
            "steps": int(steps),
            "reward": 0.0,
            "end_reason": end_reason,
            "distance": float(distance),
            "axis_error": axis_error.tolist(),
        }

    def close(self):
        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except Exception:
            pass
