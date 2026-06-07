import math
import os

import numpy as np

import config


def resolve_model_path(model_path=None):
    candidates = []
    if model_path:
        candidates.append(model_path)
        if not model_path.endswith(".zip"):
            candidates.append(model_path + ".zip")
    candidates.append(config.DEFAULT_DDPG_MODEL)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No DDPG checkpoint found. Tried: {candidates}")


class DDPGSegmentExecutor:
    """Lower action layer: use the external lidar DDPG policy per segment."""

    def __init__(self, model_path=None, ip_address="127.0.0.1"):
        import airsim
        from stable_baselines3 import DDPG

        self.airsim = airsim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.vehicle_name = self._detect_vehicle_name()
        self.model_path = resolve_model_path(model_path)
        print(f"[Model8] Loading DDPG lower policy: {self.model_path}")
        self.model = DDPG.load(
            self.model_path,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            },
        )
        self.observation_features = self._observation_feature_count()
        print(f"[Model8] DDPG observation features: {self.observation_features}")
        self.current_target = None
        self.current_start = None
        self.goal_distance = 1.0
        self.collision_time = 0
        self.min_distance_to_obstacles = config.DDPG_LIDAR_MAX_DISTANCE

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
                color_rgba=[0.1, 0.9, 0.1, 1.0],
                thickness=6.0,
                duration=-1.0,
                is_persistent=True,
            )
            self.client.simPlotPoints(
                airsim_points,
                color_rgba=[1.0, 0.1, 0.1, 1.0],
                size=12.0,
                duration=-1.0,
                is_persistent=True,
            )
        except Exception as exc:
            print(f"[Model8] Path visualization skipped: {exc}")

    def execute_path(self, points):
        summaries = []
        self.draw_global_path(points)

        for segment_index in range(len(points) - 1):
            target = np.array(points[segment_index + 1], dtype=np.float32)
            is_final_segment = segment_index == len(points) - 2
            start = np.array(points[segment_index], dtype=np.float32) if segment_index == 0 else self._get_actual_position()
            if segment_index > 0:
                print(f"[Model8] Segment {segment_index} starts from actual pos {start} "
                      f"(planned waypoint was {points[segment_index]})")

            summary = self.execute_segment(segment_index, start, target, is_final_segment=is_final_segment)
            summaries.append(summary)
            if not summary["arrived"] and config.STOP_ON_SEGMENT_FAILURE:
                print(f"[Model8] Stop after failed segment {segment_index}.")
                break

        return summaries

    def execute_segment(self, segment_index, start, target, is_final_segment=False):
        print(f"[Model8] Segment {segment_index}: {start} -> {target}")
        self._reset_to_segment_start(start, target)

        for step in range(config.DDPG_SEGMENT_MAX_STEPS):
            obs = self._get_obs()
            action, _ = self.model.predict(obs, deterministic=config.DDPG_DETERMINISTIC_POLICY)
            self._apply_action(np.asarray(action, dtype=np.float32).reshape(-1))

            current = self._get_actual_position()
            distance = float(np.linalg.norm(target - current))
            axis_error = np.abs(target - current)
            collision = self._has_collision() or self.min_distance_to_obstacles < config.DDPG_CRASH_DISTANCE
            if collision:
                print(f"\n[Model8] Segment {segment_index} collision, dist={distance:.2f}")
                return self._summary(segment_index, False, step + 1, "collision", distance, axis_error)

            if self._segment_reached(distance, axis_error, is_final_segment):
                print(f"\n[Model8] Segment {segment_index} reached in {step + 1} steps, "
                      f"dist={distance:.2f}, axis_error={axis_error}")
                return self._summary(segment_index, True, step + 1, "arrived", distance, axis_error)

        current = self._get_actual_position()
        distance = float(np.linalg.norm(target - current))
        axis_error = np.abs(target - current)
        print(f"\n[Model8] Segment {segment_index} timeout, dist={distance:.2f}")
        return self._summary(segment_index, False, config.DDPG_SEGMENT_MAX_STEPS, "segment_timeout", distance, axis_error)

    def _reset_to_segment_start(self, start, target):
        self.current_start = np.asarray(start, dtype=np.float32)
        self.current_target = np.asarray(target, dtype=np.float32)
        self.goal_distance = max(float(np.linalg.norm((self.current_target - self.current_start)[:2])), 1e-6)

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

    def _get_obs(self):
        obstacles = self._directional_lidar_distances()
        depth_features = self._depth_features() if self.observation_features == 10 else None
        current = self._get_actual_position()
        distance_xy = float(np.linalg.norm((self.current_target - current)[:2]))
        relative_z = float(current[2] - self.current_target[2])
        if self.observation_features == 10:
            dist_norm = np.clip(distance_xy / self.goal_distance * 255.0, 0.0, 255.0)
            vertical_norm = (relative_z / config.DDPG_MAX_VERTICAL_DIFFERENCE / 2.0 + 0.5) * 255.0
            vertical_norm = np.clip(vertical_norm, 0.0, 255.0)
            state = np.array([dist_norm, vertical_norm], dtype=np.float32) / 255.0
            obs = np.concatenate((depth_features, state, obstacles), axis=0).astype(np.float32)
        else:
            dist_norm = np.clip(distance_xy / self.goal_distance * config.DDPG_MAX_DEPTH, 0.0, config.DDPG_MAX_DEPTH)
            vertical_norm = (relative_z / config.DDPG_MAX_VERTICAL_DIFFERENCE / 2.0 + 0.5) * config.DDPG_MAX_DEPTH
            vertical_norm = np.clip(vertical_norm, 0.0, config.DDPG_MAX_DEPTH)
            state = np.array([dist_norm, vertical_norm], dtype=np.float32) / config.DDPG_MAX_DEPTH
            obs = np.concatenate((obstacles, state), axis=0).astype(np.float32)
        return obs.reshape((1, self.observation_features))

    def _observation_feature_count(self):
        shape = getattr(self.model.observation_space, "shape", None)
        if shape is None:
            return 5
        if len(shape) >= 2:
            return int(shape[-1])
        if len(shape) == 1:
            return int(shape[0])
        return 5

    def _depth_features(self):
        depth = self._get_depth_image()
        if depth.size == 0:
            return np.zeros(5, dtype=np.float32)
        self.min_distance_to_obstacles = min(self.min_distance_to_obstacles, float(np.min(depth)))
        depth_scaled = np.clip(depth, 0.0, config.DDPG_MAX_DEPTH) / config.DDPG_MAX_DEPTH
        inverse_depth = 1.0 - depth_scaled
        chunks = np.array_split(inverse_depth, 5, axis=1)
        return np.array([float(np.max(chunk)) for chunk in chunks], dtype=np.float32)

    def _get_depth_image(self):
        try:
            responses = self.client.simGetImages(
                [self.airsim.ImageRequest("0", self.airsim.ImageType.DepthVis, True)],
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)
        if not responses or responses[0].width <= 0 or responses[0].height <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        data = np.asarray(responses[0].image_data_float, dtype=np.float32)
        expected = int(responses[0].width) * int(responses[0].height)
        if data.size != expected:
            side = int(math.sqrt(data.size))
            if side * side == data.size:
                return data.reshape((side, side)) * 100.0
            return np.zeros((0, 0), dtype=np.float32)
        return data.reshape((int(responses[0].height), int(responses[0].width))) * 100.0

    def _directional_lidar_distances(self):
        points = self._get_lidar_points()
        if points.size == 0:
            distances = np.ones(3, dtype=np.float32) * config.DDPG_LIDAR_MAX_DISTANCE
        else:
            front, left, right = [], [], []
            for point in points:
                angle = math.degrees(math.atan2(float(point[1]), float(point[0])))
                distance = math.sqrt(float(point[0] ** 2 + point[1] ** 2))
                if -45.0 <= angle < 45.0:
                    front.append(distance)
                elif -90.0 <= angle < -45.0:
                    left.append(distance)
                elif 45.0 <= angle < 90.0:
                    right.append(distance)
            distances = np.array([
                np.mean(front) if front else config.DDPG_LIDAR_MAX_DISTANCE,
                np.mean(left) if left else config.DDPG_LIDAR_MAX_DISTANCE,
                np.mean(right) if right else config.DDPG_LIDAR_MAX_DISTANCE,
            ], dtype=np.float32)
        self.min_distance_to_obstacles = float(np.min(distances))
        return np.clip(distances, 0.0, config.DDPG_LIDAR_MAX_DISTANCE) / config.DDPG_LIDAR_MAX_DISTANCE

    def _get_lidar_points(self):
        lidar_names = ("LLidar1", "Lidar1", "")
        for lidar_name in lidar_names:
            try:
                if lidar_name:
                    lidar_data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=self.vehicle_name)
                else:
                    lidar_data = self.client.getLidarData(vehicle_name=self.vehicle_name)
            except Exception:
                continue
            if len(lidar_data.point_cloud) > 2:
                points = np.array(lidar_data.point_cloud, dtype=np.float32)
                return points.reshape((int(points.shape[0] / 3), 3))
        return np.zeros((0, 3), dtype=np.float32)

    def _apply_action(self, action):
        forward = float(action[0]) * config.DDPG_ACTION_SPEED_SCALE if action.size > 0 else 0.0
        yaw_rate = float(action[-1]) * config.DDPG_ACTION_YAW_SCALE if action.size > 1 else 0.0
        yaw = self._current_yaw() + yaw_rate * config.DDPG_ACTION_DURATION
        vx = forward * math.cos(yaw)
        vy = forward * math.sin(yaw)
        target_z = float(self.current_target[2])
        self.client.moveByVelocityZAsync(
            vx,
            vy,
            target_z,
            config.DDPG_ACTION_DURATION,
            drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
            vehicle_name=self.vehicle_name,
        ).join()

    def _current_yaw(self):
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        return float(self.airsim.to_eularian_angles(pose.orientation)[2])

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
