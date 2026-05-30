import csv
import os
import sys
import time

import numpy as np

# Add Model2 to path for AirSimBridge reuse.
MODEL2_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Model2",
)
if MODEL2_PATH not in sys.path:
    sys.path.insert(0, MODEL2_PATH)

from uav_env.observation import build_observation, get_ray_directions
from uav_env.action import local_target_to_world, clip_action
from uav_env.reward import compute_reward
from ego_planner.ego_planner import EGOPlanner
import config


class UAVRLOEnv:
    """
    Gym-compatible environment for RLoPlanner with AirSim+UE5.

    Observation: [yaw, pitch, 25 rangefinder distances, relative target info]
    Action: continuous 6D [x, y, z, phi, theta, psi] local target
    Reward: paper-consistent distance reward + local-target obstacle penalty
    """

    def __init__(self, planner_mode=None):
        try:
            from airsim_client.airsim_bridge import AirSimBridge

            self.bridge = AirSimBridge()
            self.use_airsim = True
            try:
                vehicles = self.bridge.client.listVehicles()
                self.vehicle_name = vehicles[0] if vehicles else "Drone1"
            except Exception:
                self.vehicle_name = "Drone1"
            print(f"Detected vehicle: '{self.vehicle_name}'")
            self._prime_airsim_vehicle()
        except Exception as e:
            print(f"[WARN] AirSim not available, using simulation mode: {e}")
            self.bridge = None
            self.use_airsim = False
            self.vehicle_name = None
            self._init_sim()

        self.planner = EGOPlanner(config)
        self.planner_mode = (planner_mode or getattr(config, "PLANNER_MODE", "ego")).lower()
        if self.planner_mode not in {"ego", "straight"}:
            raise ValueError(f"Unsupported planner mode: {self.planner_mode}")
        self.ray_directions = get_ray_directions(
            hfov=config.RANGE_HFOV,
            vfov=config.RANGE_VFOV,
            rays_h=config.RANGE_RAYS_H,
            rays_v=config.RANGE_RAYS_V,
        )

        self.goal = None
        self.start_pos = None
        self.prev_pos = None
        self.current_pos = None
        self.current_vel = None
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.step_count = 0
        self.collision_count = 0
        self.trajectory = []
        self.sim_obstacles = []  # Always initialized, used in both modes
        self.prev_yaw = 0.0  # For computing yaw rate
        self.prev_action = None
        self.last_local_target = None
        self.current_episode = 0
        self.total_train_steps = 0
        self.dataset_points_by_region = self._load_dataset_points()

    def _rotate_body_rays_to_world(self, ray_directions, yaw):
        """Rotate body-frame sensor rays into the world frame around z."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        r_z = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ])
        return np.array([r_z @ d for d in ray_directions])

    def _init_sim(self):
        """Initialize simple simulation state for offline testing."""
        self.sim_pos = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        self.sim_vel = np.zeros(3, dtype=np.float32)
        self.sim_yaw = 0.0
        self.sim_obstacles = []

    def _load_dataset_points(self):
        """Load navigation points from dataset/relative_coordinates.csv grouped by region."""
        csv_path = config.DATASET_CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Navigation dataset not found: {csv_path}")

        region_points = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_fields = {"x", "y", "z", "region"}
            if not reader.fieldnames or not required_fields.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Dataset {csv_path} must contain columns {sorted(required_fields)}; "
                    f"got {reader.fieldnames}"
                )

            for row in reader:
                region = str(int(float(row["region"])))
                point = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float32,
                )
                region_points.setdefault(region, []).append(point)

        region_points = {region: points for region, points in region_points.items() if points}
        if len(region_points) < 2:
            raise ValueError(
                f"Dataset {csv_path} must contain at least 2 regions with points; "
                f"got {list(region_points.keys())}"
            )

        print(
            f"[DATASET] Loaded {sum(len(points) for points in region_points.values())} points "
            f"across {len(region_points)} regions from {csv_path}"
        )
        return region_points

    def _sample_start_and_goal(self):
        """Sample one point from each of two different regions."""
        regions = sorted(self.dataset_points_by_region.keys())
        start_region, goal_region = np.random.choice(regions, size=2, replace=False)
        start_pos = self.dataset_points_by_region[start_region][
            np.random.randint(len(self.dataset_points_by_region[start_region]))
        ].copy()
        goal_pos = self.dataset_points_by_region[goal_region][
            np.random.randint(len(self.dataset_points_by_region[goal_region]))
        ].copy()
        return start_pos, goal_pos, start_region, goal_region

    def _get_attitude_from_airsim(self):
        """Extract pitch and yaw angles from the AirSim vehicle quaternion."""
        state = self.bridge.client.getMultirotorState(vehicle_name=self.vehicle_name)
        q = state.kinematics_estimated.orientation
        sinp = 2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        yaw = np.arctan2(
            2 * (q.w_val * q.z_val + q.x_val * q.y_val),
            1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val),
        )
        return float(pitch), float(yaw)

    def _check_collision_airsim(self):
        """Check if UAV has collided."""
        try:
            return self.bridge.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        except Exception:
            return False

    def _prime_airsim_vehicle(self):
        """
        Mirror Model1's startup sequence so AirSim reattaches the follow view to
        the active multirotor before training starts.
        """
        if not self.use_airsim:
            return

        try:
            self.bridge.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.bridge.client.armDisarm(True, vehicle_name=self.vehicle_name)
            self.bridge.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass

        try:
            self.bridge.client.client.call("setApiControlTimeout", 0.0, self.vehicle_name)
        except Exception:
            pass

    def _reset_airsim_to_start(self, start_pos, goal_pos):
        """Reset AirSim and place the UAV at the sampled start point facing the goal."""
        import airsim

        self.bridge.client.reset()
        self.bridge.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.bridge.client.armDisarm(True, vehicle_name=self.vehicle_name)

        direction = np.array(goal_pos, dtype=np.float32) - np.array(start_pos, dtype=np.float32)
        yaw = float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
        pose = airsim.Pose(
            airsim.Vector3r(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
            airsim.to_quaternion(0.0, 0.0, yaw),
        )
        self.bridge.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)

        # Match Model1's more conservative stabilization window so attached cameras
        # and the rendered chase/top views move with the teleported vehicle reliably.
        time.sleep(0.5)
        self.bridge.client.moveByVelocityAsync(
            0.0,
            0.0,
            0.0,
            1.0,
            vehicle_name=self.vehicle_name,
        ).join()
        time.sleep(1.0)

        # Force one image query after teleport so camera/render transforms refresh immediately.
        try:
            self.bridge.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            pass

        state = self.bridge.get_state()
        self.current_pos = state["position"].copy()
        self.current_vel = state["velocity"].copy()
        self.current_pitch, self.current_yaw = self._get_attitude_from_airsim()

    def reset(self, goal=None):
        """Reset environment for a new episode."""
        sampled_start, sampled_goal, start_region, goal_region = self._sample_start_and_goal()
        self.start_pos = sampled_start.copy()
        self.goal = sampled_goal.copy() if goal is None else np.array(goal, dtype=np.float32)
        self.start_region = start_region
        self.goal_region = goal_region

        if self.use_airsim:
            self._reset_airsim_to_start(self.start_pos, self.goal)
        else:
            self.sim_pos = self.start_pos.copy()
            self.sim_vel = np.zeros(3, dtype=np.float32)
            direction = self.goal - self.sim_pos
            self.sim_yaw = float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
            self.current_pos = self.sim_pos.copy()
            self.current_vel = self.sim_vel.copy()
            self.current_pitch = 0.0
            self.current_yaw = self.sim_yaw

        self.prev_pos = self.current_pos.copy()
        self.prev_action = None
        self.step_count = 0
        self.collision_count = 0
        self.trajectory = [self.current_pos.copy()]
        self.prev_yaw = self.current_yaw

        print(
            f"[RESET] Start(region={self.start_region}): {self.current_pos}, "
            f"Goal(region={self.goal_region}): {self.goal}, PlannerMode={self.planner_mode}"
        )
        return self._get_obs()

    def step(self, action):
        """
        Execute one RL step.

        Args:
            action: (6,) local target from RSPG policy

        Returns:
            obs, reward, done, info
        """
        action = clip_action(
            action,
            pos_min=config.ACTION_POS_MIN,
            pos_max=config.ACTION_POS_MAX,
        )
        prev_pos_for_reward = self.prev_pos.copy()
        prev_vel_for_reward = self.current_vel.copy()
        prev_action_for_reward = None if self.prev_action is None else self.prev_action.copy()

        world_target, target_angles = local_target_to_world(
            self.current_pos,
            self.current_yaw,
            action,
        )

        if self.use_airsim:
            self._update_planner_from_airsim()
        else:
            self._update_planner_from_sim()

        # Paper Eq. 7 uses d_min from the decision-making local target to the
        # nearest obstacle, not from the UAV position after executing the plan.
        min_dist, _ = self.planner.voxel_grid.get_distance_with_obstacle(world_target)
        if min_dist is None:
            min_dist = config.RANGE_MAX

        has_collision = False
        if self.planner_mode == "ego":
            spline = self.planner.plan(
                current_pos=self.current_pos,
                current_vel=self.current_vel,
                local_target=np.concatenate([world_target, target_angles]),
            )

            has_collision, _ = self.planner.check_collision(spline)
            if has_collision:
                self.collision_count += 1

            if self.use_airsim:
                self._execute_trajectory_airsim(spline)
                if self._check_collision_airsim():
                    self.collision_count += 1
                    has_collision = True
            else:
                self._execute_trajectory_sim(spline)
        else:
            if self.use_airsim:
                has_collision = self._execute_straight_airsim(world_target)
            else:
                has_collision = self._execute_straight_sim(world_target)
            if has_collision:
                self.collision_count += 1

        dist_to_goal = np.linalg.norm(self.current_pos - self.goal)
        self.step_count += 1
        done, reason = self._check_done(dist_to_goal)

        reward, reward_components = compute_reward(
            self.current_pos, prev_pos_for_reward, self.goal,
            min_dist_to_obstacle=min_dist,
            sigma=config.REWARD_SIGMA,
            beta=config.REWARD_BETA,
            current_vel=self.current_vel,
            prev_vel=prev_vel_for_reward,
            action=action,
            prev_action=prev_action_for_reward,
            collided=has_collision,
            terminal_reason=reason,
            safe_altitude=config.REWARD_SAFE_ALTITUDE,
            altitude_weight=config.REWARD_ALTITUDE_WEIGHT,
            descent_weight=config.REWARD_DESCENT_WEIGHT,
            smooth_action_weight=config.REWARD_SMOOTH_ACTION_WEIGHT,
            speed_smooth_weight=config.REWARD_SPEED_SMOOTH_WEIGHT,
            collision_penalty=config.REWARD_COLLISION_PENALTY,
            ground_penalty=config.REWARD_GROUND_PENALTY,
            out_of_bounds_penalty=config.REWARD_OUT_OF_BOUNDS_PENALTY,
            goal_bonus=config.REWARD_GOAL_BONUS,
            return_components=True,
        )

        self.prev_pos = self.current_pos.copy()
        self.prev_action = action.copy()
        self.trajectory.append(self.current_pos.copy())

        info = {
            "collision": has_collision,
            "collision_count": self.collision_count,
            "dist_to_goal": dist_to_goal,
            "altitude": max(-float(self.current_pos[2]), 0.0),
            "reason": reason,
            "step": self.step_count,
            "reward_components": reward_components,
            "executed_action": action.copy(),
            "local_target": world_target.copy(),
            "local_target_obstacle_dist": min_dist,
            "start_region": self.start_region,
            "goal_region": self.goal_region,
        }

        self.last_local_target = world_target.astype(np.float32).copy()
        self._render_dashboard(
            reward=reward,
            done=done,
            info=info,
        )
        obs = self._get_obs()
        return obs, reward, done, info

    def set_total_train_steps(self, value):
        self.total_train_steps = int(value)

    def set_current_episode(self, value):
        self.current_episode = int(value)

    def _render_dashboard(self, reward, done, info):
        """Print a one-line live dashboard similar to Model1."""
        pos = self.current_pos if self.current_pos is not None else np.zeros(3, dtype=np.float32)
        vel = self.current_vel if self.current_vel is not None else np.zeros(3, dtype=np.float32)
        local_target = self.last_local_target if self.last_local_target is not None else np.zeros(3, dtype=np.float32)

        print(
            f"\r[Episode {self.current_episode:4d} | Step {self.step_count:4d} | Total {self.total_train_steps:7d}] "
            f"Pos: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}) | "
            f"Goal: ({self.goal[0]:6.1f}, {self.goal[1]:6.1f}, {self.goal[2]:6.1f}) | "
            f"LocalTarget: ({local_target[0]:6.1f}, {local_target[1]:6.1f}, {local_target[2]:6.1f}) | "
            f"Mode: {self.planner_mode:8s} | "
            f"Vel: ({vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}) | "
            f"Dist: {info['dist_to_goal']:6.2f} | "
            f"ObsDist: {info['local_target_obstacle_dist']:5.2f} | "
            f"Rwd: {reward:7.3f}",
            end="",
            flush=True,
        )

        if done:
            reason = info.get("reason", "done")
            print(f" -> Episode End: {reason}")

    def _get_obs(self):
        """Get current observation vector."""
        if self.use_airsim:
            ray_distances = self._get_rangefinder_airsim()
            orientation = np.array([0.0, self.current_pitch, self.current_yaw], dtype=np.float32)
        else:
            ray_distances = self._get_rangefinder_sim()
            orientation = np.array([0.0, 0.0, self.sim_yaw], dtype=np.float32)

        return build_observation(
            state={"position": self.current_pos, "orientation": orientation},
            goal=self.goal,
            ray_directions=self.ray_directions,
            ray_distances=ray_distances,
        )

    def _get_rangefinder_airsim(self):
        """Project multi-channel LiDAR data onto the paper's 5x5 ray grid."""
        distances = np.full(len(self.ray_directions), config.RANGE_MAX, dtype=np.float32)

        lidar_data = None
        lidar_names = (config.PRIMARY_LIDAR_NAME,) + tuple(config.FALLBACK_LIDAR_NAMES)
        for name in lidar_names:
            try:
                lidar_data = self.bridge.client.getLidarData(
                    lidar_name=name,
                    vehicle_name=self.vehicle_name,
                )
                if lidar_data and len(lidar_data.point_cloud) >= 3:
                    break
                lidar_data = None
            except Exception:
                lidar_data = None

        if lidar_data is not None:
            distances = self._project_lidar_to_rays(lidar_data.point_cloud)

        return distances

    def _lidar_to_360(self, point_cloud):
        """Compatibility helper for older single-line LiDAR code paths."""
        bins = np.ones(360, dtype=np.float32) * config.RANGE_MAX
        if len(point_cloud) < 3:
            return bins

        points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
        for pt in points:
            dist = float(np.linalg.norm(pt[:2]))
            if dist < 1e-3 or dist > config.RANGE_MAX:
                continue
            angle = np.degrees(np.arctan2(pt[1], pt[0]))
            idx = int((angle + 180) % 360) % 360
            if dist < bins[idx]:
                bins[idx] = dist
        return bins

    def _project_lidar_to_rays(self, point_cloud):
        """Map AirSim LiDAR point cloud onto the 25 sensing directions used by the paper."""
        distances = np.full(len(self.ray_directions), config.RANGE_MAX, dtype=np.float32)
        points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return distances

        h_step = np.radians(config.RANGE_HFOV) / max(config.RANGE_RAYS_H - 1, 1)
        v_step = np.radians(config.RANGE_VFOV) / max(config.RANGE_RAYS_V - 1, 1)
        half_diag = np.sqrt((0.5 * h_step) ** 2 + (0.5 * v_step) ** 2) * config.RANGE_MATCH_MARGIN
        cosine_threshold = np.cos(half_diag)

        ray_dirs = self.ray_directions.astype(np.float32)
        ray_dirs = ray_dirs / np.maximum(np.linalg.norm(ray_dirs, axis=1, keepdims=True), 1e-6)

        for pt in points:
            dist = float(np.linalg.norm(pt))
            if dist < 1e-3 or dist > config.RANGE_MAX:
                continue

            direction = pt / dist
            dots = ray_dirs @ direction
            best_idx = int(np.argmax(dots))
            if dots[best_idx] >= cosine_threshold and dist < distances[best_idx]:
                distances[best_idx] = dist

        return distances

    def _get_rangefinder_sim(self):
        """Simulate rangefinder distances against known obstacles."""
        distances = np.full(len(self.ray_directions), config.RANGE_MAX, dtype=np.float32)
        origin = self.sim_pos
        world_rays = self._rotate_body_rays_to_world(self.ray_directions, self.sim_yaw)

        for i, direction in enumerate(world_rays):
            for obstacle_pos, obstacle_radius in self.sim_obstacles:
                oc = origin - obstacle_pos
                a = np.dot(direction, direction)
                b = 2 * np.dot(oc, direction)
                c = np.dot(oc, oc) - obstacle_radius ** 2
                discriminant = b * b - 4 * a * c

                if discriminant >= 0:
                    t = (-b - np.sqrt(discriminant)) / (2 * a)
                    if 0 < t < distances[i]:
                        distances[i] = t

        return distances

    def _update_planner_from_airsim(self):
        """Update planner voxel grid from AirSim sensor data."""
        ray_distances = self._get_rangefinder_airsim()
        world_rays = self._rotate_body_rays_to_world(self.ray_directions, self.current_yaw)
        self.planner.update_obstacles(
            self.current_pos, world_rays, ray_distances,
            max_range=config.RANGE_MAX,
        )

    def _update_planner_from_sim(self):
        """Update planner voxel grid from simulation obstacles."""
        self.planner.voxel_grid.clear()
        for obs_pos, obs_radius in self.sim_obstacles:
            self.planner.voxel_grid.mark_occupied_sphere(obs_pos, obs_radius)

    @staticmethod
    def _wrap_angle_rad(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _compute_forward_flight_command(self, error_world, max_speed):
        """
        Convert world-frame target error into a forward-only body-frame command.
        The UAV must turn toward the local target instead of drifting sideways.
        """
        dist = float(np.linalg.norm(error_world))
        if dist <= 0.05:
            return 0.0, 0.0, 0.0, self.current_yaw

        desired_yaw = float(np.arctan2(error_world[1], error_world[0]))
        yaw_delta = self._wrap_angle_rad(desired_yaw - self.current_yaw)
        yaw_rate = float(np.clip(yaw_delta / 0.05, -np.pi, np.pi))

        desired_speed = min(dist * 2.0, max_speed)
        alignment = max(0.0, np.cos(yaw_delta))
        forward_speed = float(desired_speed * alignment)
        vz = float(np.clip(error_world[2] * 2.0, -max_speed, max_speed))
        return forward_speed, 0.0, vz, yaw_rate

    def _integrate_forward_only_sim(self, target_pos, dt=0.05, max_speed=None):
        """
        Apply the same non-holonomic command model in simulation mode:
        yaw toward target, then move forward in the body x direction with no side slip.
        """
        max_speed = max_speed or config.UAV_MAX_SPEED
        error_world = target_pos - self.sim_pos
        vx, _, vz, yaw_rate = self._compute_forward_flight_command(error_world, max_speed)

        self.sim_yaw = self._wrap_angle_rad(self.sim_yaw + yaw_rate * dt)
        world_vel = np.array(
            [
                vx * np.cos(self.sim_yaw),
                vx * np.sin(self.sim_yaw),
                vz,
            ],
            dtype=np.float32,
        )
        next_pos = self.sim_pos + world_vel * dt
        return next_pos, world_vel

    def _execute_trajectory_airsim(self, spline):
        """Execute B-spline trajectory in AirSim using velocity commands."""
        import airsim

        if spline is None:
            return

        duration = min(spline.duration, config.PLANNER_DT * 5)
        n_steps = max(int(duration / 0.05), 1)
        max_speed = config.UAV_MAX_SPEED

        for i in range(n_steps):
            t = i / n_steps * duration
            target_pos, _ = self.planner.get_control_command(t)
            if target_pos is None:
                continue

            error_world = target_pos - self.current_pos
            vx, vy, vz, yaw_rate = self._compute_forward_flight_command(error_world, max_speed)

            try:
                self.bridge.client.moveByVelocityBodyFrameAsync(
                    vx=float(vx),
                    vy=float(vy),
                    vz=float(vz),
                    duration=0.05,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
                    vehicle_name=self.vehicle_name,
                ).join()
            except Exception:
                pass

            state = self.bridge.get_state()
            self.current_pos = state["position"].copy()
            self.current_vel = state["velocity"].copy()
            self.current_pitch, self.current_yaw = self._get_attitude_from_airsim()

    def _execute_straight_airsim(self, target_pos):
        """Move directly toward the local target without trajectory optimization."""
        import airsim

        duration = config.PLANNER_DT * 5
        n_steps = max(int(duration / 0.05), 1)
        max_speed = config.UAV_MAX_SPEED

        for _ in range(n_steps):
            error_world = target_pos - self.current_pos
            vx, vy, vz, yaw_rate = self._compute_forward_flight_command(error_world, max_speed)

            try:
                self.bridge.client.moveByVelocityBodyFrameAsync(
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    duration=0.05,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                    vehicle_name=self.vehicle_name,
                ).join()
            except Exception:
                pass

            state = self.bridge.get_state()
            self.current_pos = state["position"].copy()
            self.current_vel = state["velocity"].copy()
            self.current_pitch, self.current_yaw = self._get_attitude_from_airsim()
            if self._check_collision_airsim():
                return True
        return False

    def _execute_trajectory_sim(self, spline):
        """Execute B-spline trajectory in simulation with forward-only kinematics."""
        if spline is None:
            return

        duration = min(spline.duration, config.PLANNER_DT * 5)
        n_steps = max(int(duration / 0.05), 1)

        for i in range(n_steps):
            t = i / n_steps * duration
            pos, vel = self.planner.get_control_command(t)
            if pos is None:
                continue

            next_pos, world_vel = self._integrate_forward_only_sim(pos, dt=0.05, max_speed=config.UAV_MAX_SPEED)
            for obs_pos, obs_radius in self.sim_obstacles:
                if np.linalg.norm(next_pos - obs_pos) < obs_radius + config.UAV_RADIUS:
                    self.collision_count += 1
                    return

            self.sim_pos = next_pos
            self.sim_vel = world_vel

        self.current_pos = self.sim_pos.copy()
        self.current_vel = self.sim_vel.copy()
        self.current_pitch = 0.0
        self.current_yaw = self.sim_yaw

    def _execute_straight_sim(self, target_pos):
        """Move directly toward the local target in simulation mode with no side slip."""
        duration = config.PLANNER_DT * 5
        n_steps = max(int(duration / 0.05), 1)
        max_speed = config.UAV_MAX_SPEED

        for _ in range(n_steps):
            next_pos, world_vel = self._integrate_forward_only_sim(target_pos, dt=0.05, max_speed=max_speed)
            for obs_pos, obs_radius in self.sim_obstacles:
                if np.linalg.norm(next_pos - obs_pos) < obs_radius + config.UAV_RADIUS:
                    self.collision_count += 1
                    return True

            self.sim_pos = next_pos
            self.sim_vel = world_vel

        self.current_pos = self.sim_pos.copy()
        self.current_vel = self.sim_vel.copy()
        self.current_pitch = 0.0
        self.current_yaw = self.sim_yaw
        return False

    def _check_done(self, dist_to_goal):
        """Check if episode should terminate."""
        if dist_to_goal < config.UAV_ARRIVE_DIST:
            return True, "arrived"

        if self.collision_count >= 3:
            return True, "collision"

        if self.step_count >= config.TRAIN_MAX_STEPS:
            return True, "timeout"

        z = self.current_pos[2]
        altitude = max(-float(z), 0.0)
        if altitude <= 0.3:
            return True, "ground"

        if z > config.UAV_ALTITUDE_MIN or z < -config.UAV_ALTITUDE_MAX * 2:
            return True, "out_of_bounds"

        return False, ""

    def add_sim_obstacle(self, position, radius=1.0):
        """Add an obstacle for simulation mode."""
        self.sim_obstacles.append((np.array(position, dtype=np.float32), float(radius)))

    def clear_sim_obstacles(self):
        """Clear simulation obstacles."""
        self.sim_obstacles.clear()
