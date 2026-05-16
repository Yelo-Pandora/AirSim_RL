import sys
import os
import numpy as np

# Add Model2 to path for AirSimBridge reuse.
# __file__ = .../Network/Model3/uav_env/uav_env.py
# We need to go up 3 levels to Network/, then into Model2/
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
    Action: continuous 6D [x, y, z, φ, θ, ψ] local target
    Reward: distance + obstacle penalty
    """

    def __init__(self):
        # Initialize AirSim bridge
        try:
            from airsim_client.airsim_bridge import AirSimBridge
            self.bridge = AirSimBridge()
            self.use_airsim = True
            # Detect vehicle name (same as Model1)
            try:
                vehicles = self.bridge.client.listVehicles()
                self.vehicle_name = vehicles[0] if vehicles else "Drone1"
            except Exception:
                self.vehicle_name = "Drone1"
            print(f"Detected vehicle: '{self.vehicle_name}'")
        except Exception as e:
            print(f"[WARN] AirSim not available, using simulation mode: {e}")
            self.bridge = None
            self.use_airsim = False
            self.vehicle_name = None
            self._init_sim()

        self.planner = EGOPlanner(config)
        self.ray_directions = get_ray_directions(
            hfov=config.RANGE_HFOV, vfov=config.RANGE_VFOV,
            rays_h=config.RANGE_RAYS_H, rays_v=config.RANGE_RAYS_V,
        )

        self.goal = None
        self.prev_pos = None
        self.current_pos = None
        self.current_vel = None
        self.current_yaw = 0.0
        self.step_count = 0
        self.collision_count = 0
        self.trajectory = []
        self.sim_obstacles = []  # Always initialized, used in both modes
        self.prev_yaw = 0.0  # For computing yaw rate
        self.prev_action = None

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
        self.sim_pos = np.array([0.0, 0.0, -5.0])
        self.sim_vel = np.array([1.0, 0.0, 0.0])
        self.sim_yaw = 0.0
        self.sim_obstacles = []

    def _get_yaw_from_airsim(self):
        """Extract yaw angle from AirSim client quaternion."""
        import airsim
        state = self.bridge.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        yaw = np.arctan2(
            2 * (q.w_val * q.z_val + q.x_val * q.y_val),
            1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        )
        return yaw

    def _check_collision_airsim(self):
        """Check if UAV has collided."""
        try:
            return self.bridge.client.simGetCollisionInfo().has_collided
        except Exception:
            return False

    def reset(self, goal=None):
        """Reset environment for new episode."""
        if self.use_airsim:
            self.bridge.client.reset()
            self.bridge.client.enableApiControl(True)
            self.bridge.client.armDisarm(True)
            self.bridge.takeoff()
            state = self.bridge.get_state()
            self.current_pos = state["position"].copy()
            self.current_vel = state["velocity"].copy()
            self.current_yaw = self._get_yaw_from_airsim()
        else:
            self.sim_pos = np.array([0.0, 0.0, -5.0])
            self.sim_vel = np.array([1.0, 0.0, 0.0])
            self.sim_yaw = 0.0
            self.current_pos = self.sim_pos.copy()
            self.current_vel = self.sim_vel.copy()
            self.current_yaw = self.sim_yaw

        if goal is None:
            self.goal = np.array([
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15),
                np.random.uniform(config.UAV_ALTITUDE_MAX * -0.8, config.UAV_ALTITUDE_MAX * -0.2),
            ])
        else:
            self.goal = np.array(goal)

        self.prev_pos = self.current_pos.copy()
        self.prev_action = None
        self.step_count = 0
        self.collision_count = 0
        self.trajectory = [self.current_pos.copy()]
        self.prev_yaw = self.current_yaw

        print(f"[RESET] Goal: {self.goal}, Start: {self.current_pos}")
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

        # Convert body-frame action to world-frame local target
        world_target, target_angles = local_target_to_world(
            self.current_pos, self.current_yaw, action
        )

        # Update planner's obstacle map
        if self.use_airsim:
            self._update_planner_from_airsim()
        else:
            self._update_planner_from_sim()

        # Paper Eq. 7 uses d_min from the decision-making local target to the
        # nearest obstacle, not from the UAV position after executing the plan.
        min_dist, _ = self.planner.voxel_grid.get_distance_with_obstacle(world_target)
        if min_dist is None:
            min_dist = config.RANGE_MAX

        # Plan trajectory with EGO-Planner
        spline = self.planner.plan(
            current_pos=self.current_pos,
            current_vel=self.current_vel,
            local_target=np.concatenate([world_target, target_angles]),
        )

        # Check for collisions in planned trajectory
        has_collision, _ = self.planner.check_collision(spline)
        if has_collision:
            self.collision_count += 1

        # Execute trajectory segment
        if self.use_airsim:
            self._execute_trajectory_airsim(spline)
        else:
            self._execute_trajectory_sim(spline)

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
        }

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation vector."""
        if self.use_airsim:
            ray_distances = self._get_rangefinder_airsim()
            orientation = np.array([0.0, 0.0, self.current_yaw])
        else:
            ray_distances = self._get_rangefinder_sim()
            orientation = np.array([0.0, 0.0, self.sim_yaw])

        return build_observation(
            state={"position": self.current_pos, "orientation": orientation},
            goal=self.goal,
            ray_directions=self.ray_directions,
            ray_distances=ray_distances,
        )

    def _get_rangefinder_airsim(self):
        """Get 25 rangefinder distances using LiDAR + distance sensors from settings.json."""
        import airsim

        distances = np.full(len(self.ray_directions), config.RANGE_MAX)

        # === LiDAR: horizontal 360° single-line ===
        lidar_data = None
        for name in ["Lidar1", "LLidar1"]:
            try:
                lidar_data = self.bridge.client.getLidarData(
                    lidar_name=name, vehicle_name=self.vehicle_name
                )
                if lidar_data and len(lidar_data.point_cloud) >= 3:
                    break
                lidar_data = None
            except Exception:
                lidar_data = None

        if lidar_data is not None:
            # Build 360-bin horizontal distance array (same as Model1)
            lidar_360 = self._lidar_to_360(lidar_data.point_cloud)
            # Sample 5 horizontal angles for the center row of our 5x5 grid
            h_angles = np.linspace(
                -np.radians(config.RANGE_HFOV / 2),
                np.radians(config.RANGE_HFOV / 2),
                config.RANGE_RAYS_H,
            )
            # LiDAR angle 0 = forward, maps to index ~180 (or 0 depending on convention)
            for i, h_angle in enumerate(h_angles):
                angle_deg = np.degrees(h_angle)
                idx = int((angle_deg + 180) % 360) % 360
                center_v = config.RANGE_RAYS_V // 2
                ray_idx = i * config.RANGE_RAYS_V + center_v
                if lidar_360[idx] < config.RANGE_MAX:
                    distances[ray_idx] = lidar_360[idx]

        # === Fill non-center rows with max range (single-line LiDAR can't provide vertical) ===
        # This is a limitation of the current sensor setup.
        # For full 5x5 coverage, add a multi-channel LiDAR to settings.json.

        return distances

    def _lidar_to_360(self, point_cloud):
        """Convert LiDAR point cloud to 360-bin horizontal distance array.
        Same approach as Model1's lidar_points_to_360.
        """
        bins = np.ones(360, dtype=np.float32) * config.RANGE_MAX
        if len(point_cloud) < 3:
            return bins

        # Point cloud is flat [x1,y1,z1, x2,y2,z2, ...]
        points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
        for pt in points:
            angle = np.degrees(np.arctan2(pt[1], pt[0]))
            idx = int((angle + 180) % 360) % 360
            dist = np.linalg.norm(pt[:2])
            if dist < bins[idx]:
                bins[idx] = dist
        return bins

    def _get_rangefinder_sim(self):
        """Simulate rangefinder distances against known obstacles."""
        distances = np.full(len(self.ray_directions), config.RANGE_MAX)
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
            target_pos, target_vel = self.planner.get_control_command(t)
            if target_pos is None:
                continue

            # Compute velocity toward target position
            error = target_pos - self.current_pos
            dist = np.linalg.norm(error)
            if dist > 0.05:
                vel_cmd = error / dist * min(dist * 2.0, max_speed)
            else:
                vel_cmd = np.zeros(3)

            # Compute target yaw from movement direction so sensors face travel direction
            if np.linalg.norm(vel_cmd[:2]) > 0.1:
                target_yaw = np.arctan2(vel_cmd[1], vel_cmd[0])
            else:
                target_yaw = self.current_yaw

            # Compute yaw rate for smooth rotation
            yaw_delta = target_yaw - self.current_yaw
            while yaw_delta > np.pi:
                yaw_delta -= 2 * np.pi
            while yaw_delta < -np.pi:
                yaw_delta += 2 * np.pi
            yaw_rate = yaw_delta / 0.05
            yaw_rate = np.clip(yaw_rate, -np.pi, np.pi)

            # The spline and AirSim state are both in world/NED coordinates, so
            # execute world-frame velocities and only use yaw_mode for heading.
            try:
                self.bridge.client.moveByVelocityAsync(
                    vx=float(vel_cmd[0]), vy=float(vel_cmd[1]),
                    vz=float(vel_cmd[2]), duration=0.05,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
                ).join()
            except Exception:
                pass

        # Update state from AirSim
        state = self.bridge.get_state()
        self.current_pos = state["position"].copy()
        self.current_vel = state["velocity"].copy()
        self.current_yaw = self._get_yaw_from_airsim()

    def _execute_trajectory_sim(self, spline):
        """Execute B-spline trajectory in simple simulation."""
        if spline is None:
            return

        duration = min(spline.duration, config.PLANNER_DT * 5)
        n_steps = max(int(duration / 0.05), 1)

        for i in range(n_steps):
            t = i / n_steps * duration
            pos, vel = self.planner.get_control_command(t)
            if pos is not None:
                for obs_pos, obs_radius in self.sim_obstacles:
                    if np.linalg.norm(pos - obs_pos) < obs_radius + config.UAV_RADIUS:
                        self.collision_count += 1
                        return

                self.sim_pos = pos.copy()
                self.sim_vel = vel.copy() if vel is not None else np.zeros(3)
                if np.linalg.norm(self.sim_vel[:2]) > 1e-3:
                    self.sim_yaw = np.arctan2(self.sim_vel[1], self.sim_vel[0])

        self.current_pos = self.sim_pos.copy()
        self.current_vel = self.sim_vel.copy()
        self.current_yaw = self.sim_yaw

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
        self.sim_obstacles.append((np.array(position), radius))

    def clear_sim_obstacles(self):
        """Clear simulation obstacles."""
        self.sim_obstacles.clear()
