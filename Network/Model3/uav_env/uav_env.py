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
        except Exception as e:
            print(f"[WARN] AirSim not available, using simulation mode: {e}")
            self.bridge = None
            self.use_airsim = False
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
        self.step_count = 0
        self.collision_count = 0
        self.trajectory = [self.current_pos.copy()]

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
        action = clip_action(action)

        # Convert body-frame action to world-frame local target
        world_target, target_angles = local_target_to_world(
            self.current_pos, self.current_yaw, action
        )

        # Update planner's obstacle map
        if self.use_airsim:
            self._update_planner_from_airsim()
        else:
            self._update_planner_from_sim()

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

        # Compute reward
        min_dist, _ = self.planner.voxel_grid.get_distance_with_obstacle(self.current_pos)
        if min_dist is None:
            min_dist = config.RANGE_MAX

        reward = compute_reward(
            self.current_pos, self.prev_pos, self.goal,
            min_dist_to_obstacle=min_dist,
            sigma=config.REWARD_SIGMA,
            beta=config.REWARD_BETA,
        )

        # Bonus for reaching goal
        dist_to_goal = np.linalg.norm(self.current_pos - self.goal)
        if dist_to_goal < config.UAV_ARRIVE_DIST:
            reward += 100.0

        self.prev_pos = self.current_pos.copy()
        self.step_count += 1
        self.trajectory.append(self.current_pos.copy())

        # Check done conditions
        done, reason = self._check_done(dist_to_goal)

        info = {
            "collision": has_collision,
            "collision_count": self.collision_count,
            "dist_to_goal": dist_to_goal,
            "reason": reason,
            "step": self.step_count,
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
        """Get rangefinder distances via AirSim depth image + distance sensor fallback."""
        import airsim

        distances = np.full(len(self.ray_directions), config.RANGE_MAX)

        # Try depth image first — extract distances in ray directions
        try:
            responses = self.bridge.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, True),
            ])
            for resp in responses:
                if resp.height > 0 and resp.width > 0:
                    # DepthPerspective returns raw float depth in meters
                    depth_1d = np.frombuffer(resp.image_data_float, dtype=np.float32)
                    depth_2d = depth_1d.reshape(resp.height, resp.width)
                    distances = self._extract_ray_distances_from_depth(depth_2d, self.ray_directions)
                    return distances
        except Exception as e:
            print(f"[WARN] Depth image failed: {e}")

        # Fallback: try distance sensor if configured in settings.json
        try:
            ds = self.bridge.client.getDistanceSensorData()
            if ds.distance < 1e5:  # Valid reading
                distances[:] = ds.distance
        except Exception:
            print(f"[WARN] Distance sensor not configured in settings.json")
            print("[WARN] Using max range for all rays — add a Distance sensor to settings.json")

        return distances

    def _extract_ray_distances_from_depth(self, depth_image, ray_directions):
        """
        Extract distance in each ray direction from a depth image.
        depth_image: (H, W) array in meters
        ray_directions: (N, 3) array in camera frame
        """
        import airsim

        h, w = depth_image.shape
        distances = np.full(len(ray_directions), config.RANGE_MAX)

        # Get camera info
        try:
            camera_info = self.bridge.client.simGetCameraInfo(
                "0", vehicle_name="", image_type=airsim.ImageType.Scene
            )
            fx = camera_info.proj_mat[0]
            fy = camera_info.proj_mat[5]
            cx = w / 2.0
            cy = h / 2.0
        except Exception:
            # Default FOV ~90 degrees
            fx = fy = w / 2.0
            cx = cy = w / 2.0

        # Sample depth along each ray direction
        for i, direction in enumerate(ray_directions):
            # Project ray direction to pixel coordinates
            # direction is in body frame, assume camera points forward (x-axis)
            if direction[0] <= 0:
                continue  # Ray pointing backward, skip
            pixel_x = int(cx + (direction[1] / direction[0]) * fx)
            pixel_y = int(cy + (direction[2] / direction[0]) * fy)
            if 0 <= pixel_x < w and 0 <= pixel_y < h:
                d = depth_image[pixel_y, pixel_x]
                if 0 < d < config.RANGE_MAX:
                    distances[i] = d

        return distances

    def _get_rangefinder_sim(self):
        """Simulate rangefinder distances against known obstacles."""
        distances = np.full(len(self.ray_directions), config.RANGE_MAX)
        origin = self.sim_pos

        for i, direction in enumerate(self.ray_directions):
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
        self.planner.update_obstacles(
            self.current_pos, self.ray_directions, ray_distances,
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

            # Use body-frame velocity control
            try:
                self.bridge.client.moveByVelocityBodyFrameAsync(
                    vx=float(vel_cmd[0]), vy=float(vel_cmd[1]),
                    vz=float(vel_cmd[2]), duration=0.05,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                )
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
        if z > config.UAV_ALTITUDE_MIN or z < -config.UAV_ALTITUDE_MAX * 2:
            return True, "out_of_bounds"

        return False, ""

    def add_sim_obstacle(self, position, radius=1.0):
        """Add an obstacle for simulation mode."""
        self.sim_obstacles.append((np.array(position), radius))

    def clear_sim_obstacles(self):
        """Clear simulation obstacles."""
        self.sim_obstacles.clear()
