"""
Main EGO-Planner orchestration module.
Integrates B-spline, voxel grid, A*, optimization, and refinement.
"""

import numpy as np
from typing import Optional, Tuple, Dict

from .bspline import BSplineTrajectory
from .voxel_grid import LocalVoxelGrid
from .guiding_path import AStarPlanner
from .optimizer import optimize_trajectory, time_reallocation, curve_fitting


class EGOPlanner:
    """
    EGO-Planner: ESDF-free gradient-based local planner for quadrotors.

    Planning loop:
    1. Generate initial B-spline trajectory toward target
    2. Detect collisions with voxel grid
    3. Build {p, v} obstacle pairs via A* guiding paths
    4. Optimize with L-BFGS to push trajectory out of obstacles
    5. Rebound: repeat until trajectory is collision-free
    6. Time reallocation + curve fitting if dynamical limits violated
    """

    def __init__(self, config):
        """
        Args:
            config: configuration object with EGO parameters
        """
        self.config = config
        self.pb = config.PB
        self.nc = config.NC
        self.init_dt = config.INIT_DT
        self.horizon = config.HORIZON
        self.sf = config.SF
        self.trajectory_radius = config.TRAJECTORY_RADIUS

        # Cost weights
        self.weights = {
            'lambda_s': config.LAMBDA_S,
            'lambda_c': config.LAMBDA_C,
            'lambda_d': config.LAMBDA_D,
            'lambda_f': config.LAMBDA_F,
        }

        # Physical limits
        self.limits = {
            'v_max': config.V_MAX,
            'a_max': config.A_MAX,
            'j_max': config.J_MAX,
        }

        # Voxel grid and A* planner
        self.voxel_grid = LocalVoxelGrid(
            resolution=config.VOXEL_RES,
            half_size=(config.GRID_HALF_X, config.GRID_HALF_Y, config.GRID_HALF_Z)
        )
        self.astar = AStarPlanner(
            voxel_grid=self.voxel_grid,
            neighborhood=config.ASTAR_NEIGHBORHOOD,
            safety_radius=self.trajectory_radius,
            heuristic_weight=config.ASTAR_HEURISTIC_WEIGHT,
        )

        # State tracking
        self.last_trajectory: Optional[BSplineTrajectory] = None
        self.collision_count = 0
        self.replan_count = 0
        self.feasibility_ratio = 1.0

    def plan(self, current_pos: np.ndarray,
             current_vel: np.ndarray,
             target_dir: np.ndarray,
             speed: float,
             goal_pos: Optional[np.ndarray] = None) -> BSplineTrajectory:
        """
        Generate a collision-free trajectory from current state toward target.

        Args:
            current_pos: (3,) current drone position
            current_vel: (3,) current drone velocity
            target_dir: (3,) normalized direction toward target
            speed: desired speed (0-1 normalized, maps to V_MAX)
            goal_pos: (3,) optional goal position for trajectory endpoint

        Returns:
            BSplineTrajectory: optimized trajectory
        """
        # Update voxel grid center
        self.voxel_grid.reset(current_pos)

        # Determine trajectory endpoint
        if goal_pos is not None:
            # Direct toward goal if within horizon
            dist_to_goal = np.linalg.norm(goal_pos - current_pos)
            if dist_to_goal < self.horizon:
                target_point = goal_pos.copy()
            else:
                target_point = current_pos + target_dir * self.horizon
        else:
            target_point = current_pos + target_dir * self.horizon

        max_speed = self.limits['v_max'] * max(0.1, min(1.0, speed))

        # Generate initial B-spline trajectory
        cps = self._generate_initial_trajectory(
            current_pos, current_vel, target_point, max_speed)

        # Optimize: collision avoidance loop
        cps = optimize_trajectory(
            cps, self.init_dt, self.voxel_grid, self.astar,
            self.weights, self.limits, self.sf)

        trajectory = BSplineTrajectory(cps, self.init_dt, self.pb)

        # Check if collision-free
        self.collision_count = self._count_collisions(trajectory)

        # Time reallocation if needed
        new_dt, needs_reallocation = time_reallocation(
            trajectory,
            self.limits['v_max'],
            self.limits['a_max'],
            self.limits['j_max'],
            margin=self.config.TIME_REALLOCATION_MARGIN,
        )

        if needs_reallocation:
            # Curve fitting with new time allocation
            cps_fit = curve_fitting(
                cps, self.init_dt, new_dt,
                self.config.FITTING_A, self.config.FITTING_B,
                current_pos, current_vel,
                self.limits['v_max'],
                self.limits['a_max'],
                self.limits['j_max'],
                self.weights,
                self.limits,
            )
            trajectory = BSplineTrajectory(cps_fit, new_dt, self.pb)

        # Final collision check
        self.feasibility_ratio = self._compute_feasibility_ratio(trajectory)
        self.last_trajectory = trajectory

        return trajectory

    def _generate_initial_trajectory(self, start_pos: np.ndarray,
                                     start_vel: np.ndarray,
                                     target_point: np.ndarray,
                                     speed: float) -> np.ndarray:
        """
        Generate initial B-spline control points as a straight line
        from start to target, respecting velocity boundary condition.
        """
        nc = self.nc
        direction = target_point - start_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
            dist = 1.0
        direction = direction / dist

        # Space control points along the line
        cps = np.zeros((nc, 3), dtype=np.float64)
        spacing = self.init_dt * speed

        for i in range(nc):
            t = i * spacing
            cps[i] = start_pos + direction * min(t, dist)

        # Adjust first control point for velocity boundary condition
        # V0 = (Q1 - Q0) / dt = start_vel => Q0 = Q1 - dt * start_vel
        cps[0] = cps[1] - self.init_dt * start_vel

        return cps

    def _count_collisions(self, trajectory: BSplineTrajectory) -> int:
        """Count number of trajectory samples that collide with obstacles."""
        samples = trajectory.sample_points(50)
        count = 0
        for pt in samples:
            if self.voxel_grid.is_occupied(pt, self.trajectory_radius * 0.5):
                count += 1
        return count

    def _compute_feasibility_ratio(self, trajectory: BSplineTrajectory) -> float:
        """Compute the ratio of feasible control points (0=none, 1=all)."""
        vel_cps = trajectory.velocity_cps
        accel_cps = trajectory.accel_cps
        jerk_cps = trajectory.jerk_cps

        total = len(vel_cps) + len(accel_cps) + len(jerk_cps)
        if total == 0:
            return 1.0

        violations = 0
        violations += np.sum(np.abs(vel_cps) > self.limits['v_max'])
        violations += np.sum(np.abs(accel_cps) > self.limits['a_max'])
        violations += np.sum(np.abs(jerk_cps) > self.limits['j_max'])

        return 1.0 - violations / (total * 3)  # 3 dimensions

    def is_trajectory_safe(self, trajectory: Optional[BSplineTrajectory] = None,
                           radius: Optional[float] = None) -> bool:
        """Check if trajectory is collision-free."""
        if trajectory is None:
            trajectory = self.last_trajectory
        if trajectory is None:
            return False
        if radius is None:
            radius = self.trajectory_radius
        samples = trajectory.sample_points(50)
        return self.voxel_grid.is_collision_free(samples, radius)

    def get_trajectory_segment(self, trajectory: BSplineTrajectory,
                               t_start: float, t_end: float) -> Tuple[np.ndarray, float]:
        """
        Extract a segment of the trajectory for execution.

        Returns:
            (velocity, duration): velocity command and execution duration
        """
        t_end = min(t_end, trajectory.duration)
        if t_start >= t_end:
            pos = trajectory.evaluate(t_start)
            return np.zeros(3), 0.0

        p1 = trajectory.evaluate(t_start)
        p2 = trajectory.evaluate(t_end)
        duration = t_end - t_start

        if duration > 1e-6:
            velocity = (p2 - p1) / duration
        else:
            velocity = np.zeros(3)

        return velocity, duration

    def get_planner_info(self) -> np.ndarray:
        """Return planner status as a 4-dim vector for RL observation."""
        traj_length = 0.0
        if self.last_trajectory is not None:
            samples = self.last_trajectory.sample_points(50)
            if len(samples) > 1:
                diffs = np.diff(samples, axis=0)
                traj_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        return np.array([
            traj_length,
            float(self.collision_count),
            self.feasibility_ratio,
            float(self.replan_count),
        ], dtype=np.float32)

    def reset(self):
        """Reset planner state."""
        self.last_trajectory = None
        self.collision_count = 0
        self.replan_count = 0
        self.feasibility_ratio = 1.0
