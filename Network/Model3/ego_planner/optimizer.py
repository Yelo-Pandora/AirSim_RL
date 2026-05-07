import numpy as np
from .bspline import BSpline
from .cost_functions import (
    cost_smoothness, cost_collision, cost_dynamic_feasibility, cost_local_target
)


class TrajectoryOptimizer:
    """
    Gradient-based B-spline trajectory optimizer (EGO-Planner style).
    Minimizes: J = λ1*Js + λ2*Jc + λ3*Jd + λ4*Jlp
    """

    def __init__(self, config):
        self.opt_iters = config.PLANNER_OPTIM_ITERS
        self.lr = config.PLANNER_LR
        self.w_smooth = config.COST_SMOOTH_WEIGHT
        self.w_collision = config.COST_COLLISION_WEIGHT
        self.w_dynamic = config.COST_DYNAMIC_WEIGHT
        self.w_local_target = config.COST_LOCAL_TARGET_WEIGHT
        self.lp_pos_weight = config.LOCAL_TARGET_POS_WEIGHT
        self.lp_vel_weight = config.LOCAL_TARGET_VEL_WEIGHT
        self.v_max = config.UAV_MAX_SPEED

    def optimize(self, start_pos, start_vel, target_pos, target_vel,
                 voxel_grid, init_ctrl_points=None):
        """
        Optimize a B-spline trajectory from start to target.

        Args:
            start_pos: (3,) start position
            start_vel: (3,) start velocity
            target_pos: (3,) local target position
            target_vel: (3,) desired velocity at target (usually zero or along direction)
            voxel_grid: VoxelGrid with obstacle info
            init_ctrl_points: optional initial control points

        Returns:
            Optimized BSpline trajectory
        """
        n_ctrl = 10
        if init_ctrl_points is not None:
            ctrl = np.array(init_ctrl_points, dtype=np.float64)
        else:
            ctrl = self._init_control_points(start_pos, start_vel, target_pos, n_ctrl)

        spline = BSpline(ctrl, order=4, dt=0.2)

        for iteration in range(self.opt_iters):
            total_grad = np.zeros_like(ctrl)

            # Smoothness cost
            c_s, g_s = cost_smoothness(spline)
            total_grad += self.w_smooth * g_s

            # Collision cost
            c_c, g_c = cost_collision(spline, voxel_grid, radius=0.5, n_samples=30)
            total_grad += self.w_collision * g_c

            # Dynamic feasibility cost
            c_d, g_d = cost_dynamic_feasibility(spline, v_max=self.v_max, n_samples=20)
            total_grad += self.w_dynamic * g_d

            # Local target cost
            c_lp, g_lp = cost_local_target(
                spline, target_pos, target_vel,
                weight_pos=self.lp_pos_weight,
                weight_vel=self.lp_vel_weight
            )
            total_grad += self.w_local_target * g_lp

            # Gradient descent step
            ctrl = ctrl - self.lr * total_grad

            # Handle NaN/inf in gradients
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=100.0, neginf=-100.0)

            # Enforce boundary conditions: first/last control points fixed
            ctrl[0] = start_pos
            ctrl[-1] = target_pos

            # Update spline
            spline.set_control_points(ctrl)

        return spline

    def _init_control_points(self, start, start_vel, target, n_ctrl):
        """Initialize control points as a straight line from start to target."""
        ctrl = np.zeros((n_ctrl, 3))
        direction = target - start
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            ctrl[:] = start
        else:
            unit_dir = direction / dist
            for i in range(n_ctrl):
                frac = i / (n_ctrl - 1)
                ctrl[i] = start + frac * direction
                ctrl[i] += frac * start_vel * 0.5  # slight velocity bias
        return ctrl
