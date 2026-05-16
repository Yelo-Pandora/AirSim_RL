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
        self.a_max = 3.0
        self.bspline_dt = config.BSPLINE_DT

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
        if init_ctrl_points is not None:
            ctrl = np.array(init_ctrl_points, dtype=np.float64)
            n_ctrl = len(ctrl)
        else:
            ctrl = self._init_control_points(start_pos, start_vel, target_pos, n_ctrl=10)

        spline = BSpline(ctrl, order=4, dt=self.bspline_dt)

        for iteration in range(self.opt_iters):
            total_grad = np.zeros_like(ctrl)

            # Smoothness cost
            c_s, g_s = cost_smoothness(spline)
            total_grad += self.w_smooth * g_s

            # Collision cost (fewer samples for speed)
            c_c, g_c = cost_collision(spline, voxel_grid, radius=0.5, n_samples=15)
            total_grad += self.w_collision * g_c

            # Dynamic feasibility cost
            c_d, g_d = cost_dynamic_feasibility(
                spline,
                v_max=self.v_max,
                a_max=self.a_max,
                n_samples=20,
            )
            total_grad += self.w_dynamic * g_d

            # Local target cost
            c_lp, g_lp = cost_local_target(
                spline, target_pos, target_vel,
                weight_pos=self.lp_pos_weight,
                weight_vel=self.lp_vel_weight
            )
            total_grad += self.w_local_target * g_lp

            # Gradient clipping
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm > 10.0:
                total_grad = total_grad / grad_norm * 10.0

            # Gradient descent step
            ctrl = ctrl - self.lr * total_grad

            # Handle NaN/inf in gradients
            ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=100.0, neginf=-100.0)

            # Enforce boundary conditions: first/last control points fixed
            ctrl[0] = start_pos
            ctrl[-1] = target_pos

            # Update spline
            spline.set_control_points(ctrl)

            # EGO-Planner style time reallocation: relax timing when dynamics are violated.
            if (iteration + 1) % 5 == 0:
                max_vel, max_acc = self._estimate_dynamic_extrema(spline)
                if max_vel > self.v_max * 1.05 or max_acc > self.a_max * 1.05:
                    scale = max(max_vel / max(self.v_max, 1e-6), np.sqrt(max_acc / max(self.a_max, 1e-6)))
                    spline.set_dt(spline.dt * min(max(scale, 1.05), 1.5))

        return spline

    def _init_control_points(self, start, start_vel, target, n_ctrl):
        """Initialize control points as a smooth path from start to target."""
        ctrl = np.zeros((n_ctrl, 3))
        direction = target - start
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            ctrl[:] = start
        else:
            unit_dir = direction / dist
            for i in range(n_ctrl):
                # Use smooth interpolation (sigmoid-like) to avoid clustering
                frac = i / (n_ctrl - 1)
                ctrl[i] = start + frac * direction
        return ctrl

    def _estimate_dynamic_extrema(self, spline, n_samples=30):
        max_vel = 0.0
        max_acc = 0.0
        for k in range(n_samples + 1):
            t = k / max(n_samples, 1) * spline.duration
            vel = spline.eval_derivative(t, order=1)
            acc = spline.eval_derivative(t, order=2)
            max_vel = max(max_vel, float(np.linalg.norm(vel)))
            max_acc = max(max_acc, float(np.linalg.norm(acc)))
        return max_vel, max_acc
