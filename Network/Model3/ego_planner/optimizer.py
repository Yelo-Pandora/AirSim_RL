"""
L-BFGS optimizer with time reallocation and anisotropic curve fitting.
Follows EGO-Planner Algorithm 2 (Rebound Planning).
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Dict

from .bspline import BSplineTrajectory
from .voxel_grid import LocalVoxelGrid
from .guiding_path import AStarPlanner
from .cost_functions import compute_total_cost_and_gradient


def _build_obstacle_pairs(cps: np.ndarray, dt: float,
                          voxel_grid: LocalVoxelGrid,
                          astar: AStarPlanner,
                          sf: float = 0.5,
                          known_obs: Optional[List] = None) -> Tuple[List, bool]:
    """
    Detect colliding control points and build {p, v} obstacle pairs.

    For each colliding segment:
    1. Run A* to find collision-free guiding path Gamma
    2. For each control point, find anchor point p and repulsive direction v

    Args:
        cps: (Nc, 3) control points
        dt: time interval
        voxel_grid: local voxel grid
        astar: A* planner
        sf: safety clearance
        known_obs: previously known obstacle pairs (for incremental update)

    Returns:
        (obstacle_pairs, has_collision)
        obstacle_pairs[i] = [(p_j, v_j), ...] for control point i
    """
    nc = len(cps)
    obstacle_pairs = [[] for _ in range(nc)]

    # Restore previous pairs for control points that are now clear
    if known_obs is not None:
        for i in range(nc):
            obstacle_pairs[i] = list(known_obs[i])

    # Find colliding control points
    colliding_indices = []
    for i in range(nc):
        if voxel_grid.is_occupied(cps[i], radius=sf * 0.5):
            colliding_indices.append(i)

    if not colliding_indices:
        return obstacle_pairs, False

    # Find contiguous colliding segments
    segments = []
    if colliding_indices:
        seg_start = colliding_indices[0]
        seg_end = colliding_indices[0]
        for idx in colliding_indices[1:]:
            if idx == seg_end + 1:
                seg_end = idx
            else:
                segments.append((seg_start, seg_end))
                seg_start = idx
                seg_end = idx
        segments.append((seg_start, seg_end))

    # For each colliding segment, find guiding path and {p, v} pairs
    for seg_start, seg_end in segments:
        # Get start and end of colliding segment in world space
        seg_mid = cps[(seg_start + seg_end) // 2]

        # Find nearest free voxel to the colliding point as A* target
        # Use the segment endpoints as A* start/goal, trying to route around
        # We search from the first colliding point to a point beyond the segment
        if seg_start > 0:
            start_pt = cps[seg_start - 1]
        else:
            start_pt = seg_mid

        if seg_end < nc - 1:
            end_pt = cps[seg_end + 1]
        else:
            end_pt = seg_mid + (seg_mid - cps[max(0, seg_start - 1)])

        path = astar.plan(start_pt, end_pt)
        if path is None or len(path) < 2:
            # Fallback: use a simple direction away from obstacles
            for i in range(seg_start, seg_end + 1):
                v = np.array([1.0, 0.0, 0.0])  # default repulsive direction
                p = cps[i] - v * sf
                obstacle_pairs[i].append((p, v))
            continue

        # For each control point in the segment, find {p, v} pair
        for i in range(seg_start, seg_end + 1):
            # Compute tangent direction Ri = (Qi+1 - Qi-1) / (2*dt)
            if i > 0 and i < nc - 1:
                tangent = (cps[i + 1] - cps[i - 1]) / (2.0 * dt)
            elif i > 0:
                tangent = cps[i] - cps[i - 1]
            elif i < nc - 1:
                tangent = cps[i + 1] - cps[i]
            else:
                tangent = np.array([1.0, 0.0, 0.0])

            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                tangent = tangent / tangent_norm

            # Find the closest point on guiding path to cps[i]
            path_arr = np.array(path)
            dists = np.linalg.norm(path_arr - cps[i], axis=1)
            closest_idx = np.argmin(dists)
            p = path_arr[closest_idx]

            # Repulsive direction: from control point toward the guiding path point
            v = p - cps[i]
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-6:
                v = v / v_norm
            else:
                v = np.array([1.0, 0.0, 0.0])

            obstacle_pairs[i].append((p, v))

    return obstacle_pairs, True


def optimize_trajectory(cps_init: np.ndarray, dt: float,
                        voxel_grid: LocalVoxelGrid,
                        astar: AStarPlanner,
                        weights: Dict[str, float],
                        limits: Dict[str, float],
                        sf: float = 0.5,
                        max_iterations: int = 30) -> np.ndarray:
    """
    Optimize trajectory control points to avoid collisions.
    Implements EGO-Planner Algorithm 2 (Rebound Planning loop).

    Args:
        cps_init: (Nc, 3) initial control points
        dt: time interval
        voxel_grid: local voxel grid
        astar: A* planner
        weights: cost weights dict
        limits: physical limits dict
        sf: safety clearance
        max_iterations: maximum optimization iterations

    Returns:
        (Nc, 3) optimized control points
    """
    nc = len(cps_init)
    cps = cps_init.copy()
    known_obs = None

    for iteration in range(max_iterations):
        # Build obstacle pairs
        obstacle_pairs, has_collision = _build_obstacle_pairs(
            cps, dt, voxel_grid, astar, sf, known_obs)

        if not has_collision:
            break

        # Store obstacle pairs for next iteration (incremental)
        known_obs = obstacle_pairs

        # Define objective function for scipy
        def objective(q_flat):
            q = q_flat.reshape(nc, 3)
            cost, _ = compute_total_cost_and_gradient(
                q, dt, obstacle_pairs, weights, limits, sf)
            return cost

        def gradient(q_flat):
            q = q_flat.reshape(nc, 3)
            _, grad = compute_total_cost_and_gradient(
                q, dt, obstacle_pairs, weights, limits, sf)
            return grad.ravel()

        # Optimize with L-BFGS-B
        q0 = cps.ravel()
        result = minimize(
            objective, q0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 20,
                'ftol': 1e-8,
                'gtol': 1e-6,
            }
        )

        cps = result.x.reshape(nc, 3)

    return cps


def time_reallocation(trajectory: BSplineTrajectory,
                      v_max: float, a_max: float, j_max: float,
                      margin: float = 1.2) -> Tuple[float, bool]:
    """
    Compute new time interval based on limit exceeding ratio.
    Paper formula (14)(15): re = max{|Vi/vm|, sqrt(|Aj/am|), cbrt(|Jk/jm|), 1}

    Args:
        trajectory: current B-spline trajectory
        v_max, a_max, j_max: derivative limits
        margin: safety margin multiplier

    Returns:
        (new_dt, needs_reallocation)
    """
    vel_cps = trajectory.velocity_cps  # (Nc-1, 3)
    accel_cps = trajectory.accel_cps  # (Nc-2, 3)
    jerk_cps = trajectory.jerk_cps  # (Nc-3, 3)

    re = 1.0

    if len(vel_cps) > 0:
        v_ratios = np.abs(vel_cps) / v_max
        re = max(re, np.max(v_ratios))

    if len(accel_cps) > 0:
        a_ratios = np.sqrt(np.abs(accel_cps) / a_max)
        re = max(re, np.max(a_ratios))

    if len(jerk_cps) > 0:
        j_ratios = np.cbrt(np.abs(jerk_cps) / j_max)
        re = max(re, np.max(j_ratios))

    re *= margin
    new_dt = re * trajectory.dt

    needs_reallocation = re > 1.01
    return new_dt, needs_reallocation


def curve_fitting(cps_safe: np.ndarray, dt_safe: float,
                  dt_new: float, a_axis: float, b_axis: float,
                  boundary_pos: np.ndarray, boundary_vel: np.ndarray,
                  v_max: float, a_max: float, j_max: float,
                  weights: Dict[str, float],
                  limits: Dict[str, float],
                  max_iterations: int = 30) -> np.ndarray:
    """
    Anisotropic curve fitting: generate new trajectory with dt_new
    that fits the safe trajectory while adjusting smoothness and feasibility.

    Paper formula (17)(18):
    Jf = integral[da^2/a^2 + dr^2/b^2] d_alpha

    Args:
        cps_safe: (Nc, 3) safe trajectory control points
        dt_safe: time interval of safe trajectory
        dt_new: new (larger) time interval
        a_axis: semi-major axis (axial direction penalty)
        b_axis: semi-minor axis (radial direction penalty)
        boundary_pos: (3,) boundary position constraint
        boundary_vel: (3,) boundary velocity constraint
        v_max, a_max, j_max: physical limits
        weights: cost weights
        limits: physical limits dict
        max_iterations: max optimization iterations

    Returns:
        (Nc, 3) fitted control points
    """
    nc = len(cps_safe)

    # Initial guess: rescale from safe trajectory
    # With uniform B-spline, we can use the same control points as initial guess
    cps_init = cps_safe.copy()

    # Create safe trajectory for reference
    safe_traj = BSplineTrajectory(cps_safe, dt_safe)

    # Sample reference trajectory
    n_samples = nc * 4
    t_ref = np.linspace(0, safe_traj.duration, n_samples)
    ref_points = safe_traj.evaluate_many(t_ref)
    ref_velocities = np.array([
        safe_traj.evaluate(t + 0.01) - safe_traj.evaluate(t - 0.01)
        for t in t_ref
    ]) / 0.02

    # Normalize velocities for tangent direction
    for i in range(n_samples):
        norm = np.linalg.norm(ref_velocities[i])
        if norm > 1e-6:
            ref_velocities[i] /= norm

    def fitting_cost_and_grad(q_flat):
        q = q_flat.reshape(nc, 3)
        new_traj = BSplineTrajectory(q, dt_new)

        # Sample new trajectory
        new_t = np.linspace(0, new_traj.duration, n_samples)
        new_points = new_traj.evaluate_many(new_t)

        # Anisotropic fitting cost
        cost = 0.0
        grad_fitting = np.zeros_like(q)

        for i in range(n_samples):
            diff = new_points[i] - ref_points[i % len(ref_points)]
            tangent = ref_velocities[i % len(ref_velocities)]

            # Axial displacement (along tangent)
            da = np.dot(diff, tangent)
            # Radial displacement (perpendicular to tangent)
            dr_sq = np.dot(diff, diff) - da * da
            dr_sq = max(0.0, dr_sq)

            cost += (da * da) / (a_axis * a_axis) + dr_sq / (b_axis * b_axis)

            # Gradient (approximate via finite differences for fitting term)
            # d(da^2)/dQ = 2*da * (dtangent/dQ) -> simplified:
            # d(diff)/dQ_k = basis_k * I
            # We use numerical gradient for the fitting term

        # Add smoothness and feasibility costs (analytical)
        smooth_cost, smooth_grad = _simple_smoothness_cost(q, dt_new)
        feas_cost, feas_grad = _simple_feasibility_cost(
            q, dt_new, v_max, a_max, j_max)

        total_cost = weights.get('lambda_f', 20.0) * cost + \
                     weights.get('lambda_s', 1.0) * smooth_cost + \
                     weights.get('lambda_d', 5.0) * feas_cost

        # Numerical gradient for fitting term
        eps = 1e-5
        fit_grad = np.zeros_like(q)
        for k in range(nc):
            for dim in range(3):
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[k, dim] += eps
                q_minus[k, dim] -= eps

                t_plus = BSplineTrajectory(q_plus, dt_new)
                t_minus = BSplineTrajectory(q_minus, dt_new)

                pts_plus = t_plus.evaluate_many(new_t)
                pts_minus = t_minus.evaluate_many(new_t)

                grad_val = 0.0
                for i in range(n_samples):
                    diff_p = pts_plus[i] - ref_points[i % len(ref_points)]
                    diff_m = pts_minus[i] - ref_points[i % len(ref_points)]
                    tang = ref_velocities[i % len(ref_velocities)]

                    da_p = np.dot(diff_p, tang)
                    da_m = np.dot(diff_m, tang)
                    dr_p = max(0.0, np.dot(diff_p, diff_p) - da_p * da_p)
                    dr_m = max(0.0, np.dot(diff_m, diff_m) - da_m * da_m)

                    cost_p = (da_p * da_p) / (a_axis * a_axis) + dr_p / (b_axis * b_axis)
                    cost_m = (da_m * da_m) / (a_axis * a_axis) + dr_m / (b_axis * b_axis)
                    grad_val += (cost_p - cost_m) / (2 * eps)

                fit_grad[k, dim] = grad_val

        total_grad = weights.get('lambda_f', 20.0) * fit_grad + \
                     weights.get('lambda_s', 1.0) * smooth_grad + \
                     weights.get('lambda_d', 5.0) * feas_grad

        return total_cost, total_grad

    q0 = cps_init.ravel()
    result = minimize(
        lambda q: fitting_cost_and_grad(q)[0],
        q0,
        jac=lambda q: fitting_cost_and_grad(q)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iterations, 'ftol': 1e-6, 'gtol': 1e-5}
    )

    return result.x.reshape(nc, 3)


def _simple_smoothness_cost(cps: np.ndarray, dt: float):
    """Simplified smoothness cost for curve fitting (acceleration only)."""
    nc = len(cps)
    vel_cps = (cps[1:] - cps[:-1]) / dt
    accel_cps = (vel_cps[1:] - vel_cps[:-1]) / dt
    cost = np.sum(accel_cps ** 2)

    grad = np.zeros_like(cps)
    inv_dt2 = 1.0 / (dt * dt)
    for i in range(nc - 2):
        ai = accel_cps[i]
        g = 2.0 * ai * inv_dt2
        grad[i] += g
        grad[i + 1] -= 2.0 * g
        grad[i + 2] += g
    return cost, grad


def _simple_feasibility_cost(cps: np.ndarray, dt: float,
                             v_max: float, a_max: float, j_max: float):
    """Simplified feasibility cost for curve fitting (quadratic penalty)."""
    nc = len(cps)
    vel_cps = (cps[1:] - cps[:-1]) / dt
    accel_cps = (vel_cps[1:] - vel_cps[:-1]) / dt
    jerk_cps = (accel_cps[1:] - accel_cps[:-1]) / dt

    cost = 0.0
    grad = np.zeros_like(cps)

    derivs = [
        (vel_cps, v_max, 1),
        (accel_cps, a_max, 2),
        (jerk_cps, j_max, 3),
    ]

    for deriv, limit, order in derivs:
        for i in range(len(deriv)):
            for dim in range(3):
                cr = deriv[i, dim]
                excess = abs(cr) - limit
                if excess > 0:
                    cost += excess ** 2
                    sign = 1.0 if cr > 0 else -1.0

                    # Back-propagate gradient
                    if order == 1:
                        inv = 2.0 * excess * sign / dt
                        grad[i, dim] -= inv
                        grad[i + 1, dim] += inv
                    elif order == 2:
                        inv = 2.0 * excess * sign / (dt * dt)
                        grad[i, dim] += inv
                        grad[i + 1, dim] -= 2.0 * inv
                        grad[i + 2, dim] += inv
                    elif order == 3:
                        inv = 2.0 * excess * sign / (dt * dt * dt)
                        grad[i, dim] -= inv
                        grad[i + 1, dim] += 3.0 * inv
                        grad[i + 2, dim] -= 3.0 * inv
                        grad[i + 3, dim] += inv

    return cost, grad
