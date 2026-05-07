import numpy as np


def cost_smoothness(spline, n_samples=20):
    """
    J_s: Smoothness cost — penalize 4th derivative (snap) of B-spline.
    Approximated by penalizing 2nd derivative (acceleration) differences.
    Returns: (cost, gradient w.r.t. control_points)
    """
    ctrl = spline.control_points
    grad = np.zeros_like(ctrl)
    cost = 0.0

    for i in range(len(ctrl) - 2):
        diff = ctrl[i] - 2 * ctrl[i + 1] + ctrl[i + 2]
        cost += np.dot(diff, diff)
        grad[i] += 2 * diff
        grad[i + 1] += -4 * diff
        grad[i + 2] += 2 * diff

    return cost, grad


def cost_collision(spline, voxel_grid, radius=0.5, n_samples=30):
    """
    J_c: Collision cost — penalize trajectory points inside obstacles.
    Uses the voxel grid for obstacle queries.
    Returns: (cost, gradient w.r.t. control_points)
    """
    duration = spline.duration
    cost = 0.0
    ctrl_grad = np.zeros_like(spline.control_points)

    for k in range(n_samples):
        t = k / n_samples * duration
        pt = spline.eval(t)
        vel = spline.eval_derivative(t, order=1)
        speed = np.linalg.norm(vel)
        if speed < 1e-6:
            continue
        vel_norm = vel / speed

        dist, obstacle_pt = voxel_grid.get_distance_with_obstacle(pt)

        if dist < radius:
            penalty = (radius - dist) ** 2
            cost += penalty
            dir_grad = 2 * (radius - dist) * (pt - obstacle_pt) / (np.linalg.norm(pt - obstacle_pt) + 1e-8)
            db = spline_basis_grad_at_ctrl(t, spline, k_index=None)
            for i in range(len(spline.control_points)):
                bi = _basis_derivative(i, spline.order, t + spline.knots[0], spline.knots)
                ctrl_grad[i] += dir_grad * bi

    return cost, ctrl_grad


def cost_dynamic_feasibility(spline, v_max=2.1, a_max=3.0, n_samples=20):
    """
    J_d: Dynamic feasibility cost — penalize velocity and acceleration exceeding limits.
    Returns: (cost, gradient w.r.t. control_points)
    """
    duration = spline.duration
    cost = 0.0
    ctrl_grad = np.zeros_like(spline.control_points)
    eps = 1e-6

    for k in range(n_samples):
        t = k / n_samples * duration
        vel = spline.eval_derivative(t, order=1)
        acc = spline.eval_derivative(t, order=2)

        # Clamp extreme values
        vel = np.clip(vel, -100, 100)
        acc = np.clip(acc, -1000, 1000)

        v_norm = np.linalg.norm(vel)
        if v_norm > v_max:
            penalty = (v_norm - v_max) ** 2
            cost += penalty
            grad_dir = 2 * (v_norm - v_max) * vel / max(v_norm, eps)
            for i in range(len(spline.control_points)):
                bi = _basis_derivative(i, spline.order, t + spline.knots[0], spline.knots)
                ctrl_grad[i] += grad_dir * bi

        a_norm = np.linalg.norm(acc)
        if a_norm > a_max:
            penalty = (a_norm - a_max) ** 2
            cost += penalty
            grad_dir = 2 * (a_norm - a_max) * acc / max(a_norm, eps)
            for i in range(len(spline.control_points)):
                bi = _basis_second_derivative(i, spline.order, t + spline.knots[0], spline.knots)
                ctrl_grad[i] += grad_dir * bi

    return cost, ctrl_grad


def cost_local_target(spline, target_pos, target_vel, t_end=None,
                     weight_pos=1.0, weight_vel=0.5):
    """
    J_lp: Local target cost — penalize deviation of trajectory end from local target.
    Eq. 18: J_lp = λp * ||p(tlp) - plp||^2 + λv * ||v(tlp) - vlp||^2
    Returns: (cost, gradient w.r.t. control_points)
    """
    if t_end is None:
        t_end = spline.duration

    pos_err = spline.eval(t_end) - target_pos
    vel_err = spline.eval_derivative(t_end, order=1) - target_vel

    cost = weight_pos * np.dot(pos_err, pos_err) + weight_vel * np.dot(vel_err, vel_err)

    ctrl_grad = np.zeros_like(spline.control_points)
    for i in range(len(spline.control_points)):
        t_k = t_end + spline.knots[0]
        bi = _basis_function(i, spline.order, t_k, spline.knots)
        bi_d = _basis_derivative(i, spline.order, t_k, spline.knots)
        ctrl_grad[i] += 2 * weight_pos * pos_err * bi + 2 * weight_vel * vel_err * bi_d

    return cost, ctrl_grad


def _basis_function(i, k, u, knots):
    """De Boor-Cox basis function."""
    if k == 1:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    term1 = ((u - knots[i]) / denom1 * _basis_function(i, k - 1, u, knots)) if denom1 > 1e-10 else 0.0
    term2 = ((knots[i + k] - u) / denom2 * _basis_function(i + 1, k - 1, u, knots)) if denom2 > 1e-10 else 0.0
    return term1 + term2


def _basis_derivative(i, k, u, knots):
    """First derivative of basis function."""
    if k <= 1:
        return 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    term1 = (k - 1) / denom1 * _basis_function(i, k - 1, u, knots) if denom1 > 1e-10 else 0.0
    term2 = (k - 1) / denom2 * _basis_function(i + 1, k - 1, u, knots) if denom2 > 1e-10 else 0.0
    return term1 - term2


def _basis_second_derivative(i, k, u, knots):
    """Second derivative of basis function."""
    if k <= 2:
        return 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    term1 = (k - 1) / denom1 * _basis_derivative(i, k - 1, u, knots) if denom1 > 1e-10 else 0.0
    term2 = (k - 1) / denom2 * _basis_derivative(i + 1, k - 1, u, knots) if denom2 > 1e-10 else 0.0
    return term1 - term2


def spline_basis_grad_at_ctrl(t, spline, k_index=None):
    """Helper: get basis function values at t for gradient computation."""
    t_k = t + spline.knots[0]
    return np.array([_basis_function(i, spline.order, t_k, spline.knots)
                     for i in range(len(spline.control_points))])
