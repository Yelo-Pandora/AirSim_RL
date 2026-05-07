"""
Cost functions and gradients for EGO-Planner trajectory optimization.
Implements smoothness (Js), collision (Jc), and feasibility (Jd) terms
with analytical gradients, following the EGO-Planner paper formulas.
"""

import numpy as np
from typing import List, Tuple, Dict


def _compute_accel_jerk_cps(cps: np.ndarray, dt: float):
    """Compute acceleration and jerk control points from position control points."""
    vel_cps = (cps[1:] - cps[:-1]) / dt
    accel_cps = (vel_cps[1:] - vel_cps[:-1]) / dt
    jerk_cps = (accel_cps[1:] - accel_cps[:-1]) / dt
    return accel_cps, jerk_cps


def smoothness_cost(cps: np.ndarray, dt: float) -> Tuple[float, np.ndarray]:
    """
    Smoothness penalty: Js = sum(||Ai||^2) + sum(||Ji||^2)
    Minimizes acceleration and jerk along the trajectory.

    Args:
        cps: (Nc, 3) control points
        dt: time interval

    Returns:
        (cost, gradient) where gradient has shape (Nc, 3)
    """
    nc = len(cps)
    grad = np.zeros_like(cps)

    # Velocity and acceleration control points
    vel_cps = (cps[1:] - cps[:-1]) / dt  # (nc-1, 3)
    accel_cps = (vel_cps[1:] - vel_cps[:-1]) / dt  # (nc-2, 3)
    jerk_cps = (accel_cps[1:] - accel_cps[:-1]) / dt  # (nc-3, 3)

    # Cost: sum of squared norms
    cost = np.sum(accel_cps ** 2) + np.sum(jerk_cps ** 2)

    # Gradient of Js w.r.t. each control point
    # Acceleration term: Ai = (Vi+1 - Vi) / dt = (Qi+2 - 2*Qi+1 + Qi) / dt^2
    # d||Ai||^2 / dQi = 2 * Ai * dAi/dQi
    # dAi/dQi = [1, -2, 1] / dt^2 at positions [i, i+1, i+2]

    inv_dt2 = 1.0 / (dt * dt)
    inv_dt4 = inv_dt2 * inv_dt2  # for jerk

    # Acceleration gradient contribution
    # Ai = (Qi+2 - 2*Qi+1 + Qi) * inv_dt2
    # d||Ai||^2/dQi = 2*Ai * inv_dt2  (at Qi)
    # d||Ai||^2/dQi+1 = 2*Ai * (-2*inv_dt2)  (at Qi+1)
    # d||Ai||^2/dQi+2 = 2*Ai * inv_dt2  (at Qi+2)
    for i in range(nc - 2):
        ai = accel_cps[i]  # (3,)
        g_coeff = 2.0 * ai * inv_dt2
        if i < nc:
            grad[i] += g_coeff
        if i + 1 < nc:
            grad[i + 1] += -2.0 * g_coeff
        if i + 2 < nc:
            grad[i + 2] += g_coeff

    # Jerk gradient contribution
    # Ji = (Ai+1 - Ai) / dt = (Qi+3 - 3*Qi+2 + 3*Qi+1 - Qi) / dt^3
    # d||Ji||^2/dQi = 2*Ji * dJi/dQi
    inv_dt3 = inv_dt2 / dt
    for i in range(nc - 3):
        ji = jerk_cps[i]  # (3,)
        g_coeff = 2.0 * ji * inv_dt3
        if i < nc:
            grad[i] += -g_coeff
        if i + 1 < nc:
            grad[i + 1] += 3.0 * g_coeff
        if i + 2 < nc:
            grad[i + 2] += -3.0 * g_coeff
        if i + 3 < nc:
            grad[i + 3] += g_coeff

    return cost, grad


def collision_cost(cps: np.ndarray,
                   obstacle_pairs: List[List[Tuple[np.ndarray, np.ndarray]]],
                   sf: float) -> Tuple[float, np.ndarray]:
    """
    Collision penalty using repulsive forces from obstacle pairs.

    For each control point Qi with {p, v} pairs:
      dij = (Qi - pij) . vij   (distance to obstacle)
      cij = sf - dij
      jc(i,j) = piecewise(cij)

    Args:
        cps: (Nc, 3) control points
        obstacle_pairs: list of lists, obstacle_pairs[i] contains [(p_j, v_j), ...] for Qi
        sf: safety clearance

    Returns:
        (cost, gradient) where gradient has shape (Nc, 3)
    """
    nc = len(cps)
    cost = 0.0
    grad = np.zeros_like(cps)

    for i in range(nc):
        if i >= len(obstacle_pairs) or not obstacle_pairs[i]:
            continue

        for p_j, v_j in obstacle_pairs[i]:
            dij = np.dot(cps[i] - p_j, v_j)
            cij = sf - dij

            if cij <= 0:
                # No penalty
                continue
            elif cij <= sf:
                # Cubic region: cij^3
                cost += cij ** 3
                # Gradient: dJc/dQi = -3*cij^2 * vij
                grad[i] += -3.0 * (cij ** 2) * v_j
            else:
                # Quadratic region: 3*sf*cij^2 - 3*sf^2*cij + sf^3
                cost += 3.0 * sf * cij * cij - 3.0 * sf * sf * cij + sf ** 3
                # Gradient: dJc/dQi = -(6*sf*cij - 3*sf^2) * vij
                grad[i] += -(6.0 * sf * cij - 3.0 * sf * sf) * v_j

    return cost, grad


def _f_penalty_and_grad(cr: float, cm: float, cj_split: float,
                        lam: float) -> Tuple[float, float]:
    """
    Piecewise feasibility penalty F and its derivative for a single dimension.

    Regions:
      cr <= -cj:     a1*cr^2 + b1*cr + c1
      -cj < cr < -lc: (-lc - cr)^3
      -lc <= cr <= lc: 0
      lc < cr < cj:  (cr - lc)^3
      cr >= cj:      a2*cr^2 + b2*cr + c2

    Where lc = lam * cm.
    """
    lc = lam * cm  # elastic limit

    if -lc <= cr <= lc:
        return 0.0, 0.0
    elif -cj_split < cr < -lc:
        diff = -lc - cr
        return diff ** 3, -3.0 * diff ** 2
    elif lc < cr < cj_split:
        diff = cr - lc
        return diff ** 3, 3.0 * diff ** 2
    elif cr <= -cj_split:
        # Quadratic region matching value and derivative at -cj_split
        val_neg, grad_neg = _f_penalty_and_grad(-cj_split + 1e-10, cm, cj_split, lam)
        a1 = 3.0 * (-cj_split + lc) ** 2 / (2.0 * (-cj_split))
        b1 = -2.0 * a1 * (-cj_split)
        c1 = val_neg - a1 * (-cj_split) ** 2 - b1 * (-cj_split)
        return a1 * cr ** 2 + b1 * cr + c1, 2.0 * a1 * cr + b1
    else:  # cr >= cj_split
        a2 = 3.0 * (cj_split - lc) ** 2 / (2.0 * cj_split)
        b2 = -2.0 * a2 * cj_split
        val_pos, grad_pos = _f_penalty_and_grad(cj_split - 1e-10, cm, cj_split, lam)
        c2 = val_pos - a2 * cj_split ** 2 - b2 * cj_split
        return a2 * cr ** 2 + b2 * cr + c2, 2.0 * a2 * cr + b2


def feasibility_cost(cps: np.ndarray, dt: float,
                     v_max: float, a_max: float, j_max: float,
                     lam: float = 0.9,
                     split_ratio: float = 1.5) -> Tuple[float, np.ndarray]:
    """
    Feasibility penalty: restricts velocity, acceleration, jerk on each dimension.

    Jd = sum_i [wv * F(Vi) + wa * F(Ai) + wj * F(Ji)]

    Args:
        cps: (Nc, 3) control points
        dt: time interval
        v_max, a_max, j_max: derivative limits
        lam: elastic coefficient (< 1)
        split_ratio: cj = split_ratio * cm for quadratic/cubic split point

    Returns:
        (cost, gradient) where gradient has shape (Nc, 3)
    """
    nc = len(cps)
    grad = np.zeros_like(cps)

    vel_cps = (cps[1:] - cps[:-1]) / dt  # (nc-1, 3)
    accel_cps = (vel_cps[1:] - vel_cps[:-1]) / dt  # (nc-2, 3)
    jerk_cps = (accel_cps[1:] - accel_cps[:-1]) / dt  # (nc-3, 3)

    limits = [
        (vel_cps, v_max, 1.0),   # velocity
        (accel_cps, a_max, 1.0),  # acceleration
        (jerk_cps, j_max, 1.0),   # jerk
    ]

    cost = 0.0
    for deriv_cps, cm, weight in limits:
        cj_split = split_ratio * cm
        n_deriv = len(deriv_cps)

        for i in range(n_deriv):
            for dim in range(3):
                cr = deriv_cps[i, dim]
                val, dval = _f_penalty_and_grad(cr, cm, cj_split, lam)
                cost += weight * val

                # Back-propagate gradient to control points
                # For velocity: Vi = (Qi+1 - Qi) / dt
                # For acceleration: Ai = (Qi+2 - 2*Qi+1 + Qi) / dt^2
                # For jerk: Ji = (Qi+3 - 3*Qi+2 + 3*Qi+1 - Qi) / dt^3

                g_dim = weight * dval

                if deriv_cps is vel_cps:
                    # dVi/dQi = -1/dt, dVi/dQi+1 = 1/dt
                    inv = g_dim / dt
                    grad[i, dim] -= inv
                    grad[i + 1, dim] += inv
                elif deriv_cps is accel_cps:
                    # dAi/dQi = 1/dt^2, dAi/dQi+1 = -2/dt^2, dAi/dQi+2 = 1/dt^2
                    inv = g_dim / (dt * dt)
                    grad[i, dim] += inv
                    grad[i + 1, dim] -= 2.0 * inv
                    grad[i + 2, dim] += inv
                elif deriv_cps is jerk_cps:
                    inv = g_dim / (dt * dt * dt)
                    grad[i, dim] -= inv
                    grad[i + 1, dim] += 3.0 * inv
                    grad[i + 2, dim] -= 3.0 * inv
                    grad[i + 3, dim] += inv

    return cost, grad


def compute_total_cost_and_gradient(
        cps: np.ndarray, dt: float,
        obstacle_pairs: List[List[Tuple[np.ndarray, np.ndarray]]],
        weights: Dict[str, float],
        limits: Dict[str, float],
        sf: float = 0.5,
        lam: float = 0.9) -> Tuple[float, np.ndarray]:
    """
    Compute the total cost and gradient as weighted sum of Js, Jc, Jd.

    Args:
        cps: (Nc, 3) control points
        dt: time interval
        obstacle_pairs: collision obstacle pairs per control point
        weights: {'lambda_s': float, 'lambda_c': float, 'lambda_d': float}
        limits: {'v_max': float, 'a_max': float, 'j_max': float}
        sf: safety clearance
        lam: elastic coefficient for feasibility

    Returns:
        (total_cost, total_gradient) where gradient has shape (Nc, 3)
    """
    nc = len(cps)
    total_cost = 0.0
    total_grad = np.zeros_like(cps)

    # Smoothness
    if weights.get('lambda_s', 0) > 0:
        js, gs = smoothness_cost(cps, dt)
        total_cost += weights['lambda_s'] * js
        total_grad += weights['lambda_s'] * gs

    # Collision
    if weights.get('lambda_c', 0) > 0 and obstacle_pairs:
        jc, gc = collision_cost(cps, obstacle_pairs, sf)
        total_cost += weights['lambda_c'] * jc
        total_grad += weights['lambda_c'] * gc

    # Feasibility
    if weights.get('lambda_d', 0) > 0:
        jd, gd = feasibility_cost(cps, dt,
                                  limits.get('v_max', 5.0),
                                  limits.get('a_max', 8.0),
                                  limits.get('j_max', 20.0),
                                  lam)
        total_cost += weights['lambda_d'] * jd
        total_grad += weights['lambda_d'] * gd

    return total_cost, total_grad
