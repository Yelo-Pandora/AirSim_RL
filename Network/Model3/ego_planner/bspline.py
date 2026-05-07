import numpy as np


def _basis_function(i, k, u, knots):
    """Compute the i-th B-spline basis function of order k at parameter u."""
    if k == 1:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    term1 = ((u - knots[i]) / denom1 * _basis_function(i, k - 1, u, knots)) if denom1 > 1e-10 else 0.0
    term2 = ((knots[i + k] - u) / denom2 * _basis_function(i + 1, k - 1, u, knots)) if denom2 > 1e-10 else 0.0
    return term1 + term2


def _basis_derivative(i, k, u, knots):
    """First derivative of the i-th B-spline basis function of order k at u."""
    if k <= 1:
        return 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    term1 = (k - 1) / denom1 * _basis_function(i, k - 1, u, knots) if denom1 > 1e-10 else 0.0
    term2 = (k - 1) / denom2 * _basis_function(i + 1, k - 1, u, knots) if denom2 > 1e-10 else 0.0
    return term1 - term2


def _basis_second_derivative(i, k, u, knots):
    """Second derivative of the i-th B-spline basis function of order k at u."""
    if k <= 2:
        return 0.0
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]
    denom3 = knots[i + k - 1] - knots[i + 1]
    term1 = (k - 1) / denom1 * _basis_derivative(i, k - 1, u, knots) if denom1 > 1e-10 else 0.0
    term2 = (k - 1) / denom2 * _basis_derivative(i + 1, k - 1, u, knots) if denom2 > 1e-10 else 0.0
    return term1 - term2


class BSpline:
    def __init__(self, control_points, order=4, dt=0.2):
        """
        Uniform B-spline trajectory.

        Args:
            control_points: (N, 3) array of control points
            order: spline order (4 = cubic)
            dt: time interval between control points
        """
        self.control_points = np.array(control_points, dtype=np.float64)
        self.order = order
        self.n_ctrl = len(self.control_points)
        self.dt = dt
        self.knots = self._uniform_knots()
        self.duration = self.knots[-1] - self.knots[0]

    def _uniform_knots(self):
        """Create uniform knot vector with appropriate time scaling."""
        # Scale dt based on control point spacing to keep velocities reasonable
        # Compute approximate distance between consecutive control points
        max_dist = 0.0
        for i in range(len(self.control_points) - 1):
            d = np.linalg.norm(self.control_points[i + 1] - self.control_points[i])
            max_dist = max(max_dist, d)

        # Adjust dt so that velocity ≈ max_dist/dt stays under ~5 m/s initially
        # Use dt that gives reasonable initial velocity
        if max_dist > 0:
            adaptive_dt = max_dist / 3.0  # ~3 m/s initial speed
        else:
            adaptive_dt = self.dt

        n = self.n_ctrl + 2 * self.order
        knots = np.zeros(n)
        for i in range(n):
            knots[i] = max(0, i - self.order + 1) * adaptive_dt
        return knots

    def eval(self, t):
        """Evaluate position at time t."""
        t += self.knots[0]
        pt = np.zeros(3)
        for i in range(self.n_ctrl):
            b = _basis_function(i, self.order, t, self.knots)
            pt += b * self.control_points[i]
        return pt

    def eval_derivative(self, t, order=1):
        """Evaluate the `order`-th derivative at time t."""
        t += self.knots[0]
        pt = np.zeros(3)
        for i in range(self.n_ctrl):
            if order == 1:
                b = _basis_derivative(i, self.order, t, self.knots)
            elif order == 2:
                b = _basis_second_derivative(i, self.order, t, self.knots)
            elif order == 3:
                b = (_basis_second_derivative(i, self.order, t, self.knots)
                     if self.order > 3 else 0.0)
            else:
                b = 0.0
            pt += b * self.control_points[i]
        return pt

    def trajectory(self, n_samples=100):
        """Return sampled trajectory as (n_samples, 3) array."""
        ts = np.linspace(0, self.duration, n_samples)
        return np.array([self.eval(t) for t in ts])

    def get_control_points(self):
        return self.control_points.copy()

    def set_control_points(self, new_ctrl):
        self.control_points = np.array(new_ctrl, dtype=np.float64)
