"""
B-spline trajectory representation and derivative computation.
Uniform B-spline of degree pb with convex hull property.
"""

import numpy as np
from typing import List, Tuple


class BSplineTrajectory:
    """Uniform B-spline curve parameterized by control points."""

    def __init__(self, control_points: np.ndarray, dt: float, degree: int = 3):
        """
        Args:
            control_points: (Nc, 3) array of control point positions
            dt: uniform time interval between knots
            degree: B-spline degree (default 3 = cubic)
        """
        self.Q = np.asarray(control_points, dtype=np.float64)
        self.dt = dt
        self.degree = degree
        self.nc = len(self.Q)  # number of control points
        self.duration = (self.nc - 1) * dt

    @property
    def n_segments(self) -> int:
        return max(0, self.nc - 1)

    def _basis_coeffs(self, t_normalized: float) -> np.ndarray:
        """
        Compute de Boor basis coefficients for a normalized parameter t in [0, 1]
        within a single span. For cubic B-spline, returns 4 coefficients.
        """
        pb = self.degree
        t = np.clip(t_normalized, 0.0, 1.0)

        if pb == 3:
            t2 = t * t
            t3 = t2 * t
            coeffs = np.array([
                (1 - 3*t + 3*t2 - t3) / 6.0,
                (4 - 6*t2 + 3*t3) / 6.0,
                (1 + 3*t + 3*t2 - 3*t3) / 6.0,
                t3 / 6.0
            ])
        elif pb == 2:
            t2 = t * t
            coeffs = np.array([
                (1 - 2*t + t2) / 2.0,
                (1 + 2*t - 2*t2) / 2.0,
                t2 / 2.0
            ])
        elif pb == 1:
            coeffs = np.array([1 - t, t])
        else:
            coeffs = np.ones(pb + 1) / (pb + 1)

        return coeffs

    def _find_span(self, t: float) -> Tuple[int, float]:
        """
        Find the span index and local parameter for time t.
        Returns (span_index, t_normalized) where t_normalized in [0, 1].
        """
        t = np.clip(t, 0.0, self.duration - 1e-10)
        span_idx = int(t / self.dt)
        span_idx = min(span_idx, self.nc - 2)
        t_local = (t - span_idx * self.dt) / self.dt
        return span_idx, t_local

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the B-spline position at time t."""
        span_idx, t_local = self._find_span(t)

        coeffs = self._basis_coeffs(t_local)
        pos = np.zeros(3, dtype=np.float64)

        for i in range(self.degree + 1):
            cp_idx = span_idx - self.degree + i
            cp_idx = max(0, min(cp_idx, self.nc - 1))
            pos += coeffs[i] * self.Q[cp_idx]

        return pos

    def evaluate_many(self, times: np.ndarray) -> np.ndarray:
        """Evaluate the B-spline at multiple time points. Returns (N, 3)."""
        return np.array([self.evaluate(t) for t in times])

    def get_derivative_control_points(self, order: int = 1) -> np.ndarray:
        """
        Compute control points of the k-th derivative curve.
        The k-th derivative of a B-spline is still a B-spline.

        For order=1 (velocity): Vi = (Qi+1 - Qi) / dt
        For order=2 (acceleration): Ai = (Vi+1 - Vi) / dt
        For order=3 (jerk): Ji = (Ai+1 - Ai) / dt
        """
        cps = self.Q.copy()
        for _ in range(order):
            if len(cps) < 2:
                return np.zeros((0, 3), dtype=np.float64)
            cps = (cps[1:] - cps[:-1]) / self.dt
        return cps

    @property
    def velocity_cps(self) -> np.ndarray:
        """Control points of velocity curve. Shape: (Nc-1, 3)."""
        return self.get_derivative_control_points(1)

    @property
    def accel_cps(self) -> np.ndarray:
        """Control points of acceleration curve. Shape: (Nc-2, 3)."""
        return self.get_derivative_control_points(2)

    @property
    def jerk_cps(self) -> np.ndarray:
        """Control points of jerk curve. Shape: (Nc-3, 3)."""
        return self.get_derivative_control_points(3)

    def get_control_points(self) -> np.ndarray:
        return self.Q.copy()

    def sample_points(self, n_samples: int = 50) -> np.ndarray:
        """Sample n_samples points along the trajectory. Returns (N, 3)."""
        if self.duration <= 0:
            return self.Q[:1]
        times = np.linspace(0, self.duration, n_samples)
        return self.evaluate_many(times)

    def clone_with_control_points(self, new_cps: np.ndarray) -> 'BSplineTrajectory':
        """Create a new BSplineTrajectory with different control points but same dt and degree."""
        return BSplineTrajectory(new_cps, self.dt, self.degree)

    def total_duration(self) -> float:
        return self.duration

    def __repr__(self) -> str:
        return (f"BSplineTrajectory(nc={self.nc}, dt={self.dt:.3f}, "
                f"degree={self.degree}, duration={self.duration:.3f}s)")
