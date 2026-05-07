"""
Local 3D voxel grid built from depth camera and LiDAR data.
Used for collision detection and A* pathfinding in EGO-Planner.
"""

import numpy as np
from typing import Tuple, List, Optional


class LocalVoxelGrid:
    """3D voxel grid centered on the drone for local obstacle representation."""

    def __init__(self, resolution: float = 0.2,
                 half_size: Tuple[int, int, int] = (15, 15, 5)):
        """
        Args:
            resolution: voxel size in meters
            half_size: (hx, hy, hz) half-extent in voxels
        """
        self.res = resolution
        self.half_size = half_size
        self.full_size = (2 * half_size[0], 2 * half_size[1], 2 * half_size[2])
        self.grid = np.zeros(self.full_size, dtype=np.bool_)
        self.origin = np.zeros(3, dtype=np.float64)  # world position of grid center

    def reset(self, center: np.ndarray):
        """Reset grid and set new center position."""
        self.grid.fill(False)
        self.origin = center.copy()

    def _world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        local = point - self.origin
        vx = int(local[0] / self.res + self.half_size[0])
        vy = int(local[1] / self.res + self.half_size[1])
        vz = int(local[2] / self.res + self.half_size[2])
        return vx, vy, vz

    def _voxel_to_world(self, vx: int, vy: int, vz: int) -> np.ndarray:
        """Convert voxel indices to world coordinates (center of voxel)."""
        return np.array([
            (vx - self.half_size[0] + 0.5) * self.res + self.origin[0],
            (vy - self.half_size[1] + 0.5) * self.res + self.origin[1],
            (vz - self.half_size[2] + 0.5) * self.res + self.origin[2],
        ], dtype=np.float64)

    def _is_in_bounds(self, vx: int, vy: int, vz: int) -> bool:
        return (0 <= vx < self.full_size[0] and
                0 <= vy < self.full_size[1] and
                0 <= vz < self.full_size[2])

    def set_voxel(self, point: np.ndarray, occupied: bool = True):
        """Mark the voxel containing `point` as occupied or free."""
        vx, vy, vz = self._world_to_voxel(point)
        if self._is_in_bounds(vx, vy, vz):
            self.grid[vx, vy, vz] = occupied

    def is_occupied(self, point: np.ndarray, radius: float = 0.0) -> bool:
        """
        Check if the voxel at `point` is occupied.
        If radius > 0, checks a sphere of voxels around the point.
        """
        if radius > 0:
            return self._check_sphere(point, radius)
        vx, vy, vz = self._world_to_voxel(point)
        if not self._is_in_bounds(vx, vy, vz):
            return False  # Out of bounds = unknown, not occupied
        return bool(self.grid[vx, vy, vz])

    def _check_sphere(self, center: np.ndarray, radius: float) -> bool:
        """Check if any voxel within radius of center is occupied."""
        r_voxels = int(np.ceil(radius / self.res))
        cx, cy, cz = self._world_to_voxel(center)
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    if dx*dx + dy*dy + dz*dz <= r_voxels*r_voxels:
                        vx, vy, vz = cx + dx, cy + dy, cz + dz
                        if self._is_in_bounds(vx, vy, vz) and self.grid[vx, vy, vz]:
                            return True
        return False

    def is_collision_free(self, points: np.ndarray, radius: float = 0.5) -> bool:
        """
        Check if all points are collision-free with a safety radius.
        Args:
            points: (N, 3) array of points to check
            radius: safety radius in meters (cylinder pipe check)
        Returns:
            True if all points are clear
        """
        for i in range(len(points)):
            if self.is_occupied(points[i], radius):
                return False
        return True

    def update_from_depth_image(self, depth_img: np.ndarray,
                                drone_pos: np.ndarray,
                                drone_quat: Optional[np.ndarray] = None,
                                fov_h: float = 90.0,
                                fov_v: float = 60.0,
                                max_depth: float = 20.0):
        """
        Project depth image points into 3D and mark occupied voxels.

        Args:
            depth_img: (H, W) array of depth values in meters
            drone_pos: (3,) drone position in world coordinates
            drone_quat: (4,) quaternion [w, x, y, z] for camera orientation
            fov_h: horizontal FOV in degrees
            fov_v: vertical FOV in degrees
            max_depth: maximum depth to consider
        """
        h, w = depth_img.shape
        cx, cy = w / 2.0, h / 2.0
        fx = w / (2.0 * np.tan(np.radians(fov_h / 2.0)))
        fy = h / (2.0 * np.tan(np.radians(fov_v / 2.0)))

        # Build rotation matrix from quaternion if provided
        if drone_quat is not None:
            R = self._quat_to_rot(drone_quat)
        else:
            R = np.eye(3)

        # Sample depth image at reduced resolution for efficiency
        step = max(1, w // 60)  # ~60 samples horizontally
        u_coords = np.arange(0, w, step)
        v_coords = np.arange(0, h, step)
        uu, vv = np.meshgrid(u_coords, v_coords)
        uu, vv = uu.ravel(), vv.ravel()

        depths = depth_img[vv, uu]
        valid = (depths > 0.1) & (depths < max_depth)
        uu, vv, depths = uu[valid], vv[valid], depths[valid]

        if len(depths) == 0:
            return

        # Convert pixel coords to camera-frame 3D points
        x_cam = (uu - cx) * depths / fx
        y_cam = (vv - cy) * depths / fy
        z_cam = depths

        cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to world frame (AirSim NED)
        world_points = (R @ cam_points.T).T + drone_pos  # (N, 3)

        # Mark occupied voxels
        for pt in world_points:
            self.set_voxel(pt, True)

    def update_from_lidar(self, lidar_points: np.ndarray,
                          drone_pos: np.ndarray,
                          max_range: float = 20.0):
        """
        Add LiDAR point cloud to voxel grid.

        Args:
            lidar_points: (N, 3) array of points in world coordinates
                          or (360,) array of distances per angle
            drone_pos: (3,) drone position
            max_range: maximum range to consider
        """
        if lidar_points.ndim == 1:
            # Distance array: convert to 3D points assuming horizontal scan
            n = len(lidar_points)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                d = lidar_points[i]
                if 0.1 < d < max_range:
                    points[i, 0] = d * np.cos(angles[i])
                    points[i, 1] = d * np.sin(angles[i])
                    points[i, 2] = 0.0  # horizontal scan
            lidar_points = points

        # Filter by range and add to grid
        dists = np.linalg.norm(lidar_points, axis=1)
        valid = (dists > 0.1) & (dists < max_range)
        world_points = lidar_points[valid] + drone_pos

        for pt in world_points:
            self.set_voxel(pt, True)

    def get_occupied_voxels(self) -> np.ndarray:
        """Return world coordinates of all occupied voxels. (N, 3)."""
        indices = np.argwhere(self.grid)
        if len(indices) == 0:
            return np.empty((0, 3), dtype=np.float64)
        world_pts = np.array([self._voxel_to_world(ix, iy, iz)
                              for ix, iy, iz in indices])
        return world_pts

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the voxel grid in world coords."""
        min_corner = self.origin - np.array([
            self.half_size[0] * self.res,
            self.half_size[1] * self.res,
            self.half_size[2] * self.res,
        ])
        max_corner = self.origin + np.array([
            self.half_size[0] * self.res,
            self.half_size[1] * self.res,
            self.half_size[2] * self.res,
        ])
        return min_corner, max_corner

    @staticmethod
    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
        ])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.full_size
