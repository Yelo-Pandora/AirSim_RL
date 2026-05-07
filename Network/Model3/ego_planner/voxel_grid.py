import numpy as np


class VoxelGrid:
    """
    Compact 3D voxel grid for obstacle representation.
    ESDF-free: only stores occupied voxels, computes distance on demand.
    """

    def __init__(self, resolution=0.2, grid_size=50, origin=None):
        """
        Args:
            resolution: voxel size in meters
            grid_size: number of cells per dimension
            origin: (3,) array, world origin of the grid
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.origin = np.array(origin, dtype=np.float64) if origin is not None else np.zeros(3)
        # Sparse set of occupied voxel indices: set of (ix, iy, iz) tuples
        self.occupied = set()
        # For caching nearest obstacle lookups
        self._cache = {}

    def world_to_voxel(self, point):
        """Convert world coordinates to voxel indices."""
        idx = ((point - self.origin) / self.resolution).astype(int)
        return tuple(idx.tolist())

    def voxel_to_world(self, voxel_idx):
        """Convert voxel indices to world coordinates (center of voxel)."""
        return self.origin + (np.array(voxel_idx, dtype=np.float64) + 0.5) * self.resolution

    def is_occupied(self, point):
        """Check if a point is inside an occupied voxel."""
        idx = self.world_to_voxel(point)
        return idx in self.occupied

    def mark_occupied(self, point):
        """Mark the voxel containing `point` as occupied."""
        idx = self.world_to_voxel(point)
        self.occupied.add(idx)

    def mark_occupied_sphere(self, center, radius):
        """Mark all voxels within a sphere as occupied."""
        r_voxels = int(np.ceil(radius / self.resolution))
        cx, cy, cz = self.world_to_voxel(center)
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    idx = (cx + dx, cy + dy, cz + dz)
                    pt = self.voxel_to_world(idx)
                    if np.linalg.norm(pt - center) <= radius:
                        self.occupied.add(idx)

    def mark_line(self, start, end):
        """Mark voxels along a line segment (3D Bresenham-style)."""
        start_vox = self.world_to_voxel(start)
        end_vox = self.world_to_voxel(end)
        dx = abs(end_vox[0] - start_vox[0])
        dy = abs(end_vox[1] - start_vox[1])
        dz = abs(end_vox[2] - start_vox[2])
        steps = max(dx, dy, dz, 1)

        x0, y0, z0 = start_vox
        x1, y1, z1 = end_vox

        for t in range(steps + 1):
            frac = t / steps
            ix = int(x0 + (x1 - x0) * frac)
            iy = int(y0 + (y1 - y0) * frac)
            iz = int(z0 + (z1 - z0) * frac)
            self.occupied.add((ix, iy, iz))

    def get_distance_with_obstacle(self, point, search_radius=2.0):
        """
        Compute signed distance from point to nearest occupied voxel.
        Returns: (distance, nearest_obstacle_point)
        Uses BFS-like search in voxel space.
        """
        if not self.occupied:
            return search_radius, None

        start_vox = self.world_to_voxel(point)
        visited = set()
        queue = [(start_vox, 0)]
        visited.add(start_vox)

        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        while queue:
            current, dist = queue.pop(0)
            pt = self.voxel_to_world(current)
            actual_dist = np.linalg.norm(pt - point)

            if current in self.occupied:
                return actual_dist, pt

            if dist * self.resolution > search_radius:
                continue

            for d in directions:
                neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return search_radius, None

    def clear(self):
        """Clear all occupied voxels."""
        self.occupied.clear()
        self._cache.clear()

    def load_from_rangefinder(self, origin, rays, max_range=10.0):
        """
        Populate voxel grid from rangefinder data.
        rays: list of (direction, distance) tuples
        """
        self.clear()
        for direction, distance in rays:
            if distance >= max_range:
                continue
            end_pt = origin + direction * distance
            self.mark_line(origin, end_pt)
            # Mark the endpoint as occupied
            self.mark_occupied(end_pt)
