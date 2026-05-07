import numpy as np


class VoxelGrid:
    """
    Compact obstacle representation for ESDF-free planning.
    Stores obstacle spheres directly for fast distance queries.
    """

    def __init__(self, resolution=0.2, grid_size=50, origin=None):
        self.resolution = resolution
        self.grid_size = grid_size
        self.origin = np.array(origin, dtype=np.float64) if origin is not None else np.zeros(3)
        # List of (center_position, radius) tuples
        self.obstacles = []

    def mark_occupied(self, point, radius=None):
        """Mark a point as occupied (treat as a small sphere)."""
        if radius is None:
            radius = self.resolution
        self.obstacles.append((np.array(point, dtype=np.float64), float(radius)))

    def mark_occupied_sphere(self, center, radius):
        """Mark a sphere as occupied."""
        self.obstacles.append((np.array(center, dtype=np.float64), float(radius)))

    def mark_line(self, start, end):
        """Mark a line segment as occupied (series of small spheres)."""
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)
        dist = np.linalg.norm(end - start)
        n = max(int(dist / self.resolution), 1)
        for i in range(n):
            t = i / n
            pt = start + t * (end - start)
            self.obstacles.append((pt, self.resolution * 0.5))

    def get_distance_with_obstacle(self, point, search_radius=2.0):
        """
        Compute distance from point to nearest obstacle surface.
        Returns: (distance, nearest_obstacle_center)
        """
        if not self.obstacles:
            return search_radius, None

        pt = np.array(point, dtype=np.float64)
        min_dist = search_radius
        nearest_pt = None

        for obs_center, obs_radius in self.obstacles:
            vec = pt - obs_center
            dist = np.linalg.norm(vec) - obs_radius
            if dist < min_dist:
                min_dist = max(dist, 0.0)
                nearest_pt = obs_center

        return min_dist, nearest_pt

    def clear(self):
        """Clear all obstacles."""
        self.obstacles.clear()

    def load_from_rangefinder(self, origin, rays, max_range=10.0):
        """
        Populate obstacles from rangefinder data.
        Each ray endpoint becomes a small obstacle sphere.
        """
        self.clear()
        for direction, distance in rays:
            if distance >= max_range:
                continue
            end_pt = origin + direction * distance
            self.obstacles.append((end_pt, self.resolution * 2))
