import math

import numpy as np


class OccupancyGrid:
    """2D x/y occupancy grid for A* planning in AirSim's world frame."""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution)) + 1
        self.occupied = np.zeros((self.height, self.width), dtype=bool)

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def world_to_cell(self, point):
        x = int(round((float(point[0]) - self.x_min) / self.resolution))
        y = int(round((float(point[1]) - self.y_min) / self.resolution))
        return x, y

    def cell_to_world(self, cell, z=None):
        x = self.x_min + cell[0] * self.resolution
        y = self.y_min + cell[1] * self.resolution
        if z is None:
            return np.array([x, y], dtype=np.float32)
        return np.array([x, y, z], dtype=np.float32)

    def is_free(self, cell):
        if not self.in_bounds(cell):
            return False
        x, y = cell
        return not bool(self.occupied[y, x])

    def set_occupied(self, cell, occupied=True):
        if self.in_bounds(cell):
            x, y = cell
            self.occupied[y, x] = occupied

    def mark_disc(self, center_xy, radius):
        center = self.world_to_cell(center_xy)
        radius_cells = int(math.ceil(float(radius) / self.resolution))
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                self.set_occupied((center[0] + dx, center[1] + dy), True)

    def clear_disc(self, center_xy, radius):
        center = self.world_to_cell(center_xy)
        radius_cells = int(math.ceil(float(radius) / self.resolution))
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                self.set_occupied((center[0] + dx, center[1] + dy), False)

    def nearest_free(self, cell, max_radius):
        if self.is_free(cell):
            return cell

        max_cells = int(math.ceil(float(max_radius) / self.resolution))
        best = None
        best_dist = float("inf")
        for radius in range(1, max_cells + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if not self.is_free(candidate):
                        continue
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best = candidate
                        best_dist = dist
            if best is not None:
                return best
        return None

    def inflated_copy(self, radius):
        inflated = OccupancyGrid(self.x_min, self.x_max, self.y_min, self.y_max, self.resolution)
        inflated.occupied[:] = self.occupied
        ys, xs = np.where(self.occupied)
        radius_cells = int(math.ceil(float(radius) / self.resolution))
        for x, y in zip(xs, ys):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        inflated.set_occupied((x + dx, y + dy), True)
        return inflated

    def occupied_count(self):
        return int(self.occupied.sum())
