import heapq
import math

import numpy as np

import config


class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution)) + 1
        self.occupied = np.zeros((self.height, self.width), dtype=bool)

    def world_to_cell(self, point):
        return (
            int(round((float(point[0]) - self.x_min) / self.resolution)),
            int(round((float(point[1]) - self.y_min) / self.resolution)),
        )

    def cell_to_world(self, cell, z):
        return np.array(
            [self.x_min + cell[0] * self.resolution, self.y_min + cell[1] * self.resolution, z],
            dtype=np.float32,
        )

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell):
        if not self.in_bounds(cell):
            return False
        return not bool(self.occupied[cell[1], cell[0]])

    def set_occupied(self, cell, occupied=True):
        if self.in_bounds(cell):
            self.occupied[cell[1], cell[0]] = occupied

    def mark_disc(self, xy, radius):
        center = self.world_to_cell(xy)
        radius_cells = int(math.ceil(float(radius) / self.resolution))
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    self.set_occupied((center[0] + dx, center[1] + dy), True)

    def clear_disc(self, xy, radius):
        center = self.world_to_cell(xy)
        radius_cells = int(math.ceil(float(radius) / self.resolution))
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    self.set_occupied((center[0] + dx, center[1] + dy), False)

    def is_line_clear(self, point_a, point_b):
        """Return True if the straight line from point_a to point_b (xy)
        stays entirely within free cells of the occupancy grid.

        Uses Bresenham's algorithm for a robust, gap-free cell walk.
        """
        cell_a = self.world_to_cell(point_a)
        cell_b = self.world_to_cell(point_b)

        if not self.in_bounds(cell_a) or not self.in_bounds(cell_b):
            return False

        if cell_a == cell_b:
            return True

        # Bresenham line walk
        x0, y0 = cell_a
        x1, y1 = cell_b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self.occupied[y0, x0]:
                return False
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            if not self.in_bounds((x0, y0)):
                return False
        return True

    def nearest_free(self, cell, max_radius):
        if self.is_free(cell):
            return cell
        max_cells = int(math.ceil(float(max_radius) / self.resolution))
        for radius in range(1, max_cells + 1):
            best = None
            best_dist = float("inf")
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

    def occupied_ratio(self):
        return float(self.occupied.sum()) / float(self.width * self.height)


class OccupancyAStarPlanner:
    """Upper planner that derives local target points from an AirSim occupancy grid."""

    def __init__(self, client):
        self.client = client  # required for ray-scanned occupancy grid

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds = self._bounds(start, goal)
        last_cells = None

        if self.client is None:
            raise RuntimeError("Ray-scanned occupancy grid requires an AirSim client.")

        # Build the base grid once using top-down ray scanning for precise obstacle
        # geometry, then apply varying safety margin inflations as fallbacks.
        base_grid = self._build_grid_from_rays(bounds, start, goal)
        if base_grid is None:
            raise RuntimeError("Failed to build ray-scanned occupancy grid.")

        _, max_safety = config.OCCUPANCY_RADIUS_FALLBACKS[0]
        for _, safety_margin in config.OCCUPANCY_RADIUS_FALLBACKS:
            # A* needs start/goal free; work on a copy so base_grid stays intact for LOS.
            grid = self._inflate_grid(base_grid, safety_margin)
            grid = self._clear_start_goal(grid, start, goal)
            raw_start = grid.world_to_cell(start)
            raw_goal = grid.world_to_cell(goal)
            start_cell = grid.nearest_free(raw_start, config.OCCUPANCY_NEAREST_FREE_RADIUS)
            goal_cell = grid.nearest_free(raw_goal, config.OCCUPANCY_NEAREST_FREE_RADIUS)
            last_cells = (start_cell, goal_cell)
            print(
                f"[Model6] Occupancy A*: grid={grid.width}x{grid.height}, "
                f"occupied={grid.occupied_ratio():.1%}, safety_margin={safety_margin:.1f}"
            )
            if start_cell is None or goal_cell is None:
                continue

            cells = self._astar(grid, start_cell, goal_cell)
            if cells is None:
                continue

            raw_points = self._cells_to_points(grid, cells, start, goal)
            # Use base_grid (intact buildings, no clear_disc) for LOS checks.
            local_targets = self._extract_local_targets(base_grid, raw_points)
            return {
                "points": local_targets,
                "node_ids": [f"astar_{index}" for index in range(len(local_targets))],
                "regions": ["occupancy"] * len(local_targets),
                "k_neighbors": 0,
                "max_edge_distance": 0.0,
                "path_length": self.path_length(local_targets),
                "planner": "occupancy",
                "grid_cells": len(cells),
            }

        raise RuntimeError(f"Occupancy A* failed. Last cells={last_cells}")

    def _bounds(self, start, goal):
        return (
            config.OCCUPANCY_MIN_X,
            config.OCCUPANCY_MAX_X,
            config.OCCUPANCY_MIN_Y,
            config.OCCUPANCY_MAX_Y,
        )

    def _build_grid_from_rays(self, bounds, start, goal):
        """
        Build a precise occupancy grid by casting vertical rays from high above
        down through every cell.  Any cell where a ray hits an obstacle before
        reaching the ground is marked occupied.

        This captures the true 2D footprint of all buildings — L-shapes,
        U-shapes, courtyards, thin walls — without relying on disk approximations.
        """
        import time
        import airsim

        x_min, x_max, y_min, y_max = bounds
        grid = OccupancyGrid(x_min, x_max, y_min, y_max, config.OCCUPANCY_RESOLUTION)

        ray_origin_z = max(x_max - x_min, y_max - y_min) * 2.0  # high enough to clear all buildings
        ground_z = max(float(start[2]), float(goal[2]), 0.0) + 20.0  # below this is "free space"

        total_cells = grid.width * grid.height
        marked = 0
        t0 = time.time()

        for cy in range(grid.height):
            world_y = grid.y_min + cy * grid.resolution
            for cx in range(grid.width):
                world_x = grid.x_min + cx * grid.resolution
                origin = airsim.Vector3r(float(world_x), float(world_y), float(ray_origin_z))
                direction = airsim.Vector3r(0.0, 0.0, 1.0)  # straight down (positive z = down in NED)

                try:
                    hit = self.client.simCastRay(origin, direction)
                    if hit.distance > 0 and hit.distance < (ground_z - ray_origin_z):
                        grid.set_occupied((cx, cy), True)
                        marked += 1
                except Exception:
                    pass

            if (cy + 1) % 20 == 0 or cy == grid.height - 1:
                elapsed = time.time() - t0
                pct = (cy + 1) / grid.height * 100
                print(f"\r[Model6] Ray-scanning: {pct:.1f}% ({cy+1}/{grid.height}), marked={marked}", end="", flush=True)

        print(f"\n[Model6] Ray-scanning done in {time.time()-t0:.1f}s: {marked}/{total_cells} occupied ({grid.occupied_ratio():.1%})")

        return grid

    def _inflate_grid(self, base_grid, safety_margin):
        """Copy the base grid and dilate occupied cells by safety_margin."""
        inflated = OccupancyGrid(
            base_grid.x_min, base_grid.x_max,
            base_grid.y_min, base_grid.y_max,
            base_grid.resolution,
        )
        inflated.occupied[:] = base_grid.occupied

        radius_cells = int(math.ceil(float(safety_margin) / base_grid.resolution))
        ys, xs = np.where(base_grid.occupied)
        for x, y in zip(xs, ys):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        inflated.set_occupied((x + dx, y + dy), True)

        return inflated

    def _clear_start_goal(self, grid, start, goal):
        """Return a copy of *grid* with discs cleared around start and goal
        for A* pathfinding.  Leaves the original grid untouched.
        """
        working = OccupancyGrid(
            grid.x_min, grid.x_max,
            grid.y_min, grid.y_max,
            grid.resolution,
        )
        working.occupied[:] = grid.occupied
        working.clear_disc(start[:2], config.OCCUPANCY_NEAREST_FREE_RADIUS)
        working.clear_disc(goal[:2], config.OCCUPANCY_NEAREST_FREE_RADIUS)
        return working

    def _astar(self, grid, start, goal):
        if not grid.is_free(start) or not grid.is_free(goal):
            return None
        open_heap = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct(came_from, current)
            closed.add(current)

            for neighbor in self._neighbors(current):
                if not grid.is_free(neighbor):
                    continue
                step_cost = math.hypot(neighbor[0] - current[0], neighbor[1] - current[1])
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                heapq.heappush(open_heap, (priority, neighbor))
        return None

    def _neighbors(self, cell):
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if config.OCCUPANCY_ALLOW_DIAGONAL:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in steps:
            yield cell[0] + dx, cell[1] + dy

    @staticmethod
    def _reconstruct(came_from, current):
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return cells

    def _cells_to_points(self, grid, cells, start, goal):
        total_xy = max(float(np.linalg.norm(goal[:2] - start[:2])), 1e-6)
        points = []
        for cell in cells:
            xy = grid.cell_to_world(cell, 0.0)
            progress = float(np.linalg.norm(xy[:2] - start[:2])) / total_xy
            z = float(start[2] + np.clip(progress, 0.0, 1.0) * (goal[2] - start[2]))
            points.append(np.array([xy[0], xy[1], z], dtype=np.float32))
        points[0] = start
        points[-1] = goal
        return points

    @staticmethod
    def _find_free_waypoint(grid, last_selected, raw_points, current_index):
        """Walk backwards from current_index to find the furthest point that
        has clear LOS to last_selected. Returns index or None.
        """
        for i in range(current_index - 1, 0, -1):
            if grid.is_line_clear(last_selected, raw_points[i]):
                return i
        return None

    def _extract_local_targets(self, grid, raw_points):
        """Extract sparse local targets from the dense A* path.

        Keeps turning points and maintains a minimum spacing, but ensures
        that the straight-line segment between any two consecutive targets
        does NOT cross occupied cells (LOS check).
        """
        if len(raw_points) <= 2:
            return raw_points
        selected = [raw_points[0]]
        last_selected = raw_points[0]
        previous_direction = None

        for index in range(1, len(raw_points) - 1):
            current = raw_points[index]
            next_point = raw_points[index + 1]
            direction = np.sign(next_point[:2] - current[:2]).astype(np.int32)
            dist_from_last = float(np.linalg.norm(current - last_selected))
            is_turn = previous_direction is not None and not np.array_equal(direction, previous_direction)
            should_keep_turn = config.LOCAL_TARGET_KEEP_TURNS and is_turn and dist_from_last >= config.LOCAL_TARGET_MIN_SPACING
            should_keep_spacing = dist_from_last >= config.LOCAL_TARGET_SPACING

            if should_keep_turn or should_keep_spacing:
                if grid.is_line_clear(last_selected, current):
                    # Straight line is clear — keep per original rules.
                    selected.append(current)
                    last_selected = current
                else:
                    # LOS blocked: walk backwards from current to find the
                    # furthest free point and insert it as an intermediate target.
                    backtrack = self._find_free_waypoint(grid, last_selected, raw_points, index)
                    if backtrack is not None:
                        selected.append(raw_points[backtrack])
                        last_selected = raw_points[backtrack]
                    selected.append(current)
                    last_selected = current

            previous_direction = direction

        if float(np.linalg.norm(raw_points[-1] - selected[-1])) < config.LOCAL_TARGET_MIN_SPACING and len(selected) > 1:
            selected[-1] = raw_points[-1]
        else:
            selected.append(raw_points[-1])
        return selected

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))))
