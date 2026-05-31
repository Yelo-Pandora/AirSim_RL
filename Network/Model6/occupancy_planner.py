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

    def __init__(self, obstacle_centers, client=None):
        self.obstacle_centers = np.asarray(obstacle_centers, dtype=np.float32)
        self.client = client  # optional AirSim client for reachability validation

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds = self._bounds(start, goal)
        last_cells = None

        for obstacle_radius, safety_margin in config.OCCUPANCY_RADIUS_FALLBACKS:
            grid = self._build_grid(bounds, start, goal, obstacle_radius, safety_margin)
            raw_start = grid.world_to_cell(start)
            raw_goal = grid.world_to_cell(goal)
            start_cell = grid.nearest_free(raw_start, config.OCCUPANCY_NEAREST_FREE_RADIUS)
            goal_cell = grid.nearest_free(raw_goal, config.OCCUPANCY_NEAREST_FREE_RADIUS)
            last_cells = (start_cell, goal_cell)
            print(
                f"[Model6] Occupancy A*: grid={grid.width}x{grid.height}, "
                f"occupied={grid.occupied_ratio():.1%}, radius={obstacle_radius:.1f}, margin={safety_margin:.1f}"
            )
            if start_cell is None or goal_cell is None:
                continue

            cells = self._astar(grid, start_cell, goal_cell)
            if cells is None:
                continue

            raw_points = self._cells_to_points(grid, cells, start, goal)
            local_targets = self._extract_local_targets(raw_points)

            # Reachability validation: only check suspicious segments where A*
            # detoured significantly (grid path >> straight line), then cast a
            # single ray to confirm.
            if self.client is not None and not self._validate_reachability(raw_points, local_targets, grid):
                print(f"[Model6] Reachability validation failed; trying next fallback...")
                continue

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
        points = [start, goal]
        if len(self.obstacle_centers) > 0:
            points.append(self.obstacle_centers)
        stacked = np.vstack(points)
        x_min = max(config.OCCUPANCY_MIN_X, float(np.min(stacked[:, 0])) - config.OCCUPANCY_BOUNDS_MARGIN)
        x_max = min(config.OCCUPANCY_MAX_X, float(np.max(stacked[:, 0])) + config.OCCUPANCY_BOUNDS_MARGIN)
        y_min = max(config.OCCUPANCY_MIN_Y, float(np.min(stacked[:, 1])) - config.OCCUPANCY_BOUNDS_MARGIN)
        y_max = min(config.OCCUPANCY_MAX_Y, float(np.max(stacked[:, 1])) + config.OCCUPANCY_BOUNDS_MARGIN)
        return x_min, x_max, y_min, y_max

    def _build_grid(self, bounds, start, goal, obstacle_radius, safety_margin):
        grid = OccupancyGrid(*bounds, config.OCCUPANCY_RESOLUTION)
        for center in self.obstacle_centers:
            grid.mark_disc(center[:2], obstacle_radius + safety_margin)
        grid.clear_disc(start[:2], obstacle_radius + safety_margin)
        grid.clear_disc(goal[:2], obstacle_radius + safety_margin)
        return grid

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

    def _extract_local_targets(self, raw_points):
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
                selected.append(current)
                last_selected = current
            previous_direction = direction

        if float(np.linalg.norm(raw_points[-1] - selected[-1])) < config.LOCAL_TARGET_MIN_SPACING and len(selected) > 1:
            selected[-1] = raw_points[-1]
        else:
            selected.append(raw_points[-1])
        return selected

    def _validate_reachability(self, raw_points, local_targets, grid):
        """
        For consecutive waypoints with a significant detour ratio (grid path
        >> straight line), cast a ray to verify line-of-sight in the real UE5
        scene.  Catches courtyard traps where the waypoint is inside an enclosed
        area the drone cannot actually reach.
        """
        detour_threshold = 0.6  # Euclidean / grid_path ratio below which we suspect detour

        for i in range(len(local_targets) - 1):
            wp_from = np.asarray(local_targets[i], dtype=np.float32)
            wp_to = np.asarray(local_targets[i + 1], dtype=np.float32)
            euclidean = float(np.linalg.norm(wp_to[:2] - wp_from[:2]))
            if euclidean < 5.0:
                continue

            idx_from = self._find_closest_point_index(raw_points, wp_from)
            idx_to = self._find_closest_point_index(raw_points, wp_to)
            if idx_from is None or idx_to is None:
                continue

            lo, hi = min(idx_from, idx_to), max(idx_from, idx_to)
            if hi - lo < 2:
                continue

            grid_path_len = self._path_length_2d(raw_points[lo:hi + 1])
            if grid_path_len < euclidean * 1.05:
                continue

            ratio = euclidean / grid_path_len
            if ratio < detour_threshold:
                if not self._ray_segment_clear(wp_from, wp_to):
                    print(
                        f"[Model6]   Segment {i}->{i+1} blocked: "
                        f"euclidean={euclidean:.1f}m, grid_path={grid_path_len:.1f}m, "
                        f"ratio={ratio:.2f}"
                    )
                    return False

        return True

    def _find_closest_point_index(self, points, target):
        best_idx = None
        best_dist = float("inf")
        for idx, p in enumerate(points):
            d = float(np.linalg.norm(p[:2] - target[:2]))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx if best_dist < 5.0 else None

    @staticmethod
    def _path_length_2d(points):
        total = 0.0
        for i in range(1, len(points)):
            total += float(np.linalg.norm(points[i][:2] - points[i - 1][:2]))
        return total

    def _ray_segment_clear(self, wp_from, wp_to):
        import airsim

        direction_vec = wp_to - wp_from
        distance = float(np.linalg.norm(direction_vec[:2]))
        if distance < 1e-6:
            return True
        direction = airsim.Vector3r(
            float(direction_vec[0] / distance),
            float(direction_vec[1] / distance),
            0.0,
        )
        try:
            hit = self.client.simCastRay(
                airsim.Vector3r(float(wp_from[0]), float(wp_from[1]), float(wp_from[2])),
                direction,
            )
            if hit.distance < distance * 0.9:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))))
