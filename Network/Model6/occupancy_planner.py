import heapq
import math

import numpy as np

import config


def _matches_any(name, patterns):
    lower = name.lower()
    return any(pattern in lower for pattern in patterns)


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

    def mark_bbox(self, center_xy, half_size_xy, margin=0.0):
        """Mark a rectangular bounding box as occupied.

        Args:
            center_xy: (x, y) center of the obstacle
            half_size_xy: (hx, hy) half-extent of the obstacle
            margin: additional safety margin to add around the obstacle
        """
        cx, cy = center_xy
        hx, hy = half_size_xy
        hxm, hym = hx + margin, hy + margin
        # Compute cell bounds
        c_min_x = self.world_to_cell((cx - hxm, cy - hym))
        c_max_x = self.world_to_cell((cx + hxm, cy + hym))
        for cell_x in range(min(c_min_x[0], c_max_x[0]), max(c_min_x[0], c_max_x[0]) + 1):
            for cell_y in range(min(c_min_x[1], c_max_x[1]), max(c_min_x[1], c_max_x[1]) + 1):
                self.set_occupied((cell_x, cell_y), True)

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
    """Upper planner that derives local target points from an AirSim occupancy grid.

    Obstacles are detected by casting vertical rays top-down for each grid
    cell, using AirSim's simTestLineOfSightBetweenPoints. If the line from a
    point high above the scene to a point just above the ground is blocked,
    that cell is marked occupied.

    A* runs on the inflated working copy while LOS checks use the intact base
    grid so that start/goal clear-disc never hides buildings from the
    visibility test.
    """

    def __init__(self, client):
        self.client = client

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds = self._bounds(start, goal)
        last_cells = None

        if self.client is None:
            raise RuntimeError("Occupancy grid requires an AirSim client.")

        base_grid = self._build_grid_from_objects(bounds)
        if base_grid is None:
            raise RuntimeError("Failed to build occupancy grid from scene objects.")
        if config.OCCUPANCY_REQUIRE_NONEMPTY_MAP and int(base_grid.occupied.sum()) == 0:
            raise RuntimeError("Occupancy grid is empty; raycast/building detection failed.")

        print(
            f"[Model6] Occupancy base_grid={base_grid.width}x{base_grid.height}, "
            f"occupied={base_grid.occupied.sum()} cells ({base_grid.occupied_ratio():.1%})"
        )

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
            # Use base_grid (intact buildings, no clear_disc) for all LOS / occupied checks.
            try:
                local_targets = self._extract_local_targets(base_grid, raw_points)
            except RuntimeError as exc:
                print(f"[Model6] Reject occupancy path at safety_margin={safety_margin:.1f}: {exc}")
                continue
            if len(local_targets) < 2:
                print(f"[Model6] Reject occupancy path at safety_margin={safety_margin:.1f}: insufficient local targets")
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
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        min_x = min(float(start[0]), float(goal[0])) - config.OCCUPANCY_BOUNDS_MARGIN
        max_x = max(float(start[0]), float(goal[0])) + config.OCCUPANCY_BOUNDS_MARGIN
        min_y = min(float(start[1]), float(goal[1])) - config.OCCUPANCY_BOUNDS_MARGIN
        max_y = max(float(start[1]), float(goal[1])) + config.OCCUPANCY_BOUNDS_MARGIN

        span_x = max_x - min_x
        span_y = max_y - min_y
        min_span = float(config.OCCUPANCY_MIN_PLAN_SPAN)
        if span_x < min_span:
            pad = 0.5 * (min_span - span_x)
            min_x -= pad
            max_x += pad
        if span_y < min_span:
            pad = 0.5 * (min_span - span_y)
            min_y -= pad
            max_y += pad

        return (
            max(config.OCCUPANCY_MIN_X, min_x),
            min(config.OCCUPANCY_MAX_X, max_x),
            max(config.OCCUPANCY_MIN_Y, min_y),
            min(config.OCCUPANCY_MAX_Y, max_y),
        )

    def _build_grid_from_objects(self, bounds):
        """Build the occupancy grid by vertical top-down LOS checks.

        NED coordinates: more negative z means higher altitude. We treat the
        world as approximately planar and cast one vertical line segment per
        grid cell from high above to just above the ground plane.

        If the raycast path produces an empty map in the current AirSim build
        or scene, fall back to rasterising scene objects by pose + scale so we
        never continue planning on an all-free grid.
        """
        x_min, x_max, y_min, y_max = bounds
        grid = OccupancyGrid(x_min, x_max, y_min, y_max, config.OCCUPANCY_RESOLUTION)
        top_z = config.OCCUPANCY_GROUND_Z - config.OCCUPANCY_RAY_ABOVE_GROUND
        bottom_z = config.OCCUPANCY_GROUND_Z - config.OCCUPANCY_GROUND_CLEARANCE

        print(
            f"[Model6] Building occupancy grid: {grid.width}x{grid.height}, "
            f"x=[{grid.x_min:.1f}, {grid.x_max:.1f}], y=[{grid.y_min:.1f}, {grid.y_max:.1f}], "
            f"ray_z=[{top_z:.1f} -> {bottom_z:.1f}]"
        )

        count = 0
        for cell_y in range(grid.height):
            if cell_y % max(int(config.OCCUPANCY_RAY_PROGRESS_ROWS), 1) == 0 or cell_y == grid.height - 1:
                print(f"[Model6] Occupancy raycast progress: row {cell_y + 1}/{grid.height}")
            for cell_x in range(grid.width):
                world_x = grid.x_min + cell_x * grid.resolution
                world_y = grid.y_min + cell_y * grid.resolution
                if self._vertical_path_blocked(world_x, world_y, top_z, bottom_z):
                    grid.set_occupied((cell_x, cell_y), True)
                    count += 1

        print(
            f"[Model6] Raycast occupancy: {count} cells occupied "
            f"({grid.occupied_ratio():.1%})"
        )

        if count == 0:
            print("[Model6] Raycast occupancy is empty, fallback to scene-object occupancy.")
            fallback_grid = self._build_grid_from_scene_objects(bounds)
            if int(fallback_grid.occupied.sum()) > 0:
                return fallback_grid

        return grid

    def _build_grid_from_scene_objects(self, bounds):
        """Fallback occupancy builder using scene object pose + scale."""
        x_min, x_max, y_min, y_max = bounds
        grid = OccupancyGrid(x_min, x_max, y_min, y_max, config.OCCUPANCY_RESOLUTION)

        obstacle_names = self._collect_obstacle_names()
        if not obstacle_names:
            print("[Model6] WARNING: No scene obstacles detected for fallback occupancy.")
            return grid

        marked = 0
        skipped = 0
        for name in obstacle_names:
            try:
                pose = self.client.simGetObjectPose(name)
                scale = self.client.simGetObjectScale(name)
            except Exception:
                skipped += 1
                continue

            px = float(pose.position.x_val)
            py = float(pose.position.y_val)
            pz = float(pose.position.z_val)
            if not all(math.isfinite(v) for v in (px, py, pz)):
                skipped += 1
                continue
            if px == 0.0 and py == 0.0 and pz == 0.0:
                skipped += 1
                continue

            sx = abs(float(scale.x_val))
            sy = abs(float(scale.y_val))
            half_x = max(sx / 2.0, 0.5)
            half_y = max(sy / 2.0, 0.5)
            grid.mark_bbox((px, py), (half_x, half_y), margin=config.OCCUPANCY_OBSTACLE_RADIUS)
            marked += 1

        print(
            f"[Model6] Fallback object occupancy: {marked} obstacles marked, "
            f"{skipped} skipped, {grid.occupied.sum()} cells ({grid.occupied_ratio():.1%})"
        )
        return grid

    def _vertical_path_blocked(self, x, y, top_z, bottom_z):
        """Return True when a vertical LOS from sky to near-ground is blocked."""
        import airsim

        start = airsim.Vector3r(float(x), float(y), float(top_z))
        end = airsim.Vector3r(float(x), float(y), float(bottom_z))
        try:
            has_los = self.client.simTestLineOfSightBetweenPoints(start, end)
        except Exception:
            return True
        return not bool(has_los)

    def _raycast_cell(self, x, y, ground_z):
        """Find highest obstruction z at (x, y) using LOS binary search.

        Returns the z of the highest blocking point, or None if the path
        from high above to ground is completely clear.

        NED: lower z = higher altitude. Binary search finds the transition
        between clear (high) and blocked (low) z values.
        """
        import airsim

        hi_z = ground_z - config.OCCUPANCY_RAY_ABOVE_GROUND  # high above (more negative)
        lo_z = ground_z  # at ground (more positive)

        highest_clear = hi_z
        lowest_blocked = None

        for _ in range(20):
            if abs(lo_z - hi_z) < 0.5:
                break
            mid_z = (hi_z + lo_z) / 2.0
            try:
                pt = airsim.Vector3r(x, y, mid_z)
                has_los = self.client.simTestLineOfSightToPoint(pt)
            except Exception:
                has_los = False

            if has_los:
                # Clear to mid_z → obstruction is below
                highest_clear = mid_z
                hi_z = mid_z
            else:
                # Blocked at mid_z → obstruction at or above
                lowest_blocked = mid_z
                lo_z = mid_z

        if lowest_blocked is None:
            return None
        return highest_clear

    def _collect_obstacle_names(self):
        """Reuse the same pattern matching logic as waypoint_safety."""
        names = set()
        for pattern in config.OBSTACLE_OBJECT_PATTERNS:
            try:
                names.update(self.client.simListSceneObjects(f".*{pattern}.*"))
            except Exception:
                continue

        try:
            all_names = self.client.simListSceneObjects(".*")
        except Exception:
            return sorted(names)

        for name in all_names:
            try:
                if int(self.client.simGetSegmentationObjectID(name)) == int(config.OBSTACLE_SEGMENTATION_ID):
                    names.add(name)
                    continue
            except Exception:
                pass
            if _matches_any(name, config.OBSTACLE_OBJECT_PATTERNS):
                names.add(name)
        return sorted(names)

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
        clear_radius = float(config.OCCUPANCY_START_GOAL_CLEAR_RADIUS)
        if clear_radius > 0.0:
            working.clear_disc(start[:2], clear_radius)
            working.clear_disc(goal[:2], clear_radius)
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

    def _extract_local_targets(self, grid, raw_points):
        """Extract sparse local targets from the dense A* path.

        Walks the dense path and selects points that are at least
        LOCAL_TARGET_SPACING apart while maintaining clear LOS.  Turning
        points (direction changes beyond LOCAL_TARGET_MIN_SPACING) are
        preserved to guide the TD3 controller around obstacles.

        When LOS between last_selected and the furthest candidate is blocked,
        the furthest free point with clear LOS from last_selected is inserted
        as an intermediate waypoint.

        Every candidate point is verified free (not inside an obstacle).
        The goal is also checked and moved to the nearest free cell if needed.
        """
        if len(raw_points) <= 2:
            return raw_points

        selected = [raw_points[0]]
        last_selected = raw_points[0]
        previous_direction = None

        for index in range(1, len(raw_points) - 1):
            current = raw_points[index]

            # Skip occupied cells
            cell = grid.world_to_cell(current)
            if not grid.is_free(cell):
                previous_direction = None
                continue

            next_point = raw_points[index + 1]
            direction = np.sign(next_point[:2] - current[:2]).astype(np.int32)
            dist_from_last = float(np.linalg.norm(current - last_selected))
            is_turn = (
                previous_direction is not None
                and not np.array_equal(direction, previous_direction)
            )
            should_keep_turn = (
                config.LOCAL_TARGET_KEEP_TURNS
                and is_turn
                and dist_from_last >= config.LOCAL_TARGET_MIN_SPACING
            )
            should_keep_spacing = dist_from_last >= config.LOCAL_TARGET_SPACING

            if should_keep_turn or should_keep_spacing:
                if grid.is_line_clear(last_selected, current):
                    selected.append(current)
                    last_selected = current
                else:
                    # LOS blocked: insert the furthest free point with clear LOS.
                    backtrack_found = False
                    for k in range(index - 1, 0, -1):
                        bt = raw_points[k]
                        bt_cell = grid.world_to_cell(bt)
                        if grid.is_free(bt_cell) and grid.is_line_clear(last_selected, bt):
                            selected.append(bt)
                            last_selected = bt
                            backtrack_found = True
                            break
                    # Only append current if it's free AND reachable from last_selected.
                    cur_cell = grid.world_to_cell(current)
                    if grid.is_free(cur_cell) and grid.is_line_clear(last_selected, current):
                        selected.append(current)
                        last_selected = current
                    elif not backtrack_found:
                        # No viable backtrack and current is unreachable — just
                        # keep current as last_selected so the loop can retry
                        # from here with a different candidate.
                        pass
                    previous_direction = None

            previous_direction = direction

        # Handle goal: check if inside an obstacle
        goal_point = raw_points[-1]
        goal_cell = grid.world_to_cell(goal_point)
        if not grid.is_free(goal_cell):
            free_goal_cell = grid.nearest_free(goal_cell, config.OCCUPANCY_NEAREST_FREE_RADIUS)
            if free_goal_cell is not None:
                goal_world = grid.cell_to_world(free_goal_cell, float(goal_point[2]))
                print(
                    f"[Model6] WARNING: Goal is inside an obstacle! "
                    f"Moved to nearest free cell: {goal_world}"
                )
                if len(selected) > 1:
                    selected[-1] = goal_world
                else:
                    selected.append(goal_world)
            else:
                # No free cell found within search radius — try a larger one
                # before giving up.  If still nothing, append goal anyway and
                # let the lower-level executor / safety layer handle it.
                free_goal_cell = grid.nearest_free(goal_cell, config.OCCUPANCY_NEAREST_FREE_RADIUS * 2)
                if free_goal_cell is not None:
                    goal_world = grid.cell_to_world(free_goal_cell, float(goal_point[2]))
                    print(
                        f"[Model6] WARNING: Goal is inside an obstacle! "
                        f"Moved to nearest free cell (extended search): {goal_world}"
                    )
                    if len(selected) > 1:
                        selected[-1] = goal_world
                    else:
                        selected.append(goal_world)
                else:
                    print(
                        f"[Model6] WARNING: Goal is inside an obstacle and no free "
                        f"cell found within {config.OCCUPANCY_NEAREST_FREE_RADIUS * 2}m. "
                        f"Appending goal anyway — executor will handle it."
                    )
                    raise RuntimeError(
                        f"Goal remains inside an obstacle and no free cell was found within "
                        f"{config.OCCUPANCY_NEAREST_FREE_RADIUS * 2}m."
                    )
        elif not grid.is_line_clear(selected[-1], goal_point):
            raise RuntimeError(
                f"Final direct segment to goal is blocked from {selected[-1]} to {goal_point}."
            )
        elif float(np.linalg.norm(goal_point - selected[-1])) < config.LOCAL_TARGET_MIN_SPACING and len(selected) > 1:
            selected[-1] = goal_point
        else:
            selected.append(goal_point)

        return selected

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))))
