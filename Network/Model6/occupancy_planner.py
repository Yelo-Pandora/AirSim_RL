import heapq
import json
import math
import os

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


class TopDownOccupancyGrid(OccupancyGrid):
    """Occupancy grid loaded from a top-down map.

    The saved map uses image cells, with the AirSim local NED origin anchored
    at the vehicle spawn pixel.  For the current top-down camera orientation,
    increasing rows move toward local -x and increasing columns move toward
    local +y.
    """

    def __init__(self, metadata, occupied):
        self.metadata = metadata
        coverage = metadata["coverage"]
        meters_per_pixel = metadata["meters_per_pixel"]
        self.resolution_x = float(meters_per_pixel["x"])
        self.resolution_y = float(meters_per_pixel["y"])
        if abs(self.resolution_x - self.resolution_y) > 1e-6:
            raise RuntimeError(
                "Top-down occupancy map requires square pixels; got "
                f"x={self.resolution_x}, y={self.resolution_y}"
            )
        self.resolution = self.resolution_x
        self.occupied = np.asarray(occupied, dtype=bool)
        self.height, self.width = self.occupied.shape
        frame = metadata.get("coordinate_frame", {})
        if frame.get("origin") == "vehicle_spawn":
            origin = frame["origin_pixel"]
            self.origin_row = float(origin["row"])
            self.origin_col = float(origin["col"])
        else:
            # Legacy metadata was written before the image row direction was
            # corrected.  Infer the spawn origin from the local coverage bounds.
            self.origin_row = float(coverage["x_max"]) / self.resolution_x
            self.origin_col = -float(coverage["y_min"]) / self.resolution_y
        self.x_max = self.origin_row * self.resolution_x
        self.x_min = self.x_max - (self.height - 1) * self.resolution_x
        self.y_min = -self.origin_col * self.resolution_y
        self.y_max = self.y_min + (self.width - 1) * self.resolution_y

    def to_metadata(self):
        return {
            "coverage": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
            "meters_per_pixel": {
                "x": self.resolution_x,
                "y": self.resolution_y,
            },
            "coordinate_frame": {
                "origin": "vehicle_spawn",
                "origin_pixel": {
                    "row": self.origin_row,
                    "col": self.origin_col,
                },
            },
        }

    def world_to_cell(self, point):
        return (
            int(round(self.origin_row - float(point[0]) / self.resolution_x)),
            int(round(self.origin_col + float(point[1]) / self.resolution_y)),
        )

    def cell_to_world(self, cell, z):
        return np.array(
            [
                (self.origin_row - cell[0]) * self.resolution_x,
                (cell[1] - self.origin_col) * self.resolution_y,
                z,
            ],
            dtype=np.float32,
        )

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def is_free(self, cell):
        if not self.in_bounds(cell):
            return False
        return not bool(self.occupied[cell[0], cell[1]])

    def set_occupied(self, cell, occupied=True):
        if self.in_bounds(cell):
            self.occupied[cell[0], cell[1]] = occupied

    def is_line_clear(self, point_a, point_b):
        cell_a = self.world_to_cell(point_a)
        cell_b = self.world_to_cell(point_b)

        if not self.in_bounds(cell_a) or not self.in_bounds(cell_b):
            return False
        if cell_a == cell_b:
            return True

        row0, col0 = cell_a
        row1, col1 = cell_b
        d_row = abs(row1 - row0)
        d_col = abs(col1 - col0)
        step_row = 1 if row0 < row1 else -1
        step_col = 1 if col0 < col1 else -1
        err = d_row - d_col

        while True:
            if self.occupied[row0, col0]:
                return False
            if row0 == row1 and col0 == col1:
                break
            e2 = 2 * err
            if e2 > -d_col:
                err -= d_col
                row0 += step_row
            if e2 < d_row:
                err += d_row
                col0 += step_col
            if not self.in_bounds((row0, col0)):
                return False
        return True

    def occupied_ratio(self):
        return float(self.occupied.sum()) / float(self.height * self.width)


class OccupancyAStarPlanner:
    """Upper planner that derives local target points from an AirSim occupancy grid.

    The runtime path loads a prebuilt top-down occupancy map. Older LOS and
    scene-object pose/scale builders remain as fallbacks only.

    A* runs on the inflated working copy while LOS checks use the intact base
    grid so that start/goal clear-disc never hides buildings from the
    visibility test.
    """

    def __init__(self, client):
        self.client = client
        self._enforce_target_clearance = True

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        bounds = self._bounds(start, goal)
        last_cells = None

        if self.client is None:
            if not config.OCCUPANCY_USE_TOPDOWN_MAP:
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
            clearance_cost = self._build_clearance_cost(grid)
            raw_start = grid.world_to_cell(start)
            raw_goal = grid.world_to_cell(goal)
            start_cell, goal_cell = self._nearest_reachable_endpoint_cells(
                grid,
                raw_start,
                raw_goal,
                config.OCCUPANCY_NEAREST_FREE_RADIUS,
                clearance_cost=clearance_cost,
            )
            last_cells = (start_cell, goal_cell)
            print(
                f"[Model6] Occupancy A*: grid={grid.width}x{grid.height}, "
                f"occupied={grid.occupied_ratio():.1%}, safety_margin={safety_margin:.1f}"
            )
            if start_cell is None or goal_cell is None:
                continue

            cells = self._astar(grid, start_cell, goal_cell, clearance_cost=clearance_cost)
            if cells is None:
                continue

            raw_points = self._cells_to_points(
                grid,
                cells,
                start,
                goal,
                keep_original_endpoints=False,
            )
            # Use base_grid (intact buildings, no clear_disc) for all LOS / occupied checks.
            try:
                local_targets = self._extract_local_targets(base_grid, raw_points)
            except RuntimeError as exc:
                print(f"[Model6] Reject occupancy path at safety_margin={safety_margin:.1f}: {exc}")
                continue
            if len(local_targets) < 2:
                print(f"[Model6] Reject occupancy path at safety_margin={safety_margin:.1f}: insufficient local targets")
                continue
            return self._make_plan(local_targets, cells, "occupancy")

        if config.OCCUPANCY_LEGACY_FALLBACK:
            print("[Model6] Safe occupancy A* failed; retrying legacy A* without clearance penalties.")
            legacy_plan = self._plan_legacy(base_grid, start, goal)
            if legacy_plan is not None:
                return legacy_plan

        raise RuntimeError(f"Occupancy A* failed. Last cells={last_cells}")

    def _make_plan(self, local_targets, cells, planner):
        return {
            "points": local_targets,
            "node_ids": [f"astar_{index}" for index in range(len(local_targets))],
            "regions": ["occupancy"] * len(local_targets),
            "k_neighbors": 0,
            "max_edge_distance": 0.0,
            "path_length": self.path_length(local_targets),
            "planner": planner,
            "grid_cells": len(cells),
        }

    def _plan_legacy(self, base_grid, start, goal):
        last_cells = None
        previous_enforcement = self._enforce_target_clearance
        self._enforce_target_clearance = False
        try:
            for _, safety_margin in config.OCCUPANCY_RADIUS_FALLBACKS:
                grid = self._inflate_grid(base_grid, safety_margin)
                grid = self._clear_start_goal(grid, start, goal)
                raw_start = grid.world_to_cell(start)
                raw_goal = grid.world_to_cell(goal)
                start_cell, goal_cell = self._nearest_reachable_endpoint_cells(
                    grid,
                    raw_start,
                    raw_goal,
                    config.OCCUPANCY_NEAREST_FREE_RADIUS,
                )
                last_cells = (start_cell, goal_cell)
                print(
                    f"[Model6] Legacy occupancy A*: grid={grid.width}x{grid.height}, "
                    f"occupied={grid.occupied_ratio():.1%}, safety_margin={safety_margin:.1f}"
                )
                if start_cell is None or goal_cell is None:
                    continue

                cells = self._astar(grid, start_cell, goal_cell)
                if cells is None:
                    continue

                raw_points = self._cells_to_points(
                    grid,
                    cells,
                    start,
                    goal,
                    keep_original_endpoints=False,
                )
                try:
                    local_targets = self._extract_local_targets(base_grid, raw_points)
                except RuntimeError as exc:
                    print(f"[Model6] Reject legacy occupancy path at safety_margin={safety_margin:.1f}: {exc}")
                    continue
                if len(local_targets) < 2:
                    print(f"[Model6] Reject legacy occupancy path at safety_margin={safety_margin:.1f}: insufficient local targets")
                    continue
                print("[Model6] WARNING: using legacy occupancy fallback; local targets may be closer to obstacles.")
                return self._make_plan(local_targets, cells, "occupancy_legacy")
        finally:
            self._enforce_target_clearance = previous_enforcement

        print(f"[Model6] Legacy occupancy A* also failed. Last cells={last_cells}")
        return None

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
        """Build the occupancy grid from scene-object bounds.

        The old top-down LOS path is kept only as an optional fallback because
        LOS can ignore building collision in the current scene.
        """
        if config.OCCUPANCY_USE_TOPDOWN_MAP:
            topdown_grid = self._build_grid_from_topdown_map()
            if int(topdown_grid.occupied.sum()) > 0:
                return topdown_grid
            print("[Model6] Top-down occupancy map is empty.")

        if config.OCCUPANCY_USE_OBJECT_BOUNDS:
            object_grid = self._build_grid_from_scene_objects(bounds)
            if int(object_grid.occupied.sum()) > 0:
                return object_grid
            print("[Model6] Object-bound occupancy is empty.")
            if not config.OCCUPANCY_USE_LOS_FALLBACK:
                return object_grid

        return self._build_grid_from_los(bounds)

    def _build_grid_from_topdown_map(self):
        metadata_path = config.OCCUPANCY_TOPDOWN_METADATA
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)

        occupancy_path = metadata["files"]["occupancy_npy"]
        if not os.path.isabs(occupancy_path):
            occupancy_path = os.path.join(config.PROJECT_ROOT, occupancy_path)
        occupied = np.load(occupancy_path)
        grid = TopDownOccupancyGrid(metadata, occupied)
        print(
            f"[Model6] Top-down occupancy: {grid.height}x{grid.width}, "
            f"x=[{grid.x_min:.1f}, {grid.x_max:.1f}], "
            f"y=[{grid.y_min:.1f}, {grid.y_max:.1f}], "
            f"occupied={grid.occupied.sum()} cells ({grid.occupied_ratio():.1%})"
        )
        return grid

    def _build_grid_from_los(self, bounds):
        """Build the occupancy grid by vertical top-down LOS checks."""
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
        """Fallback occupancy builder using scene-object pose + scale."""
        x_min, x_max, y_min, y_max = bounds
        grid = OccupancyGrid(x_min, x_max, y_min, y_max, config.OCCUPANCY_RESOLUTION)

        obstacle_names = self._collect_obstacle_names()
        if not obstacle_names:
            print("[Model6] WARNING: No scene obstacles detected for fallback occupancy.")
            return grid

        marked = 0
        skipped = 0
        for name in obstacle_names:
            obstacle_bounds = self._object_bounds_xy_from_pose_scale(name)
            if obstacle_bounds is None:
                skipped += 1
                continue

            center_xy, half_size_xy = obstacle_bounds
            grid.mark_bbox(
                center_xy,
                half_size_xy,
                margin=config.OCCUPANCY_OBSTACLE_RADIUS,
            )
            marked += 1

        print(
            f"[Model6] Object occupancy: {marked} pose/scale obstacles marked, "
            f"{skipped} skipped, {grid.occupied.sum()} cells "
            f"({grid.occupied_ratio():.1%})"
        )
        return grid

    def _object_bounds_xy_from_pose_scale(self, name):
        try:
            pose = self.client.simGetObjectPose(name)
            scale = self.client.simGetObjectScale(name)
        except Exception:
            return None

        px = float(pose.position.x_val)
        py = float(pose.position.y_val)
        pz = float(pose.position.z_val)
        if not all(math.isfinite(v) for v in (px, py, pz)):
            return None
        if px == 0.0 and py == 0.0 and pz == 0.0:
            return None

        sx = abs(float(scale.x_val))
        sy = abs(float(scale.y_val))
        center_xy = (px, py)
        half_size_xy = (max(sx / 2.0, 0.5), max(sy / 2.0, 0.5))
        return center_xy, half_size_xy

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
        """Collect obstacle actors by name, with optional segmentation lookup."""
        names = set()
        for pattern in config.OBSTACLE_OBJECT_PATTERNS:
            try:
                names.update(self.client.simListSceneObjects(f".*{pattern}.*"))
            except Exception:
                continue

        if not config.OCCUPANCY_INCLUDE_SEGMENTATION_OBSTACLES:
            return sorted(names)

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
        if isinstance(base_grid, TopDownOccupancyGrid):
            inflated = TopDownOccupancyGrid(base_grid.to_metadata(), base_grid.occupied.copy())
            if config.OCCUPANCY_TOPDOWN_PREINFLATED:
                return inflated
        else:
            inflated = OccupancyGrid(
                base_grid.x_min, base_grid.x_max,
                base_grid.y_min, base_grid.y_max,
                base_grid.resolution,
            )
            inflated.occupied[:] = base_grid.occupied

        radius_cells = int(math.ceil(float(safety_margin) / base_grid.resolution))
        rows, cols = np.where(base_grid.occupied)
        for row, col in zip(rows, cols):
            for d_row in range(-radius_cells, radius_cells + 1):
                for d_col in range(-radius_cells, radius_cells + 1):
                    if d_row * d_row + d_col * d_col <= radius_cells * radius_cells:
                        inflated.set_occupied((row + d_row, col + d_col), True)

        return inflated

    def _clear_start_goal(self, grid, start, goal):
        """Return a copy of *grid* with discs cleared around start and goal
        for A* pathfinding.  Leaves the original grid untouched.
        """
        if isinstance(grid, TopDownOccupancyGrid):
            working = TopDownOccupancyGrid(grid.to_metadata(), grid.occupied.copy())
        else:
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

    def _nearest_reachable_endpoint_cells(self, grid, raw_start, raw_goal, radius, clearance_cost=None):
        start_cell = grid.nearest_free(raw_start, radius)
        goal_cell = grid.nearest_free(raw_goal, radius)
        if start_cell is not None and goal_cell is not None:
            if self._astar(grid, start_cell, goal_cell, clearance_cost=clearance_cost) is not None:
                return start_cell, goal_cell

        max_radius = max(int(math.ceil(radius * 3.0)), int(math.ceil(radius + 24.0)))
        start_candidates = self._free_candidates_by_component(grid, raw_start, max_radius)
        goal_candidates = self._free_candidates_by_component(grid, raw_goal, max_radius)
        if not start_candidates or not goal_candidates:
            return start_cell, goal_cell

        goal_by_component = {}
        for distance, cell, component in goal_candidates:
            goal_by_component.setdefault(component, []).append((distance, cell))

        best = None
        for start_distance, start_candidate, component in start_candidates:
            if component not in goal_by_component:
                continue
            goal_distance, goal_candidate = goal_by_component[component][0]
            score = start_distance + goal_distance
            if best is None or score < best[0]:
                best = (score, start_candidate, goal_candidate)

        if best is None:
            return start_cell, goal_cell

        print(
            "[Model6] Endpoint nearest-free cells are disconnected; "
            f"using reachable cells start={best[1]}, goal={best[2]}, "
            f"search_radius={max_radius}."
        )
        return best[1], best[2]

    def _free_candidates_by_component(self, grid, center, radius):
        components = self._free_components(grid)
        center_row, center_col = center
        radius = int(math.ceil(radius))
        row_min = max(0, center_row - radius)
        row_max = min(grid.height - 1, center_row + radius)
        col_min = max(0, center_col - radius)
        col_max = min(grid.width - 1, center_col + radius)
        candidates = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                component = int(components[row, col])
                if component < 0:
                    continue
                distance = math.hypot(row - center_row, col - center_col)
                if distance <= radius:
                    candidates.append((distance, (row, col), component))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _free_components(self, grid):
        components = np.full(grid.occupied.shape, -1, dtype=np.int32)
        component_id = 0
        for row in range(grid.height):
            for col in range(grid.width):
                if grid.occupied[row, col] or components[row, col] >= 0:
                    continue
                stack = [(row, col)]
                components[row, col] = component_id
                while stack:
                    cur_row, cur_col = stack.pop()
                    for next_row, next_col in self._neighbors((cur_row, cur_col)):
                        if not grid.in_bounds((next_row, next_col)):
                            continue
                        if grid.occupied[next_row, next_col] or components[next_row, next_col] >= 0:
                            continue
                        components[next_row, next_col] = component_id
                        stack.append((next_row, next_col))
                component_id += 1
        return components

    def _build_clearance_cost(self, grid):
        min_clearance = self._effective_target_clearance(grid, config.OCCUPANCY_ASTAR_MIN_CLEARANCE)
        preferred_clearance = self._effective_target_clearance(grid, config.OCCUPANCY_ASTAR_PREFERRED_CLEARANCE)
        if preferred_clearance <= 0.0 or preferred_clearance <= min_clearance:
            return None

        radius_cells = int(math.ceil(preferred_clearance / grid.resolution))
        cost = np.zeros(grid.occupied.shape, dtype=np.float32)
        occupied_axis0, occupied_axis1 = np.where(grid.occupied)
        for axis0, axis1 in zip(occupied_axis0, occupied_axis1):
            occupied_cell = (axis0, axis1) if isinstance(grid, TopDownOccupancyGrid) else (axis1, axis0)
            for d0 in range(-radius_cells, radius_cells + 1):
                for d1 in range(-radius_cells, radius_cells + 1):
                    distance = math.hypot(d0, d1) * grid.resolution
                    if distance > preferred_clearance:
                        continue
                    near_cell = (occupied_cell[0] + d0, occupied_cell[1] + d1)
                    if not grid.in_bounds(near_cell):
                        continue
                    if not grid.is_free(near_cell):
                        continue
                    if distance < min_clearance:
                        self._set_cell_cost(grid, cost, near_cell, np.inf)
                    else:
                        penalty = (preferred_clearance - distance) / (preferred_clearance - min_clearance)
                        old_cost = self._cell_cost(grid, cost, near_cell)
                        self._set_cell_cost(grid, cost, near_cell, max(old_cost, float(penalty)))
        return cost

    @staticmethod
    def _cell_cost(grid, cost, cell):
        if isinstance(grid, TopDownOccupancyGrid):
            return float(cost[cell[0], cell[1]])
        return float(cost[cell[1], cell[0]])

    @staticmethod
    def _set_cell_cost(grid, cost, cell, value):
        if isinstance(grid, TopDownOccupancyGrid):
            cost[cell[0], cell[1]] = value
        else:
            cost[cell[1], cell[0]] = value

    def _astar(self, grid, start, goal, clearance_cost=None):
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
                if clearance_cost is not None:
                    cell_clearance_cost = self._cell_cost(grid, clearance_cost, neighbor)
                    if math.isinf(cell_clearance_cost):
                        if neighbor != goal and neighbor != start:
                            continue
                    else:
                        step_cost += step_cost * config.OCCUPANCY_ASTAR_CLEARANCE_COST_WEIGHT * cell_clearance_cost
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

    def _cells_to_points(self, grid, cells, start, goal, keep_original_endpoints=True):
        total_xy = max(float(np.linalg.norm(goal[:2] - start[:2])), 1e-6)
        points = []
        for cell in cells:
            xy = grid.cell_to_world(cell, 0.0)
            progress = float(np.linalg.norm(xy[:2] - start[:2])) / total_xy
            z = float(start[2] + np.clip(progress, 0.0, 1.0) * (goal[2] - start[2]))
            points.append(np.array([xy[0], xy[1], z], dtype=np.float32))
        if keep_original_endpoints:
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
                if not self._enforce_target_clearance:
                    if grid.is_line_clear(last_selected, current):
                        selected.append(current)
                        last_selected = current
                    else:
                        self._append_reachable_target_or_bridge(grid, selected, raw_points, last_selected, current, index)
                        last_selected = selected[-1]
                        previous_direction = None
                    previous_direction = direction
                    continue

                safe_current = self._safe_target_near_index(
                    grid,
                    raw_points,
                    index,
                    last_selected,
                    min_distance=config.LOCAL_TARGET_MIN_SPACING,
                )
                if safe_current is not None:
                    selected.append(safe_current)
                    last_selected = safe_current
                else:
                    # LOS blocked: insert the furthest free point with clear LOS.
                    backtrack_found = False
                    for k in range(index - 1, 0, -1):
                        bt = raw_points[k]
                        bt_cell = grid.world_to_cell(bt)
                        if (
                            grid.is_free(bt_cell)
                            and self._has_target_clearance(grid, bt)
                            and grid.is_line_clear(last_selected, bt)
                        ):
                            selected.append(bt)
                            last_selected = bt
                            backtrack_found = True
                            break
                    # Only append current if it's free AND reachable from last_selected.
                    cur_cell = grid.world_to_cell(current)
                    if (
                        grid.is_free(cur_cell)
                        and self._has_target_clearance(grid, current)
                        and grid.is_line_clear(last_selected, current)
                    ):
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
            self._append_blocked_tail(grid, selected, raw_points)
        elif float(np.linalg.norm(goal_point - selected[-1])) < config.LOCAL_TARGET_MIN_SPACING and len(selected) > 1:
            selected[-1] = goal_point
        else:
            selected.append(goal_point)

        return selected

    def _append_reachable_target_or_bridge(self, grid, selected, raw_points, last_selected, current, index):
        backtrack_found = False
        for k in range(index - 1, 0, -1):
            bt = raw_points[k]
            bt_cell = grid.world_to_cell(bt)
            if grid.is_free(bt_cell) and grid.is_line_clear(last_selected, bt):
                selected.append(bt)
                last_selected = bt
                backtrack_found = True
                break
        cur_cell = grid.world_to_cell(current)
        if grid.is_free(cur_cell) and grid.is_line_clear(last_selected, current):
            selected.append(current)
        elif not backtrack_found:
            pass

    def _has_target_clearance(self, grid, point, clearance=None):
        if clearance is None:
            clearance = config.LOCAL_TARGET_CLEARANCE
        clearance = self._effective_target_clearance(grid, clearance)
        cell = grid.world_to_cell(point)
        if not grid.is_free(cell):
            return False
        if clearance <= 0.0:
            return True

        radius_cells = int(math.ceil(clearance / grid.resolution))
        for d0 in range(-radius_cells, radius_cells + 1):
            for d1 in range(-radius_cells, radius_cells + 1):
                if math.hypot(d0, d1) * grid.resolution > clearance:
                    continue
                neighbor = (cell[0] + d0, cell[1] + d1)
                if not grid.in_bounds(neighbor) or not grid.is_free(neighbor):
                    return False
        return True

    @staticmethod
    def _effective_target_clearance(grid, clearance):
        clearance = float(clearance)
        if not isinstance(grid, TopDownOccupancyGrid):
            return clearance
        inflated = float(grid.metadata.get("inflate_meters", 0.0))
        return max(0.0, clearance - inflated)

    def _safe_target_near_index(self, grid, raw_points, preferred_index, last_selected, min_distance=0.0):
        max_offset = max(int(math.ceil(config.LOCAL_TARGET_SPACING / grid.resolution)), 1)
        last_raw_index = len(raw_points) - 2
        for offset in range(0, max_offset + 1):
            candidate_indices = [preferred_index] if offset == 0 else [
                preferred_index - offset,
                preferred_index + offset,
            ]
            for candidate_index in candidate_indices:
                if candidate_index < 1 or candidate_index > last_raw_index:
                    continue
                candidate = raw_points[candidate_index]
                if float(np.linalg.norm(candidate - last_selected)) < float(min_distance):
                    continue
                if not self._has_target_clearance(grid, candidate):
                    continue
                if grid.is_line_clear(last_selected, candidate):
                    return candidate
        return None

    def _append_blocked_tail(self, grid, selected, raw_points):
        goal_point = raw_points[-1]
        guard = 0
        while not grid.is_line_clear(selected[-1], goal_point):
            guard += 1
            if guard > len(raw_points):
                raise RuntimeError(
                    f"Final direct segment to goal is blocked from {selected[-1]} "
                    f"to {goal_point}."
                )

            bridge = None
            last_cell = grid.world_to_cell(selected[-1])
            for index in range(len(raw_points) - 2, 0, -1):
                candidate = raw_points[index]
                candidate_cell = grid.world_to_cell(candidate)
                if candidate_cell == last_cell:
                    break
                if (
                    grid.is_free(candidate_cell)
                    and (
                        not self._enforce_target_clearance
                        or self._has_target_clearance(grid, candidate)
                    )
                    and grid.is_line_clear(selected[-1], candidate)
                ):
                    bridge = candidate
                    break

            if bridge is None:
                raise RuntimeError(
                    f"Final direct segment to goal is blocked from {selected[-1]} "
                    f"to {goal_point}, and no clear bridge point was found."
                )

            if float(np.linalg.norm(bridge - selected[-1])) < 1e-3:
                raise RuntimeError(
                    f"Final direct segment to goal is blocked from {selected[-1]} "
                    f"to {goal_point}, and bridge search stalled."
                )
            selected.append(bridge)

        if float(np.linalg.norm(goal_point - selected[-1])) < config.LOCAL_TARGET_MIN_SPACING and len(selected) > 1:
            selected[-1] = goal_point
        else:
            selected.append(goal_point)

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))))
