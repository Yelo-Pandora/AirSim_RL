#!/usr/bin/env python
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

MODEL5_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL5_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)
MODEL2_PATH = os.path.join(NETWORK_DIR, "Model2")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODEL5_DIR not in sys.path:
    sys.path.insert(0, MODEL5_DIR)
if MODEL2_PATH not in sys.path:
    sys.path.insert(0, MODEL2_PATH)

import config
from astar import astar, simplify_path
from grid_map import OccupancyGrid
from segmentation_rules import SegmentationRules


class AStarAirSimNavigator:
    def __init__(self):
        from airsim_client.airsim_bridge import AirSimBridge

        self.bridge = AirSimBridge()
        self.client = self.bridge.client
        vehicles = self.client.listVehicles()
        self.vehicle_name = vehicles[0] if vehicles else config.VEHICLE_NAME
        self.segmentation = SegmentationRules(self.client)
        self.segmentation.apply()

    def load_dataset_points(self):
        regions = {}
        with open(config.DATASET_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                region = str(int(float(row.get("region", 0))))
                point = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float32,
                )
                regions.setdefault(region, []).append(point)
        return regions

    def sample_task(self):
        regions = self.load_dataset_points()
        keys = sorted(regions.keys())
        if len(keys) < 2:
            raise RuntimeError("Need at least two dataset regions for start/goal sampling.")
        start_region, goal_region = np.random.choice(keys, size=2, replace=False)
        start = regions[start_region][np.random.randint(len(regions[start_region]))].copy()
        goal = regions[goal_region][np.random.randint(len(regions[goal_region]))].copy()
        start[2] = config.CRUISE_ALTITUDE_Z
        goal[2] = config.CRUISE_ALTITUDE_Z
        return start, goal, start_region, goal_region

    def reset_vehicle(self, start, goal):
        import airsim

        print("[Model5] Resetting vehicle...")
        direction = goal - start
        yaw = math.atan2(float(direction[1]), float(direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
        pose = airsim.Pose(
            airsim.Vector3r(float(start[0]), float(start[1]), float(start[2])),
            airsim.to_quaternion(0.0, 0.0, yaw),
        )
        print("[Model5] Calling AirSim reset...")
        self.client.reset()
        print("[Model5] Re-enabling API control...")
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        print("[Model5] Teleporting to start pose...")
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)
        print("[Model5] Stabilizing at start...")
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.5, vehicle_name=self.vehicle_name).join()
        print("[Model5] Vehicle ready.")

    def _collect_obstacle_centers(self):
        print("[Model5] Collecting obstacle objects from segmentation/name rules...")
        obstacle_names = self.segmentation.obstacle_objects()
        print(f"[Model5] Found {len(obstacle_names)} candidate obstacle objects. Querying poses...")

        centers = []
        for idx, name in enumerate(obstacle_names, start=1):
            try:
                pose = self.client.simGetObjectPose(name)
            except Exception:
                continue
            x = float(pose.position.x_val)
            y = float(pose.position.y_val)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if x == 0.0 and y == 0.0:
                continue
            centers.append((x, y))
            if idx % 100 == 0:
                print(f"[Model5] Pose queries: {idx}/{len(obstacle_names)}, usable_centers={len(centers)}")
        return obstacle_names, centers

    def build_grid_from_centers(self, start, goal, obstacle_centers, obstacle_radius, safety_margin):
        grid = OccupancyGrid(
            config.MAP_X_MIN,
            config.MAP_X_MAX,
            config.MAP_Y_MIN,
            config.MAP_Y_MAX,
            config.MAP_RESOLUTION,
        )

        marked = 0
        for x, y in obstacle_centers:
            grid.mark_disc((x, y), obstacle_radius)
            marked += 1

        inflated = grid.inflated_copy(safety_margin)
        inflated.clear_disc(start[:2], safety_margin + obstacle_radius)
        inflated.clear_disc(goal[:2], safety_margin + obstacle_radius)
        occupied_ratio = inflated.occupied_count() / float(inflated.width * inflated.height)
        print(
            f"[Model5] Map: {inflated.width}x{inflated.height}, "
            f"marked={marked}, obstacle_radius={obstacle_radius:.2f}, "
            f"safety_margin={safety_margin:.2f}, occupied={inflated.occupied_count()} "
            f"({occupied_ratio:.1%})"
        )
        return inflated

    def plan(self, start, goal):
        _, obstacle_centers = self._collect_obstacle_centers()
        last_start_cell = None
        last_goal_cell = None
        cells = None
        grid = None

        for obstacle_radius, safety_margin in config.PLANNING_RADIUS_SCHEDULE:
            grid = self.build_grid_from_centers(
                start,
                goal,
                obstacle_centers,
                obstacle_radius=obstacle_radius,
                safety_margin=safety_margin,
            )
            raw_start_cell = grid.world_to_cell(start)
            raw_goal_cell = grid.world_to_cell(goal)
            start_cell = grid.nearest_free(raw_start_cell, config.NEAREST_FREE_SEARCH_RADIUS)
            goal_cell = grid.nearest_free(raw_goal_cell, config.NEAREST_FREE_SEARCH_RADIUS)
            last_start_cell = start_cell
            last_goal_cell = goal_cell
            if start_cell is None or goal_cell is None:
                print(
                    f"[Model5] No nearby free start/goal cell "
                    f"(start={raw_start_cell}->{start_cell}, goal={raw_goal_cell}->{goal_cell})"
                )
                continue
            if start_cell != raw_start_cell or goal_cell != raw_goal_cell:
                print(
                    f"[Model5] Snapped cells: start {raw_start_cell}->{start_cell}, "
                    f"goal {raw_goal_cell}->{goal_cell}"
                )

            cells = astar(grid, start_cell, goal_cell, allow_diagonal=config.ALLOW_DIAGONAL)
            if cells is not None:
                break
            print("[Model5] A* failed at this inflation level; trying a less conservative map...")

        if cells is None or grid is None:
            raise RuntimeError(
                f"A* failed from {last_start_cell} to {last_goal_cell} even after fallback inflation. "
                "The object-center map is likely over-blocked or the map bounds miss the route."
            )

        cells = simplify_path(cells)
        waypoints = [grid.cell_to_world(cell, z=config.CRUISE_ALTITUDE_Z) for cell in cells]
        waypoints[-1] = goal.astype(np.float32)
        print(f"[Model5] Planned {len(cells)} grid waypoints.")
        return waypoints

    def get_state(self):
        return self.bridge.get_state()

    def _yaw_from_velocity(self, velocity, fallback=0.0):
        if np.linalg.norm(velocity[:2]) < 1e-6:
            return fallback
        return math.atan2(float(velocity[1]), float(velocity[0]))

    def fly_waypoints(self, waypoints):
        import airsim

        print("[Model5] Executing path...")
        for idx, waypoint in enumerate(waypoints[1:], start=1):
            start_time = time.time()
            while True:
                state = self.get_state()
                pos = state["position"].astype(np.float32)
                error = waypoint - pos
                dist = float(np.linalg.norm(error))
                if dist <= config.WAYPOINT_REACHED_DIST:
                    break
                if time.time() - start_time > config.WAYPOINT_TIMEOUT_SEC:
                    print(f"\n[Model5] Waypoint {idx} timeout at dist={dist:.2f}m")
                    break

                direction = error / max(dist, 1e-6)
                speed = min(config.MAX_SPEED, dist)
                vel = direction * speed
                yaw = self._yaw_from_velocity(vel)
                self.client.moveByVelocityAsync(
                    float(vel[0]),
                    float(vel[1]),
                    float(vel[2]),
                    config.CONTROL_DT,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw)),
                    vehicle_name=self.vehicle_name,
                ).join()

                collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
                print(
                    f"\r[Model5] WP {idx:03d}/{len(waypoints)-1:03d} "
                    f"pos=({pos[0]:6.1f},{pos[1]:6.1f},{pos[2]:5.1f}) "
                    f"dist={dist:6.2f}",
                    end="",
                    flush=True,
                )
                if collision.has_collided:
                    print("\n[Model5] Collision detected; stopping.")
                    self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.2, vehicle_name=self.vehicle_name).join()
                    return False
        print("\n[Model5] Path execution complete.")
        return True

    def run(self, start=None, goal=None):
        if start is None or goal is None:
            start, goal, start_region, goal_region = self.sample_task()
        else:
            start = np.array(start, dtype=np.float32)
            goal = np.array(goal, dtype=np.float32)
            start_region = "manual"
            goal_region = "manual"
        print(f"[Model5] Start({start_region})={start}, Goal({goal_region})={goal}")
        self.reset_vehicle(start, goal)
        waypoints = self.plan(start, goal)
        return self.fly_waypoints(waypoints)


def main():
    parser = argparse.ArgumentParser(description="Run Model5 pure A* navigation in AirSim.")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    navigator = AStarAirSimNavigator()
    navigator.run(start=args.start, goal=args.goal)


if __name__ == "__main__":
    main()
