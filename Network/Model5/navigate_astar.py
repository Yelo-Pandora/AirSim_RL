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
            z = float(pose.position.z_val)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
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

        cells = simplify_path(cells, grid=grid)
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

        if len(waypoints) < 2:
            print("[Model5] No waypoints to fly.")
            return True

        print("[Model5] Executing path with real-time segmentation obstacle detection...")

        goal = waypoints[-1]

        for idx in range(1, len(waypoints)):
            wp_xy = np.array([waypoints[idx][0], waypoints[idx][1], config.CRUISE_ALTITUDE_Z], dtype=np.float32)
            if not self._fly_to_point_smart(wp_xy, goal):
                return False

        # Descend to final goal
        print(f"[Model5] Descending from z={config.CRUISE_ALTITUDE_Z:.1f} to goal z={goal[2]:.1f}")
        if not self._fly_to_point(goal):
            return False

        print("\n[Model5] Path execution complete.")
        return True

    def _get_seg_obstacle_ratio(self):
        """
        Get the fraction of obstacle pixels in the center region of the forward segmentation camera.
        Uses grayscale mode (single channel) where pixel value = segmentation object ID.
        Returns: float in [0, 1], or None on failure.
        """
        import airsim

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, True, False)
            ], vehicle_name=self.vehicle_name)
            if not responses or not responses[0].image_data_uint8:
                return None

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            h, w = responses[0].height, responses[0].width
            img = img1d.reshape(h, w)

            # Crop center region
            ch = int(h * config.SEG_CAM_CENTER_CROP)
            cw = int(w * config.SEG_CAM_CENTER_CROP)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            center = img[y0:y0+ch, x0:x0+cw]

            # In grayscale mode, pixel value = segmentation object ID
            # Obstacles have seg_id=2 (set by segmentation_rules.py)
            # Ground/background has seg_id=0
            # Free objects (sidewalk, street) have seg_id=1
            obstacle_mask = (center == 2)

            return float(obstacle_mask.mean())
        except Exception as e:
            print(f"\n[Model5] Seg camera error: {e}")
            return None

    def _get_depth_obstacle_ratio(self):
        """
        Get the fraction of close-range obstacle pixels in the forward depth camera.
        Returns: float in [0, 1] representing fraction of pixels closer than DEPTH_CAM_OBSTACLE_DIST,
                 or None on failure.
        """
        import airsim

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest('0', airsim.ImageType.DepthVis, True, False)
            ], vehicle_name=self.vehicle_name)
            if not responses or not responses[0].image_data_float:
                return None

            # image_data_float is already a list of floats when pixels_as_float=True
            depth1d = np.array(responses[0].image_data_float, dtype=np.float32)
            h, w = responses[0].height, responses[0].width
            depth = depth1d.reshape(h, w)

            # Crop center region
            ch = int(h * config.DEPTH_CAM_CENTER_CROP)
            cw = int(w * config.DEPTH_CAM_CENTER_CROP)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            center_depth = depth[y0:y0+ch, x0:x0+cw]

            # Filter out invalid depths (0 or inf)
            valid = (center_depth > 0.1) & (center_depth < 100.0)
            if not valid.any():
                return None

            # Fraction of valid pixels that are closer than threshold
            too_close = (center_depth <= config.DEPTH_CAM_OBSTACLE_DIST) & valid
            return float(too_close.sum() / valid.sum())
        except Exception as e:
            print(f"\n[Model5] Depth camera error: {e}")
            return None

    def _should_climb(self):
        """Check if either depth or segmentation camera sees obstacles blocking forward path."""
        # Depth camera check (primary, measures actual distance)
        depth_ratio = self._get_depth_obstacle_ratio()
        if depth_ratio is not None and depth_ratio > 0.05:
            print(f"\n  [Depth] Close obstacle ratio {depth_ratio:.1%} -> CLIMB")
            return True

        # Segmentation camera check (backup, detects obstacle type)
        seg_ratio = self._get_seg_obstacle_ratio()
        if seg_ratio is not None:
            should = seg_ratio > config.SEG_CAM_OBSTACLE_THRESHOLD
            if should:
                print(f"\n  [Seg] Obstacle ratio {seg_ratio:.1%} > {config.SEG_CAM_OBSTACLE_THRESHOLD:.0%} -> CLIMB")
            return should

        return False

    def _fly_to_point_smart(self, target_xy, goal):
        """
        Fly toward target_xy while using depth+segmentation cameras to detect obstacles.
        If obstacle detected ahead, STOP horizontal motion and climb to SAFE_ALTITUDE_Z.
        When close to goal XY, descend to goal altitude.
        """
        import airsim

        start_time = time.time()
        cycle_count = 0
        climb_triggered = False

        while True:
            state = self.get_state()
            pos = state["position"].astype(np.float32)
            error = target_xy - pos
            error[2] = 0
            dist_xy = float(np.linalg.norm(error[:2]))

            if dist_xy <= config.WAYPOINT_REACHED_DIST:
                break
            if time.time() - start_time > config.WAYPOINT_TIMEOUT_SEC * 2:
                print(f"\n[Model5] Point timeout at dist={dist_xy:.2f}m, z={pos[2]:.1f}")
                break

            cycle_count += 1

            # === Obstacle detection (every N cycles) ===
            obstacle_detected = False
            if cycle_count % config.DEPTH_CAM_CHECK_INTERVAL == 0:
                if pos[2] > config.SAFE_ALTITUDE_Z:  # below safe altitude, check for obstacles
                    obstacle_detected = self._should_climb()
                    if obstacle_detected:
                        print(f"[Model5] Obstacle ahead! Climbing from z={pos[2]:.1f} to {config.SAFE_ALTITUDE_Z:.1f}...")
                        self._climb_to_safe_altitude()
                        climb_triggered = True
                        # Reset timeout after climbing — climb takes time
                        start_time = time.time()
                        cycle_count = 0
                        # Re-check position after climb
                        state = self.get_state()
                        pos = state["position"].astype(np.float32)
                        error = target_xy - pos
                        error[2] = 0
                        dist_xy = float(np.linalg.norm(error[:2]))
                        if dist_xy <= config.WAYPOINT_REACHED_DIST:
                            break

            # Determine desired altitude
            dist_to_goal = float(np.linalg.norm(goal[:2] - pos[:2]))
            if dist_to_goal < 10.0:
                # Within 10m of goal XY: descend gradually
                descend_frac = dist_to_goal / 10.0
                desired_z = config.CRUISE_ALTITUDE_Z + (config.SAFE_ALTITUDE_Z - config.CRUISE_ALTITUDE_Z) * descend_frac
                desired_z = max(config.CRUISE_ALTITUDE_Z, desired_z)
            elif climb_triggered and pos[2] > config.SAFE_ALTITUDE_Z:
                desired_z = config.SAFE_ALTITUDE_Z
            else:
                desired_z = pos[2]

            # === Compute velocity ===
            if obstacle_detected:
                # Just finished climbing: zero horizontal velocity, only Z adjustment
                vel = np.array([0.0, 0.0, config.SAFE_ALTITUDE_Z - pos[2]], dtype=np.float32)
                yaw = 0.0
            else:
                # Normal flight: move toward XY target with Z adjustment
                target_pos = np.array([target_xy[0], target_xy[1], desired_z], dtype=np.float32)
                vel_error = target_pos - pos
                vel_dist = float(np.linalg.norm(vel_error))
                if vel_dist > 1e-6:
                    direction = vel_error / vel_dist
                    speed = min(config.MAX_SPEED, vel_dist)
                    vel = direction * speed
                else:
                    vel = np.zeros(3)
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
                f"\r[Model5] pos=({pos[0]:6.1f},{pos[1]:6.1f},{pos[2]:5.1f}) "
                f"dist={dist_xy:6.2f}",
                end="",
                flush=True,
            )
            if collision.has_collided:
                print("\n[Model5] Collision detected; stopping.")
                self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.2, vehicle_name=self.vehicle_name).join()
                return False
        return True

    def _climb_to_safe_altitude(self):
        """
        Climb directly to SAFE_ALTITUDE_Z using a sustained upward velocity.
        Uses negative z velocity because AirSim uses NED coordinates (z down).
        """
        import airsim

        climb_start = time.time()
        climb_timeout = 60
        climb_speed = 3.0  # m/s upward

        while time.time() - climb_start < climb_timeout:
            state = self.get_state()
            pos = state["position"].astype(np.float32)

            if pos[2] <= config.SAFE_ALTITUDE_Z:
                print(f"[Model5] Reached safe altitude z={pos[2]:.1f}")
                return

            # Fly straight UP (negative z = up in NED)
            self.client.moveByVelocityAsync(
                0.0, 0.0, -climb_speed, config.CONTROL_DT,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.vehicle_name,
            ).join()

            # Log progress periodically
            elapsed = time.time() - climb_start
            if int(elapsed * 10) % 50 == 0:
                print(f"  [Climb] z={pos[2]:.1f} -> target {config.SAFE_ALTITUDE_Z:.1f}")

        print(f"[Model5] Climb timeout at z={pos[2]:.1f}")

    def _fly_to_point(self, target):
        """Fly to a single target point with velocity control. Returns False on collision."""
        import airsim

        start_time = time.time()
        while True:
            state = self.get_state()
            pos = state["position"].astype(np.float32)
            error = target - pos
            dist = float(np.linalg.norm(error))
            if dist <= config.WAYPOINT_REACHED_DIST:
                break
            if time.time() - start_time > config.WAYPOINT_TIMEOUT_SEC:
                print(f"\n[Model5] Point timeout at dist={dist:.2f}m")
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
                f"\r[Model5] pos=({pos[0]:6.1f},{pos[1]:6.1f},{pos[2]:5.1f}) "
                f"dist={dist:6.2f}",
                end="",
                flush=True,
            )
            if collision.has_collided:
                print("\n[Model5] Collision detected; stopping.")
                self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.2, vehicle_name=self.vehicle_name).join()
                return False
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
