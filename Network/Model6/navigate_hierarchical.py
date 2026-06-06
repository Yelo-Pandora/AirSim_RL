#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np

MODEL6_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)

import config
if config.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, config.PROJECT_ROOT)

from graph_planner import WaypointGraphPlanner
from occupancy_planner import OccupancyAStarPlanner
from td3_executor import TD3SegmentExecutor


def print_plan(plan):
    print(
        f"[Model6] Upper {plan.get('planner', 'csv')} A*: {len(plan['points'])} local targets, "
        f"path_length={plan['path_length']:.2f}"
    )
    for index, point in enumerate(plan["points"]):
        print(f"  [{index:02d}] {plan['node_ids'][index]} region={plan['regions'][index]} point={point}")


def randomize_intermediate_target_altitudes(plan):
    """Randomize intermediate local target altitude for safer cruise."""
    if not config.LOCAL_TARGET_RANDOMIZE_INTERMEDIATE_ALTITUDE:
        return plan

    points = plan.get("points", [])
    if len(points) <= 2:
        return plan

    min_altitude, max_altitude = config.LOCAL_TARGET_INTERMEDIATE_ALTITUDE_RANGE
    min_altitude = float(min_altitude)
    max_altitude = float(max_altitude)
    if min_altitude > max_altitude:
        min_altitude, max_altitude = max_altitude, min_altitude

    adjusted_points = [
        np.array(point, dtype=np.float32).copy()
        for point in points
    ]
    rng = np.random.default_rng()
    altitudes = rng.uniform(
        min_altitude,
        max_altitude,
        size=len(adjusted_points) - 2,
    )
    for point, altitude in zip(adjusted_points[1:-1], altitudes):
        point[2] = float(config.OCCUPANCY_GROUND_Z - altitude)

    adjusted_plan = dict(plan)
    adjusted_plan["points"] = adjusted_points
    adjusted_plan["path_length"] = path_length(adjusted_points)
    ned_low = config.OCCUPANCY_GROUND_Z - max_altitude
    ned_high = config.OCCUPANCY_GROUND_Z - min_altitude
    print(
        "[Model6] Intermediate local target altitude randomized: "
        f"{min_altitude:.1f}-{max_altitude:.1f}m AGL "
        f"(NED z {ned_low:.1f} to {ned_high:.1f})."
    )
    return adjusted_plan


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            np.linalg.norm(points[index] - points[index - 1])
            for index in range(1, len(points))
        )
    )


def create_airsim_client():
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def main():
    parser = argparse.ArgumentParser(description="Model6 hierarchical A* + TD3 UAV navigation.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--td3-model", type=str, default=None, help="Model1 TD3 checkpoint path")
    parser.add_argument(
        "--planner",
        choices=["occupancy", "csv"],
        default=config.UPPER_PLANNER,
        help="Upper planner type. occupancy does not use dataset waypoint candidates.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Only run upper A* and print local targets")
    parser.add_argument(
        "--validate-waypoints",
        action="store_true",
        help="Connect to AirSim and filter local targets using segmentation obstacle poses",
    )
    args = parser.parse_args()

    start = np.array(args.start, dtype=np.float32)
    goal = np.array(args.goal, dtype=np.float32)

    executor = None
    safety = None
    client = None

    # Occupancy planner needs a raw AirSim client (pose + scale of scene objects).
    # CSV planner's waypoint validation uses AirSimWaypointSafety.
    if args.planner == "occupancy" or args.validate_waypoints or (not args.plan_only and config.VALIDATE_WAYPOINTS_WITH_AIRSIM):
        from waypoint_safety import AirSimWaypointSafety

        if args.plan_only:
            client = create_airsim_client()
            if args.planner == "occupancy":
                # Occupancy planner only needs the raw client, no safety wrapping.
                pass
            else:
                safety = AirSimWaypointSafety(client)
        else:
            executor = TD3SegmentExecutor(model_path=args.td3_model)
            client = executor.env.client
            if args.planner != "occupancy":
                safety = AirSimWaypointSafety(client)

    if args.planner == "occupancy":
        if client is None:
            raise RuntimeError("Occupancy planner requires an AirSim connection.")
        planner = OccupancyAStarPlanner(client=client)
    else:
        waypoint_filter = safety.is_safe if safety is not None else None
        planner = WaypointGraphPlanner(waypoint_filter=waypoint_filter)
    plan = planner.plan(start, goal)
    plan = randomize_intermediate_target_altitudes(plan)
    print_plan(plan)

    if args.plan_only:
        if executor is not None:
            executor.close()
        return

    if executor is None:
        executor = TD3SegmentExecutor(model_path=args.td3_model)
    try:
        summaries = executor.execute_path(plan["points"])
    finally:
        executor.close()

    arrived_segments = sum(1 for item in summaries if item["arrived"])
    print(f"[Model6] Finished {arrived_segments}/{len(plan['points']) - 1} segments.")


if __name__ == "__main__":
    main()
