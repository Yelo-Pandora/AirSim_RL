#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np

MODEL6_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)

from graph_planner import WaypointGraphPlanner
from td3_executor import TD3SegmentExecutor
import config


def print_plan(plan):
    print(
        f"[Model6] Upper A*: {len(plan['points'])} local targets, "
        f"path_length={plan['path_length']:.2f}, "
        f"k={plan['k_neighbors']}, max_edge={plan['max_edge_distance']:.1f}"
    )
    for index, point in enumerate(plan["points"]):
        print(f"  [{index:02d}] {plan['node_ids'][index]} region={plan['regions'][index]} point={point}")


def main():
    parser = argparse.ArgumentParser(description="Model6 hierarchical A* + TD3 UAV navigation.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--td3-model", type=str, default=None, help="Model1 TD3 checkpoint path")
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
    waypoint_filter = None
    if args.validate_waypoints or (not args.plan_only and config.VALIDATE_WAYPOINTS_WITH_AIRSIM):
        from waypoint_safety import AirSimWaypointSafety

        executor = TD3SegmentExecutor(model_path=args.td3_model)
        safety = AirSimWaypointSafety(executor.env.client)
        waypoint_filter = safety.is_safe

    planner = WaypointGraphPlanner(waypoint_filter=waypoint_filter)
    plan = planner.plan(start, goal)
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
