#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np

MODEL8_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL8_DIR not in sys.path:
    sys.path.insert(0, MODEL8_DIR)

import config

if config.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, config.PROJECT_ROOT)
if config.MODEL6_DIR not in sys.path:
    sys.path.insert(0, config.MODEL6_DIR)

from ddpg_executor import DDPGSegmentExecutor
from graph_planner import WaypointGraphPlanner
from occupancy_planner import OccupancyAStarPlanner


def print_plan(plan):
    print(
        f"[Model8] Upper {plan.get('planner', 'csv')} A*: {len(plan['points'])} local targets, "
        f"path_length={plan['path_length']:.2f}"
    )
    for index, point in enumerate(plan["points"]):
        print(f"  [{index:02d}] {plan['node_ids'][index]} region={plan['regions'][index]} point={point}")


def create_airsim_client():
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def main():
    parser = argparse.ArgumentParser(description="Model8 hierarchical A* + external DDPG UAV navigation.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--ddpg-model", type=str, default=None, help="External DDPG checkpoint path")
    parser.add_argument(
        "--planner",
        choices=["occupancy", "csv"],
        default="occupancy",
        help="Upper planner type. occupancy reuses Model6's occupancy A*.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Only run upper A* and print local targets")
    parser.add_argument(
        "--validate-waypoints",
        action="store_true",
        help="Connect to AirSim and filter CSV local targets using segmentation obstacle poses",
    )
    args = parser.parse_args()

    start = np.array(args.start, dtype=np.float32)
    goal = np.array(args.goal, dtype=np.float32)

    executor = None
    safety = None
    client = None

    if args.planner == "occupancy" or args.validate_waypoints:
        if args.plan_only:
            client = create_airsim_client()
        else:
            executor = DDPGSegmentExecutor(model_path=args.ddpg_model)
            client = executor.client

    if args.planner == "occupancy":
        if client is None:
            raise RuntimeError("Occupancy planner requires an AirSim connection.")
        planner = OccupancyAStarPlanner(client=client)
    else:
        if args.validate_waypoints:
            from waypoint_safety import AirSimWaypointSafety

            safety = AirSimWaypointSafety(client)
        waypoint_filter = safety.is_safe if safety is not None else None
        planner = WaypointGraphPlanner(waypoint_filter=waypoint_filter)

    plan = planner.plan(start, goal)
    print_plan(plan)

    if args.plan_only:
        if executor is not None:
            executor.close()
        return

    if executor is None:
        executor = DDPGSegmentExecutor(model_path=args.ddpg_model)
    try:
        summaries = executor.execute_path(plan["points"])
    finally:
        executor.close()

    arrived_segments = sum(1 for item in summaries if item["arrived"])
    print(f"[Model8] Finished {arrived_segments}/{len(plan['points']) - 1} segments.")


if __name__ == "__main__":
    main()
