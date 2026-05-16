#!/usr/bin/env python

import argparse
import os
import sys
import time

import numpy as np

MODEL4_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL4_DIR not in sys.path:
    sys.path.insert(0, MODEL4_DIR)

import airsim
import config
from task_source import Model4TaskSource


def _vec3(point):
    return airsim.Vector3r(float(point[0]), float(point[1]), float(point[2]))


def _teleport_vehicle(client, vehicle_name, point, look_at=None, settle_sec=1.0):
    yaw = 0.0
    if look_at is not None:
        delta = np.array(look_at, dtype=np.float32) - np.array(point, dtype=np.float32)
        if np.linalg.norm(delta[:2]) > 1e-6:
            yaw = float(np.arctan2(delta[1], delta[0]))

    pose = airsim.Pose(
        _vec3(point),
        airsim.to_quaternion(0.0, 0.0, yaw),
    )
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
    time.sleep(settle_sec)


def _plot_route(client, task, duration, point_size, persistent):
    route_points = [task["start"]] + list(task["route_subgoals"])
    airsim_points = [_vec3(p) for p in route_points]
    route_line = [_vec3(p) for p in [task["start"]] + list(task["route_points"])]

    client.simFlushPersistentMarkers()

    client.simPlotPoints(
        [airsim_points[0]],
        color_rgba=[0.0, 1.0, 0.0, 1.0],
        size=float(point_size) * 1.4,
        duration=float(duration),
        is_persistent=persistent,
    )
    if len(airsim_points) > 1:
        client.simPlotPoints(
            airsim_points[1:-1],
            color_rgba=[1.0, 0.85, 0.1, 1.0],
            size=float(point_size),
            duration=float(duration),
            is_persistent=persistent,
        )
        client.simPlotPoints(
            [airsim_points[-1]],
            color_rgba=[1.0, 0.1, 0.1, 1.0],
            size=float(point_size) * 1.25,
            duration=float(duration),
            is_persistent=persistent,
        )
    if len(route_line) >= 2:
        client.simPlotLineStrip(
            route_line,
            color_rgba=[0.1, 0.8, 1.0, 1.0],
            thickness=8.0,
            duration=float(duration),
            is_persistent=persistent,
        )


def _teleport_preview(source, task, pause_sec):
    client = source.client
    vehicle_name = source.vehicle_name
    preview_points = [task["start"]] + list(task["route_subgoals"])
    print("\nTeleport preview:")
    for idx, point in enumerate(preview_points):
        label = "start" if idx == 0 else ("global_goal" if idx == len(preview_points) - 1 else f"subgoal_{idx}")
        look_at = preview_points[min(idx + 1, len(preview_points) - 1)]
        print(f"  teleport -> {label}: {np.array(point)}")
        _teleport_vehicle(client, vehicle_name, point, look_at=look_at, settle_sec=pause_sec)


def run(args):
    source = Model4TaskSource(force_rebuild=args.rebuild)
    points = source.get_points()
    point_array = np.array([p["point"] for p in points], dtype=np.float32)
    clearances = np.array([p["clearance"] for p in points], dtype=np.float32)
    bottoms = np.array([p["bottom_distance"] for p in points], dtype=np.float32)
    regions = sorted({p["region"] for p in points})

    print(f"Point count: {len(points)}")
    print(f"Region bucket count: {len(regions)}")
    print(
        f"X range: {point_array[:, 0].min():.2f} .. {point_array[:, 0].max():.2f} | "
        f"Y range: {point_array[:, 1].min():.2f} .. {point_array[:, 1].max():.2f} | "
        f"Z range: {point_array[:, 2].min():.2f} .. {point_array[:, 2].max():.2f}"
    )
    print(
        f"Clearance stats: min={clearances.min():.2f}, max={clearances.max():.2f}, mean={clearances.mean():.2f}"
    )
    print(f"Bottom distance stats: min={bottoms.min():.2f}, max={bottoms.max():.2f}, mean={bottoms.mean():.2f}")
    print(f"Cache path: {config.AIRSIM_NAV_CACHE_JSON}")

    sampled_tasks = []
    for i in range(args.examples):
        task = source.sample_task()
        sampled_tasks.append(task)
        print(
            f"Task {i + 1}: start={task['start']} -> global_goal={task['global_goal']} | "
            f"regions={task['start_region']}->{task['goal_region']} | global_dist={task['distance']:.2f} | "
            f"subgoals={len(task['route_subgoals'])}"
        )
        for j, subgoal in enumerate(task["route_subgoals"], start=1):
            prev = task["start"] if j == 1 else task["route_subgoals"][j - 2]
            seg_dist = float(np.linalg.norm(np.array(subgoal) - np.array(prev)))
            tail = " (final)" if j == len(task["route_subgoals"]) else ""
            print(f"  subgoal {j}: {subgoal} | segment_dist={seg_dist:.2f}{tail}")

    preview_task = sampled_tasks[0]
    print(
        f"\nPreview task selected: global_dist={preview_task['distance']:.2f}, "
        f"subgoals={len(preview_task['route_subgoals'])}"
    )

    if args.plot_all_points:
        all_points = [
            airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))
            for p in point_array[:: max(1, args.downsample)]
        ]
        source.client.simPlotPoints(
            all_points,
            color_rgba=[0.1, 0.9, 0.2, 0.65],
            size=float(args.point_size) * 0.6,
            duration=float(args.duration),
            is_persistent=args.persistent,
        )
        print(f"Plotted {len(all_points)} cached feasible points.")

    _plot_route(
        source.client,
        preview_task,
        duration=args.duration,
        point_size=args.point_size,
        persistent=args.persistent,
    )
    print("Plotted preview route: green=start, yellow=subgoals, red=global goal, cyan=route line.")

    if args.teleport:
        _teleport_preview(source, preview_task, pause_sec=args.pause)


def main():
    parser = argparse.ArgumentParser(description="Inspect Model4 long-range tasks and teleport across subgoals")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the point cache")
    parser.add_argument("--teleport", action="store_true", help="Teleport through start/subgoals/global goal in sequence")
    parser.add_argument("--plot-all-points", action="store_true", help="Also plot all cached feasible points")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds to pause at each teleport point")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--examples", type=int, default=3, help="Print N sampled task examples")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
