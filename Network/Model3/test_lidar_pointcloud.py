#!/usr/bin/env python
"""
Visual test for the new RLoPlanner LiDAR.

This script:
1. Connects to AirSim and reads the configured multi-channel LiDAR.
2. Draws the current point cloud in the scene using simPlotPoints.
3. Prints basic point-count and distance statistics.
4. Projects the point cloud to the paper's 5x5 sensing grid and prints the 25 distances.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from uav_env.observation import get_ray_directions


def _quat_to_rot(q):
    """Quaternion to 3x3 rotation matrix."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points_to_world(points_local, lidar_pose):
    """Transform SensorLocalFrame point cloud into world coordinates."""
    if len(points_local) == 0:
        return np.empty((0, 3), dtype=np.float64)

    rotation = _quat_to_rot(lidar_pose.orientation)
    translation = np.array(
        [
            lidar_pose.position.x_val,
            lidar_pose.position.y_val,
            lidar_pose.position.z_val,
        ],
        dtype=np.float64,
    )
    return points_local @ rotation.T + translation


def _project_lidar_to_rays(points_local, ray_directions):
    """Same projection logic as Model3 env, but standalone for testing."""
    distances = np.full(len(ray_directions), config.RANGE_MAX, dtype=np.float32)
    if len(points_local) == 0:
        return distances

    h_step = np.radians(config.RANGE_HFOV) / max(config.RANGE_RAYS_H - 1, 1)
    v_step = np.radians(config.RANGE_VFOV) / max(config.RANGE_RAYS_V - 1, 1)
    half_diag = math.sqrt((0.5 * h_step) ** 2 + (0.5 * v_step) ** 2) * config.RANGE_MATCH_MARGIN
    cosine_threshold = math.cos(half_diag)

    ray_dirs = ray_directions.astype(np.float32)
    ray_dirs = ray_dirs / np.maximum(np.linalg.norm(ray_dirs, axis=1, keepdims=True), 1e-6)

    for pt in points_local:
        dist = float(np.linalg.norm(pt))
        if dist < 1e-3 or dist > config.RANGE_MAX:
            continue

        direction = pt / dist
        dots = ray_dirs @ direction
        best_idx = int(np.argmax(dots))
        if dots[best_idx] >= cosine_threshold and dist < distances[best_idx]:
            distances[best_idx] = dist

    return distances


def _detect_vehicle_name(client):
    try:
        vehicles = client.listVehicles()
        return vehicles[0] if vehicles else "Drone1"
    except Exception:
        return "Drone1"


def _fetch_lidar_data(client, vehicle_name):
    lidar_names = (config.PRIMARY_LIDAR_NAME,) + tuple(config.FALLBACK_LIDAR_NAMES)
    for lidar_name in lidar_names:
        try:
            data = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
            if data and len(data.point_cloud) >= 3:
                return lidar_name, data
        except Exception:
            pass
    return None, None


def _print_stats(lidar_name, points_local, projected_distances):
    print(f"LiDAR name: {lidar_name}")
    print(f"Point count: {len(points_local)}")
    if len(points_local) == 0:
        print("No valid LiDAR points received.")
        return

    dists = np.linalg.norm(points_local, axis=1)
    print(
        f"Distance stats (m): min={dists.min():.3f}, max={dists.max():.3f}, "
        f"mean={dists.mean():.3f}, median={np.median(dists):.3f}"
    )

    hit_count = int(np.sum(projected_distances < config.RANGE_MAX))
    print(f"Projected 5x5 hits: {hit_count}/25")
    print("Projected 5x5 distances (row-major, meters):")
    for row in range(config.RANGE_RAYS_H):
        start = row * config.RANGE_RAYS_V
        row_vals = projected_distances[start : start + config.RANGE_RAYS_V]
        print("  " + " ".join(f"{v:5.2f}" for v in row_vals))


def run(args):
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    vehicle_name = _detect_vehicle_name(client)
    print(f"Detected vehicle: {vehicle_name}")

    ray_directions = get_ray_directions(
        hfov=config.RANGE_HFOV,
        vfov=config.RANGE_VFOV,
        rays_h=config.RANGE_RAYS_H,
        rays_v=config.RANGE_RAYS_V,
    )

    try:
        client.simFlushPersistentMarkers()
    except Exception:
        pass

    end_time = time.time() + args.duration if args.duration > 0 else None

    while end_time is None or time.time() < end_time:
        lidar_name, lidar_data = _fetch_lidar_data(client, vehicle_name)
        if lidar_data is None:
            print(
                f"No LiDAR data found. Checked: {(config.PRIMARY_LIDAR_NAME,) + tuple(config.FALLBACK_LIDAR_NAMES)}"
            )
            time.sleep(args.interval)
            continue

        points_local = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        points_world = _transform_points_to_world(points_local, lidar_data.pose)
        projected_distances = _project_lidar_to_rays(points_local, ray_directions)

        _print_stats(lidar_name, points_local, projected_distances)

        if len(points_world) > 0:
            airsim_points = [
                airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))
                for p in points_world[:: max(1, args.downsample)]
            ]
            try:
                client.simPlotPoints(
                    airsim_points,
                    color_rgba=[0.1, 0.9, 1.0, 1.0],
                    size=float(args.point_size),
                    duration=float(args.interval) * 1.5,
                    is_persistent=False,
                )
            except Exception as exc:
                print(f"simPlotPoints failed: {exc}")

        print("-" * 60)
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Visualize and validate Model3 LiDAR point cloud")
    parser.add_argument("--duration", type=float, default=15.0, help="Run time in seconds; <=0 means infinite")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling / redraw interval in seconds")
    parser.add_argument("--point-size", type=float, default=8.0, help="AirSim plotted point size")
    parser.add_argument("--downsample", type=int, default=2, help="Plot every Nth point to reduce clutter")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
