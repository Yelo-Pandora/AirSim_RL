#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(TOOLS_DIR)
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)

import config
if config.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, config.PROJECT_ROOT)

import airsim


def _base_map_span():
    span_x = float(config.OCCUPANCY_MAX_X - config.OCCUPANCY_MIN_X)
    span_y = float(config.OCCUPANCY_MAX_Y - config.OCCUPANCY_MIN_Y)
    return span_x, span_y


def _base_image_size(args):
    span_x, span_y = _base_map_span()
    width = int(round(span_y / float(args.meters_per_pixel)))
    height = int(round(span_x / float(args.meters_per_pixel)))
    return width, height


def _expected_image_size(args):
    span_x, span_y = _base_map_span()
    width = int(round(span_y * float(args.capture_margin) / float(args.meters_per_pixel)))
    height = int(round(span_x * float(args.capture_margin) / float(args.meters_per_pixel)))
    return width, height


def _infer_image_size(response, value_count, args):
    expected_width, expected_height = _expected_image_size(args)
    if value_count == expected_width * expected_height:
        return expected_width, expected_height
    base_width, base_height = _base_image_size(args)
    if value_count == base_width * base_height:
        return base_width, base_height
    if response.width > 0 and response.height > 0:
        return int(response.width), int(response.height)
    raise RuntimeError(
        f"Invalid image response size: {response.width}x{response.height}"
    )


def _array_from_response(response, args):
    values = np.asarray(response.image_data_float, dtype=np.float32)
    width, height = _infer_image_size(response, values.size, args)
    expected = width * height
    if values.size != expected:
        raise RuntimeError(
            f"Depth image has {values.size} values; expected {expected} "
            f"for {width}x{height}. Response reported "
            f"{response.width}x{response.height}."
        )
    return values.reshape((height, width)), width, height


def _write_csv(path, array):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(array.tolist())


def _save_visual(path, array, valid_mask, label):
    try:
        import cv2
    except Exception as exc:
        print(f"[TopDownDepth] WARNING: cv2 unavailable, skip {label} PNG: {exc}")
        return

    visual = np.zeros(array.shape, dtype=np.uint8)
    if np.any(valid_mask):
        valid_values = array[valid_mask]
        low = float(np.percentile(valid_values, 1.0))
        high = float(np.percentile(valid_values, 99.0))
        if high <= low:
            high = low + 1.0
        filled = np.where(valid_mask, array, low)
        scaled = np.clip((filled - low) / (high - low), 0.0, 1.0)
        visual = (scaled * 255.0).astype(np.uint8)
        visual[~valid_mask] = 0

    colored = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = (0, 0, 0)
    cv2.imwrite(path, colored)


def _camera_height_for_coverage(ground_width, fov_degrees):
    half_fov = math.radians(float(fov_degrees)) * 0.5
    return (float(ground_width) * 0.5) / max(math.tan(half_fov), 1e-6)


def _map_span(margin):
    span_x, span_y = _base_map_span()
    return span_x * float(margin), span_y * float(margin)


def _crop_by_world_bounds(
        array,
        center_x,
        center_y,
        ground_x_span,
        ground_y_span,
        x_min,
        x_max,
        y_min,
        y_max):
    image_height, image_width = array.shape
    capture_x_min = center_x - ground_x_span * 0.5
    capture_x_max = center_x + ground_x_span * 0.5
    capture_y_min = center_y - ground_y_span * 0.5
    meters_per_row = ground_x_span / max(image_height, 1)
    meters_per_col = ground_y_span / max(image_width, 1)

    row0 = int(math.floor((capture_x_max - float(x_max)) / meters_per_row))
    row1 = int(math.ceil((capture_x_max - float(x_min)) / meters_per_row))
    col0 = int(math.floor((float(y_min) - capture_y_min) / meters_per_col))
    col1 = int(math.ceil((float(y_max) - capture_y_min) / meters_per_col))

    row0 = max(0, min(image_height, row0))
    row1 = max(row0 + 1, min(image_height, row1))
    col0 = max(0, min(image_width, col0))
    col1 = max(col0 + 1, min(image_width, col1))
    actual = {
        "x_min": capture_x_max - row1 * meters_per_row,
        "x_max": capture_x_max - row0 * meters_per_row,
        "y_min": capture_y_min + col0 * meters_per_col,
        "y_max": capture_y_min + col1 * meters_per_col,
    }
    return array[row0:row1, col0:col1], (col0, row0), actual


def capture(args):
    client = airsim.MultirotorClient()
    print("[TopDownDepth] Connecting to AirSim...")
    client.confirmConnection()

    center_x = args.center_x
    center_y = args.center_y
    if center_x is None:
        center_x = getattr(
            config,
            "TOPDOWN_CAMERA_CENTER_X",
            0.5 * (config.OCCUPANCY_MIN_X + config.OCCUPANCY_MAX_X),
        )
    if center_y is None:
        center_y = getattr(
            config,
            "TOPDOWN_CAMERA_CENTER_Y",
            0.5 * (config.OCCUPANCY_MIN_Y + config.OCCUPANCY_MAX_Y),
        )
    target_ground_x_span, target_ground_y_span = _map_span(args.capture_margin)
    target_aspect = target_ground_x_span / max(target_ground_y_span, 1e-6)
    camera_height = args.height
    if camera_height is None:
        camera_height = _camera_height_for_coverage(target_ground_y_span, args.fov)
    camera_z = config.OCCUPANCY_GROUND_Z - camera_height

    pose = airsim.Pose(
        airsim.Vector3r(float(center_x), float(center_y), float(camera_z)),
        airsim.to_quaternion(math.radians(-90.0), 0.0, 0.0),
    )
    client.simSetCameraPose(args.camera, pose, external=True)

    response = client.simGetImages(
        [
            airsim.ImageRequest(
                args.camera,
                airsim.ImageType.DepthPlanar,
                pixels_as_float=True,
                compress=False,
            )
        ],
        external=True,
    )[0]

    depth, image_width, image_height = _array_from_response(response, args)
    response_aspect = image_height / max(image_width, 1)
    finite = np.isfinite(depth)
    positive = depth > 0.0
    plausible = depth < camera_height * 2.0
    valid = finite & positive & plausible
    height = np.full(depth.shape, np.nan, dtype=np.float32)
    height[valid] = camera_height - depth[valid]

    horizontal_span = 2.0 * camera_height * math.tan(math.radians(args.fov) * 0.5)
    ground_y_span = horizontal_span
    ground_x_span = horizontal_span * response_aspect
    crop_center_x = 0.5 * (config.OCCUPANCY_MIN_X + config.OCCUPANCY_MAX_X)
    crop_center_y = 0.5 * (config.OCCUPANCY_MIN_Y + config.OCCUPANCY_MAX_Y)
    save_x_min = config.OCCUPANCY_MIN_X - float(args.save_x_min_extra)
    save_x_max = config.OCCUPANCY_MAX_X + float(args.save_x_max_extra)
    save_y_min = config.OCCUPANCY_MIN_Y - float(args.save_y_min_extra)
    save_y_max = config.OCCUPANCY_MAX_Y + float(args.save_y_max_extra)
    if args.save_margin != 1.0:
        crop_ground_x_span, crop_ground_y_span = _map_span(args.save_margin)
        save_x_min = crop_center_x - crop_ground_x_span * 0.5
        save_x_max = crop_center_x + crop_ground_x_span * 0.5
        save_y_min = crop_center_y - crop_ground_y_span * 0.5
        save_y_max = crop_center_y + crop_ground_y_span * 0.5

    depth_saved, crop_origin, actual_coverage = _crop_by_world_bounds(
        depth, center_x, center_y, ground_x_span, ground_y_span,
        save_x_min, save_x_max, save_y_min, save_y_max)
    height_saved, _, _ = _crop_by_world_bounds(
        height, center_x, center_y, ground_x_span, ground_y_span,
        save_x_min, save_x_max, save_y_min, save_y_max)
    valid_saved, _, _ = _crop_by_world_bounds(
        valid, center_x, center_y, ground_x_span, ground_y_span,
        save_x_min, save_x_max, save_y_min, save_y_max)
    saved_ground_y_span = ground_y_span * (depth_saved.shape[1] / max(image_width, 1))
    saved_ground_x_span = ground_x_span * (depth_saved.shape[0] / max(image_height, 1))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"topdown_depth_{timestamp}")

    depth_npy = f"{prefix}_depth.npy"
    height_npy = f"{prefix}_height.npy"
    depth_csv = f"{prefix}_depth.csv"
    height_csv = f"{prefix}_height.csv"
    depth_png = f"{prefix}_depth.png"
    height_png = f"{prefix}_height.png"
    metadata_json = f"{prefix}_metadata.json"

    np.save(depth_npy, depth_saved)
    np.save(height_npy, height_saved)
    if args.write_csv:
        _write_csv(depth_csv, depth_saved)
        _write_csv(height_csv, height_saved)

    _save_visual(depth_png, depth_saved, valid_saved, "depth")
    _save_visual(height_png, height_saved, valid_saved, "height")

    metadata = {
        "camera": args.camera,
        "image_type": "DepthPlanar",
        "width": int(depth_saved.shape[1]),
        "height": int(depth_saved.shape[0]),
        "reported_width": int(response.width),
        "reported_height": int(response.height),
        "fov_degrees": float(args.fov),
        "meters_per_pixel": {
            "x": float(saved_ground_x_span / max(int(depth_saved.shape[0]), 1)),
            "y": float(saved_ground_y_span / max(int(depth_saved.shape[1]), 1)),
            "target": float(args.meters_per_pixel),
        },
        "capture": {
            "margin": float(args.capture_margin),
            "width": int(image_width),
            "height": int(image_height),
            "ground_x_span": float(ground_x_span),
            "ground_y_span": float(ground_y_span),
            "coverage": {
                "x_min": float(center_x - ground_x_span * 0.5),
                "x_max": float(center_x + ground_x_span * 0.5),
                "y_min": float(center_y - ground_y_span * 0.5),
                "y_max": float(center_y + ground_y_span * 0.5),
            },
        },
        "saved_crop": {
            "margin": float(args.save_margin),
            "width": int(depth_saved.shape[1]),
            "height": int(depth_saved.shape[0]),
            "origin_pixel_x": int(crop_origin[0]),
            "origin_pixel_y": int(crop_origin[1]),
            "ground_x_span": float(saved_ground_x_span),
            "ground_y_span": float(saved_ground_y_span),
            "requested_coverage": {
                "x_min": float(save_x_min),
                "x_max": float(save_x_max),
                "y_min": float(save_y_min),
                "y_max": float(save_y_max),
            },
        },
        "target_aspect": float(target_aspect),
        "response_aspect": float(response_aspect),
        "camera_pose_ned": {
            "x": float(center_x),
            "y": float(center_y),
            "z": float(camera_z),
            "pitch_degrees": -90.0,
            "roll_degrees": 0.0,
            "yaw_degrees": 0.0,
        },
        "ground_z": float(config.OCCUPANCY_GROUND_Z),
        "camera_height_above_ground": float(camera_height),
        "coverage": {
            "x_min": float(actual_coverage["x_min"]),
            "x_max": float(actual_coverage["x_max"]),
            "y_min": float(actual_coverage["y_min"]),
            "y_max": float(actual_coverage["y_max"]),
            "world_x_npy_order": "bottom_to_top",
            "world_y_npy_order": "left_to_right",
        },
        "model6_bounds": {
            "x_min": float(config.OCCUPANCY_MIN_X),
            "x_max": float(config.OCCUPANCY_MAX_X),
            "y_min": float(config.OCCUPANCY_MIN_Y),
            "y_max": float(config.OCCUPANCY_MAX_Y),
        },
        "files": {
            "depth_npy": depth_npy,
            "height_npy": height_npy,
            "depth_png": depth_png,
            "height_png": height_png,
            "metadata_json": metadata_json,
        },
        "stats": {
            "valid_pixels": int(valid_saved.sum()),
            "total_pixels": int(valid_saved.size),
            "depth_min": float(np.nanmin(depth_saved[valid_saved])) if np.any(valid_saved) else None,
            "depth_max": float(np.nanmax(depth_saved[valid_saved])) if np.any(valid_saved) else None,
            "height_min": float(np.nanmin(height_saved[valid_saved])) if np.any(valid_saved) else None,
            "height_max": float(np.nanmax(height_saved[valid_saved])) if np.any(valid_saved) else None,
        },
    }
    if args.write_csv:
        metadata["files"]["depth_csv"] = depth_csv
        metadata["files"]["height_csv"] = height_csv

    with open(metadata_json, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("[TopDownDepth] Capture complete.")
    print(f"[TopDownDepth] Depth PNG: {depth_png}")
    print(f"[TopDownDepth] Height PNG: {height_png}")
    print(f"[TopDownDepth] Depth NPY: {depth_npy}")
    print(f"[TopDownDepth] Height NPY: {height_npy}")
    print(f"[TopDownDepth] Metadata: {metadata_json}")
    print(
        "[TopDownDepth] Stats: "
        f"valid={metadata['stats']['valid_pixels']}/{metadata['stats']['total_pixels']}, "
        f"depth=[{metadata['stats']['depth_min']}, {metadata['stats']['depth_max']}], "
        f"height=[{metadata['stats']['height_min']}, {metadata['stats']['height_max']}]"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Capture a top-down DepthPlanar image for Astar_planner map review."
    )
    parser.add_argument("--camera", default="TopDownFollowCamera_1")
    parser.add_argument("--output-dir", default=os.path.join(MODEL6_DIR, "topdown_depth"))
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--capture-margin", type=float, default=2.0)
    parser.add_argument("--save-margin", type=float, default=1.0)
    parser.add_argument("--save-x-min-extra", type=float, default=0.0)
    parser.add_argument("--save-x-max-extra", type=float, default=0.0)
    parser.add_argument("--save-y-min-extra", type=float, default=0.0)
    parser.add_argument("--save-y-max-extra", type=float, default=0.0)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
