#!/usr/bin/env python
import argparse
import json
import math
import os
import sys

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(TOOLS_DIR)
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)

import config


def _latest_metadata(depth_dir):
    candidates = [
        os.path.join(depth_dir, name)
        for name in os.listdir(depth_dir)
        if name.endswith("_metadata.json")
    ]
    if not candidates:
        raise RuntimeError(f"No topdown metadata found in {depth_dir}")
    return max(candidates, key=os.path.getmtime)


def _dilate(mask, radius_cells):
    if radius_cells <= 0:
        return mask.copy()
    try:
        import cv2
        kernel_size = radius_cells * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    except Exception:
        dilated = mask.copy()
        ys, xs = np.where(mask)
        height, width = mask.shape
        for x, y in zip(xs, ys):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dilated[ny, nx] = True
        return dilated


def _save_occupancy_png(path, occupied, inflated):
    try:
        import cv2
    except Exception as exc:
        print(f"[TopDownOccupancy] WARNING: cv2 unavailable, skip PNG: {exc}")
        return

    image = np.full((*inflated.shape, 3), 255, dtype=np.uint8)
    image[inflated] = (0, 0, 0)
    image[occupied] = (50, 50, 220)
    cv2.imwrite(path, image)


def _coverage_from_metadata(metadata):
    coverage = dict(metadata["coverage"])
    coverage["world_x_npy_order"] = "bottom_to_top"
    coverage["world_y_npy_order"] = "left_to_right"
    return coverage


def _origin_pixel_from_coverage(coverage, meters_per_pixel):
    return {
        "row": int(round(float(coverage["x_max"]) / float(meters_per_pixel["x"]))),
        "col": int(round(-float(coverage["y_min"]) / float(meters_per_pixel["y"]))),
    }


def _world_to_pixel_from_coverage(coverage, meters_per_pixel, x, y):
    row = int(round((float(coverage["x_max"]) - float(x)) / float(meters_per_pixel["x"])))
    col = int(round((float(y) - float(coverage["y_min"])) / float(meters_per_pixel["y"])))
    return row, col


def export(args):
    depth_dir = os.path.abspath(args.depth_dir)
    metadata_path = args.metadata or _latest_metadata(depth_dir)
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    height_path = metadata["files"]["height_npy"]
    height = np.load(height_path)
    occupied = np.isfinite(height) & (height > float(args.height_threshold))
    radius_cells = int(math.ceil(float(args.inflate_meters) / float(args.meters_per_pixel)))
    inflated = _dilate(occupied, radius_cells)

    prefix = os.path.splitext(metadata_path)[0].replace("_metadata", "")
    occupancy_npy = f"{prefix}_occupancy_h{args.height_threshold:g}_infl{args.inflate_meters:g}.npy"
    occupancy_png = f"{prefix}_occupancy_h{args.height_threshold:g}_infl{args.inflate_meters:g}.png"
    occupancy_metadata = f"{prefix}_occupancy_h{args.height_threshold:g}_infl{args.inflate_meters:g}.json"

    np.save(occupancy_npy, inflated)
    _save_occupancy_png(occupancy_png, occupied, inflated)

    meters_per_pixel = metadata["meters_per_pixel"]
    coverage = _coverage_from_metadata(metadata)
    origin_pixel = _origin_pixel_from_coverage(coverage, meters_per_pixel)
    model_bounds_pixels = {
        "x_min_y_min": _world_to_pixel_from_coverage(
            coverage, meters_per_pixel, config.OCCUPANCY_MIN_X, config.OCCUPANCY_MIN_Y),
        "x_max_y_max": _world_to_pixel_from_coverage(
            coverage, meters_per_pixel, config.OCCUPANCY_MAX_X, config.OCCUPANCY_MAX_Y),
    }
    output_metadata = {
        "source_metadata": metadata_path,
        "source_height_npy": height_path,
        "height_threshold": float(args.height_threshold),
        "inflate_meters": float(args.inflate_meters),
        "inflate_cells": int(radius_cells),
        "width": int(inflated.shape[1]),
        "height": int(inflated.shape[0]),
        "occupied_pixels_raw": int(occupied.sum()),
        "occupied_pixels_inflated": int(inflated.sum()),
        "total_pixels": int(inflated.size),
        "raw_occupied_ratio": float(occupied.sum() / inflated.size),
        "inflated_occupied_ratio": float(inflated.sum() / inflated.size),
        "coverage": coverage,
        "meters_per_pixel": meters_per_pixel,
        "coordinate_frame": {
            "name": "airsim_local_ned",
            "origin": "vehicle_spawn",
            "origin_local_xy": [0.0, 0.0],
            "origin_pixel": {
                "row": int(origin_pixel["row"]),
                "col": int(origin_pixel["col"]),
            },
            "row_axis": "-x",
            "col_axis": "+y",
        },
        "pixel_axes": {
            "row": "-airsim_local_x",
            "col": "+airsim_local_y",
        },
        "model_bounds_pixels": model_bounds_pixels,
        "files": {
            "occupancy_npy": occupancy_npy,
            "occupancy_png": occupancy_png,
            "metadata_json": occupancy_metadata,
        },
    }
    with open(occupancy_metadata, "w", encoding="utf-8") as file:
        json.dump(output_metadata, file, indent=2)

    print("[TopDownOccupancy] Export complete.")
    print(f"[TopDownOccupancy] PNG: {occupancy_png}")
    print(f"[TopDownOccupancy] NPY: {occupancy_npy}")
    print(f"[TopDownOccupancy] Metadata: {occupancy_metadata}")
    print(
        "[TopDownOccupancy] Ratios: "
        f"raw={output_metadata['raw_occupied_ratio']:.2%}, "
        f"inflated={output_metadata['inflated_occupied_ratio']:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export an inflated occupancy map from the latest top-down height map."
    )
    parser.add_argument(
        "--depth-dir",
        default=os.path.join(MODEL6_DIR, "topdown_depth"),
    )
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--height-threshold", type=float, default=5.0)
    parser.add_argument("--inflate-meters", type=float, default=2.0)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
