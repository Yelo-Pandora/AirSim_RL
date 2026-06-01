#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(TOOLS_DIR)
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL6_DIR))


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _world_to_pixel(metadata, point):
    frame = metadata.get("coordinate_frame", {})
    meters_per_pixel = metadata["meters_per_pixel"]
    if frame.get("origin") == "vehicle_spawn":
        origin = frame["origin_pixel"]
        row = int(round(float(origin["row"]) - float(point[0]) / float(meters_per_pixel["x"])))
        col = int(round(float(origin["col"]) + float(point[1]) / float(meters_per_pixel["y"])))
        return row, col
    coverage = metadata["coverage"]
    row = int(round((float(coverage["x_max"]) - float(point[0])) / float(meters_per_pixel["x"])))
    col = int(round((float(point[1]) - float(coverage["y_min"])) / float(meters_per_pixel["y"])))
    return row, col


def _pixel_to_world(metadata, row, col, z):
    frame = metadata.get("coordinate_frame", {})
    meters_per_pixel = metadata["meters_per_pixel"]
    if frame.get("origin") == "vehicle_spawn":
        origin = frame["origin_pixel"]
        x = (float(origin["row"]) - row) * float(meters_per_pixel["x"])
        y = (col - float(origin["col"])) * float(meters_per_pixel["y"])
        return x, y, z
    coverage = metadata["coverage"]
    x = float(coverage["x_max"]) - row * float(meters_per_pixel["x"])
    y = float(coverage["y_min"]) + col * float(meters_per_pixel["y"])
    return x, y, z


def _nearest_free(occupied, row, col, max_radius):
    height, width = occupied.shape
    if 0 <= row < height and 0 <= col < width and not occupied[row, col]:
        return row, col
    for radius in range(1, max_radius + 1):
        best = None
        best_dist = float("inf")
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                rr, cc = row + dr, col + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if occupied[rr, cc]:
                    continue
                dist = (dr * dr + dc * dc) ** 0.5
                if dist < best_dist:
                    best = (rr, cc)
                    best_dist = dist
        if best is not None:
            return best
    return None


def _load_or_make_background(height_path, occupied):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"cv2 is required for annotation output: {exc}")

    if height_path is None or not os.path.exists(height_path):
        image = np.full((*occupied.shape, 3), 255, dtype=np.uint8)
        image[occupied] = (0, 0, 200)
        return image

    height = np.load(height_path)
    valid = np.isfinite(height)
    image = np.full((*height.shape, 3), 255, dtype=np.uint8)
    if np.any(valid):
        values = height[valid]
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.0))
        if high <= low:
            high = low + 1.0
        scaled = np.clip((np.where(valid, height, low) - low) / (high - low), 0.0, 1.0)
        gray = (scaled * 255.0).astype(np.uint8)
        image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        image[~valid] = (0, 0, 0)
    overlay = image.copy()
    overlay[occupied] = (0, 0, 200)
    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)


def annotate(args):
    occupancy_metadata_path = _resolve_path(args.occupancy_metadata)
    with open(occupancy_metadata_path, encoding="utf-8") as file:
        occupancy_metadata = json.load(file)
    source_metadata_path = _resolve_path(occupancy_metadata.get("source_metadata", ""))
    if os.path.exists(source_metadata_path):
        with open(source_metadata_path, encoding="utf-8") as file:
            source_metadata = json.load(file)
        height_path = _resolve_path(source_metadata["files"]["height_npy"])
    else:
        source_metadata = occupancy_metadata
        height_path = None

    occupied = np.load(_resolve_path(occupancy_metadata["files"]["occupancy_npy"]))
    image = _load_or_make_background(height_path, occupied)

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"cv2 is required for annotation output: {exc}")

    points = {
        "start": tuple(args.start),
        "goal": tuple(args.goal),
    }
    colors = {
        "start": (0, 255, 0),
        "goal": (255, 0, 255),
    }
    report = {
        "source_metadata": source_metadata_path,
        "occupancy_metadata": occupancy_metadata_path,
        "coverage": source_metadata["coverage"],
        "meters_per_pixel": source_metadata["meters_per_pixel"],
        "coordinate_frame": source_metadata.get("coordinate_frame"),
        "pixel_axes": {
            "row": "-airsim_local_x",
            "col": "+airsim_local_y",
        },
        "points": {},
    }

    for label, point in points.items():
        row, col = _world_to_pixel(source_metadata, point)
        in_bounds = 0 <= row < occupied.shape[0] and 0 <= col < occupied.shape[1]
        is_occupied = bool(occupied[row, col]) if in_bounds else True
        nearest = _nearest_free(occupied, row, col, args.nearest_radius)
        nearest_world = None
        if nearest is not None:
            nearest_world = _pixel_to_world(source_metadata, nearest[0], nearest[1], point[2])

        if in_bounds:
            cv2.drawMarker(
                image,
                (col, row),
                colors[label],
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.circle(image, (col, row), 7, colors[label], 2)
            cv2.putText(
                image,
                label,
                (col + 8, row - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[label],
                1,
                cv2.LINE_AA,
            )
        if nearest is not None:
            n_row, n_col = nearest
            cv2.circle(image, (n_col, n_row), 5, (255, 255, 255), 1)

        report["points"][label] = {
            "world": [float(value) for value in point],
            "pixel": {"row_x": int(row), "col_y": int(col)},
            "in_bounds": bool(in_bounds),
            "occupied": bool(is_occupied),
            "nearest_free_pixel": None if nearest is None else {
                "row_x": int(nearest[0]),
                "col_y": int(nearest[1]),
            },
            "nearest_free_world": None if nearest_world is None else [
                float(value) for value in nearest_world
            ],
        }

    output_prefix = os.path.splitext(occupancy_metadata_path)[0] + "_points"
    image_path = output_prefix + ".png"
    report_path = output_prefix + ".json"
    cv2.imwrite(image_path, image)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("[TopDownAnnotate] Annotation complete.")
    print(f"[TopDownAnnotate] Image: {image_path}")
    print(f"[TopDownAnnotate] Report: {report_path}")
    for label, data in report["points"].items():
        print(
            f"[TopDownAnnotate] {label}: pixel={data['pixel']}, "
            f"occupied={data['occupied']}, nearest_free={data['nearest_free_world']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Annotate top-down map with start/goal points.")
    parser.add_argument(
        "--occupancy-metadata",
        default=os.path.join(
            MODEL6_DIR,
            "topdown_depth",
            "topdown_depth_20260601_165808_occupancy_h10_infl2.json",
        ),
    )
    parser.add_argument("--start", type=float, nargs=3, required=True)
    parser.add_argument("--goal", type=float, nargs=3, required=True)
    parser.add_argument("--nearest-radius", type=int, default=20)
    args = parser.parse_args()
    annotate(args)


if __name__ == "__main__":
    main()
