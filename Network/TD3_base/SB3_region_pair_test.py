#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime

import torch
from stable_baselines3 import TD3


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# 优先使用仓库内的 airsim 源码与本目录模块。
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from reinforcement_network import AirSimUAVEnv


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "td3_resume_latest")
DEFAULT_TEST_CSV = os.path.join(REPO_ROOT, "dataset", "relative_coordinates.csv")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test")


def resolve_model_path(model_path):
    candidates = [model_path]
    if not model_path.endswith(".zip"):
        candidates.append(f"{model_path}.zip")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Model checkpoint not found. Tried: {', '.join(candidates)}"
    )


def sanitize_region_name(region):
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(region)).strip("_") or "unknown"


def load_region_points(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Test CSV not found: {csv_path}\n"
            f"Please add relative_coordinates.csv first."
        )

    regions = OrderedDict()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"x", "y", "z", "region"}
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_fields)}"
            )

        for csv_row_idx, row in enumerate(reader, start=2):
            region = str(row["region"]).strip() or "default"
            point = [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ]
            bucket = regions.setdefault(region, [])
            bucket.append(
                {
                    "region_index": len(bucket),
                    "csv_row": csv_row_idx,
                    "point": point,
                }
            )

    return regions


def build_region_cases(region, points):
    cases = []
    pair_count = len(points) // 2

    for pair_index in range(pair_count):
        idx_a = 2 * pair_index
        idx_b = idx_a + 1
        point_a = points[idx_a]
        point_b = points[idx_b]

        cases.append(
            {
                "region": region,
                "pair_index": pair_index,
                "direction": "forward",
                "start_meta": point_a,
                "target_meta": point_b,
            }
        )
        cases.append(
            {
                "region": region,
                "pair_index": pair_index,
                "direction": "reverse",
                "start_meta": point_b,
                "target_meta": point_a,
            }
        )

    ignored_point = points[-1] if len(points) % 2 == 1 else None
    return cases, ignored_point


def point_to_string(point):
    return f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"


def run_single_case(env, model, case):
    start_point = case["start_meta"]["point"]
    target_point = case["target_meta"]["point"]

    obs, _ = env.reset(
        options={
            "start_pos": start_point,
            "target": target_point,
            "region": case["region"],
        }
    )

    done = False
    total_reward = 0.0
    step_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return {
        "region": case["region"],
        "pair_index": case["pair_index"],
        "direction": case["direction"],
        "start_region_index": case["start_meta"]["region_index"],
        "target_region_index": case["target_meta"]["region_index"],
        "start_csv_row": case["start_meta"]["csv_row"],
        "target_csv_row": case["target_meta"]["csv_row"],
        "start_x": float(start_point[0]),
        "start_y": float(start_point[1]),
        "start_z": float(start_point[2]),
        "target_x": float(target_point[0]),
        "target_y": float(target_point[1]),
        "target_z": float(target_point[2]),
        "success": bool(step_info.get("arrived", False)),
        "end_reason": str(step_info.get("end_reason", "unknown")),
        "steps": int(env.step_count),
        "total_reward": float(total_reward),
        "final_dis2goal": float(step_info.get("dis2goal", 0.0)),
        "final_xy_dist": float(step_info.get("xy_dist", 0.0)),
        "final_z_dist": float(step_info.get("z_dist", 0.0)),
        "collision": bool(step_info.get("collision", False)),
        "crossed_border": bool(step_info.get("crossed_border", False)),
        "out_of_ceiling": bool(step_info.get("out_of_ceiling", False)),
        "lidar_min_dist": float(step_info.get("lidar_min_dist", 0.0)),
    }


def write_region_results(output_dir, run_stamp, region, results):
    os.makedirs(output_dir, exist_ok=True)
    safe_region = sanitize_region_name(region)
    csv_path = os.path.join(
        output_dir,
        f"{run_stamp}_region_{safe_region}_results.csv",
    )

    fieldnames = [
        "region",
        "pair_index",
        "direction",
        "start_region_index",
        "target_region_index",
        "start_csv_row",
        "target_csv_row",
        "start_x",
        "start_y",
        "start_z",
        "target_x",
        "target_y",
        "target_z",
        "success",
        "end_reason",
        "steps",
        "total_reward",
        "final_dis2goal",
        "final_xy_dist",
        "final_z_dist",
        "collision",
        "crossed_border",
        "out_of_ceiling",
        "lidar_min_dist",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def write_overall_summary(output_dir, run_stamp, summary):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{run_stamp}_summary.json")
    with open(summary_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path


def print_case_result(case_result):
    status = "SUCCESS" if case_result["success"] else f"FAILED({case_result['end_reason']})"
    print(
        f"    Pair {case_result['pair_index']} | {case_result['direction']:7s} | "
        f"{case_result['start_region_index']} -> {case_result['target_region_index']} | "
        f"{status} | steps={case_result['steps']:3d} | "
        f"reward={case_result['total_reward']:.2f} | "
        f"final_dis={case_result['final_dis2goal']:.2f}"
    )


def evaluate_region_pairs(
    model_path,
    csv_path,
    output_dir,
    target_region=None,
    show_visuals=False,
):
    resolved_model_path = resolve_model_path(model_path)
    region_points = load_region_points(csv_path)

    if target_region is not None:
        target_region = str(target_region)
        if target_region not in region_points:
            raise ValueError(
                f"Region {target_region} not found in {csv_path}. "
                f"Available regions: {list(region_points.keys())}"
            )
        region_points = OrderedDict([(target_region, region_points[target_region])])

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[{'=' * 60}]")
    print(f"Loading model: {resolved_model_path}")
    print(f"Using test CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Regions to test: {len(region_points)}")
    print(f"[{'=' * 60}]")

    env = AirSimUAVEnv()
    env.visualize_goal = bool(show_visuals)
    env.visualize_traj = bool(show_visuals)
    env.visualize_waypoints = False
    env.show_goal_text = False

    model = None
    try:
        model = TD3.load(resolved_model_path, env=env, device=device)
        print("Model loaded successfully.\n")

        overall_region_summaries = []
        total_cases = 0
        total_successes = 0

        for region, points in region_points.items():
            print(f"Region {region}")
            print(f"  Point count: {len(points)}")

            cases, ignored_point = build_region_cases(region, points)
            if ignored_point is not None:
                print(
                    f"  Warning: odd number of points in region {region}. "
                    f"Ignoring last point index {ignored_point['region_index']} "
                    f"(CSV row {ignored_point['csv_row']})."
                )

            if not cases:
                print("  No valid point pairs in this region, skipping.\n")
                overall_region_summaries.append(
                    {
                        "region": region,
                        "point_count": len(points),
                        "tested_cases": 0,
                        "success_count": 0,
                        "success_rate": 0.0,
                        "results_file": None,
                    }
                )
                continue

            region_results = []
            for case in cases:
                start_point = case["start_meta"]["point"]
                target_point = case["target_meta"]["point"]
                print(
                    f"  Testing pair {case['pair_index']} {case['direction']}: "
                    f"{case['start_meta']['region_index']} {point_to_string(start_point)} -> "
                    f"{case['target_meta']['region_index']} {point_to_string(target_point)}"
                )
                case_result = run_single_case(env, model, case)
                region_results.append(case_result)
                print_case_result(case_result)

            region_successes = sum(1 for item in region_results if item["success"])
            region_success_rate = (
                region_successes / len(region_results) * 100.0 if region_results else 0.0
            )
            results_file = write_region_results(output_dir, run_stamp, region, region_results)

            print(f"  Saved region results: {results_file}")
            print(
                f"  Region {region} success rate: "
                f"{region_successes}/{len(region_results)} = {region_success_rate:.2f}%\n"
            )

            total_cases += len(region_results)
            total_successes += region_successes
            overall_region_summaries.append(
                {
                    "region": region,
                    "point_count": len(points),
                    "tested_cases": len(region_results),
                    "success_count": region_successes,
                    "success_rate": region_success_rate,
                    "results_file": results_file,
                }
            )

        overall_success_rate = (
            total_successes / total_cases * 100.0 if total_cases else 0.0
        )
        summary = {
            "run_stamp": run_stamp,
            "model_path": resolved_model_path,
            "csv_path": csv_path,
            "output_dir": output_dir,
            "total_regions": len(overall_region_summaries),
            "total_test_cases": total_cases,
            "total_success_count": total_successes,
            "total_success_rate": overall_success_rate,
            "regions": overall_region_summaries,
        }
        summary_file = write_overall_summary(output_dir, run_stamp, summary)

        print(f"[{'=' * 60}]")
        print("Overall summary")
        print(f"  Total test cases: {total_cases}")
        print(f"  Total success count: {total_successes}")
        print(f"  Total success rate: {overall_success_rate:.2f}%")
        print(f"  Saved summary: {summary_file}")
        print(f"[{'=' * 60}]")

        return summary
    finally:
        try:
            env.client.armDisarm(False, vehicle_name=env.vehicle_name)
            env.client.enableApiControl(False, vehicle_name=env.vehicle_name)
        except Exception:
            pass


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate TD3_base TD3 by region pair cases from relative_coordinates.csv"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="TD3 checkpoint path. Supports with or without .zip suffix.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_TEST_CSV,
        help="Path to relative_coordinates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save per-region results.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Only test one specific region value.",
    )
    parser.add_argument(
        "--show-visuals",
        action="store_true",
        help="Enable goal and trajectory visualization during evaluation.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    evaluate_region_pairs(
        model_path=args.model_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        target_region=args.region,
        show_visuals=args.show_visuals,
    )


if __name__ == "__main__":
    main()
