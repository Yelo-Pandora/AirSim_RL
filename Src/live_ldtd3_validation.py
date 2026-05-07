import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Network", "Model1")))

import airsim
import cv2
import numpy as np

from preprocessing_utils import (
    LIDAR_EMPTY_VALUE,
    LIDAR_FRONT_INDICES,
    LIDAR_SIDE_LEFT_INDICES,
    LIDAR_SIDE_RIGHT_INDICES,
    LIDAR_REAR_INDICES,
    LIDAR_105_INDICES,
    action_to_acceleration,
    decode_depth_planar,
    downsample_depth_minpool,
    downsample_lidar_105,
    integrate_velocity_with_acceleration,
    lidar_points_to_360,
    resize_depth_for_ldtd3,
)


MAX_DISPLAY_RANGE = 20.0
WINDOW_NAME = "LD-TD3 Live Validation"
DEFAULT_START = np.array([0.0, 0.0, -2.0], dtype=np.float32)
DEFAULT_TARGET = np.array([20.0, 0.0, -2.0], dtype=np.float32)
CONTROL_DT = 0.5
ACCEL_SCALE = 2.0

IMAGE_TYPE_NAMES = {
    int(airsim.ImageType.Scene): "Scene",
    int(airsim.ImageType.DepthPlanar): "DepthPlanar",
}


def wrap_angle_deg(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


ACTION_PROBES = [
    ("+X", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    ("-X", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
    ("+Y", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ("-Y", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
    ("+Z", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ("-Z", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    ("hover", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    ("small+X", np.array([0.15, 0.0, 0.0], dtype=np.float32)),
]

def decode_scene_png(response):
    img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Scene PNG 解码失败")
    return img_bgr


def load_settings_summary(repo_root):
    path = Path(repo_root) / "settings.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}

    data = json.loads(path.read_text(encoding="utf-8"))
    vehicle_cfg = next(iter(data.get("Vehicles", {}).values()), {})
    sensors = vehicle_cfg.get("Sensors", {})
    lidar_cfg = sensors.get("Lidar1", {})
    bottom_cfg = sensors.get("DistanceBottom", {})
    top_cfg = sensors.get("DistanceTop", {})
    capture = data.get("CameraDefaults", {}).get("CaptureSettings", [])
    return {
        "exists": True,
        "path": str(path),
        "lidar": lidar_cfg,
        "bottom": bottom_cfg,
        "top": top_cfg,
        "capture": capture,
    }


def connect_client():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    try:
        vehicle_name = client.listVehicles()[0]
    except Exception:
        vehicle_name = "Drone1"
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    try:
        client.client.call("setApiControlTimeout", 0.0, vehicle_name)
    except Exception:
        pass
    return client, vehicle_name


def get_state(client, vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
    pos = np.array([state.position.x_val, state.position.y_val, state.position.z_val], dtype=np.float32)
    vel = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val], dtype=np.float32)
    acc_ang = np.array([state.angular_acceleration.x_val, state.angular_acceleration.y_val, state.angular_acceleration.z_val], dtype=np.float32)
    yaw = math.degrees(airsim.to_eularian_angles(state.orientation)[2])
    return {
        "pos": pos,
        "vel": vel,
        "ang_acc": acc_ang,
        "ang_acc_mag": float(np.linalg.norm(acc_ang)),
        "yaw_deg": wrap_angle_deg(yaw),
    }


def get_soft_alignment_yaw(target_vel, current_yaw_deg, last_commanded_yaw_deg, speed_on=0.2, speed_off=0.1, max_step_deg=20.0):
    speed_xy = float(np.linalg.norm(target_vel[:2]))
    commanded_yaw = float(last_commanded_yaw_deg)
    if speed_xy >= speed_on:
        desired_yaw_deg = math.degrees(math.atan2(float(target_vel[1]), float(target_vel[0])))
    elif speed_xy <= speed_off:
        desired_yaw_deg = commanded_yaw
    else:
        desired_yaw_deg = commanded_yaw
    yaw_delta = wrap_angle_deg(desired_yaw_deg - float(current_yaw_deg))
    yaw_delta = float(np.clip(yaw_delta, -max_step_deg, max_step_deg))
    return wrap_angle_deg(float(current_yaw_deg) + yaw_delta)


def set_pose_and_hover(client, vehicle_name, start_pos=DEFAULT_START, target=DEFAULT_TARGET):
    client.reset()
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)

    direction = target - start_pos
    yaw_rad = math.atan2(float(direction[1]), float(direction[0])) if np.linalg.norm(direction[:2]) > 1e-6 else 0.0
    pose = airsim.Pose(
        airsim.Vector3r(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
        airsim.to_quaternion(0.0, 0.0, yaw_rad),
    )
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
    time.sleep(0.4)
    client.moveByVelocityAsync(0, 0, 0, 0.8, vehicle_name=vehicle_name)
    time.sleep(0.8)


def fetch_lidar_data(client, vehicle_name):
    lidar_data = None
    for name in ("LLidar1", "Lidar1"):
        try:
            lidar_data = client.getLidarData(lidar_name=name, vehicle_name=vehicle_name)
            if lidar_data is not None and len(lidar_data.point_cloud) >= 3:
                break
        except Exception:
            continue
    if lidar_data is None or len(lidar_data.point_cloud) < 3:
        return None, np.ones(360, dtype=np.float32) * LIDAR_EMPTY_VALUE, np.ones(105, dtype=np.float32) * LIDAR_EMPTY_VALUE

    lidar_360 = lidar_points_to_360(lidar_data.point_cloud)
    lidar_105 = downsample_lidar_105(lidar_360).numpy()
    return lidar_data, lidar_360, lidar_105


def fetch_depth_and_scene(client, vehicle_name):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
    ], vehicle_name=vehicle_name)
    if len(responses) < 2:
        raise RuntimeError("simGetImages 返回数量不足")

    scene = decode_scene_png(responses[0])
    raw_depth = decode_depth_planar(responses[1])
    resized_depth = resize_depth_for_ldtd3(raw_depth)
    pooled_depth = downsample_depth_minpool(resized_depth).squeeze(0).squeeze(0).cpu().numpy()
    return scene, raw_depth, resized_depth, pooled_depth, responses[1]


def sample_lidar_sector(lidar_360, indices):
    values = lidar_360[indices]
    valid = values[values < LIDAR_EMPTY_VALUE - 1e-6]
    if len(valid) == 0:
        return {"nearest": LIDAR_EMPTY_VALUE, "mean": LIDAR_EMPTY_VALUE, "valid_bins": 0}
    return {
        "nearest": float(np.min(valid)),
        "mean": float(np.mean(valid)),
        "valid_bins": int(len(valid)),
    }


def analyze_lidar_alignment(lidar_360):
    sector_map = {
        "front": sample_lidar_sector(lidar_360, LIDAR_FRONT_INDICES),
        "left": sample_lidar_sector(lidar_360, list(range(0, 45)) + list(range(315, 360))),
        "right": sample_lidar_sector(lidar_360, list(range(135, 226))),
        "rear": sample_lidar_sector(lidar_360, list(range(225, 316))),
    }

    nearest_bin = int(np.argmin(lidar_360))
    nearest_distance = float(lidar_360[nearest_bin])
    front_density = sector_map["front"]["valid_bins"]
    rear_density = sector_map["rear"]["valid_bins"]
    front_back_ratio = float((front_density + 1) / (rear_density + 1))

    verdict = "pass"
    if nearest_distance < LIDAR_EMPTY_VALUE - 1e-6 and nearest_bin in range(180, 360) and front_density == 0 and rear_density > 10:
        verdict = "suspicious"

    return {
        "sectors": sector_map,
        "nearest_bin": nearest_bin,
        "nearest_distance": nearest_distance,
        "front_back_density_ratio": front_back_ratio,
        "verdict": verdict,
    }


def pick_depth_block(resized_depth):
    h, w = resized_depth.shape
    block_h, block_w = 16, 16
    min_block = None
    min_value = float("inf")
    for r in range(0, h, block_h):
        for c in range(0, w, block_w):
            block = resized_depth[r:r + block_h, c:c + block_w]
            block_min = float(np.min(block))
            if block_min < min_value:
                min_value = block_min
                min_block = (r // block_h, c // block_w, r, c, block)
    return min_block, min_value




def depth_to_colormap(depth):
    clipped = np.clip(depth, 0, MAX_DISPLAY_RANGE)
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)


def raw_depth_to_grayscale(depth):
    clipped = np.clip(depth, 0, MAX_DISPLAY_RANGE)
    gray = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def build_zero_mask_vis(depth):
    mask = (depth <= 1e-6).astype(np.uint8)
    vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    vis[mask > 0] = (0, 0, 255)
    return vis, int(mask.sum())


def summarize_depth_values(depth):
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "zero_count": int((depth <= 1e-6).sum()),
            "finite_count": 0,
        }
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "zero_count": int((depth <= 1e-6).sum()),
        "finite_count": int(finite.size),
    }


def verify_capture_mapping(depth_response):
    return {
        "request_order": ["Scene", "DepthPlanar"],
        "response_order": ["responses[0]", "responses[1]"],
        "scene_decode": "responses[0].image_data_uint8 -> decode_scene_png",
        "depth_decode": "responses[1].image_data_float -> decode_depth_planar",
        "depth_image_type": IMAGE_TYPE_NAMES.get(int(depth_response.image_type), str(depth_response.image_type)),
        "depth_is_float": bool(depth_response.pixels_as_float),
        "depth_size": (int(depth_response.width), int(depth_response.height)),
    }


def draw_grid(image, cell_w, cell_h, color=(255, 255, 255)):
    out = image.copy()
    h, w = out.shape[:2]
    for x in range(0, w + 1, cell_w):
        cv2.line(out, (x, 0), (x, h), color, 1)
    for y in range(0, h + 1, cell_h):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def draw_lidar_strip(lidar_360, width=720, height=180):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    values = np.clip(lidar_360, 0, MAX_DISPLAY_RANGE)
    x_scale = width / 360.0
    for idx, value in enumerate(values):
        x = int(idx * x_scale)
        bar_h = int((1.0 - value / MAX_DISPLAY_RANGE) * (height - 20))
        color = (0, 180, 255) if idx not in LIDAR_105_INDICES else (0, 255, 0)
        cv2.line(canvas, (x, height - 1), (x, height - 1 - bar_h), color, 1)
    for label, bin_idx, color in (("L", 0, (255, 255, 255)), ("F", 90, (0, 255, 255)), ("R", 180, (255, 255, 255)), ("B", 270, (255, 255, 255))):
        x = int(bin_idx * x_scale)
        cv2.line(canvas, (x, 0), (x, height), color, 1)
        cv2.putText(canvas, label, (max(0, x - 10), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "LiDAR 360 bins (F~90, L~0, R~180, B~270)", (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_lidar_105_strip(lidar_105, width=720, height=140):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    values = np.clip(lidar_105, 0, MAX_DISPLAY_RANGE)
    x_scale = width / len(values)
    sections = [(0, 45, (0, 200, 255), "front"), (45, 60, (255, 180, 0), "left flank"), (60, 75, (180, 255, 0), "right flank"), (75, 105, (255, 0, 180), "rear")]
    for start, end, color, label in sections:
        sx = int(start * x_scale)
        ex = int(end * x_scale)
        cv2.rectangle(canvas, (sx, 0), (max(sx + 1, ex - 1), 18), color, -1)
        cv2.putText(canvas, label, (sx + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for idx, value in enumerate(values):
        x = int(idx * x_scale)
        bar_h = int((1.0 - value / MAX_DISPLAY_RANGE) * (height - 25))
        cv2.line(canvas, (x, height - 1), (x, height - 1 - bar_h), (0, 255, 0), 1)
    cv2.putText(canvas, "LiDAR 105 ordering used by Model1", (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_depth_panels(raw_depth, resized_depth, pooled_depth):
    raw_gray = raw_depth_to_grayscale(raw_depth)
    raw_color = depth_to_colormap(raw_depth)
    zero_mask_vis, zero_count = build_zero_mask_vis(raw_depth)

    resized_vis = depth_to_colormap(resized_depth)
    resized_vis = draw_grid(resized_vis, 16, 16)

    pooled_up = cv2.resize(depth_to_colormap(pooled_depth), (resized_vis.shape[1], resized_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    pooled_up = draw_grid(pooled_up, 16, 16, color=(0, 0, 0))

    block_info, block_min = pick_depth_block(resized_depth)
    if block_info is not None:
        br, bc, r, c, block = block_info
        cv2.rectangle(resized_vis, (c, r), (c + 16, r + 16), (255, 255, 255), 2)
        cv2.rectangle(pooled_up, (bc * 16, br * 16), (bc * 16 + 16, br * 16 + 16), (255, 255, 255), 2)
        center_value = float(block[8, 8])
        block_text = f"block=({br},{bc}) min={block_min:.2f} center={center_value:.2f}"
    else:
        block_text = "block=n/a"

    raw_stats = summarize_depth_values(raw_depth)
    resized_stats = summarize_depth_values(resized_depth)

    raw_gray = cv2.resize(raw_gray, (320, 180), interpolation=cv2.INTER_AREA)
    raw_color = cv2.resize(raw_color, (320, 180), interpolation=cv2.INTER_AREA)
    zero_mask_vis = cv2.resize(zero_mask_vis, (320, 180), interpolation=cv2.INTER_NEAREST)
    resized_vis = cv2.resize(resized_vis, (320, 180), interpolation=cv2.INTER_AREA)
    pooled_up = cv2.resize(pooled_up, (320, 180), interpolation=cv2.INTER_NEAREST)

    cv2.putText(raw_gray, f"Raw depth gray {raw_depth.shape[1]}x{raw_depth.shape[0]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(raw_gray, f"min={raw_stats['min']:.2f} mean={raw_stats['mean']:.2f} max={raw_stats['max']:.2f}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(raw_color, "Raw depth colormap (visualization only)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(raw_color, "Do not compare this panel to raw Scene colors", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(zero_mask_vis, f"Depth==0 mask | red=zero | count={zero_count}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(zero_mask_vis, f"resized zeros={resized_stats['zero_count']}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(resized_vis, "Resized 256x144 with 16x16 grid", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(pooled_up, f"Min pooled 9x16 | {block_text}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return raw_gray, raw_color, zero_mask_vis, resized_vis, pooled_up, block_text, raw_stats


def rotate_and_capture_lidar(client, vehicle_name, yaw_delta_deg):
    before = get_state(client, vehicle_name)
    target_yaw = before["yaw_deg"] + yaw_delta_deg
    client.rotateToYawAsync(target_yaw, vehicle_name=vehicle_name).join()
    time.sleep(0.6)
    _, lidar_360, _ = fetch_lidar_data(client, vehicle_name)
    after = get_state(client, vehicle_name)
    return {
        "yaw_before": before["yaw_deg"],
        "yaw_after": after["yaw_deg"],
        "nearest_bin": int(np.argmin(lidar_360)),
        "nearest_dist": float(np.min(lidar_360)),
    }


def run_yaw_rotation_probe(client, vehicle_name):
    set_pose_and_hover(client, vehicle_name)
    results = []
    for delta in (0, 90, 180, -90):
        results.append((delta, rotate_and_capture_lidar(client, vehicle_name, delta)))
    return results


def run_action_probe(client, vehicle_name, action, yaw_override_deg=None):
    set_pose_and_hover(client, vehicle_name)
    if yaw_override_deg is not None:
        client.rotateToYawAsync(yaw_override_deg, vehicle_name=vehicle_name).join()
        time.sleep(0.6)
    before = get_state(client, vehicle_name)
    commanded_accel = action_to_acceleration(action, accel_scale=ACCEL_SCALE)
    target_vel = integrate_velocity_with_acceleration(before["vel"], commanded_accel, dt=CONTROL_DT, max_v=10.0)
    commanded_yaw_deg = get_soft_alignment_yaw(target_vel, before["yaw_deg"], before["yaw_deg"])
    target_heading_deg = before["yaw_deg"]
    if np.linalg.norm(target_vel[:2]) > 1e-6:
        target_heading_deg = wrap_angle_deg(math.degrees(math.atan2(float(target_vel[1]), float(target_vel[0]))))
    yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(commanded_yaw_deg))
    client.moveByVelocityAsync(
        float(target_vel[0]),
        float(target_vel[1]),
        float(target_vel[2]),
        CONTROL_DT,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=yaw_mode,
        vehicle_name=vehicle_name,
    )
    time.sleep(CONTROL_DT + 0.15)
    after = get_state(client, vehicle_name)
    delta_pos = after["pos"] - before["pos"]
    delta_vel = after["vel"] - before["vel"]
    expected_delta_vel = commanded_accel * CONTROL_DT
    action_norm = float(np.linalg.norm(action))
    accel_norm = float(np.linalg.norm(commanded_accel))
    speed_xy = float(np.linalg.norm(target_vel[:2]))
    dominant_axis = int(np.argmax(np.abs(delta_vel)))
    expected_axis = int(np.argmax(np.abs(commanded_accel)))
    expected_sign = int(np.sign(commanded_accel[expected_axis])) if np.abs(commanded_accel[expected_axis]) > 1e-6 else 0
    observed_sign = int(np.sign(delta_vel[expected_axis])) if np.abs(delta_vel[expected_axis]) > 1e-6 else 0
    heading_error_deg = abs(wrap_angle_deg(after["yaw_deg"] - target_heading_deg))
    yaw_change_deg = wrap_angle_deg(after["yaw_deg"] - before["yaw_deg"])
    delta_vel_error = delta_vel - expected_delta_vel
    delta_vel_error_norm = float(np.linalg.norm(delta_vel_error))

    if action_norm <= 1e-6:
        verdict = "pass" if abs(yaw_change_deg) <= 5.0 and float(np.linalg.norm(delta_vel)) <= 0.5 else "suspicious"
    elif accel_norm <= 1e-6:
        verdict = "pass" if abs(yaw_change_deg) <= 5.0 else "suspicious"
    elif speed_xy <= 0.1:
        verdict = "pass" if abs(yaw_change_deg) <= 5.0 else "suspicious"
    else:
        verdict = "pass" if dominant_axis == expected_axis and observed_sign == expected_sign else "suspicious"

    return {
        "action": action.tolist(),
        "commanded_accel": commanded_accel.tolist(),
        "expected_delta_vel": expected_delta_vel.tolist(),
        "yaw": before["yaw_deg"],
        "yaw_after": after["yaw_deg"],
        "yaw_change_deg": float(yaw_change_deg),
        "target_vel": target_vel.tolist(),
        "target_heading_deg": float(target_heading_deg),
        "commanded_yaw_deg": float(commanded_yaw_deg),
        "heading_error_deg": float(heading_error_deg),
        "speed_xy": speed_xy,
        "delta_pos": delta_pos.tolist(),
        "delta_vel": delta_vel.tolist(),
        "delta_vel_error_norm": delta_vel_error_norm,
        "dominant_axis": dominant_axis,
        "expected_axis": expected_axis,
        "expected_sign": expected_sign,
        "observed_sign": observed_sign,
        "ang_acc_mag_before": before["ang_acc_mag"],
        "ang_acc_mag_after": after["ang_acc_mag"],
        "ang_acc_dim": int(before["ang_acc"].shape[0]),
        "verdict": verdict,
    }


def run_action_probe_suite(client, vehicle_name):
    results = []
    for yaw in (0.0, 90.0):
        for label, action in ACTION_PROBES:
            probe = run_action_probe(client, vehicle_name, action, yaw_override_deg=yaw)
            probe["label"] = label
            results.append(probe)
    return results


def draw_text_block(canvas, lines, origin, line_gap=22, color=(230, 230, 230)):
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += line_gap


def build_dashboard(scene, raw_gray, raw_color, zero_mask_vis, resized_vis, pooled_vis, lidar_strip, lidar_105_strip, lidar_report, block_text, action_summary, yaw_probe_summary, state, settings_summary, runtime_depth_size, capture_report, raw_stats):
    scene = cv2.resize(scene, (480, 270), interpolation=cv2.INTER_AREA)
    raw_gray = cv2.resize(raw_gray, (320, 180), interpolation=cv2.INTER_AREA)
    raw_color = cv2.resize(raw_color, (320, 180), interpolation=cv2.INTER_AREA)
    zero_mask_vis = cv2.resize(zero_mask_vis, (320, 180), interpolation=cv2.INTER_NEAREST)
    resized_vis = cv2.resize(resized_vis, (320, 180), interpolation=cv2.INTER_AREA)
    pooled_vis = cv2.resize(pooled_vis, (320, 180), interpolation=cv2.INTER_NEAREST)
    lidar_strip = cv2.resize(lidar_strip, (960, 200), interpolation=cv2.INTER_AREA)
    lidar_105_strip = cv2.resize(lidar_105_strip, (320, 180), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((980, 1500, 3), dtype=np.uint8)
    canvas[:270, :480] = scene
    canvas[:180, 500:820] = raw_gray
    canvas[:180, 840:1160] = raw_color
    canvas[:180, 1180:1500] = zero_mask_vis
    canvas[200:380, 500:820] = resized_vis
    canvas[200:380, 840:1160] = pooled_vis
    canvas[200:380, 1180:1500] = lidar_105_strip
    canvas[410:610, :960] = lidar_strip

    cv2.putText(canvas, "Raw Scene", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Depth capture mapping", (500, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LiDAR validation", (10, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    lidar_lines = [
        f"Yaw: {state['yaw_deg']:.1f} deg | Angular acc mag: {state['ang_acc_mag']:.3f}",
        f"Nearest LiDAR bin/dist: {lidar_report['nearest_bin']} / {lidar_report['nearest_distance']:.2f}m",
        f"Front density ratio vs rear: {lidar_report['front_back_density_ratio']:.2f}",
        f"LiDAR verdict: {lidar_report['verdict']}",
        f"Front nearest/mean/bins: {lidar_report['sectors']['front']['nearest']:.2f} / {lidar_report['sectors']['front']['mean']:.2f} / {lidar_report['sectors']['front']['valid_bins']}",
        f"Left nearest/mean/bins:  {lidar_report['sectors']['left']['nearest']:.2f} / {lidar_report['sectors']['left']['mean']:.2f} / {lidar_report['sectors']['left']['valid_bins']}",
        f"Right nearest/mean/bins: {lidar_report['sectors']['right']['nearest']:.2f} / {lidar_report['sectors']['right']['mean']:.2f} / {lidar_report['sectors']['right']['valid_bins']}",
        f"Rear nearest/mean/bins:  {lidar_report['sectors']['rear']['nearest']:.2f} / {lidar_report['sectors']['rear']['mean']:.2f} / {lidar_report['sectors']['rear']['valid_bins']}",
        f"Depth runtime size: {runtime_depth_size[0]}x{runtime_depth_size[1]}",
        block_text,
        "Current frame expectation: front around bin~90",
    ]
    draw_text_block(canvas, lidar_lines, (980, 430))

    capture_lines = [
        "Capture mapping",
        f"request[0]: {capture_report['request_order'][0]} -> {capture_report['response_order'][0]}",
        f"request[1]: {capture_report['request_order'][1]} -> {capture_report['response_order'][1]}",
        capture_report['scene_decode'],
        capture_report['depth_decode'],
        f"depth response image_type: {capture_report['depth_image_type']}",
        f"depth pixels_as_float: {capture_report['depth_is_float']}",
        f"raw depth finite/min/max: {raw_stats['finite_count']} / {raw_stats['min']:.2f} / {raw_stats['max']:.2f}",
        f"raw depth mean/zero_count: {raw_stats['mean']:.2f} / {raw_stats['zero_count']}",
        "Only the middle/right depth panels are visualizations.",
        "The left scene panel is decoded from Scene PNG only.",
    ]
    draw_text_block(canvas, capture_lines, (10, 650), line_gap=22)

    action_lines = ["Action probes (accel->integrated vel)"]
    action_lines.extend(action_summary)
    action_lines.append("")
    action_lines.append("Yaw rotation probe")
    action_lines.extend(yaw_probe_summary)
    draw_text_block(canvas, action_lines, (760, 650), line_gap=20)

    settings_lines = ["Settings summary"]
    if settings_summary.get("exists"):
        lidar_cfg = settings_summary.get("lidar", {})
        settings_lines.extend([
            f"Lidar frame: {lidar_cfg.get('DataFrame', 'n/a')}",
            f"LiDAR channels: {lidar_cfg.get('NumberOfChannels', 'n/a')}",
            f"LiDAR H-FOV: {lidar_cfg.get('HorizontalFOVStart', 'n/a')}..{lidar_cfg.get('HorizontalFOVEnd', 'n/a')}",
            f"LiDAR max dist cfg: {lidar_cfg.get('MaxDistance', 'n/a')}",
            f"Empty bin default: {LIDAR_EMPTY_VALUE}",
        ])
    else:
        settings_lines.append("settings.json not found")
    draw_text_block(canvas, settings_lines, (1180, 650), line_gap=22)

    footer = "Keys: Q quit | R rerun probes"
    cv2.putText(canvas, footer, (10, 955), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def summarize_action_results(results):
    lines = []
    for result in results[:8]:
        axis_map = ["X", "Y", "Z"]
        lines.append(
            f"yaw={result['yaw']:+5.0f}->{result['yaw_after']:+5.0f} {result['label']}: {result['verdict']} dom={axis_map[result['dominant_axis']]} dverr={result['delta_vel_error_norm']:.2f} err={result['heading_error_deg']:.1f} aa={result['ang_acc_mag_after']:.2f}"
        )
    if results:
        suspicious = sum(1 for r in results if r["verdict"] != "pass")
        low_speed = [r for r in results if r["speed_xy"] <= 0.3]
        low_speed_max_yaw = max((abs(r["yaw_change_deg"]) for r in low_speed), default=0.0)
        heading_mean = float(np.mean([r["heading_error_deg"] for r in results if r["speed_xy"] >= 0.6])) if any(r["speed_xy"] >= 0.6 for r in results) else 0.0
        delta_vel_error_mean = float(np.mean([r["delta_vel_error_norm"] for r in results]))
        aa_dims = sorted({r["ang_acc_dim"] for r in results})
        lines.append(f"total={len(results)} suspicious={suspicious} head_err_mean={heading_mean:.1f} low_speed_max_yaw={low_speed_max_yaw:.1f}")
        lines.append(f"dvel_err_mean={delta_vel_error_mean:.2f} aa_dim={aa_dims} aa_mag_after_mean={np.mean([r['ang_acc_mag_after'] for r in results]):.2f}")
        lines.append("Pass criteria: delta_vel axis/sign follows commanded_accel*dt; yaw tracks integrated horizontal velocity")
        lines.append(f"Note: +Z means descend in NED")
    return lines


def summarize_yaw_probe(results):
    lines = []
    for delta, result in results:
        lines.append(
            f"rot {delta:+4d}: yaw {result['yaw_before']:+6.1f}->{result['yaw_after']:+6.1f} nearest bin={result['nearest_bin']:3d} dist={result['nearest_dist']:.2f}"
        )
    return lines


def main():
    repo_root = Path(__file__).resolve().parents[1]
    settings_summary = load_settings_summary(repo_root)
    client, vehicle_name = connect_client()

    print(f"Using vehicle: {vehicle_name}")
    if settings_summary.get("exists"):
        lidar_cfg = settings_summary.get("lidar", {})
        print(f"LiDAR config frame={lidar_cfg.get('DataFrame')} channels={lidar_cfg.get('NumberOfChannels')} h_fov={lidar_cfg.get('HorizontalFOVStart')}..{lidar_cfg.get('HorizontalFOVEnd')}")
        print(f"LiDAR config max distance={lidar_cfg.get('MaxDistance', 'n/a')} | Model1 empty bin={LIDAR_EMPTY_VALUE}")
    print("Capture mapping expectation: Scene -> responses[0], DepthPlanar -> responses[1]")

    set_pose_and_hover(client, vehicle_name)
    yaw_probe_results = run_yaw_rotation_probe(client, vehicle_name)
    action_results = run_action_probe_suite(client, vehicle_name)
    action_summary = summarize_action_results(action_results)
    yaw_probe_summary = summarize_yaw_probe(yaw_probe_results)

    while True:
        set_pose_and_hover(client, vehicle_name)
        scene, raw_depth, resized_depth, pooled_depth, depth_response = fetch_depth_and_scene(client, vehicle_name)
        _, lidar_360, lidar_105 = fetch_lidar_data(client, vehicle_name)
        state = get_state(client, vehicle_name)
        lidar_report = analyze_lidar_alignment(lidar_360)
        raw_gray, raw_color, zero_mask_vis, resized_vis, pooled_vis, block_text, raw_stats = draw_depth_panels(raw_depth, resized_depth, pooled_depth)
        capture_report = verify_capture_mapping(depth_response)
        lidar_strip = draw_lidar_strip(lidar_360)
        lidar_105_strip = draw_lidar_105_strip(lidar_105)

        dashboard = build_dashboard(
            scene,
            raw_gray,
            raw_color,
            zero_mask_vis,
            resized_vis,
            pooled_vis,
            lidar_strip,
            lidar_105_strip,
            lidar_report,
            block_text,
            action_summary,
            yaw_probe_summary,
            state,
            settings_summary,
            (depth_response.width, depth_response.height),
            capture_report,
            raw_stats,
        )
        cv2.imshow(WINDOW_NAME, dashboard)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            yaw_probe_results = run_yaw_rotation_probe(client, vehicle_name)
            action_results = run_action_probe_suite(client, vehicle_name)
            action_summary = summarize_action_results(action_results)
            yaw_probe_summary = summarize_yaw_probe(yaw_probe_results)
        time.sleep(0.2)

    client.hoverAsync(vehicle_name=vehicle_name).join()
    client.armDisarm(False, vehicle_name=vehicle_name)
    client.enableApiControl(False, vehicle_name=vehicle_name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
