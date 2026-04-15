import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import airsim
import cv2
import numpy as np
import time
import math
import csv


# =========================
# 基本配置
# =========================
VEHICLE_NAME = "Drone1"
LIDAR_NAME = "Lidar1"

# 如果你输入的是 UE 世界坐标，会按这个原点转换
UE_ORIGIN_X = 1360.0
UE_ORIGIN_Y = 940.0
UE_ORIGIN_Z = 130.0

# 数据输出文件
RL_NPY_FILE = "rl_states.npy"
RL_CSV_FILE = "rl_states.csv"
COUNT_FILE = "dataset_count.txt"

# 航点数量、速度、采样频率
WAYPOINT_COUNT = 16
FLIGHT_SPEED = 4.0
SAMPLE_INTERVAL = 0.10     # 秒
MAX_LIDAR_RANGE = 30.0


# =========================
# 坐标转换
# =========================
def ue_world_to_airsim_ned(x, y, z):
    """
    UE 世界坐标 -> AirSim 本地 NED 坐标
    规则：
    1. 以 (1360, 940, 130) 为原点
    2. 各轴除以 10
    3. z 取负
    """
    nx = (x - UE_ORIGIN_X) / 10.0
    ny = (y - UE_ORIGIN_Y) / 10.0
    nz = -((z - UE_ORIGIN_Z) / 10.0)
    return nx, ny, nz


# =========================
# 数据集保存
# =========================
def get_dataset_count():
    if os.path.exists(RL_NPY_FILE):
        data = np.load(RL_NPY_FILE)
        if data.ndim == 1:
            return 1
        return data.shape[0]
    return 0


def update_count_file(count):
    with open(COUNT_FILE, "w", encoding="utf-8") as f:
        f.write(f"当前数据集条数: {count}\n")


def append_rl_state(state_row):
    """
    state_row 格式：
    [
        pos_x, pos_y, pos_z,
        vel_x, vel_y, vel_z,
        nearest_dist,
        front_dist,
        left_dist,
        right_dist,
        dist_to_final
    ]
    """
    row = np.array(state_row, dtype=np.float32).reshape(1, -1)

    if os.path.exists(RL_NPY_FILE):
        old_data = np.load(RL_NPY_FILE)
        if old_data.ndim == 1:
            old_data = old_data.reshape(1, -1)
        new_data = np.vstack([old_data, row])
    else:
        new_data = row

    np.save(RL_NPY_FILE, new_data)

    header = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "nearest_dist",
        "front_dist",
        "left_dist",
        "right_dist",
        "dist_to_final"
    ]

    file_exists = os.path.exists(RL_CSV_FILE)
    with open(RL_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row.flatten().tolist())

    count = new_data.shape[0]
    update_count_file(count)
    return count


# =========================
# 图像解析
# =========================
def decode_scene_png(response):
    """
    Scene 图像按 PNG 压缩图解码
    """
    img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Scene PNG 解码失败，收到字节数: {img_1d.size}")
    return img_bgr


def decode_depth_float(response):
    """
    DepthPlanar 浮点深度图解析
    """
    depth_arr = np.array(response.image_data_float, dtype=np.float32)
    expected_size = response.height * response.width

    if depth_arr.size != expected_size:
        raise ValueError(
            f"Depth 数据长度不匹配: got={depth_arr.size}, expected={expected_size}"
        )

    depth_img = depth_arr.reshape(response.height, response.width)
    return depth_img


# =========================
# 雷达特征
# =========================
def extract_lidar_features(lidar_data, max_range=30.0):
    """
    从点云提取简单特征：
    nearest_dist, front_dist, left_dist, right_dist, lidar_pts

    假设 DataFrame=SensorLocalFrame 时：
    x: 前方
    y: 右方
    z: 下方
    """
    if lidar_data is None or len(lidar_data.point_cloud) < 3:
        return max_range, max_range, max_range, max_range, 0

    pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    lidar_pts = len(pts)

    if lidar_pts == 0:
        return max_range, max_range, max_range, max_range, 0

    dists = np.linalg.norm(pts, axis=1)
    nearest_dist = float(np.min(dists))

    # 前方点
    front_pts = pts[pts[:, 0] > 0]
    # 左侧点（y < 0）
    left_pts = pts[(pts[:, 0] > 0) & (pts[:, 1] < 0)]
    # 右侧点（y > 0）
    right_pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]

    front_dist = float(np.min(np.linalg.norm(front_pts, axis=1))) if len(front_pts) > 0 else max_range
    left_dist = float(np.min(np.linalg.norm(left_pts, axis=1))) if len(left_pts) > 0 else max_range
    right_dist = float(np.min(np.linalg.norm(right_pts, axis=1))) if len(right_pts) > 0 else max_range

    return nearest_dist, front_dist, left_dist, right_dist, lidar_pts


# =========================
# 工具函数
# =========================
def calc_distance(p1x, p1y, p1z, p2x, p2y, p2z):
    return math.sqrt(
        (p1x - p2x) ** 2 +
        (p1y - p2y) ** 2 +
        (p1z - p2z) ** 2
    )


def build_waypoints(start_pos, tx, ty, tz, waypoint_count=16):
    sx, sy, sz = start_pos.x_val, start_pos.y_val, start_pos.z_val
    waypoints = []
    for i in range(1, waypoint_count + 1):
        ratio = i / float(waypoint_count)
        wx = sx + (tx - sx) * ratio
        wy = sy + (ty - sy) * ratio
        wz = sz + (tz - sz) * ratio
        waypoints.append(airsim.Vector3r(wx, wy, wz))
    return waypoints


def read_target_coordinate():
    """
    支持两种输入：
    1 -> 直接输入 AirSim 本地 NED 坐标
    2 -> 输入 UE 世界坐标，自动转换
    """
    print("\n" + "=" * 60)
    print("请选择输入坐标模式：")
    print("1. AirSim 本地 NED 坐标（推荐先测试，例如 10,0,-5）")
    print("2. UE 世界坐标（自动按原点 1360,940,130 转换）")
    mode = input("请输入 1 或 2：").strip()

    if mode == "2":
        raw = input("请输入 UE 世界坐标 X,Y,Z：").strip()
        raw = raw.replace("，", ",").replace(" ", "")
        try:
            x, y, z = map(float, raw.split(","))
            tx, ty, tz = ue_world_to_airsim_ned(x, y, z)
            print(f"转换后的 AirSim 本地 NED 坐标：({tx:.2f}, {ty:.2f}, {tz:.2f})")
            print("=" * 60 + "\n")
            return tx, ty, tz
        except Exception:
            print("输入无效，默认使用 AirSim 本地坐标: (10, 0, -5)")
            print("=" * 60 + "\n")
            return 10.0, 0.0, -5.0

    raw = input("请输入 AirSim 本地 NED 坐标 X,Y,Z：").strip()
    raw = raw.replace("，", ",").replace(" ", "")
    try:
        tx, ty, tz = map(float, raw.split(","))
    except Exception:
        print("输入无效，默认使用 AirSim 本地坐标: (10, 0, -5)")
        tx, ty, tz = 10.0, 0.0, -5.0

    print("=" * 60 + "\n")
    return tx, ty, tz


# =========================
# 主逻辑
# =========================
def run_airsim_rl_monitor():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    current_count = get_dataset_count()
    update_count_file(current_count)

    print(f"当前已有数据条数: {current_count}")

    # 先读目标
    tx, ty, tz = read_target_coordinate()

    # 接管无人机
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    # 起飞
    print("正在起飞...")
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    time.sleep(2.0)
    client.hoverAsync(vehicle_name=VEHICLE_NAME).join()

    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pos = state.kinematics_estimated.position

    print("起飞后位置：")
    print(f"X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
    print(f"目标位置：X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}")

    # 生成航点
    waypoints = build_waypoints(pos, tx, ty, tz, WAYPOINT_COUNT)
    current_wp_idx = 0
    target_wp = waypoints[current_wp_idx]

    # 只在切换航点时发送一次命令
    print(f"发送第 1 个航点命令: ({target_wp.x_val:.2f}, {target_wp.y_val:.2f}, {target_wp.z_val:.2f})")
    client.moveToPositionAsync(
        target_wp.x_val,
        target_wp.y_val,
        target_wp.z_val,
        FLIGHT_SPEED,
        vehicle_name=VEHICLE_NAME
    )

    last_save_time = 0.0
    mission_done = False

    try:
        while True:
            state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
            k = state.kinematics_estimated

            pos = k.position
            vel = k.linear_velocity
            acc = k.angular_acceleration

            # 距离
            dist_to_wp = calc_distance(
                pos.x_val, pos.y_val, pos.z_val,
                target_wp.x_val, target_wp.y_val, target_wp.z_val
            )

            dist_to_final = calc_distance(
                pos.x_val, pos.y_val, pos.z_val,
                tx, ty, tz
            )

            # 到达当前航点后，才发送下一个航点命令
            if (dist_to_wp < 1.5) and (current_wp_idx < len(waypoints) - 1):
                current_wp_idx += 1
                target_wp = waypoints[current_wp_idx]

                print(
                    f"切换到第 {current_wp_idx + 1} 个航点: "
                    f"({target_wp.x_val:.2f}, {target_wp.y_val:.2f}, {target_wp.z_val:.2f})"
                )

                client.moveToPositionAsync(
                    target_wp.x_val,
                    target_wp.y_val,
                    target_wp.z_val,
                    FLIGHT_SPEED,
                    vehicle_name=VEHICLE_NAME
                )

            elif (dist_to_wp < 1.0) and (current_wp_idx == len(waypoints) - 1) and (not mission_done):
                mission_done = True
                print("已到达最终目标点，切换为悬停。")
                client.hoverAsync(vehicle_name=VEHICLE_NAME).join()

            # 雷达
            try:
                lidar_data = client.getLidarData(
                    lidar_name=LIDAR_NAME,
                    vehicle_name=VEHICLE_NAME
                )
                nearest_dist, front_dist, left_dist, right_dist, lidar_pts = extract_lidar_features(
                    lidar_data,
                    max_range=MAX_LIDAR_RANGE
                )
            except Exception as e:
                print(f"[Lidar Error] {e}")
                nearest_dist, front_dist, left_dist, right_dist, lidar_pts = (
                    MAX_LIDAR_RANGE, MAX_LIDAR_RANGE, MAX_LIDAR_RANGE, MAX_LIDAR_RANGE, 0
                )

            # 定时保存 RL 状态
            now = time.time()
            total_count = current_count
            if now - last_save_time >= SAMPLE_INTERVAL:
                rl_state = [
                    pos.x_val, pos.y_val, pos.z_val,
                    vel.x_val, vel.y_val, vel.z_val,
                    nearest_dist,
                    front_dist,
                    left_dist,
                    right_dist,
                    dist_to_final
                ]
                total_count = append_rl_state(rl_state)
                current_count = total_count
                last_save_time = now

            # 图像
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
                    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
                ], vehicle_name=VEHICLE_NAME)

                if len(responses) < 2:
                    print("[Warning] simGetImages 返回数量不足")
                    time.sleep(0.02)
                    continue

                img_bgr = decode_scene_png(responses[0])
                h, w = img_bgr.shape[:2]

                img_depth = decode_depth_float(responses[1])
                img_depth = np.clip(img_depth, 0, MAX_LIDAR_RANGE)

                img_d_vis = cv2.applyColorMap(
                    cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                img_d_vis = cv2.resize(img_d_vis, (w, h))

            except Exception as e:
                print(f"[Image Error] {e}")
                time.sleep(0.02)
                continue

            # 画面
            canvas_h = max(h, 850)
            canvas_w = w * 2 + 520
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            canvas[:h, :w] = img_bgr
            canvas[:h, w:w * 2] = img_d_vis

            x_offset = w * 2 + 30
            y_ptr = 40
            gap = 26

            panel_data = [
                (">> AIRSIM RL DASHBOARD <<", (0, 255, 255)),
                (f"TIME: {time.strftime('%H:%M:%S')}", (180, 180, 180)),
                ("-" * 42, (100, 100, 100)),

                (f"WAYPOINT: {current_wp_idx + 1} / {WAYPOINT_COUNT}", (0, 255, 0)),
                (f"DIST TO WP:    {dist_to_wp:.2f} m", (0, 200, 255)),
                (f"DIST TO END:   {dist_to_final:.2f} m", (0, 100, 255)),
                ("-" * 42, (100, 100, 100)),

                ("WORLD POSITION (m):", (255, 255, 0)),
                (f"  X: {pos.x_val:>10.2f}", (255, 255, 255)),
                (f"  Y: {pos.y_val:>10.2f}", (255, 255, 255)),
                (f"  Z: {pos.z_val:>10.2f}", (255, 255, 255)),
                ("-" * 42, (100, 100, 100)),

                ("LINEAR VELOCITY (m/s):", (255, 255, 0)),
                (f"  Vx: {vel.x_val:>10.2f}", (0, 255, 255)),
                (f"  Vy: {vel.y_val:>10.2f}", (0, 255, 255)),
                (f"  Vz: {vel.z_val:>10.2f}", (0, 255, 255)),
                ("-" * 42, (100, 100, 100)),

                ("ANGULAR ACCEL (rad/s^2):", (255, 255, 0)),
                (f"  Ax: {acc.x_val:>10.4f}", (255, 0, 255)),
                (f"  Ay: {acc.y_val:>10.4f}", (255, 0, 255)),
                (f"  Az: {acc.z_val:>10.4f}", (255, 0, 255)),
                ("-" * 42, (100, 100, 100)),

                ("LIDAR FEATURES:", (255, 255, 0)),
                (f"  POINTS:  {lidar_pts:>10}", (0, 255, 0)),
                (f"  NEAREST: {nearest_dist:>10.2f}", (0, 255, 255)),
                (f"  FRONT:   {front_dist:>10.2f}", (255, 255, 255)),
                (f"  LEFT:    {left_dist:>10.2f}", (255, 255, 255)),
                (f"  RIGHT:   {right_dist:>10.2f}", (255, 255, 255)),
                ("-" * 42, (100, 100, 100)),

                (f"DATASET SIZE: {current_count}", (255, 0, 255)),
                ("MISSION: " + ("COMPLETED" if mission_done else "RUNNING"), (0, 255, 0)),
                ("PRESS 'Q' TO QUIT", (0, 0, 255))
            ]

            for text, color in panel_data:
                cv2.putText(
                    canvas,
                    text,
                    (x_offset, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )
                y_ptr += gap

            cv2.imshow("AirSim RL Data Monitor", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            time.sleep(0.02)

    except Exception as e:
        print(f"致命错误: {e}")

    finally:
        print("释放控制权...")
        try:
            client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        except Exception:
            pass

        try:
            client.armDisarm(False, vehicle_name=VEHICLE_NAME)
            client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
        except Exception as e:
            print(f"释放控制权时出错: {e}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_airsim_rl_monitor()