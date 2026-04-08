import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import airsim

import cv2
import numpy as np
import time
import math


def run_airsim_full_data_waypoint_monitor():
    # 1. 连接与初始化
    client = airsim.MultirotorClient()
    client.confirmConnection()
    v_name = "Drone1"
    l_name = "LLidar1"

    # 2. 坐标输入 (处理中英文逗号)
    print("\n" + "=" * 40)
    user_input = input("请输入终点坐标 X, Y, Z (例如 100, 50, -20): ")
    user_input = user_input.replace('，', ',').replace(' ', '')
    try:
        tx, ty, tz = map(float, user_input.split(','))
    except:
        tx, ty, tz = 50.0, 0.0, -15.0
        print(f"输入无效，使用默认终点: {tx}, {ty}, {tz}")
    print("=" * 40 + "\n")

    # 3. 飞行准备
    client.enableApiControl(True, vehicle_name=v_name)
    client.armDisarm(True, vehicle_name=v_name)
    client.takeoffAsync(vehicle_name=v_name).join()

    # --- 核心：计算 15 个均匀分布的 Waypoints ---
    start_pos = client.getMultirotorState(vehicle_name=v_name).kinematics_estimated.position
    sx, sy, sz = start_pos.x_val, start_pos.y_val, start_pos.z_val

    # 16等分产生15个中间点 + 1个终点 = 16个目标点
    waypoints = []
    for i in range(1, 17):
        ratio = i / 16.0
        wp_x = sx + (tx - sx) * ratio
        wp_y = sy + (ty - sy) * ratio
        wp_z = sz + (tz - sz) * ratio
        waypoints.append(airsim.Vector3r(wp_x, wp_y, wp_z))

    current_wp_idx = 0

    try:
        while True:
            # --- [A] 完整运动学数据提取 ---
            state = client.getMultirotorState(vehicle_name=v_name)
            k = state.kinematics_estimated

            # 1. 位置/速度/角加速度
            pos = k.position
            vel = k.linear_velocity
            acc = k.angular_acceleration  # 角加速度

            # 2. 导航逻辑：依次飞向航点
            target_wp = waypoints[current_wp_idx]
            client.moveToPositionAsync(target_wp.x_val, target_wp.y_val, target_wp.z_val,
                                       velocity=5, vehicle_name=v_name)

            # 3. 距离计算
            dist_to_wp = math.sqrt((pos.x_val - target_wp.x_val) ** 2 + (pos.y_val - target_wp.y_val) ** 2 + (
                        pos.z_val - target_wp.z_val) ** 2)
            dist_to_final = math.sqrt((pos.x_val - tx) ** 2 + (pos.y_val - ty) ** 2 + (pos.z_val - tz) ** 2)

            # 自动切换航点
            if dist_to_wp < 2.0 and current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1

            # 4. 激光雷达
            try:
                lidar_data = client.getLidarData(lidar_name=l_name, vehicle_name=v_name)
                lidar_pts = len(lidar_data.point_cloud) // 3
            except:
                lidar_pts = 0

            # --- [B] 图像获取与对齐 (解决 Broadcast 报错) ---
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ], vehicle_name=v_name)

            if len(responses) < 2: continue

            # RGB
            img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                           responses[0].width, 3).copy()
            h, w = img_rgb.shape[:2]

            # 深度图 Resize 对齐
            img_depth = np.array(responses[1].image_data_float).reshape(responses[1].height, responses[1].width)
            img_depth[img_depth > 30] = 30
            img_d_vis = cv2.applyColorMap(cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                          cv2.COLORMAP_JET)
            img_d_vis = cv2.resize(img_d_vis, (w, h))

            # --- [C] UI 布局拼接 (增加高度防止溢出) ---
            canvas_h = max(h, 800)
            canvas = np.zeros((canvas_h, w * 2 + 450, 3), dtype=np.uint8)
            canvas[:h, :w] = img_rgb
            canvas[:h, w:w * 2] = img_d_vis

            # --- [D] 完整数据面板 (绝无省略) ---
            x_offset = w * 2 + 30
            y_ptr = 40  # 垂直起始点
            gap = 26  # 紧凑行间距

            # 组织所有数据项
            panel_data = [
                (">> AIRSIM DASHBOARD <<", (0, 255, 255)),
                (f"TIME: {time.strftime('%H:%M:%S')}", (150, 150, 150)),
                ("-" * 38, (100, 100, 100)),

                (f"WAYPOINT: {current_wp_idx + 1} / 16", (0, 255, 0)),
                (f"DIST TO WP:  {dist_to_wp:.2f} m", (0, 200, 255)),
                (f"DIST TO END: {dist_to_final:.2f} m", (0, 100, 255)),
                ("-" * 38, (100, 100, 100)),

                ("WORLD POSITION (m):", (255, 255, 0)),
                (f"  X: {pos.x_val:>10.2f}", (255, 255, 255)),
                (f"  Y: {pos.y_val:>10.2f}", (255, 255, 255)),
                (f"  Z: {pos.z_val:>10.2f}", (255, 255, 255)),
                ("-" * 38, (100, 100, 100)),

                ("LINEAR VELOCITY (m/s):", (255, 255, 0)),
                (f"  Vx: {vel.x_val:>10.2f}", (0, 255, 255)),
                (f"  Vy: {vel.y_val:>10.2f}", (0, 255, 255)),
                (f"  Vz: {vel.z_val:>10.2f}", (0, 255, 255)),
                ("-" * 38, (100, 100, 100)),

                ("ANGULAR ACCEL (r/s^2):", (255, 255, 0)),
                (f"  Ax: {acc.x_val:>10.4f}", (255, 0, 255)),
                (f"  Ay: {acc.y_val:>10.4f}", (255, 0, 255)),
                (f"  Az: {acc.z_val:>10.4f}", (255, 0, 255)),
                ("-" * 38, (100, 100, 100)),

                (f"LIDAR POINTS: {lidar_pts:>10}", (0, 255, 0)),
                ("MISSION: " + ("COMPLETED" if (current_wp_idx == 15 and dist_to_wp < 1.5) else "RUNNING"),
                 (0, 255, 0)),
                ("-" * 38, (100, 100, 100)),
                ("PRESS 'Q' TO QUIT", (0, 0, 255))
            ]

            for text, color in panel_data:
                cv2.putText(canvas, text, (x_offset, y_ptr),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y_ptr += gap

            cv2.imshow("AirSim Professional Data Monitor", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"致命错误: {e}")
    finally:
        print("释放控制权...")
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_airsim_full_data_waypoint_monitor()