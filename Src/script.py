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
    l_name = "Lidar1"

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

    # --- 核心：计算 15 个 Waypoints ---
    start_pos = client.getMultirotorState(vehicle_name=v_name).kinematics_estimated.position
    sx, sy, sz = start_pos.x_val, start_pos.y_val, start_pos.z_val

    # 16等分产生15个中间点 + 1个终点 = 16个点
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
            # --- [A] 完整传感器数据提取 ---
            state = client.getMultirotorState(vehicle_name=v_name)
            k = state.kinematics_estimated
            pos, vel, acc = k.position, k.linear_velocity, k.angular_acceleration

            # 导航逻辑
            target_wp = waypoints[current_wp_idx]
            client.moveToPositionAsync(target_wp.x_val, target_wp.y_val, target_wp.z_val,
                                       velocity=5, vehicle_name=v_name)

            # 计算到当前航点的距离
            dist_to_wp = math.sqrt((pos.x_val - target_wp.x_val) ** 2 + (pos.y_val - target_wp.y_val) ** 2 + (
                        pos.z_val - target_wp.z_val) ** 2)

            # 航点切换逻辑
            if dist_to_wp < 2.0 and current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1

            # 激光雷达
            try:
                lidar_data = client.getLidarData(lidar_name=l_name, vehicle_name=v_name)
                lidar_pts = len(lidar_data.point_cloud) // 3
            except:
                lidar_pts = 0

            # --- [B] 图像获取与分辨率对齐 (防止报错的关键) ---
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ], vehicle_name=v_name)

            if len(responses) < 2: continue

            # RGB 处理
            img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                           responses[0].width, 3).copy()
            h, w = img_rgb.shape[:2]

            # 深度图处理 + Resize 对齐
            img_depth = np.array(responses[1].image_data_float).reshape(responses[1].height, responses[1].width)
            img_depth[img_depth > 30] = 30
            img_d_norm = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_d_vis = cv2.applyColorMap(img_d_norm, cv2.COLORMAP_JET)
            img_d_vis = cv2.resize(img_d_vis, (w, h))  # 强制对齐

            # --- [C] UI 布局拼接 ---
            canvas_h = max(h, 750)  # 确保高度足够显示所有数据
            canvas = np.zeros((canvas_h, w * 2 + 420, 3), dtype=np.uint8)
            canvas[:h, :w] = img_rgb
            canvas[:h, w:w * 2] = img_d_vis

            # --- [D] 完整数据面板绘制 ---
            x_offset = w * 2 + 25
            y_start = 40
            dy = 28  # 行间距

            data_panel = [
                (">> MEGA DATA MONITOR <<", (0, 255, 255)),
                (f"TIME: {time.strftime('%H:%M:%S')}", (150, 150, 150)),
                ("-" * 35, (100, 100, 100)),
                (f"WP PROGRESS: {current_wp_idx + 1} / 16", (0, 255, 0)),
                (f"DIST TO WP:  {dist_to_wp:>8.2f} m", (0, 165, 255)),
                ("-" * 35, (100, 100, 100)),
                ("POSITION (World):", (255, 255, 0)),
                (f"  X: {pos.x_val:>8.2f}", (255, 255, 255)),
                (f"  Y: {pos.y_val:>8.2f}", (255, 255, 255)),
                (f"  Z: {pos.z_val:>8.2f}", (255, 255, 255)),
                ("-" * 35, (100, 100, 100)),
                ("LINEAR VEL (m/s):", (255, 255, 0)),
                (f"  Vx: {vel.x_val:>8.2f}", (0, 255, 255)),
                (f"  Vy: {vel.y_val:>8.2f}", (0, 255, 255)),
                (f"  Vz: {vel.z_val:>8.2f}", (0, 255, 255)),
                ("-" * 35, (100, 100, 100)),
                ("ANGULAR ACC (r/s^2):", (255, 255, 0)),
                (f"  Ax: {acc.x_val:>8.4f}", (255, 0, 255)),
                (f"  Ay: {acc.y_val:>8.4f}", (255, 0, 255)),
                (f"  Az: {acc.z_val:>8.4f}", (255, 0, 255)),
                ("-" * 35, (100, 100, 100)),
                (f"LIDAR POINTS: {lidar_pts:>8}", (0, 255, 0)),
                ("STATUS: " + ("ARRIVED" if (current_wp_idx == 15 and dist_to_wp < 1) else "FLYING"), (0, 255, 0)),
                ("-" * 35, (100, 100, 100)),
                ("Press 'Q' to Emergency Stop", (0, 0, 255))
            ]

            for i, (text, color) in enumerate(data_panel):
                cv2.putText(canvas, text, (x_offset, y_start + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            cv2.imshow("AirSim All-in-One Controller", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("清理资源中...")
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_airsim_full_data_waypoint_monitor()