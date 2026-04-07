import airsim
import cv2
import numpy as np
import time
import math


def run_airsim_mega_monitor():
    # 1. 连接与初始化
    client = airsim.MultirotorClient()
    client.confirmConnection()
    v_name = "Drone1"
    l_name = "Lidar1"  # 请确保与 settings.json 一致

    # 2. 坐标输入处理 (兼容中英文逗号)
    user_input = input("请输入目标坐标 X, Y, Z (如 15,15,-20): ")
    user_input = user_input.replace('，', ',').replace(' ', '')
    try:
        tx, ty, tz = map(float, user_input.split(','))
    except:
        tx, ty, tz = 20.0, 20.0, -10.0
        print(f"输入解析失败，使用默认值: {tx}, {ty}, {tz}")

    client.enableApiControl(True, vehicle_name=v_name)
    client.armDisarm(True, vehicle_name=v_name)
    client.takeoffAsync(vehicle_name=v_name).join()

    # 异步开始导航
    client.moveToPositionAsync(tx, ty, tz, velocity=3, vehicle_name=v_name)

    try:
        while True:
            # --- [A] 数据提取 ---
            state = client.getMultirotorState(vehicle_name=v_name)
            k = state.kinematics_estimated

            # 1. 坐标与距离
            pos = k.position
            dist = math.sqrt((pos.x_val - tx) ** 2 + (pos.y_val - ty) ** 2 + (pos.z_val - tz) ** 2)

            # 2. 三轴线速度 (Linear Velocity)
            lv = k.linear_velocity

            # 3. 三轴角加速度 (Angular Acceleration)
            aa = k.angular_acceleration

            # 4. 激光雷达数据
            try:
                lidar_data = client.getLidarData(lidar_name=l_name, vehicle_name=v_name)
                lidar_pts = len(lidar_data.point_cloud) // 3
            except:
                lidar_pts = 0

            # 5. 图像获取
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ], vehicle_name=v_name)

            if len(responses) < 2: continue

            # --- [B] 图像处理 ---
            # 1. 处理 RGB 转换
            img_rgb_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_raw.reshape(responses[0].height, responses[0].width, 3).copy()

            # 获取 RGB 的尺寸作为基准
            target_h, target_w = img_rgb.shape[:2]

            # 2. 处理 深度图 转换
            img_depth_raw = np.array(responses[1].image_data_float).reshape(responses[1].height, responses[1].width)
            img_depth_raw[img_depth_raw > 30] = 30
            img_depth_norm = cv2.normalize(img_depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_depth_vis_raw = cv2.applyColorMap(img_depth_norm, cv2.COLORMAP_JET)

            # --- 核心修复：强制对齐尺寸 ---
            # 将深度图拉伸/缩小到和 RGB 一样大，防止拼接报错
            img_depth_vis = cv2.resize(img_depth_vis_raw, (target_w, target_h))
            # --- [C] UI 布局拼接 ---
            h, w = img_rgb.shape[:2]
            # 创建宽大的画布：两张图的宽度 + 420像素的侧边栏
            canvas_h = max(h, 720)
            canvas_w = w * 2 + 420
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 放置图像
            canvas[:h, :w] = img_rgb
            canvas[:h, w:w * 2] = img_depth_vis

            # --- [D] 绘制右侧数据面板 (DataItems) ---
            db_x = w * 2 + 25
            line_y = 40
            line_step = 28  # 行间距

            # 准备数据项列表: (文本, 颜色)
            data_items = [
                (">> FLIGHT SYSTEM MONITOR <<", (0, 255, 255)),
                (f"TIME: {time.strftime('%H:%M:%S')}", (200, 200, 200)),
                ("-" * 35, (100, 100, 100)),

                ("MISSION TARGET", (255, 255, 255)),
                (f"  TARGET POS: {tx:.1f}, {ty:.1f}, {tz:.1f}", (255, 255, 255)),
                (f"  DISTANCE TO GO: {dist:>8.2f} m", (0, 165, 255)),
                ("-" * 35, (100, 100, 100)),

                ("REAL-TIME POSITION (m)", (255, 255, 0)),
                (f"  POS X: {pos.x_val:>8.2f}", (0, 255, 0)),
                (f"  POS Y: {pos.y_val:>8.2f}", (0, 255, 0)),
                (f"  POS Z: {pos.z_val:>8.2f}", (0, 255, 0)),
                ("-" * 35, (100, 100, 100)),

                ("LINEAR VELOCITY (m/s)", (255, 255, 0)),
                (f"  Vx: {lv.x_val:>8.2f}", (0, 255, 255)),
                (f"  Vy: {lv.y_val:>8.2f}", (0, 255, 255)),
                (f"  Vz: {lv.z_val:>8.2f}", (0, 255, 255)),
                ("-" * 35, (100, 100, 100)),

                ("ANGULAR ACCEL (rad/s^2)", (255, 255, 0)),
                (f"  Ax: {aa.x_val:>8.4f}", (255, 0, 255)),
                (f"  Ay: {aa.y_val:>8.4f}", (255, 0, 255)),
                (f"  Az: {aa.z_val:>8.4f}", (255, 0, 255)),
                ("-" * 35, (100, 100, 100)),

                (f"LIDAR POINTS: {lidar_pts:>8}", (0, 255, 0)),
                ("STATUS: " + ("NAVIGATING" if dist > 1 else "GOAL REACHED"), (0, 255, 0)),
                ("-" * 35, (100, 100, 100)),
                ("Press 'Q' to End Session", (0, 0, 255))
            ]

            # 循环绘制文字
            for text, color in data_items:
                cv2.putText(canvas, text, (db_x, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                line_y += line_step

            # --- [E] 显示窗口 ---
            cv2.imshow("AirSim All-in-One Dashboard", canvas)

            # 到达判定与安全退出
            if dist < 0.5:
                client.hoverAsync(vehicle_name=v_name)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"运行时发生错误: {e}")
    finally:
        # 退出时释放控制权，防止出现 IOLoop 错误
        print("正在清理并退出...")
        try:
            client.armDisarm(False, vehicle_name=v_name)
            client.enableApiControl(False, vehicle_name=v_name)
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_airsim_mega_monitor()