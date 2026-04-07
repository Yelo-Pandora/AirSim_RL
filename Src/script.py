import airsim
import cv2
import numpy as np
import time
import math


def run_airsim_trace_monitor():
    # 1. 连接 AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    v_name = "Drone1"
    l_name = "Lidar1"  # 请确保 settings.json 中也是这个名字

    # --- 2. 轨迹绘制参数初始化 ---
    last_pose = None
    trace_color_rgba = [255, 0, 0, 1.0]  # 纯红色 (R, G, B, A)
    trace_thickness = 10.0  # 线条厚度
    trace_duration = 60.0  # 线条存留时间 (秒)
    # -----------------------------------

    # 3. 获取用户输入的坐标
    print("\n" + "=" * 30)
    try:
        target_input = input("请输入目标坐标 X, Y, Z (例如: 20, 20, -10): ")
        tx, ty, tz = map(float, target_input.split(','))
    except:
        print("输入格式错误，使用默认坐标: 20, 20, -10")
        tx, ty, tz = 20.0, 20.0, -10.0
    print("=" * 30 + "\n")

    # 4. 飞行初始化
    client.enableApiControl(True, vehicle_name=v_name)
    client.armDisarm(True, vehicle_name=v_name)
    client.takeoffAsync(vehicle_name=v_name).join()

    # 开始异步飞向目标
    print(f"正在飞向目标点: ({tx}, {ty}, {tz})...")
    client.moveToPositionAsync(tx, ty, tz, velocity=3, vehicle_name=v_name)

    try:
        while True:
            # --- 数据采集 ---
            state = client.getMultirotorState(vehicle_name=v_name)
            current_pose = state.kinematics_estimated.position  # 这是 airsim.Vector3r 对象
            vel = state.kinematics_estimated.linear_velocity

            # --- 核心新增：绘制红色轨迹 ---
            if last_pose is not None:
                # 在当前位置打一个红色的点，持续60秒
                # 即使没有 simDrawLines，通常也有 simPlotPoints
                try:
                    client.simPlotPoints(
                        points=[current_pose],
                        color_rgba=[255, 0, 0, 1.0],
                        size=15,
                        duration=60,
                        is_persistent=False
                    )
                except AttributeError:
                    # 如果连这个都没有，我们就在控制台提醒一下，转为2D轨迹预览
                    pass

            last_pose = current_pose
            # --------------------------------

            # --- 图像处理与 UI 拼接 (保持之前的逻辑) ---
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ], vehicle_name=v_name)

            if len(responses) < 2: continue

            img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb.reshape(responses[0].height, responses[0].width, 3).copy()

            img_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height,
                                                                                          responses[1].width)
            img_depth[img_depth > 30] = 30
            img_depth_vis = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX)
            img_depth_vis = cv2.applyColorMap(np.uint8(img_depth_vis), cv2.COLORMAP_JET)

            try:
                lidar_data = client.getLidarData(lidar_name=l_name, vehicle_name=v_name)
                lidar_pts = len(lidar_data.point_cloud) // 3
            except:
                lidar_pts = 0

            h1, w1 = img_rgb.shape[:2]
            h2, w2 = img_depth_vis.shape[:2]
            canvas_h = max(h1, h2, 550)
            panel_w = 380
            final_view = np.zeros((canvas_h, w1 + w2 + panel_w, 3), dtype=np.uint8)

            final_view[:h1, :w1] = img_rgb
            final_view[:h2, w1:w1 + w2] = img_depth_vis

            speed = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
            dist_to_target = math.sqrt(
                (current_pose.x_val - tx) ** 2 + (current_pose.y_val - ty) ** 2 + (current_pose.z_val - tz) ** 2)

            db_x = w1 + w2 + 25
            data_items = [
                (">> NAVIGATION & TRACE <<", (0, 0, 255)),  # 标题改为红色
                (f"TIME: {time.strftime('%H:%M:%S')}", (200, 200, 200)),
                ("-" * 30, (100, 100, 100)),
                (f"TARGET POS: ({tx:.1f}, {ty:.1f}, {tz:.1f})", (255, 255, 255)),
                (f"DIST TO TG: {dist_to_target:>8.2f} m", (0, 165, 255)),
                ("-" * 30, (100, 100, 100)),
                (f"CURRENT X: {current_pose.x_val:>8.2f} m", (0, 255, 0)),
                (f"CURRENT Y: {current_pose.y_val:>8.2f} m", (0, 255, 0)),
                (f"CURRENT Z: {current_pose.z_val:>8.2f} m", (0, 255, 0)),
                ("-" * 30, (100, 100, 100)),
                (f"SPEED:     {speed:>8.2f} m/s", (255, 0, 255)),
                (f"LIDAR PTS: {lidar_pts:>8}", (0, 255, 0)),
                ("-" * 30, (100, 100, 100)),
                ("MISSION STATUS:", (0, 255, 255)),
                ("NAVIGATING..." if dist_to_target > 1 else "GOAL REACHED!", (0, 255, 0)),
                ("-" * 30, (100, 100, 100)),
                ("HINT: Red line indicates path", (0, 0, 255))
            ]

            for i, (text, color) in enumerate(data_items):
                cv2.putText(final_view, text, (db_x, 45 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            cv2.imshow("AirSim Nav Hub with Trace", final_view)

            if dist_to_target < 0.5:
                client.hoverAsync(vehicle_name=v_name)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        client.armDisarm(False, vehicle_name=v_name)
        client.enableApiControl(False, vehicle_name=v_name)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_airsim_trace_monitor()