import torch
import math
import csv
import random
import os
import gymnasium as gym
import numpy as np
import airsim
import time
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from deep_network import LDTED3FeatureExtractor
from preprocessing_utils import (
    decode_depth_planar,
    resize_depth_for_ldtd3,
    downsample_depth_minpool,
    lidar_points_to_360,
    downsample_lidar_105,
    action_to_acceleration,
    integrate_velocity_with_acceleration,
)

# 基于特征提取的网络自定义 SB3 要用到的特征提取器
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # SB3 最终输出维度是 74
        super().__init__(observation_space, features_dim=74)
        self.extractor = LDTED3FeatureExtractor()

    def forward(self, observations) -> torch.Tensor:
        # SB3 会自动把环境返回的 NumPy 字典转换为 PyTorch Tensor
        depth = observations["depth"]
        lidar = observations["lidar"]
        kin = observations["kinematics"]
        return self.extractor(depth, lidar, kin)

 # gym环境
class AirSimUAVEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 从 CSV 加载目标位置数据，起点默认固定为 [0.0, 0.0, 0.0]
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.positions_file = os.path.join(base_dir, "dataset", "relative_coordinates_export.csv")
        self.position_pairs = []
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过第一行表头 (x,y,z)
                    for row in reader:
                        if len(row) >= 3:
                            # 读取 CSV 中的终点坐标
                            target_pos = [float(row[0]), float(row[1]), float(row[2])]
                            # 组装数据字典：起点固定，终点来自 CSV
                            self.position_pairs.append({
                                "start": [0.0, 0.0, 0.0],
                                "target": target_pos
                            })
                print(f"成功从 CSV 加载 {len(self.position_pairs)} 组目标坐标数据")
            except Exception as e:
                print(f"读取 CSV 文件时出错: {e}")
        # 兜底逻辑：如果文件不存在或内容为空，使用默认值
        if not self.position_pairs:
            self.position_pairs = [{"start": [0.0, 0.0, 0.0], "target": [20.0, 0.0, -2.0]}]
            print(f"警告：未找到或未能读取 {self.positions_file}，使用默认测试位置。")



        # 参数初始化
        # 动作空间: 3维连续加速度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # 状态空间: Dict字典形式，包含下采样后的深度图、精简后的雷达数据、无人机自身的运动参数
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 9, 16), dtype=np.float32),
            "lidar": spaces.Box(low=-np.inf, high=np.inf, shape=(105,), dtype=np.float32),
            "kinematics": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        })


        # 实例化airsim
        self.client = airsim.MultirotorClient() # 实例化客户端
        self.client.confirmConnection()         # 阻塞等待，直到连上虚幻引擎
        # 自动检测无人机名称，兼容不同的 settings.json 配置
        self.vehicle_name = "Drone1" # 默认值
        try:
            vehicles = self.client.listVehicles()
            self.vehicle_name = vehicles[0] if vehicles else "Drone1"
        except:
            pass
        print(f"检测到无人机: '{self.vehicle_name}'")
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)      # 获取无人机的 API 控制权
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)             # 解锁无人机电机 (Ready to fly)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()         # 自动起飞，脱离地面
        # 禁用悬停安全机制，防止 "API call was not received" 频繁出现并中断动作
        # 在某些版本的 AirSim 中，该方法叫 setApiControlTimeout 或类似名称，若不支持可通过捕捉异常跳过
        try:
            self.client.client.call('setApiControlTimeout', 0.0, self.vehicle_name)
        except Exception:
            pass


        self.step_count = 0                                                         # 统计当前回合走过的步数
        self.max_steps = 150                                                        # 允许的最大步数
        self.current_target = np.array([20.0, 0.0, -2.0], dtype=np.float32)   # 默认目标
        self.current_start_pos = np.array([0.0, 0.0, -2.0], dtype=np.float32) # 默认起点
        self.start_rel_pos = self.current_target - self.current_start_pos           # 开始的相对位置
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))                 # 起点与终点的距离
        self.dt = 0.5                                                               # 步长时间
        self.accel_scale = 2.0                                                      # 让 action=1 在 0.5s 步长下不再只产生 0.5 的单步速度增量
        self.waypoints = []                                                         # 设置的中途点的坐标
        self.passed_waypoints_mask = np.zeros(16, dtype=bool)                # 15个中间点 + 1个终点
        self.current_yaw_deg = 0.0
        self.last_commanded_yaw_deg = 0.0
        self.yaw_align_speed_on = 0.2
        self.yaw_align_speed_off = 0.1
        self.max_yaw_step_deg = 20.0
        self.depth_norm_max = 20.0
        self.lidar_norm_max = 20.0
        self.position_norm_max = 80.0
        self.velocity_norm_max = 10.0
        self.angular_acc_norm_max = 10.0
        self.vertical_distance_norm_max = 40.0


    def _normalize_depth_obs(self, depth_tensor):
        return torch.clamp(depth_tensor / self.depth_norm_max, 0.0, 1.0)

    def _normalize_lidar_obs(self, lidar_tensor):
        return torch.clamp(lidar_tensor / self.lidar_norm_max, 0.0, 1.0)

    def _normalize_kinematics_obs(self, kin_data):
        kin = kin_data.astype(np.float32).copy()
        kin[0:3] = np.clip(kin[0:3] / self.position_norm_max, -1.0, 1.0)
        kin[3] = np.clip(kin[3] / self.position_norm_max, 0.0, 1.0)
        kin[4:7] = np.clip(kin[4:7] / self.velocity_norm_max, -1.0, 1.0)
        kin[7] = np.clip(kin[7] / self.angular_acc_norm_max, 0.0, 1.0)
        kin[8:10] = np.clip(kin[8:10] / self.vertical_distance_norm_max, 0.0, 1.0)
        return kin


    # 深度图下采样函数
    def downsample_depth(self, depth_img):
        """
        根据论文公式(1)，对 144x256 的深度图进行 16x16 的下采样，取最小值。
        """
        return downsample_depth_minpool(depth_img)


    # 激光雷达数据获取与处理
    def _get_lidar_data(self):
        # 尝试获取雷达数据，兼容不同的命名 (script.py 用 LLidar1, settings.json 用 Lidar1)
        lidar_data = None
        try:
            lidar_data = self.client.getLidarData(lidar_name="LLidar1", vehicle_name=self.vehicle_name)
        except Exception:
            pass

        if lidar_data is None or len(lidar_data.point_cloud) < 3:
            try:
                lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=self.vehicle_name)
            except Exception:
                pass

        if lidar_data is None or len(lidar_data.point_cloud) < 3:
            # 如果依然失败或点云为空，返回默认的最大距离 (例如 20.0)
            return np.ones(360, dtype=np.float32) * 20.0

        return lidar_points_to_360(lidar_data.point_cloud)


    # 激光雷达下采样函数
    def downsample_lidar(self, lidar_data):
        """
        根据论文 Fig.4 描述的极坐标下采样：
        - 前方 (45°-134°): 每 2° 采样一次 (45个点)
        - 前侧方 (0°-44°, 135°-179°): 每 3° 采样一次 (30个点)
        - 后方 (180°-359°): 每 6° 采样一次 (30个点)
        总计: 45 + 30 + 30 = 105 个点

        注意：此处的 LiDAR 配置为水平 360° 全景雷达 (-180° 到 180°)，
        返回的 lidar_data 是经过 360 个角度 bin 聚合后的最小距离数组。
        """
        return downsample_lidar_105(lidar_data)



    @staticmethod
    def _wrap_angle_deg(angle_deg):
        return ((angle_deg + 180.0) % 360.0) - 180.0

    def _compute_soft_aligned_yaw_deg(self, target_vel):
        # yaw 对齐的是积分后的期望水平速度方向，而不是 raw action 本身。
        speed_xy = float(np.linalg.norm(target_vel[:2]))
        commanded_yaw = float(self.last_commanded_yaw_deg)

        if speed_xy >= self.yaw_align_speed_on:
            desired_yaw_deg = math.degrees(math.atan2(float(target_vel[1]), float(target_vel[0])))
        elif speed_xy <= self.yaw_align_speed_off:
            desired_yaw_deg = commanded_yaw
        else:
            desired_yaw_deg = commanded_yaw

        yaw_delta = self._wrap_angle_deg(desired_yaw_deg - float(self.current_yaw_deg))
        yaw_delta = float(np.clip(yaw_delta, -self.max_yaw_step_deg, self.max_yaw_step_deg))
        commanded_yaw = self._wrap_angle_deg(float(self.current_yaw_deg) + yaw_delta)
        self.last_commanded_yaw_deg = commanded_yaw
        return commanded_yaw


    # 获取环境需要的信息的函数，包括状态向量和一些判别信息
    def _get_obs(self):

        # 获取深度图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ], vehicle_name=self.vehicle_name)
        # 成功获取到深度图
        if responses and responses[0].width > 0:
            raw_depth = decode_depth_planar(responses[0])
            # 将获取到的深度图通过下采样变维到 256*144 大小
            raw_depth = resize_depth_for_ldtd3(raw_depth)
        # 没获取到深度图，基于一个默认值
        else:
            raw_depth = np.ones((144, 256), dtype=np.float32) * 20.0


        # 获取激光雷达数据
        raw_lidar = self._get_lidar_data()


        # 获取运动学状态
        # 状态向量
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        # 无人机的当前的绝对位置
        pos = state.kinematics_estimated.position
        # 无人机当前的朝向
        orient = state.kinematics_estimated.orientation
        _, _, yaw_rad = airsim.to_eularian_angles(orient)
        self.current_yaw_deg = self._wrap_angle_deg(math.degrees(yaw_rad))
        # 无人机当前的线速度
        vel = state.kinematics_estimated.linear_velocity
        # 无人机当前的角加速度
        acc_ang = state.kinematics_estimated.angular_acceleration

        # 计算 kinematics 向量 (10维)
        # 无人机的绝对位置，转换成np数组
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        # 无人机和当前目的地的相对位置，此处为从无人机指向目的地
        rel_pos = self.current_target - drone_pos
        # 相对距离
        dis2goal = np.linalg.norm(rel_pos)
        # 无人机当前的线速度
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        # 无人机当前的角加速度大小
        angular_acc_mag = np.linalg.norm([acc_ang.x_val, acc_ang.y_val, acc_ang.z_val])
        # 尝试获取上下距离传感器数据
        try:
            # 优先尝试获取底部距离传感器数据 (名称需与 settings.json 对应)
            bottom_sensor = self.client.getDistanceSensorData(distance_sensor_name="DistanceBottom", vehicle_name=self.vehicle_name)
            # 下距离
            dis_z_bottom = bottom_sensor.distance
        except Exception:
                dis_z_bottom = float(abs(pos.z_val))
        try:
            # 至于上方的距离，同样优先尝试顶部距离传感器
            top_sensor = self.client.getDistanceSensorData(distance_sensor_name="DistanceTop", vehicle_name=self.vehicle_name)
            # 上距离
            dis_z_top = top_sensor.distance
        except Exception:
            dis_z_top = float(abs(10.0 - abs(pos.z_val)))
        # 组装状态向量（10维）
        kin_data = np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],
            dis2goal,
            velocity[0], velocity[1], velocity[2],
            angular_acc_mag,
            dis_z_bottom,
            dis_z_top
        ], dtype=np.float32)
        # 保存原始物理量，step/reset 中的奖励、终止和位置推导都必须使用未归一化数据。
        self.current_rel_pos = rel_pos.astype(np.float32)
        self.current_dis2goal = float(dis2goal)
        self.current_velocity = velocity
        self.current_angular_acc = float(angular_acc_mag)
        self.current_dis_z_bottom = float(dis_z_bottom)
        self.current_dis_z_top = float(dis_z_top)

        depth_obs = self._normalize_depth_obs(self.downsample_depth(raw_depth)).squeeze(0).cpu().numpy()
        lidar_obs = self._normalize_lidar_obs(self.downsample_lidar(raw_lidar)).cpu().numpy()
        kin_obs = self._normalize_kinematics_obs(kin_data)

        return {
            "depth": depth_obs,
            "lidar": lidar_obs,
            "kinematics": kin_obs
        }

    def step(self, action):
        self.step_count += 1

        # 动作先映射为加速度命令，再在一个 RL 控制周期内积分成末端目标速度。
        commanded_accel = action_to_acceleration(action, accel_scale=self.accel_scale)
        target_vel = integrate_velocity_with_acceleration(
            self.current_velocity,
            commanded_accel,
            dt=self.dt,
            max_v=10.0,
        )


        # AirSim 边界仍复用速度接口，但这里的 target_vel 已经来自加速度语义积分。
        yaw_mode = airsim.YawMode(
            is_rate=False,
            yaw_or_rate=float(self._compute_soft_aligned_yaw_deg(target_vel)),
        )

        # 取消 join() 阻塞，直接发送指令。
        self.client.moveByVelocityAsync(
            float(target_vel[0]),
            float(target_vel[1]),
            float(target_vel[2]),
            float(self.dt),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        )

        # 强制补充发送一个 ping 或者心跳包，告诉 AirSim 我们还在，不要进入安全悬停模式
        try:
            self.client.ping()
        except:
            pass
        # 手动进行短暂休眠来模拟步长间隔，这能让物理引擎运转且不会阻塞 API
        time.sleep(self.dt)
        # [统一调用] 在动作执行完后，统一调用一次 _get_obs() 获取所有最新数据
        obs = self._get_obs()


        # 获取动作执行后的无人机信息。这里必须使用 _get_obs 缓存的原始物理量，而不是归一化后的网络观测。
        rel_pos = self.current_rel_pos.copy()
        dis2goal = self.current_dis2goal
        velocity = self.current_velocity.copy()
        angular_acc = self.current_angular_acc
        dis_z_bottom = self.current_dis_z_bottom
        dis_z_top = self.current_dis_z_top
        progress = self.last_dis2goal - dis2goal
        self.last_dis2goal = dis2goal  # 更新记忆
        # 在 _get_obs() 中: rel_pos = target - drone_pos
        # 逆向推导无人机位置，省去 getMultirotorState 调用
        drone_pos = self.current_target - rel_pos


        # 检查航点经过逻辑
        passed_waypoint_id = 0
        is_first_arrival = False
        # 检查是否进入了新的航点范围
        for i, wp in enumerate(self.waypoints):
            if not self.passed_waypoints_mask[i]:
                dist_to_wp = np.linalg.norm(drone_pos - wp)
                if dist_to_wp < 2.0: # 航点判定半径 2m
                    self.passed_waypoints_mask[i] = True
                    passed_waypoint_id = i + 1
                    is_first_arrival = True
                    # 也可以选择标记之前的所有航点也为已通过
                    for j in range(i):
                        self.passed_waypoints_mask[j] = True
                    break
        # 执行动作后的速度，用来计算当前运动方向和目标的夹角
        v_magnitude = float(np.linalg.norm(velocity))
        if dis2goal > 1e-5 and v_magnitude > 1e-5:
            cos_theta = np.dot(velocity, rel_pos) / (v_magnitude * dis2goal)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_to_target = np.degrees(np.arccos(cos_theta))
        else:
            angle_to_target = 0.0
        # 是否到达终点位置
        arrived = bool(dis2goal < 1.5)

        if self.start_dist > 1e-5:
            cross_product = np.cross(rel_pos, self.start_rel_pos)
            perpendicular_dist = np.linalg.norm(cross_product) / self.start_dist
            crossed_border = bool(perpendicular_dist > 7.0)
        else:
            perpendicular_dist = 0.0
            crossed_border = False
        # 添加一个飞行限高，当无人机飞出指定高度直接结束回合
        out_of_ceiling = bool(drone_pos[2] < -50.0)
        # 是否发生碰撞
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)

        info = {
            'collision': collision_info.has_collided,
            'arrived': arrived,
            'crossed_border': crossed_border,
            'out_of_ceiling': out_of_ceiling,
            'passed_waypoint': passed_waypoint_id,
            'is_first_arrival': is_first_arrival,
            'dis2goal': dis2goal,
            'progress': progress,  # 将进度传给 reward 计算函数
            'angle_to_target': angle_to_target,
            'dis_z_bottom': dis_z_bottom,
            'dis_z_top': dis_z_top,
            'angular_acc': angular_acc,
            'v_magnitude': v_magnitude,
            'perpendicular_dist': perpendicular_dist,
        }
        # 收尾部分
        reward = self._calculate_reward(info)
        terminated = info['collision'] or info['arrived'] or info['out_of_ceiling'] or info['crossed_border']
        truncated = self.step_count >= self.max_steps
        self._render_dashboard(action, drone_pos, velocity, reward, terminated, truncated, info)


        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 如果 options 中没有指定具体的 target，则从数据集中随机选一个
        options = options or {}
        if "start_pos" in options and "target" in options:
            start_pos = options["start_pos"]
            target_pos = options["target"]
        else:
            selected_pair = random.choice(self.position_pairs)
            start_pos = selected_pair["start"]
            target_pos = selected_pair["target"]

        self.current_start_pos = np.array(start_pos, dtype=np.float32)
        self.current_target = np.array(target_pos, dtype=np.float32)
        # 重置airsim环境
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        # 传送无人机
        # 在 AirSim 中，Z轴负方向是向上。
        # 当设置为 -2.0 时，指的是“相对于出生点向上 2 米”
        start_z = float(start_pos[2]) if float(start_pos[2]) < 0 else -10.0
        self.current_start_pos[2] = start_z
        # 计算起点到终点的二维方向，获取初始偏航角(Yaw)
        direction = self.current_target - self.current_start_pos
        if np.linalg.norm([direction[0], direction[1]]) > 1e-5:
            initial_yaw_rad = math.atan2(direction[1], direction[0])
        else:
            initial_yaw_rad = 0.0
        initial_yaw_deg = math.degrees(initial_yaw_rad)
        self.current_yaw_deg = initial_yaw_deg
        self.last_commanded_yaw_deg = initial_yaw_deg
        pose = airsim.Pose(
            airsim.Vector3r(float(start_pos[0]), float(start_pos[1]), start_z),
            airsim.to_quaternion(0.0, 0.0, initial_yaw_rad),
        )
        # 注意：必须调用 simSetVehiclePose，并等待它生效
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)
        time.sleep(0.5) # 给物理引擎一点时间来应用位置突变
        # 确保每次重置后处于悬停状态
        self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.vehicle_name)
        time.sleep(1.0) # 延长重置等待时间，确保飞机被稳稳托在空中

        # 生成 16 个等间距航点 (15个中间点 + 1个终点)
        self.waypoints = []
        for i in range(1, 17):
            ratio = i / 16.0
            # 确保航点的 Z 轴也是浮点数
            wp = self.current_start_pos + (self.current_target - self.current_start_pos) * ratio
            self.waypoints.append(wp)
        self.passed_waypoints_mask = np.zeros(16, dtype=bool)


        # reset 后不再注入额外初速度，首拍运动完全交给策略动作决定。
        obs = self._get_obs()
        self.start_rel_pos = self.current_rel_pos.copy()

        # 使用目标和起点的绝对距离来计算 start_dist
        # 防止 kin[0:3] 相对坐标原点偏差导致 D 计算过小
        self.start_dist = float(np.linalg.norm(self.current_target - self.current_start_pos))

        if self.start_dist < 1e-5:
            self.start_dist = 1.0
        self.last_dis2goal = self.start_dist  # 记录初始距离

        return obs, {'start': self.current_start_pos, 'target': self.current_target}

    def _calculate_reward(self, info):
        """按照论文公式 (2)-(14) 实现"""
        progress = info.get('progress', 0.0)
        r_progress = 25.0 * progress

        # R_path (公式 8)
        # 接近程度奖励
        r_proximity = 40 * (info.get('passed_waypoint', 0) / 15) if info.get('is_first_arrival', False) else 0
        # 到达奖励
        r_arrive = 500 if info['arrived'] else 0

        d = info['dis2goal']
        D = max(self.start_dist, 1.0)
        # 步数奖励（实际上是距离奖励）
        if d > D:
            r_step = -0.2 * (d / D)
        elif 3 <= d <= D:
            r_step = -0.01 * d
        else:
            r_step = -0.03

        # Direction奖励
        a_abs = abs(info['angle_to_target'])
        if a_abs < 10:
            r_direction = 0.3
        elif 10 <= a_abs < 30:
            r_direction = 0.1
        elif 30 <= a_abs < 45:
            r_direction = -0.2 * (a_abs / 180.0)
        else:
            r_direction = -0.5 * (a_abs / 180.0)

        # 保持原有 crossed_border 终止/惩罚机制，不在这里加入连续偏航越界惩罚。
        r_border = -400 if info['crossed_border'] else 0
        r_path = 1.5 * r_proximity + r_step + r_direction + r_arrive + r_border

        # R_collision (公式 11)
        r_failure = -500 if info['collision'] or info.get('out_of_ceiling', False) else 0
        dis_z_bottom = info['dis_z_bottom']
        dis_z_top = info['dis_z_top']
        # 与地面障碍物距离定义的奖励
        if 0.5 <= dis_z_bottom < 1:
            r_z_bottom = -0.05
        elif 0.3 <= dis_z_bottom < 0.5:
            r_z_bottom = -0.1
        elif dis_z_bottom < 0.3:
            r_z_bottom = -0.2/dis_z_bottom
        else:
            r_z_bottom = 0
        # 与上方障碍物距离定义的奖励
        if 0.5 <= dis_z_top < 1:
            r_z_top = -0.05
        elif 0.3 <= dis_z_top < 0.5:
            r_z_top = -0.1
        elif dis_z_top < 0.3:
            r_z_top = -0.2/dis_z_top
        else:
            r_z_top = 0
        # 总距离奖励
        r_z = r_z_bottom + r_z_top
        r_collision = r_failure + r_z

        # R_stabilization (公式 14)
        # 角加速度稳定性奖励
        r_ang = -0.02 * info['angular_acc']
        # 速度奖励
        v = info['v_magnitude']
        if v < 0.5:
            r_vel = -0.2
        elif 0.5 <= v < 2 or v > 8:
            r_vel = -0.05
        else:
            r_vel = 0
        r_stabilization = r_ang + r_vel

        r_survival = 0.5 if not (info['collision'] or info.get('out_of_ceiling', False)) else 0.0

        return r_progress + r_path + r_collision + r_stabilization + r_survival

    def _render_dashboard(self, action, drone_pos, velocity, reward, terminated, truncated, info):
        """
        在终端实时格式化输出无人机飞行状态仪表盘
        """
        # 实际使用的加速度命令
        actual_acc = action_to_acceleration(action)

        # 使用 \r 回到行首覆盖输出，flush=True 强制刷新缓冲区
        print(f"\r[Step {self.step_count:4d}] "
              f"Target: ({self.current_target[0]:5.1f}, {self.current_target[1]:5.1f}, {self.current_target[2]:5.1f}) | "
              f"Pos: ({drone_pos[0]:5.1f}, {drone_pos[1]:5.1f}, {drone_pos[2]:5.1f}) | "
              f"Vel: ({velocity[0]:5.1f}, {velocity[1]:5.1f}, {velocity[2]:5.1f}) | "
              f"Acc: ({actual_acc[0]:5.1f}, {actual_acc[1]:5.1f}, {actual_acc[2]:5.1f}) | "
              f"Rwd: {reward:6.2f} ", end="", flush=True)

        # 当一个回合结束（撞机、到达终点或超时）时，打印一个换行符，以免下一回合的输出挤在一起
        if terminated or truncated:
            end_reason = "ARRIVED!" if info['arrived'] else ("CRASHED!" if info['collision'] else "TIMEOUT!")
            print(f" -> Episode End: {end_reason}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("=== 开始测试 AirSimUAVEnv 环境与传感器 ===")
    print("=" * 50)
    try:
        env = AirSimUAVEnv()
        print("\n[1] 环境初始化成功！正在重置环境 (Reset)...")
        obs, info = env.reset()

        print("\n[2] 获取到的初始观测 (Observation) 数据:")

        # ------------------------------------------------
        # 1. 测试 Depth 深度图
        # ------------------------------------------------
        depth = obs["depth"]
        print(
            f"  [Depth] 维度: {depth.shape}, Min: {np.min(depth):.2f}, Max: {np.max(depth):.2f}, Mean: {np.mean(depth):.2f}")
        if np.all(depth == 0):
            print("    -> ❌ 警告: 深度图全为 0！可能 AirSim 未能正常渲染深度图通道。")
        elif np.all(depth == 20.0) or np.all(depth == -20.0):
            print(
                "    -> ❌ 警告: 深度图全为极大值！通常是因为未获取到图像，触发了代码里的兜底逻辑。请检查 settings.json 相机配置。")
        else:
            print("    -> ✅ 深度图数据分布看似正常。")

        # ------------------------------------------------
        # 2. 测试 LiDAR 激光雷达
        # ------------------------------------------------
        lidar = obs["lidar"]
        print(
            f"  [LiDAR] 维度: {lidar.shape}, Min: {np.min(lidar):.2f}, Max: {np.max(lidar):.2f}, Mean: {np.mean(lidar):.2f}")
        if np.all(lidar == 0):
            print("    -> ❌ 警告: LiDAR 数据全为 0！可能传感器未开启或未能有效捕获点云。")
        elif np.all(lidar == 20.0):
            print(
                "    -> ⚠️ 警告: LiDAR 数据全为最大量程 20.0！可能无人机周围 20 米内空无一物，或者 LiDAR 接口调用失败触发了兜底。建议在复杂地形中再测一次。")
        else:
            print("    -> ✅ LiDAR 点云数据分布看似正常。")

        # ------------------------------------------------
        # 3. 测试 Kinematics 运动学向量
        # ------------------------------------------------
        kin = obs["kinematics"]
        print(f"  [Kinematics] 维度: {kin.shape}")
        print(f"    - Rel Pos (相对目标位置) : [{kin[0]:.2f}, {kin[1]:.2f}, {kin[2]:.2f}]")
        print(f"    - Dis2Goal (距离终点)  : {kin[3]:.2f}")
        print(f"    - Velocity (当前速度)    : [{kin[4]:.2f}, {kin[5]:.2f}, {kin[6]:.2f}]")
        print(f"    - Angular Acc (角加速度): {kin[7]:.2f}")
        print(f"    - Dis Z Bottom (距地面) : {kin[8]:.2f}")
        print(f"    - Dis Z Top (距顶空)    : {kin[9]:.2f}")
        if np.all(kin == 0):
            print("    -> ❌ 警告: 运动学状态全为 0！API 调用可能出现严重问题。")
        else:
            print("    -> ✅ 运动学特征提取正常。")

        # ------------------------------------------------
        # 4. 执行单步测试
        # ------------------------------------------------
        print("\n[3] 正在执行单步随机动作测试 (Step)...")
        # 模拟模型输出的一个向前、向右的加速度指令
        test_action = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(test_action)

        print(f"\n[4] Step 执行结果概览:")
        print(f"  - 动作 (Action)  : {test_action}")
        print(f"  - 奖励 (Reward)  : {reward:.4f}")
        print(f"  - 死亡 (Terminated) : {terminated}")
        print(f"  - 碰撞 (Collision)  : {step_info.get('collision', False)}")
        print(f"  - 越界 (Out of Ceiling) : {step_info.get('out_of_ceiling', False)}")
        print(f"  - 新的距目标距离 : {step_info['dis2goal']:.2f}")

        print("\n" + "=" * 50)
        print("✅ 测试流程全部顺利执行完毕！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback

        traceback.print_exc()