import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import airsim
import time
import cv2
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from deep_network import LDTED3FeatureExtractor

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

        # 动作空间: 3维连续加速度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 状态空间: 改为 Dict 字典形式！
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 9, 16), dtype=np.float32),
            "lidar": spaces.Box(low=-np.inf, high=np.inf, shape=(105,), dtype=np.float32),
            "kinematics": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        })

        # 实例化airsim
        self.client = airsim.MultirotorClient() # 实例化客户端
        self.client.confirmConnection()         # 阻塞等待，直到连上虚幻引擎
        
        # 自动检测车辆名称，兼容不同的 settings.json 配置
        vehicles = self.client.listVehicles()
        self.vehicle_name = vehicles[0] if vehicles else ""
        print(f"检测到无人机: '{self.vehicle_name}'")

        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)      # 获取无人机的 API 控制权
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)             # 解锁无人机电机 (Ready to fly)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()         # 自动起飞，脱离地面
        
        # 禁用悬停安全机制，防止 "API call was not received" 频繁出现并中断动作
        # 在某些版本的 AirSim 中，该方法叫 setApiControlTimeout 或类似名称，若不支持可通过捕捉异常跳过
        try:
            self.client.client.call('setApiControlTimeout', 0.0, self.vehicle_name)
        except Exception:
            # 如果不支持该 RPC 调用，则在每次 action 后立即重申控制权
            pass

        self.step_count = 0                                                         # 统计当前回合走过的步数
        self.max_steps = 150                                                        # 允许的最大步数
        self.current_target = np.array([20.0, 0.0, -2.0], dtype=np.float32)        # 默认目标
        self.current_start_pos = np.array([0.0, 0.0, -2.0], dtype=np.float32)       # 默认起点
        self.start_rel_pos = self.current_target - self.current_start_pos           # 开始的相对位置
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))                 # 起点与终点的距离

        self.last_velocity = np.zeros(3, dtype=np.float32)
        self.dt = 0.1 # 步长时间
        self.waypoints = []
        self.current_wp_idx = 0
        self.passed_waypoints_mask = np.zeros(16, dtype=bool) # 15个中间点 + 1个终点

    # 深度图下采样函数
    def downsample_depth(self, depth_img):
        """
        根据论文公式(1)，对 144x256 的深度图进行 16x16 的下采样，取最小值。
        """
        # 使用 PyTorch 的 max_pool2d 实现 min_pooling (取负数再取最大)
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
        downsampled = -F.max_pool2d(-depth_tensor, kernel_size=16, stride=16)
        return downsampled  # 输出尺寸 [1, 1, 9, 16]

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
            return np.ones(360) * 20.0
        
        # 将点云转换为距离数据 (极坐标)
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # 计算每个点到无人机的距离
        dists = np.linalg.norm(points, axis=1)
        # 计算每个点的水平角度 (0-360)
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        angles = (angles + 360) % 360
        
        # 简单处理：将360度划分为360个bin，取每个bin内的最小值
        lidar_360 = np.ones(360) * 20.0 # 默认最大距离20m
        for i in range(len(dists)):
            angle_idx = int(angles[i]) % 360
            if dists[i] < lidar_360[angle_idx]:
                lidar_360[angle_idx] = dists[i]
        return lidar_360

    # 激光雷达下采样函数
    def downsample_lidar(self, lidar_data):
        """
        根据论文 Fig.4 描述的极坐标下采样：
        - 前方 (45°-134°): 每 2° 采样一次 (45个点)
        - 前侧方 (0°-44°, 135°-179°): 每 3° 采样一次 (30个点)
        - 后方 (180°-359°): 每 6° 采样一次 (30个点)
        总计: 45 + 30 + 30 = 105 个点
        """
        indices = []
        # 前方
        indices.extend(range(45, 135, 2))
        # 前侧方
        indices.extend(range(0, 45, 3))
        indices.extend(range(135, 180, 3))
        # 后方
        indices.extend(range(180, 360, 6))

        downsampled = lidar_data[indices]
        return torch.from_numpy(downsampled).float()

    def _get_obs(self):
        # 1. 获取深度图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ], vehicle_name=self.vehicle_name)
        if responses and responses[0].width > 0:
            raw_depth = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # Resize to 144x256 as required by downsample_depth
            raw_depth = cv2.resize(raw_depth, (256, 144))
        else:
            raw_depth = np.ones((144, 256)) * 20.0

        # 2. 获取激光雷达数据
        raw_lidar = self._get_lidar_data()

        # 3. 获取运动学状态
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        acc_ang = state.kinematics_estimated.angular_acceleration
        
        # 计算 kinematics 向量 (10维)
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        rel_pos = self.current_target - drone_pos
        dis2goal = np.linalg.norm(rel_pos)
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        angular_acc_mag = np.linalg.norm([acc_ang.x_val, acc_ang.y_val, acc_ang.z_val])
        
        # 使用射线探测上下距离
        # simTestLineOfSight 检查是否被遮挡，我们更想要距离。
        # 这里改用 simGetRayLength 或者简单使用 lidar 数据中垂直分量
        # 简化处理：假设环境高度限制，或者通过位置估算。
        # 为了更准确，我们可以发送两条垂直的射线（如果 AirSim 支持）
        # 实际上 AirSim 的 Lidar1 已经有 ±15 度的垂直 FOV，可以覆盖一部分。
        # 这里暂且使用位置估算 (AirSim 中 Z 轴负向向上)
        dis_z_bottom = float(abs(pos.z_val)) # 假设地面在 Z=0
        dis_z_top = float(abs(-10.0 - pos.z_val)) # 假设天花板在 Z=-10

        kin_data = np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],
            dis2goal,
            velocity[0], velocity[1], velocity[2],
            angular_acc_mag,
            dis_z_bottom,
            dis_z_top
        ], dtype=np.float32)

        return {
            "depth": self.downsample_depth(raw_depth).squeeze(0).cpu().numpy(),
            "lidar": self.downsample_lidar(raw_lidar).cpu().numpy(),
            "kinematics": kin_data
        }

    def step(self, action):
        self.step_count += 1
        
        # 将动作 (加速度) 转换为速度指令
        # action 是 [-1, 1] 之间的 3 维向量
        accel = action * 2.0 # 放大加速度范围
        # 1. 运动控制：利用上一步缓存的速度进行矢量计算，避免多余的 API 请求
        # action 是 [-1, 1] 之间的 3 维向量
        target_vel = self.last_velocity + (action * 2.0) * self.dt

        # state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        # curr_vel = state.kinematics_estimated.linear_velocity
        # target_vel = airsim.Vector3r(
        #     curr_vel.x_val + accel[0] * self.dt,
        #     curr_vel.y_val + accel[1] * self.dt,
        #     curr_vel.z_val + accel[2] * self.dt
        # )
        
        # 限制最大速度 (保持在合理范围内)
        max_v = 10.0
        # v_mag = np.linalg.norm([target_vel.x_val, target_vel.y_val, target_vel.z_val])
        # if v_mag > max_v:
        #     target_vel.x_val = (target_vel.x_val / v_mag) * max_v
        #     target_vel.y_val = (target_vel.y_val / v_mag) * max_v
        #     target_vel.z_val = (target_vel.z_val / v_mag) * max_v
        # 限制最大速度 (向量化操作，避免繁琐的拆解)
        v_mag = np.linalg.norm(target_vel)
        if v_mag > max_v:
            target_vel = (target_vel / v_mag) * 10.0

        # 取消 join() 阻塞，直接发送指令。
        # join() 会等待动画/动作执行完（如果 dt 较长），
        # 移除它可以让通信更紧凑，避免 "API call was not received" 的误报。
        self.client.moveByVelocityAsync(
            float(target_vel.x_val),
            float(target_vel.y_val),
            float(target_vel.z_val),
            float(self.dt),
            vehicle_name=self.vehicle_name
        )
        
        # 强制补充发送一个 ping 或者心跳包，告诉 AirSim 我们还在，不要进入安全悬停模式
        try:
            self.client.ping()
        except:
            pass
            
        # 手动进行短暂休眠来模拟步长间隔，这能让物理引擎运转且不会阻塞 API
        time.sleep(self.dt)

        obs = self._get_obs()
        kin = obs["kinematics"]
        
        # 获取无人机位置
        # state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        # drone_pos = np.array([state.kinematics_estimated.position.x_val,
        #                       state.kinematics_estimated.position.y_val,
        #                       state.kinematics_estimated.position.z_val])
        # 从 kinematics 向量中提取数据
        rel_pos = kin[0:3]
        dis2goal = kin[3]
        velocity = kin[4:7]
        angular_acc = kin[7]
        dis_z_bottom = kin[8]
        dis_z_top = kin[9]

        # 缓存当前速度供下一步使用
        self.last_velocity = velocity
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

        v_magnitude = float(np.linalg.norm(velocity))
        
        if dis2goal > 1e-5 and v_magnitude > 1e-5:
            cos_theta = np.dot(velocity, rel_pos) / (v_magnitude * dis2goal)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_to_target = np.degrees(np.arccos(cos_theta))
        else:
            angle_to_target = 0.0
            
        arrived = bool(dis2goal < 1.5)
        
        if self.start_dist > 1e-5:
            cross_product = np.cross(rel_pos, self.start_rel_pos)
            perpendicular_dist = np.linalg.norm(cross_product) / self.start_dist
            crossed_border = bool(perpendicular_dist > 7.0)
        else:
            crossed_border = False

        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        
        info = {
            'collision': collision_info.has_collided,
            'arrived': arrived,
            'crossed_border': crossed_border,
            'passed_waypoint': passed_waypoint_id,
            'is_first_arrival': is_first_arrival,
            'dis2goal': dis2goal,
            'angle_to_target': angle_to_target,
            'dis_z_bottom': dis_z_bottom,
            'dis_z_top': dis_z_top,
            'angular_acc': angular_acc,
            'v_magnitude': v_magnitude
        }

        reward = self._calculate_reward(info)

        terminated = info['collision'] or info['arrived']
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        options = options or {}
        # 默认起点和终点，如果在 options 中没提供
        start_pos = options.get("start_pos", [0.0, 0.0, -2.0])
        target_pos = options.get("target", [20.0, 0.0, -2.0])
        
        self.current_start_pos = np.array(start_pos, dtype=np.float32)
        self.current_target = np.array(target_pos, dtype=np.float32)

        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        # 传送无人机
        # 注意：在 AirSim 中，Z轴负方向是向上。
        # 当设置为 -2.0 时，指的是“相对于出生点向上 2 米”
        start_z = float(start_pos[2]) if float(start_pos[2]) < 0 else -2.0
        pose = airsim.Pose(
            airsim.Vector3r(float(start_pos[0]), float(start_pos[1]), start_z), 
            airsim.Quaternionr(0, 0, 0, 1)
        )
        # 注意：必须调用 simSetVehiclePose，并等待它生效
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)
        time.sleep(0.5) # 给物理引擎一点时间来应用位置突变
        
        # 确保每次重置后处于悬停状态，而不是受重力坠落
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

        obs = self._get_obs()
        kin = obs["kinematics"]
        self.start_rel_pos = kin[0:3].copy()
        
        # 使用目标和起点的绝对距离来计算 start_dist
        # 防止 kin[0:3] 相对坐标原点偏差导致 D 计算过小
        self.start_dist = float(np.linalg.norm(self.current_target - self.current_start_pos))
        
        if self.start_dist < 1e-5:
            self.start_dist = 1.0

        return obs, {'start': self.current_start_pos, 'target': self.current_target}

    def _calculate_reward(self, info):
        """按照论文公式 (2)-(14) 实现"""

        # --- R_path (公式 8) ---
        # 接近程度奖励
        r_proximity = 40 * (info.get('passed_waypoint', 0) / 15) if info.get('is_first_arrival', False) else 0
        # 到达奖励
        r_arrive = 500 if info['arrived'] else 0
        # 步数奖励
        d = info['dis2goal']
        D = self.start_dist
        if d > D:
            r_step = -0.2 * (d / D)
        elif 3 <= d <= D:
            r_step = -0.01 * d
        else:
            r_step = -0.03
        # Direction奖励
        a_abs = abs(info['angle_to_target'])
        if a_abs < 10:
            r_direction = 0.05
        elif 10 <= a_abs < 30:
            r_direction = -0.2 * (a_abs / 180.0)
        elif 30 <= a_abs < 45:
            r_direction = -0.3 * (a_abs / 180.0)
        else:
            r_direction = -0.5 * (a_abs / 180.0)
        # border奖励
        r_border = -400 if info['crossed_border'] else 0
        r_path = 1.5 * r_proximity + r_step + r_direction + r_arrive + r_border

        # --- R_collision (公式 11) ---
        r_failure = -500 if info['collision'] else 0
        dis_z_bottom = info['dis_z_bottom']  # 离地面障碍物距离
        dis_z_top = info['dis_z_top']       # 距离上方障碍物的距离
        # 与地面障碍物距离定义的奖励
        if 0.5 <= dis_z_bottom < 1:
            r_z_bottom = -0.05
        elif 0.3 <= dis_z_bottom < 0.5:
            r_z_bottom = -0.1
        elif dis_z_bottom < 0.3:
            r_z_bottom = -0.2
        else:
            r_z_bottom = 0
        # 与上方障碍物距离定义的奖励
        if 0.5 <= dis_z_top < 1:
            r_z_top = -0.05
        elif 0.3 <= dis_z_top < 0.5:
            r_z_top = -0.1
        elif dis_z_top < 0.3:
            r_z_top = -0.2
        else:
            r_z_top = 0
        # 总距离奖励
        r_z = r_z_bottom + r_z_top
        r_collision = r_failure + r_z

        # --- R_stabilization (公式 14) ---
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

        return r_path + r_collision + r_stabilization