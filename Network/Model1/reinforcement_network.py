import torch
import json
import random
import os
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

        # 加载位置对数据
        self.positions_file = "target.json" # 到时候根据情况
        if os.path.exists(self.positions_file):
            with open(self.positions_file, 'r') as f:
                self.position_pairs = json.load(f)
            print(f"成功加载 {len(self.position_pairs)} 组起点-终点位置对")
        else:
            # 如果文件不存在，使用默认值
            self.position_pairs = [{"start": [0.0, 0.0, -2.0], "target": [20.0, 0.0, -2.0]}]
            print("警告：未找到 positions.json，使用默认位置。")

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
        self.vehicle_name = "Drone1" # 默认值
        try:
            vehicles = self.client.listVehicles()
            self.vehicle_name = vehicles[0] if vehicles else ""
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
            # 如果不支持该 RPC 调用，则在每次 action 后立即重申控制权
            pass

        self.step_count = 0                                                         # 统计当前回合走过的步数
        self.max_steps = 150                                                        # 允许的最大步数
        self.current_target = np.array([20.0, 0.0, -2.0], dtype=np.float32)        # 默认目标
        self.current_start_pos = np.array([0.0, 0.0, -2.0], dtype=np.float32)       # 默认起点
        self.start_rel_pos = self.current_target - self.current_start_pos           # 开始的相对位置
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))                 # 起点与终点的距离

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
        
        注意：此处的 LiDAR 配置为水平 360° 全景雷达 (-180° 到 180°)，
        返回的 lidar_data 是经过 360 个角度 bin 聚合后的最小距离数组。
        """
        # lidar_data 是 360 度的原始距离数据
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
        orient = state.kinematics_estimated.orientation
        vel = state.kinematics_estimated.linear_velocity
        acc_ang = state.kinematics_estimated.angular_acceleration
        
        # 计算 kinematics 向量 (10维)
        drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        rel_pos = self.current_target - drone_pos
        dis2goal = np.linalg.norm(rel_pos)
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        angular_acc_mag = np.linalg.norm([acc_ang.x_val, acc_ang.y_val, acc_ang.z_val])
        
      
        # 在下采样的 105 个点中：
        # lidar[0:45] 是前方 (45°-134°)
        # lidar[45:75] 是前侧方
        # lidar[75:105] 是后方 (180°-359°)
        # 这是一个 360° 水平全景雷达（Vertical FOV ±15°），
        # 它的主要作用是避开四周墙壁，但无法直接测出正上方和正下方的垂直距离。
        
        
        try:
            # 优先尝试获取底部距离传感器数据 (名称需与 settings.json 对应)
            bottom_sensor = self.client.getDistanceSensorData(distance_sensor_name="DistanceBottom", vehicle_name=self.vehicle_name)
            dis_z_bottom = bottom_sensor.distance
        except Exception:
            try:
                # 获取无人机当前的姿态 (Orientation)
                # 在 AirSim 中，NED 坐标系下 Z 轴向下为正，
                # 因此我们在局部坐标系下向正下方发射一条射线 (0, 0, 1)。
                ray_length = 20.0
                
                # 使用 getMultirotorState 中的真实朝向，以保证无人机倾斜时射线也能垂直于无人机机腹
                drone_pose = airsim.Pose(pos, orient) 
                down_ray = self.client.simGetRayCastResult(
                    drone_pose, 
                    airsim.Vector3r(0, 0, 1), 
                    ray_length, 
                    vehicle_name=self.vehicle_name
                )
                dis_z_bottom = down_ray.distance if down_ray.has_hit else ray_length
            except Exception:
                # 如果所有接口调用失败，作为后备方案回退到对地绝对高度
                dis_z_bottom = float(abs(pos.z_val))
        
        # 至于上方的距离，同样优先尝试顶部距离传感器
        try:
            top_sensor = self.client.getDistanceSensorData(distance_sensor_name="DistanceTop", vehicle_name=self.vehicle_name)
            dis_z_top = top_sensor.distance
        except Exception:
            # 虽然是 3D 全景雷达（VerticalFOV 为 ±15°），但它无法扫描到正上方（+90°）的天空区域。
            # 由于是室外场景，实际上并没有“天花板”的物理阻挡，
            # 但为了保证安全飞行空域和避免模型无限制地往高处飞（刷步数或逃避地面障碍），
            # 我们在这里人为设定一个虚拟的“法定最高飞行高度”（比如 10 米）。
            dis_z_top = float(abs(10.0 - abs(pos.z_val)))

        kin_data = np.array([
            rel_pos[0], rel_pos[1], rel_pos[2],
            dis2goal,
            velocity[0], velocity[1], velocity[2],
            angular_acc_mag,
            dis_z_bottom,
            dis_z_top
        ], dtype=np.float32)
        
        # 保存当前绝对坐标和速度供 step 方法使用，减少 API 调用
        self.current_position = drone_pos
        self.current_velocity = velocity

        return {
            "depth": self.downsample_depth(raw_depth).squeeze(0).cpu().numpy(),
            "lidar": self.downsample_lidar(raw_lidar).cpu().numpy(),
            "kinematics": kin_data
        }

    def step(self, action):
        self.step_count += 1
        
        # 将动作 (加速度) 转换为速度指令
        accel = action * 2.0 # 放大加速度范围
        # 1. 运动控制：利用上一步缓存的速度进行矢量计算，避免多余的 API 请求
        # action 是 [-1, 1] 之间的 3 维向量
        target_vel = self.current_velocity + (action * 2.0) * self.dt

        # 限制最大速度 (保持在合理范围内)
        max_v = 10.0
        v_mag = np.linalg.norm(target_vel)
        if v_mag > max_v:
            target_vel = (target_vel / v_mag) * max_v

        # 取消 join() 阻塞，直接发送指令。
        self.client.moveByVelocityAsync(
            float(target_vel[0]),
            float(target_vel[1]),
            float(target_vel[2]),
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

        # [统一调用] 在动作执行完后，统一调用一次 _get_obs() 获取所有最新数据
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

        # 如果 options 中没有指定具体的 start/target，则从列表中随机选一组
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