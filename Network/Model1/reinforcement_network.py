import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
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

        # 实例化airsim，实际使用时用后端定义好的类来初始化
        # self.client = airsim.MultirotorClient() # 实例化客户端
        # self.client.confirmConnection()         # 阻塞等待，直到连上虚幻引擎
        # self.client.enableApiControl(True)      # 获取无人机的 API 控制权
        # self.client.armDisarm(True)             # 解锁无人机电机 (Ready to fly)

        self.step_count = 0                                                         # 统计当前回合走过的步数
        self.max_steps = 150                                                        # 允许的最大步数
        self.current_target = np.random.uniform(low=-1.0, high=1.0, size=(3,))      # 当前回合的目标位置，从后端得到
        self.current_start_pos = np.random.uniform(low=-1.0, high=1.0, size=(3,))   # 当前回合的起点位置，从后端得到
        self.start_rel_pos = self.current_target - self.current_start_pos           # 开始的相对位置
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))                 # 起点与终点的距离

    # 深度图下采样函数
    def downsample_depth(self, depth_img):
        """
        根据论文公式(1)，对 144x256 的深度图进行 16x16 的下采样，取最小值。
        """
        # 使用 PyTorch 的 max_pool2d 实现 min_pooling (取负数再取最大)
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
        downsampled = -F.max_pool2d(-depth_tensor, kernel_size=16, stride=16)
        return downsampled  # 输出尺寸 [1, 1, 9, 16]
    # 激光雷达下采样函数
    def downsample_lidar(self, lidar_data):
        """
        根据论文 Fig.4 描述的极坐标下采样：
        - 前方 (45°-134°): 每 2° 采样一次 (45个点)
        - 前侧方 (0°-44°, 135°-179°): 每 3° 采样一次 (30个点)
        - 后方 (180°-359°): 每 6° 采样一次 (30个点)
        总计: 45 + 30 + 30 = 105 个点
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
        # 模拟获取原始传感器数据，后续替换为 AirSim 接口
        raw_depth = np.random.rand(144, 256)  # 后续替换为诸如 get_img() 的获取深度图的函数
        raw_lidar = np.random.rand(360)  # 后续替换为诸如 get_LiDAR() 的获取激光雷达数据的函数
        kin_data = np.random.rand(10)  # 后续替换为诸如 get_kin_state() 的获取无人机运动状态的函数

        # 返回字典格式的 numpy 数组，送给 SB3
        return {
            "depth": self.downsample_depth(raw_depth).squeeze(0).cpu().numpy(),
            "lidar": self.downsample_lidar(raw_lidar).cpu().numpy(),
            "kinematics": kin_data.astype(np.float32)
        }

    def step(self, action):
        self.step_count += 1
        # apply_action(action) # 将动作发送给 AirSim

        obs = self._get_obs()
        kin = obs["kinematics"]
        rel_pos = kin[0:3]      # 从无人机指向目标点的向量 (如果在全局系则需转换)
        dis2goal = kin[3]       # 无人机和目标的距离
        velocity = kin[4:7]     # 速度向量
        angular_acc = kin[7]    # 角加速度
        dis_z_bottom = kin[8]   # 距底端的距离
        dis_z_top = kin[9]      # 距顶部的距离
        v_magnitude = float(np.linalg.norm(velocity))   # 速度大小
        # 计算速度方向与目标方向的夹角 (Angle to target)
        if dis2goal > 1e-5 and v_magnitude > 1e-5:
            # 这里的 rel_pos 应该是从无人机指向目标的向量。
            cos_theta = np.dot(velocity, rel_pos) / (v_magnitude * dis2goal)
            cos_theta = np.clip(cos_theta, -1.0, 1.0) # 防止浮点误差导致超出 [-1, 1]
            angle_to_target = np.degrees(np.arccos(cos_theta))
        else:
            angle_to_target = 0.0
        # 判断是否到达
        arrived = bool(dis2goal < 1.5)
        # 判断是否越界 (Crossed Border)
        # 使用向量叉乘计算无人机到直线的垂直距离: d = |r x start_r| / |start_r|
        if self.start_dist > 1e-5:
            cross_product = np.cross(rel_pos, self.start_rel_pos)
            perpendicular_dist = np.linalg.norm(cross_product) / self.start_dist
            crossed_border = bool(perpendicular_dist > 5.0)
        else:
            crossed_border = False


        # 模拟获取环境信息，真正训练的时候应当通过后端类获取,例如
        # info = self.client.simGetCollisionInfo()
        # info相较于从状态向量kinmatrics获取的信息，还需额外获取判定是否发生碰撞的collision，是否经过预设点的passed_waypoint和是否首次到达该预设点的is_first_arrival
        info = {
            'collision': False,
            'arrived': arrived,
            'crossed_border': crossed_border,
            'passed_waypoint': 0,  # 可选，如果当前步经过预设点，则指出经过哪个预设点
            'is_first_arrival': False,  # 可选，和passed_waypoints绑定，判定是否第一次经过该预设点
            'dis2goal': dis2goal,
            'angle_to_target': angle_to_target,
            'dis_z_bottom': dis_z_bottom,
            'dis_z_top': dis_z_top,
            'angular_acc': angular_acc,
            'v_magnitude': v_magnitude
        }

        reward = self._calculate_reward(info)  # 调用奖励函数

        terminated = info['collision'] or info['arrived']
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        环境重置函数。
        如果需要指定起点和终点，可以通过 options 字典传入：
        env.reset(options={"start_pos": [0, 0, -5], "target": [50, 20, -5]})
        """
        super().reset(seed=seed)

        # 重置回合（Episode）相关的内部计数器和状态
        self.step_count = 0
        # 用于 Proximity reward 的路点防刷分参数重置，预计从外部获取
        # self.highest_point_reached = 0

        # 解析 options 中的自定义参数 (起点和目标点)
        options = options or {}
        # 假设未传入时，使用预定义数值防止报错
        self.current_start_pos = np.array(options.get("start_pos", [0.0, 0.0, 0.0]))
        self.current_target = np.array(options.get("target", [20.0, 0.0, 0.0]))

        # AirSim 物理重置操作，待后续对接 AirSim API)
        # self.client.reset()
        # self.client.enableApiControl(True)

        # # 将无人机传送到 self.current_start_pos 的位置
        # position = airsim.Vector3r(self.current_start_pos[0], self.current_start_pos[1], self.current_start_pos[2])
        # self.client.simSetVehiclePose(airsim.Pose(position, airsim.to_quaternion(0, 0, 0)), True)

        # 获取环境的初始观测值
        # 来计算 kin[0:3]
        obs = self._get_obs()
        kin = obs["kinematics"]

        # 计算本回合用于奖励判定和越界判定的基准几何数据
        # 初始相对目标向量 (起点指向终点的三维向量)
        self.start_rel_pos = kin[0:3].copy()

        # 初始直线距离 (D)
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))

        # 如果起点和终点过于接近（或重合），强行给一个最小值，避免后续除 0 错误
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