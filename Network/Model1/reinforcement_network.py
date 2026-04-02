import gymnasium as gym
from gymnasium import spaces
import numpy as np
from deep_network import LDTED3FeatureExtractor
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义动作噪声
n_actions = 3 # [ax, ay, az]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# 配置网络架构
# 决策和评价网络均为一系列全连接层组成的网络
# 架构: 512 -> 512 -> 512 -> 512 -> 256 -> 128
policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 512, 512, 512, 256, 128],  # Actor 网络
        qf=[512, 512, 512, 512, 256, 128]   # Critic 网络，其中这套参数同时用于两个网络
    ),
    activation_fn=torch.nn.ReLU
)

class AirSimUAVEnv(gym.Env):
    def __init__(self):
        super(AirSimUAVEnv, self).__init__()
        # 定义动作空间：连续的 3维加速度 [ax, ay, az]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 定义观测空间：对应论文中的 74 维 state vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32)

        # 封装特征提取器
        self.extrator = LDTED3FeatureExtractor()

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass