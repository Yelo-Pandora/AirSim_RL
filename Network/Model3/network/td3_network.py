"""
TD3 feature extractor for EGO-Planner environment.
Multi-modal fusion of depth, LiDAR, kinematics, and planner info.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class EGOFeatureExtractor(nn.Module):
    """
    Multi-modal feature extractor for EGO-Planner observations.

    Processes:
    - Depth image (1, 9, 16) -> 32-dim via 3 Conv2D + FC
    - LiDAR (105,) -> 32-dim via FC layers
    - Kinematics + planner info (14,) -> passed through directly

    Output: 78-dim fused feature vector
    """

    def __init__(self):
        super().__init__()

        # Depth branch: (1, 9, 16) -> 32
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # After conv: 32 * 9 * 16 = 4608
        self.depth_fc = nn.Sequential(
            nn.Linear(32 * 9 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # LiDAR branch: (105,) -> 32
        self.lidar_fc = nn.Sequential(
            nn.Linear(105, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
        )

    def forward(self, depth: torch.Tensor, lidar: torch.Tensor,
                kinematics: torch.Tensor, planner_info: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all branches.

        Args:
            depth: (B, 1, 9, 16)
            lidar: (B, 105)
            kinematics: (B, 10)
            planner_info: (B, 4)

        Returns:
            (B, 78) fused feature vector
        """
        depth_feat = self.depth_fc(self.depth_conv(depth))
        lidar_feat = self.lidar_fc(lidar)
        kin_feat = torch.cat([kinematics, planner_info], dim=-1)

        return torch.cat([depth_feat, lidar_feat, kin_feat], dim=-1)  # 32+32+14=78


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    SB3 feature extractor wrapper.
    Combines all modalities into a 78-dim feature vector.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=78)
        self.extractor = EGOFeatureExtractor()

    def forward(self, observations) -> torch.Tensor:
        depth = observations["depth"]
        lidar = observations["lidar"]
        kinematics = observations["kinematics"]
        planner_info = observations["planner_info"]
        return self.extractor(depth, lidar, kinematics, planner_info)
