import torch
import torch.nn as nn
import torch.nn.functional as F


class LDTED3FeatureExtractor(nn.Module):
    def __init__(self):
        super(LDTED3FeatureExtractor, self).__init__()

        # 深度图像特征提取
        # 输入尺寸下采样后为 9x16
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 缩小尺寸
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 经过Conv和Flatten后的全连接层，输出 32 维特征
        # 计算经过卷积和池化后的向量维度：9x16 -> 4x8 -> 2x4. 64*2*4 = 512
        self.depth_fc = nn.Linear(512, 32)

        # 点云特征提取分支
        # 下采样后 LiDAR 输入为 105 维向量
        self.lidar_fc = nn.Sequential(
            nn.Linear(105, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # 输出 32 维特征
            nn.Linear(128, 32)
        )

    def forward(self, depth_img, lidar_points, kin_vector):
        """
        Args:
            depth_img: 下采样后的深度图 [batch, 1, 9, 16]
            lidar_points: 下采样后的点云 [batch, 105]
            kin_vector: 10 维运动状态向量 [batch, 10]
        Returns:
            state_vector: 融合后的 74 维状态向量
        """
        # 提取深度图像特征 (32维)
        d_feat = self.depth_conv(depth_img)
        d_feat = F.relu(self.depth_fc(d_feat))

        # 提取LiDAR特征 (32维)
        l_feat = self.lidar_fc(lidar_points)

        # 融合: LiDARFeature(32) + DepthFeature(32) + KinVector(10)
        # 根据 Fig.3，最终 State Vector 维度为 32 + 32 + 10 = 74
        state_vector = torch.cat([l_feat, d_feat, kin_vector], dim=1)

        return state_vector


# 测试
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 1
    mock_depth = torch.randn(batch_size, 1, 9, 16)
    mock_lidar = torch.randn(batch_size, 105)
    # 运动向量包括: pos(3), dis2dest(1), v(3), aa(1), dis_bottom(1), dis_above(1) = 10维
    mock_kin = torch.randn(batch_size, 10)

    # 初始化模型
    model = LDTED3FeatureExtractor()

    # 获取 State Vector
    state_vector = model(mock_depth, mock_lidar, mock_kin)

    print(f"Depth Feature 维度: 32")
    print(f"LiDAR Feature 维度: 32")
    print(f"Kinematic Vector 维度: 10")
    print(f"最终生成的 State Vector 维度: {state_vector.shape[1]}")  # 应该输出 74