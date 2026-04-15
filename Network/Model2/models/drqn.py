# drqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, action_dim):
        super(DRQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_img = nn.Linear(32 * 18 * 18, 128)
        self.fc = nn.Linear(128 + 9, 128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.out = nn.Linear(128, action_dim)

    def forward(self, image, state, hx=None):
        """
        image: (B, T, H, W, C)
        state: (B, T, state_dim)
        """

        B, T, H, W, C = image.shape

        image = image.reshape(B * T, H, W, C).permute(0, 3, 1, 2) / 255.0

        img_feat = self.conv(image)
        img_feat = F.relu(self.fc_img(img_feat))

        img_feat = img_feat.view(B, T, -1)

        x = torch.cat([img_feat, state], dim=2)
        x = F.relu(self.fc(x))

        x, hx = self.lstm(x, hx)

        q = self.out(x)  # (B, T, action_dim)

        return q, hx

#dqn
# class DRQN(nn.Module):
#     def __init__(self, action_dim):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 5, stride=2),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         self.fc_img = nn.Linear(32 * 18 * 18, 128)

#         self.fc = nn.Sequential(
#             nn.Linear(128 + 9, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_dim)
#         )

#     def forward(self, image, state):
#         image = image.permute(0, 3, 1, 2) / 255.0

#         img_feat = self.conv(image)
#         img_feat = self.fc_img(img_feat)

#         x = torch.cat([img_feat, state], dim=1)

#         q = self.fc(x)

#         return q