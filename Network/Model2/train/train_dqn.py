# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import os

# from uav_env.uav_nav_env import UAVNavEnv
# from models.drqn import DRQN   # ⚠️ 你要把这个文件换成 DQN版本（之前我给你的）
# from train.utils import ReplayBuffer
# import train.config as cfg

# env = UAVNavEnv()

# action_dim = env.action_space.n
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# policy_net = DRQN(action_dim).to(device)
# target_net = DRQN(action_dim).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.Adam(policy_net.parameters(), lr=cfg.LR)
# criterion = nn.MSELoss()

# buffer = ReplayBuffer()

# epsilon = cfg.EPS_START

# print("Start DQN training...")

# for ep in range(cfg.EPISODES):
#     obs = env.reset()
#     image = obs["image"]
#     state = obs["state"]

#     done = False
#     total_reward = 0

#     while not done:
#         # ε-greedy
#         if random.random() < epsilon:
#             action = random.randint(0, action_dim - 1)
#         else:
#             with torch.no_grad():
#                 img_t = torch.FloatTensor(image).unsqueeze(0).to(device)
#                 state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

#                 q_values = policy_net(img_t, state_t)
#                 action = q_values.argmax().item()

#         # 执行动作
#         next_obs, reward, done, info = env.step(action)
#         # ===== 碰撞检测 =====
#         collision_occurred = info.get('collision', False)
#         consecutive_collision_count = 0
#         COLLISION_THRESHOLD = 100
#         COLLISION_PENALTY = -1000000.0
#         if collision_occurred:
#             consecutive_collision_count += 1
#         else:
#             consecutive_collision_count = 0

#         # ===== 超阈值处理（🔥核心）=====
#         if consecutive_collision_count >= COLLISION_THRESHOLD:
#             print(f"🔥 Episode {ep} terminated due to {COLLISION_THRESHOLD} consecutive collisions!")

#             reward += COLLISION_PENALTY
#             done = True
#         next_image = next_obs["image"]
#         next_state = next_obs["state"]

#         # ✅ 存一条 transition（不是 episode）
#         buffer.push(image, state, action, reward, next_image, next_state, done)

#         image = next_image
#         state = next_state
#         total_reward += reward

#         # ======================
#         # 🔥 DQN训练
#         # ======================
#         if len(buffer) > cfg.BATCH_SIZE:
#             images, states, actions, rewards, next_images, next_states, dones = buffer.sample(cfg.BATCH_SIZE)

#             images = torch.FloatTensor(images).to(device)
#             states = torch.FloatTensor(states).to(device)
#             next_images = torch.FloatTensor(next_images).to(device)
#             next_states = torch.FloatTensor(next_states).to(device)

#             actions = torch.LongTensor(actions).to(device)
#             rewards = torch.FloatTensor(rewards).to(device)
#             dones = torch.FloatTensor(dones).to(device)

#             # 当前Q
#             q_values = policy_net(images, states)
#             q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

#             # 目标Q
#             with torch.no_grad():
#                 next_q_values = target_net(next_images, next_states)
#                 next_q_value = next_q_values.max(1)[0]

#             target = rewards + cfg.GAMMA * next_q_value * (1 - dones)

#             loss = criterion(q_value, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     # ε衰减
#     epsilon = max(cfg.EPS_END, epsilon * cfg.EPS_DECAY)

#     # target更新
#     if ep % cfg.TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

#     print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

#     # ======================
#     # 💾 保存模型
#     # ======================
#     if ep % 20 == 0:
#         os.makedirs("models/saved_models", exist_ok=True)
#         torch.save(policy_net.state_dict(), f"models/saved_models/dqn_ep{ep}.pth")

# train_drqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from uav_env.uav_nav_env import UAVNavEnv
from models.drqn import DRQN
from train.utils import ReplayBuffer
import train.config as cfg

env = UAVNavEnv()
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DRQN(action_dim).to(device)
target_net = DRQN(action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=cfg.LR)
criterion = nn.MSELoss()
buffer = ReplayBuffer()
epsilon = cfg.EPS_START
SEQ_LEN = 8  # 序列长度

print("Start DRQN training...")

for ep in range(cfg.EPISODES):
    obs = env.reset()
    image = obs["image"]
    state = obs["state"]
    done = False
    total_reward = 0
    episode_data = []

    # 连续碰撞计数器
    consecutive_collision_count = 0
    COLLISION_THRESHOLD = 100
    COLLISION_PENALTY = -1000000.0

    while not done:
        # ----- 动作选择 -----
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)  # ✅ 分支1：随机动作
        else:
            with torch.no_grad():
                img_t = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
                state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
                q_values, _ = policy_net(img_t, state_t)
                action = q_values[:, -1].argmax().item()  # ✅ 分支2：策略动作

        # ----- 执行动作 -----
        next_obs, reward, done, info = env.step(action)

        # ----- 碰撞检测（从 info 获取）-----
        collision_occurred = info.get('collision', False)

        if collision_occurred:
            consecutive_collision_count += 1
        else:
            consecutive_collision_count = 0

        # 若连续碰撞达到阈值，注入巨大负奖励并强制结束 episode
        if consecutive_collision_count >= COLLISION_THRESHOLD:
            reward += COLLISION_PENALTY
            done = True
            print(f"Episode {ep} terminated early due to {COLLISION_THRESHOLD} consecutive collisions!")

        next_image = next_obs["image"]
        next_state = next_obs["state"]

        # 存储一步经验
        episode_data.append((image, state, action, reward, next_image, next_state, done))

        image = next_image
        state = next_state
        total_reward += reward

        # ----- 训练（当经验池有足够数据时）-----
        if len(buffer) > 10:
            images, states, actions, rewards, next_images, next_states, dones = buffer.sample_sequence(
                cfg.BATCH_SIZE, SEQ_LEN
            )
            images = torch.FloatTensor(images).to(device)
            states = torch.FloatTensor(states).to(device)
            next_images = torch.FloatTensor(next_images).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # 当前 Q 值
            q_values, _ = policy_net(images, states)
            q_last = q_values[:, -1, :]
            q_value = q_last.gather(1, actions[:, -1].unsqueeze(1)).squeeze(1)

            # 目标 Q 值
            with torch.no_grad():
                next_q_values, _ = target_net(next_images, next_states)
                next_q_last = next_q_values[:, -1, :]
                next_q_value = next_q_last.max(1)[0]
                target = rewards[:, -1] + cfg.GAMMA * next_q_value * (1 - dones[:, -1])

            loss = criterion(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 将整个 episode 存入经验池
    buffer.push_episode(episode_data)

    # 更新 epsilon
    epsilon = max(cfg.EPS_END, epsilon * cfg.EPS_DECAY)

    # 更新目标网络
    if ep % cfg.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # 定期保存模型
    if ep % 10 == 0:
        os.makedirs("models/saved_models", exist_ok=True)
        torch.save(policy_net.state_dict(), f"models/saved_models/drqn_ep{ep}.pth")