# import gym
# from gym import spaces
# import numpy as np
# from airsim_client.airsim_bridge import AirSimBridge
# from .reward import compute_reward
# from .action import ACTION_SPACE
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd

# class UAVNavEnv(gym.Env):
#     def __init__(self):
#         super(UAVNavEnv, self).__init__()
#         self.bridge = AirSimBridge()
#         #读取目标点列表（确保路径正确）
#         self.goal_list = pd.read_csv("uav_env/relative_coordinates_export.csv").values
#         print(f"Loaded {len(self.goal_list)} goals.")

#         self.observation_space = spaces.Dict({
#             "image": spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
#             "state": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
#         })
#         self.action_space = spaces.Discrete(len(ACTION_SPACE))

#     def reset(self):
#         self.bridge.client.reset()
#         self.bridge.client.enableApiControl(True)
#         self.bridge.client.armDisarm(True)
#         self.bridge.takeoff()

#         # 随机选择一个目标点
#         idx = np.random.randint(0, len(self.goal_list))
#         self.goal = self.goal_list[idx]
#         # self.goal = np.array([
#         #     np.random.uniform(-15, 15),
#         #     np.random.uniform(-15, 15),
#         #     -5
#         # ])
#         print(f"[RESET] Selected goal: {self.goal}")
#         self.prev_state = self._get_state()

#         obs = self._get_obs()
#         print(f"[RESET] Goal position: {self.goal}")
#         print(f"[RESET] Initial state: {self.prev_state}")
#         return obs

#     def step(self, action_idx):
#         try:
#             # 1. 执行动作
#             action = ACTION_SPACE[action_idx]
#             self.bridge.move_by_action(action)

#             # 2. 获取状态与观测
#             state = self._get_state()
#             obs = self._get_obs()

#             # 3. 碰撞检测（鲁棒处理）
#             try:
#                 collision = self.bridge.client.simGetCollisionInfo().has_collided
#             except Exception as e:
#                 print("[WARN] Collision fetch failed:", e)
#                 collision = False

#             # 4. 计算奖励
#             seg_img = obs.get("image", None)
#             try:
#                 reward = compute_reward(
#                     state,
#                     self.prev_state,
#                     self.goal,
#                     collision=collision,
#                     seg_img=seg_img
#                 )
#             except Exception as e:
#                 print("[WARN] Reward error:", e)
#                 reward = -1.0

#             self.prev_state = state

#             # 5. 判断是否结束
#             done = self._check_done(state)

#             # 6. 调试打印（可选，训练时可注释以提速）
#             dis2goal = np.linalg.norm(state[:3] - self.goal)
#             arrived = dis2goal < 2
#             print("\n========== STEP DEBUG ==========")
#             print(f"Position: {state[:3]}")
#             print(f"Velocity: {state[3:6]}")
#             print(f"Goal: {self.goal}")
#             print(f"Distance to goal: {dis2goal:.2f}")
#             print(f"Arrived: {arrived}")
#             print(f"Collision: {collision}")
#             print(f"Done: {done}")
#             print(f"Reward: {reward:.3f}")
#             print("================================\n")

#             # ✅ 关键修改：正常返回时带上 collision 信息
#             return obs, reward, done, {'collision': collision}

#         except Exception as e:
#             # 🔥 全局异常捕获，防止训练崩溃
#             print("\n🔥 [ERROR] STEP FAILED:", e)

#             dummy_obs = {
#                 "image": np.zeros((84, 84, 3), dtype=np.uint8),
#                 "state": np.zeros_like(self.prev_state) if hasattr(self, 'prev_state') else np.zeros(12)
#             }
#             reward = -100.0
#             done = True
#             # ✅ 异常返回也提供 info 字段，标记为碰撞（视为严重失败）
#             info = {'collision': True}

#             print("⚠️ Episode terminated due to error\n")
#             return dummy_obs, reward, done, info

#     def _get_state(self):
#         s = self.bridge.get_state()
#         pos = s["position"]
#         vel = s["velocity"]
#         rel_goal = self.goal - pos
#         return np.concatenate([pos, vel, rel_goal])

#     def _check_done(self, state):
#         pos = state[:3]
#         # 到达目标
#         if np.linalg.norm(pos - self.goal) < 2:
#             return True
#         # 高度超出安全范围（地面以上 0 米或低于 -200 米）
#         if pos[2] > 0 or pos[2] < -200:
#             return True
#         return False

#     def _get_obs(self):
#         s = self.bridge.get_state()
#         pos = s["position"]
#         vel = s["velocity"]

#         img = self.bridge.get_segmentation()
#         img = cv2.resize(img, (84, 84))

#         # 只显示一次分割图像用于调试
#         if not hasattr(self, "debug_shown"):
#             plt.imshow(img)
#             plt.title("Segmentation Debug")
#             plt.show()
#             self.debug_shown = True

#         obs_dict = {
#             "image": img,
#             "state": np.concatenate([pos, vel, self.goal - pos])
#         }
#         return obs_dict


# uav_env.py
import gym
from gym import spaces
import numpy as np
from airsim_client.airsim_bridge import AirSimBridge
from .reward import compute_reward
from .action import ACTION_SPACE
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class UAVNavEnv(gym.Env):
    def __init__(self):
        super(UAVNavEnv, self).__init__()
        self.bridge = AirSimBridge()
        # 读取目标点列表（确保路径正确）
        # self.goal_list = pd.read_csv("uav_env/relative_coordinates_export.csv").values
        # print(f"Loaded {len(self.goal_list)} goals.")

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(len(ACTION_SPACE))

    def reset(self):
        self.bridge.client.reset()
        self.bridge.client.enableApiControl(True)
        self.bridge.client.armDisarm(True)
        self.bridge.takeoff()

        # 随机选择一个目标点
        # idx = np.random.randint(0, len(self.goal_list))
        # self.goal = self.goal_list[idx]
        self.goal = np.array([
            np.random.uniform(-15, 15),
            np.random.uniform(-15, 15),
            -5
        ])
        print(f"[RESET] Selected goal: {self.goal}")
        self.prev_state = self._get_state()

        obs = self._get_obs()
        print(f"[RESET] Goal position: {self.goal}")
        print(f"[RESET] Initial state: {self.prev_state}")
        return obs

    def step(self, action_idx):
        try:
            # 1. 执行动作
            action = ACTION_SPACE[action_idx]
            self.bridge.move_by_action(action)

            # 2. 获取状态与观测
            state = self._get_state()
            obs = self._get_obs()

            # 3. 碰撞检测（鲁棒处理）
            try:
                collision = self.bridge.client.simGetCollisionInfo().has_collided
            except Exception as e:
                print("[WARN] Collision fetch failed:", e)
                collision = False

            # 4. 计算奖励
            seg_img = obs.get("image", None)
            try:
                reward = compute_reward(
                    state,
                    self.prev_state,
                    self.goal,
                    collision=collision,
                    seg_img=seg_img
                )
            except Exception as e:
                print("[WARN] Reward error:", e)
                reward = -1.0

            self.prev_state = state

            # 5. 判断是否结束
            done = self._check_done(state)

            # 6. 调试打印（可选，训练时可注释以提速）
            dis2goal = np.linalg.norm(state[:3] - self.goal)
            arrived = dis2goal < 2
            print("\n========== STEP DEBUG ==========")
            print(f"Position: {state[:3]}")
            print(f"Velocity: {state[3:6]}")
            print(f"Goal: {self.goal}")
            print(f"Distance to goal: {dis2goal:.2f}")
            print(f"Arrived: {arrived}")
            print(f"Collision: {collision}")
            print(f"Done: {done}")
            print(f"Reward: {reward:.3f}")
            print("================================\n")

            # ✅ 关键修改：正常返回时带上 collision 信息
            return obs, reward, done, {'collision': collision}

        except Exception as e:
            # 🔥 全局异常捕获，防止训练崩溃
            print("\n🔥 [ERROR] STEP FAILED:", e)

            dummy_obs = {
                "image": np.zeros((84, 84, 3), dtype=np.uint8),
                "state": np.zeros_like(self.prev_state) if hasattr(self, 'prev_state') else np.zeros(12)
            }
            reward = -100.0
            done = True
            # ✅ 异常返回也提供 info 字段，标记为碰撞（视为严重失败）
            info = {'collision': True}

            print("⚠️ Episode terminated due to error\n")
            return dummy_obs, reward, done, info

    def _get_state(self):
        s = self.bridge.get_state()
        pos = s["position"]
        vel = s["velocity"]
        rel_goal = self.goal - pos
        return np.concatenate([pos, vel, rel_goal])

    def _check_done(self, state):
        pos = state[:3]
        # 到达目标
        if np.linalg.norm(pos - self.goal) < 2:
            return True
        # 高度超出安全范围（地面以上 0 米或低于 -200 米）
        if pos[2] > 0 or pos[2] < -200:
            return True
        return False

    def _get_obs(self):
        s = self.bridge.get_state()
        pos = s["position"]
        vel = s["velocity"]

        img = self.bridge.get_segmentation()
        img = cv2.resize(img, (84, 84))

        # 只显示一次分割图像用于调试
        if not hasattr(self, "debug_shown"):
            plt.imshow(img)
            plt.title("Segmentation Debug")
            plt.show()
            self.debug_shown = True

        obs_dict = {
            "image": img,
            "state": np.concatenate([pos, vel, self.goal - pos])
        }
        return obs_dict