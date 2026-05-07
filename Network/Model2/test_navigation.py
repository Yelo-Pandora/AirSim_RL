# test_drqn.py
import torch
import time
import numpy as np
from uav_env.uav_nav_env import UAVNavEnv
from models.drqn import DRQN

MODEL_PATH = "models/saved_models/drqn_ep30.pth"
MAX_STEPS = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UAVNavEnv()
action_dim = env.action_space.n

model = DRQN(action_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded.")

obs = env.reset()
image = obs["image"]  # shape: (84, 84, 3)
state = obs["state"]  # shape: (12,)

done = False
step = 0
total_reward = 0
hx = None
trajectory = []

while not done and step < MAX_STEPS:
    with torch.no_grad():
        # ⭐ 构造符合 (B, T, H, W, C) 的输入
        img_t = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 84, 84, 3]
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # [1, 1, 12]
        img_t = img_t.to(device)
        state_t = state_t.to(device)

        q_values, hx = model(img_t, state_t, hx)  # q_values: [1, 1, action_dim]
        action = q_values[0, -1].argmax().item()  # 取最后一个时间步的Q值

    next_obs, reward, done, info = env.step(action)
    next_image = next_obs["image"]
    next_state = next_obs["state"]

    total_reward += reward
    step += 1

    trajectory.append({
        "step": step,
        "action": action,
        "reward": reward,
        "state": state
    })

    print(f"[Step {step}] Action: {action}, Reward: {reward:.2f}")

    image = next_image
    state = next_state
    time.sleep(0.05)

print("\n===== TEST RESULT =====")
print(f"Steps: {step}")
print(f"Total Reward: {total_reward:.2f}")

if step < MAX_STEPS:
    print("Episode ended early. Reason: ", end="")
    if env.bridge.client.simGetCollisionInfo().has_collided:
        print("Collision!")
    else:
        pos = state[:3]
        if np.linalg.norm(pos - env.goal) < 2:
            print("Goal reached!")
        else:
            print("Height out of bounds or other termination.")
else:
    print("Max steps reached (timeout).")

print("\nTrajectory (last 10 steps):")
for t in trajectory[-10:]:
    print(t)