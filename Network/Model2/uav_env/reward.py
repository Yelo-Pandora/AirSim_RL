# reward.py
import numpy as np

def compute_reward(state, prev_state, goal, collision=False, seg_img=None):
    pos = state[:3]
    prev_pos = prev_state[:3]
    vel = state[3:6]

    # ========================
    # 🎯 距离推进（核心唯一正反馈）
    # ========================
    dist = np.linalg.norm(goal - pos)
    prev_dist = np.linalg.norm(goal - prev_pos)

    dist_delta = prev_dist - dist
    reward = 0.0

    if dist_delta > 0:
        reward += dist_delta * 50.0   # 靠近目标 → 强奖励
    else:
        reward += dist_delta * 10.0   # 远离目标 → 轻惩罚

    # ========================
    # 🚫 防“绕圈 / 停滞”（关键🔥）
    # ========================
    if abs(dist_delta) < 0.05:
        reward -= 1.0

    # ========================
    # 🏁 到达目标
    # ========================
    if dist < 2.0:
        reward += 1000.0

    # ========================
    # 💥 碰撞惩罚
    # ========================
    if collision:
        reward -= 500.0

    # ========================
    # ⏳ 时间惩罚（防拖延）
    # ========================
    reward -= 1

    # ========================
    # 🚫 原地不动惩罚
    # ========================
    speed = np.linalg.norm(vel)
    if speed < 0.5:
        reward -= 1.0

    # ========================
    # 🚀 鼓励合理速度（但不能被利用）
    # ========================
    reward += min(speed, 3.0) * 0.3

    # ========================
    # 🎯 水平方向对齐（辅助，不主导）
    # ========================
    horizontal_vel = np.array([vel[0], vel[1], 0.0])
    horizontal_goal = np.array([goal[0] - pos[0], goal[1] - pos[1], 0.0])

    norm_hv = np.linalg.norm(horizontal_vel)
    norm_hg = np.linalg.norm(horizontal_goal)

    if norm_hv > 1e-3 and norm_hg > 1e-3:
        cos_sim = np.dot(horizontal_vel, horizontal_goal) / (norm_hv * norm_hg)
        reward += cos_sim * 1.5   # 适中即可，不能压过距离奖励

    # ========================
    # 🚁 高度安全约束
    # ========================
    height = pos[2]

    # 太低（接近地面）
    if height > -3.0:
        reward -= 200.0 * (height + 3.0) ** 2

    # 太高（飞丢）
    if height < -25.0:
        reward -= 2.0

    # ========================
    # 🧱 简单语义避障（轻量版）
    # ========================
    if seg_img is not None:
        try:
            h, w = seg_img.shape[:2]
            center_pixel = seg_img[h // 2, w // 2]

            semantic_id = center_pixel[0] if hasattr(center_pixel, "__len__") else center_pixel

            # 2 = obstacle
            if semantic_id == 2:
                reward -= 10.0

        except Exception:
            pass  # 防止 segmentation 偶发错误炸掉训练

    return reward