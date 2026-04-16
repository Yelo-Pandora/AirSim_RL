import os
import torch
import numpy as np

# 导入你写好的环境和模型
from reinforcement_network import AirSimUAVEnv
from stable_baselines3 import TD3


def evaluate_model(model_path, num_episodes=5, custom_start=None, custom_target=None):
    print(f"[{'=' * 50}]")
    print(f"开始加载模型: {model_path}")

    # 1. 初始化 AirSim 环境
    env = AirSimUAVEnv()

    # 2. 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")

    try:
        model = TD3.load(model_path, env=env, device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print(f"[{'=' * 50}]")
    print("开始进行测试飞行...")

    # 3. 开始测试循环
    for ep in range(num_episodes):
        print(f"\n--- 测试回合 (Episode) {ep + 1}/{num_episodes} ---")

        # 🟢 核心修改：打包 options 参数传入指定的坐标
        reset_options = {}
        if custom_start is not None and custom_target is not None:
            reset_options = {
                "start_pos": custom_start,
                "target": custom_target
            }
            print(f"强制指定路线: 起点 {custom_start} -> 终点 {custom_target}")
        else:
            print("未指定坐标，将从 CSV 数据集中随机抽取路线...")

        # 将 options 传入 reset 函数
        obs, info = env.reset(options=reset_options)

        done = False
        episode_reward = 0.0

        while not done:
            # deterministic=True 关闭探索噪声，输出确定性最优动作
            action, _states = model.predict(obs, deterministic=True)

            # 在环境中执行动作
            obs, reward, terminated, truncated, step_info = env.step(action)

            episode_reward += reward
            done = terminated or truncated

        # 回合结束，打印结果
        end_reason = "成功到达终点！" if step_info.get('arrived', False) else \
            ("发生碰撞！" if step_info.get('collision', False) else \
                 ("飞出限高！" if step_info.get('out_of_ceiling', False) else \
                      ("越界！" if step_info.get('crossed_border', False) else "超时！")))

        print(f"\n回合 {ep + 1} 结束 | 总奖励: {episode_reward:.2f} | 结束原因: {end_reason}")

    print(f"\n[{'=' * 50}]")
    print("测试全部完成！")

    # 测试结束后清理环境控制权
    env.client.armDisarm(False, vehicle_name=env.vehicle_name)
    env.client.enableApiControl(False, vehicle_name=env.vehicle_name)


if __name__ == "__main__":
    MODEL_PATH = "td3_airsim_uav_model"

    # 🌟 在这里自定义你的测试起点和终点
    # 注意：高度(Z轴)在 AirSim 中，负值表示在空中 (例如 -2.0 表示高出地面 2 米)
    TEST_START = [0.0, 0.0, -2.0]
    TEST_TARGET = [30.0, 15.0, -5.0]

    # 如果你想随机测试 CSV 里的数据，只需把 custom_start 和 custom_target 设为 None
    # evaluate_model(MODEL_PATH, num_episodes=5, custom_start=None, custom_target=None)

    evaluate_model(
        model_path=MODEL_PATH,
        num_episodes=3,  # 测试回合数
        custom_start=TEST_START,  # 传入自定义起点
        custom_target=TEST_TARGET  # 传入自定义终点
    )