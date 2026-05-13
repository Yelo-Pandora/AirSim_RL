import os
import numpy as np
import torch
import sys

# 确保能优先找到根目录下的 airsim 源码
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import airsim
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from reinforcement_network import AirSimUAVEnv
from reinforcement_network import CustomCombinedExtractor
from stable_baselines3.common.noise import NormalActionNoise


class ConsecutiveArrivalSaveCallback(BaseCallback):
    def __init__(self, save_dir, formal_prefix="td3_arrived10", resume_name="td3_resume_latest", verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.formal_prefix = formal_prefix
        self.resume_name = resume_name
        self.last_saved_step = -1

    def _init_callback(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        self.training_env.env_method("set_total_train_steps", self.num_timesteps)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos or not dones:
            return True

        info = infos[0]
        done = bool(dones[0])
        streak = int(info.get("consecutive_arrivals", 0))

        if done:
            resume_path = os.path.join(self.save_dir, self.resume_name)
            self.model.save(resume_path)
            if self.verbose > 0:
                print(f"\n[Resume Save] 回合结束，已保存续训快照: {resume_path}.zip")

            if streak >= 10 and self.num_timesteps != self.last_saved_step:
                model_path = os.path.join(self.save_dir, f"{self.formal_prefix}_{self.num_timesteps}_steps")
                self.model.save(model_path)
                self.last_saved_step = self.num_timesteps
                self.training_env.env_method("set_consecutive_arrivals", 0)
                if self.verbose > 0:
                    print(f"\n[Checkpoint] 连续 10 次 arrived，已保存正式权重: {model_path}.zip")

        return True

    def _on_training_end(self):
        resume_path = os.path.join(self.save_dir, self.resume_name)
        self.model.save(resume_path)
        if self.verbose > 0:
            print(f"\n[Resume Save] 训练结束，已保存续训快照: {resume_path}.zip")


# ==========================================
# 训练主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建环境并包裹 Monitor (用于记录训练日志，如 episode reward)
    env = AirSimUAVEnv()
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)  # Monitor 可以记录每个回合的奖励和长度

    # SB3 推荐把环境打包成向量化环境
    vec_env = DummyVecEnv([lambda: env])

    # 定义动作噪声 (Action Noise)
    # TD3 是 Off-Policy 确定性算法，必须依靠外部添加的噪声来进行动作空间的探索
    n_actions = env.action_space.shape[-1]
    # 提高探索强度，避免过早塌缩到固定方向动作模板
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # 配置自定义特征提取器
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(),  # 当前不需要额外参数
        net_arch=[512, 512, 512, 512, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    save_dir = "./checkpoints"
    resume_path = os.path.join(save_dir, "td3_resume_latest.zip")
    callback = ConsecutiveArrivalSaveCallback(save_dir=save_dir)

    # 4. 实例化或恢复 TD3 模型
    print(f"Using device: {device}")
    if os.path.exists(resume_path):
        print(f"Loading existing checkpoint: {resume_path}")
        model = TD3.load(
            resume_path,
            env=vec_env,
            device=device,
            action_noise=action_noise,
        )
    else:
        model = TD3(
            policy="MultiInputPolicy",  # 因为 observation_space 是 Dict，须用 MultiInputPolicy
            env=vec_env,                        # 训练环境
            learning_rate=1e-4,                 # 学习率
            buffer_size=2 ** 18,                # 经验回放池大小
            learning_starts=5000,               # 先积累更多多样经验，再开始更新网络
            batch_size=256,                     # 每次采样的批次大小
            gamma=0.986,                        # 学习衰减率
            tau=0.005,                          # 软更新系数
            action_noise=action_noise,          # 注入探索噪声
            policy_delay=2,                     # 每几轮更新一次决策参数
            target_policy_noise=0.2,            # 策略噪声
            target_noise_clip=0.5,              # 策略噪声裁剪范围
            policy_kwargs=policy_kwargs,        # 注入自定义的网络特征提取器
            tensorboard_log="./tb_logs/",      # TensorBoard 日志保存路径
            device=device,                      # 使用 cuda
            verbose=0                           # 打印详细输出
        )

    # 开始训练
    print("Starting training...")
    total_timesteps = 300000  # 设置训练的总步数
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="TD3_AirSim_Run1",  # TensorBoard 中显示的实验名称
        callback=callback,
        reset_num_timesteps=not os.path.exists(resume_path),
        # progress_bar=True  # 显示进度条 (需安装 tqdm)
    )


if __name__ == "__main__":
    main()
