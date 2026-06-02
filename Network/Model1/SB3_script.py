import os
import numpy as np
import torch
import sys
import time
import random

# 确保能优先找到根目录下的 airsim 源码
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import airsim
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from reinforcement_network import AirSimUAVEnv
from reinforcement_network import CustomCombinedExtractor
from stable_baselines3.common.noise import ActionNoise


CURRICULUM_REGION_ORDER = ("1", "2", "3", "4")
CURRICULUM_STAGE_STEPS = 200_000
ACTION_NOISE_INITIAL_SIGMA = 0.2
ACTION_NOISE_FINAL_SIGMA = 0.08
ACTION_NOISE_DECAY_STEPS = 800_000


class LinearDecayNormalActionNoise(ActionNoise):
    def __init__(self, mean, initial_sigma, final_sigma, decay_steps):
        self.mean = np.array(mean, dtype=np.float32)
        self.initial_sigma = np.array(initial_sigma, dtype=np.float32)
        self.final_sigma = np.array(final_sigma, dtype=np.float32)
        self.decay_steps = max(int(decay_steps), 1)
        self.num_calls = 0

    def __call__(self):
        progress = min(float(self.num_calls) / float(self.decay_steps), 1.0)
        sigma = self.initial_sigma + progress * (self.final_sigma - self.initial_sigma)
        self.num_calls += 1
        return np.random.normal(self.mean, sigma).astype(np.float32)

    def reset(self):
        pass

    def set_step(self, step):
        self.num_calls = max(int(step), 0)

    def __repr__(self):
        return (
            f"LinearDecayNormalActionNoise(initial_sigma={self.initial_sigma}, "
            f"final_sigma={self.final_sigma}, decay_steps={self.decay_steps}, "
            f"num_calls={self.num_calls})"
        )


class TrainingTeleportResetWrapper(gym.Wrapper):
    """Keep training resets teleport-based while the env reset stays deployment-safe."""

    def __init__(self, env):
        super().__init__(env)
        self.total_train_steps = 0
        self.region_pairs = self._build_region_pairs()
        self.pending_reverse_route = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "start_pos" in options and "target" in options:
            start_pos = np.array(options["start_pos"], dtype=np.float32)
            target_pos = np.array(options["target"], dtype=np.float32)
            region = options.get("region", "manual")
        else:
            start_pos, target_pos, region = self._next_curriculum_route()

        self.env.stabilize_before_reset()
        start_pos = self._teleport_to_training_start(start_pos)

        env_options = {
            "start_pos": start_pos.tolist(),
            "target": target_pos.tolist(),
            "region": region,
            "skip_stabilization": True,
        }
        return self.env.reset(seed=seed, options=env_options)

    def _build_region_pairs(self):
        region_pairs = {}
        for region, points in self.env.region_points.items():
            pairs = []
            pair_count = len(points) // 2
            for pair_index in range(pair_count):
                start_index = pair_index * 2
                point_a = np.array(points[start_index], dtype=np.float32)
                point_b = np.array(points[start_index + 1], dtype=np.float32)
                pairs.append((pair_index, point_a, point_b))
            if pairs:
                region_pairs[str(region)] = pairs
            if len(points) % 2 == 1:
                print(f"[Curriculum Warning] Region {region} has an unmatched last point; ignoring it.")

        if not region_pairs:
            raise ValueError("No paired training points found in relative_coordinates.csv.")
        return region_pairs

    def _next_curriculum_route(self):
        if self.pending_reverse_route is not None:
            route = self.pending_reverse_route
            self.pending_reverse_route = None
            return route

        active_regions = self._active_regions()
        active_pairs = [
            (region, pair_index, point_a, point_b)
            for region in active_regions
            for pair_index, point_a, point_b in self.region_pairs.get(region, [])
        ]
        if not active_pairs:
            active_pairs = [
                (region, pair_index, point_a, point_b)
                for region, pairs in self.region_pairs.items()
                for pair_index, point_a, point_b in pairs
            ]

        region, pair_index, point_a, point_b = random.choice(active_pairs)
        if random.random() < 0.5:
            first = (point_a, point_b, region)
            second = (point_b, point_a, region)
        else:
            first = (point_b, point_a, region)
            second = (point_a, point_b, region)

        self.pending_reverse_route = second
        print(
            f"[Curriculum] steps={self.total_train_steps} active={active_regions} "
            f"region={region} pair={pair_index}"
        )
        return first

    def _active_regions(self):
        stage_count = int(self.total_train_steps // CURRICULUM_STAGE_STEPS) + 1
        stage_count = max(1, min(stage_count, len(CURRICULUM_REGION_ORDER)))
        return [
            region for region in CURRICULUM_REGION_ORDER[:stage_count]
            if region in self.region_pairs
        ]

    def _teleport_to_training_start(self, start_pos):
        start_pos = np.array(start_pos, dtype=np.float32)
        if float(start_pos[2]) >= 0.0:
            start_pos[2] = -10.0

        yaw_rad = self._current_yaw_rad()
        pose = airsim.Pose(
            airsim.Vector3r(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
            airsim.to_quaternion(0.0, 0.0, yaw_rad),
        )

        try:
            self.env.client.reset()
            self.env.client.enableApiControl(True, vehicle_name=self.env.vehicle_name)
            self.env.client.armDisarm(True, vehicle_name=self.env.vehicle_name)
            self.env.client.simSetVehiclePose(pose, True, vehicle_name=self.env.vehicle_name)
            time.sleep(0.5)
            self.env.client.moveByVelocityAsync(
                0.0,
                0.0,
                0.0,
                0.5,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.env.vehicle_name,
            ).join()
        except Exception as e:
            print(f"[Training Reset Warning] 训练起点传送失败: {e}")
        return start_pos

    def _current_yaw_rad(self):
        try:
            state = self.env.client.getMultirotorState(vehicle_name=self.env.vehicle_name)
            _, _, yaw_rad = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            return float(yaw_rad)
        except Exception:
            return 0.0

    def set_total_train_steps(self, value):
        self.total_train_steps = int(value)
        return self.env.set_total_train_steps(value)

    def set_consecutive_arrivals(self, value):
        return self.env.set_consecutive_arrivals(value)


def has_tensorboard():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


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
            os.makedirs(self.save_dir, exist_ok=True)
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
        os.makedirs(self.save_dir, exist_ok=True)
        resume_path = os.path.join(self.save_dir, self.resume_name)
        self.model.save(resume_path)
        if self.verbose > 0:
            print(f"\n[Resume Save] 训练结束，已保存续训快照: {resume_path}.zip")


# ==========================================
# 训练主程序
# ==========================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensorboard_enabled = has_tensorboard()
    # 创建环境并包裹 Monitor (用于记录训练日志，如 episode reward)
    env = TrainingTeleportResetWrapper(AirSimUAVEnv())
    log_dir = os.path.join(script_dir, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)  # Monitor 可以记录每个回合的奖励和长度

    # SB3 推荐把环境打包成向量化环境
    vec_env = DummyVecEnv([lambda: env])

    # 定义动作噪声 (Action Noise)
    # TD3 是 Off-Policy 确定性算法，必须依靠外部添加的噪声来进行动作空间的探索
    n_actions = env.action_space.shape[-1]
    action_noise = LinearDecayNormalActionNoise(
        mean=np.zeros(n_actions),
        initial_sigma=ACTION_NOISE_INITIAL_SIGMA * np.ones(n_actions),
        final_sigma=ACTION_NOISE_FINAL_SIGMA * np.ones(n_actions),
        decay_steps=ACTION_NOISE_DECAY_STEPS,
    )

    # 配置自定义特征提取器
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(),  # 当前不需要额外参数
        net_arch=[512, 512, 512, 512, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    save_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    resume_path = os.path.join(save_dir, "td3_resume_latest.zip")
    callback = ConsecutiveArrivalSaveCallback(save_dir=save_dir)

    # 4. 实例化或恢复 TD3 模型
    print(f"Using device: {device}")
    print(f"Checkpoint directory: {save_dir}")
    if tensorboard_enabled:
        print(f"TensorBoard logging enabled: {log_dir}")
    else:
        print("TensorBoard not installed, disabling TensorBoard logging.")
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
            tensorboard_log=log_dir if tensorboard_enabled else None,  # TensorBoard ??????
            device=device,                      # 使用 cuda
            verbose=0                           # 打印详细输出
        )

    # 开始训练
    if hasattr(model.action_noise, "set_step"):
        model.action_noise.set_step(int(model.num_timesteps))
    vec_env.env_method("set_total_train_steps", int(model.num_timesteps))

    print("Starting training...")
    total_timesteps = 800000  # 4-region curriculum: unlock one new region every 20w steps
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="TD3_AirSim_Run1" if tensorboard_enabled else None,  # TensorBoard ????????
        callback=callback,
        reset_num_timesteps=not os.path.exists(resume_path),
        # progress_bar=True  # 显示进度条 (需安装 tqdm)
    )


if __name__ == "__main__":
    main()
