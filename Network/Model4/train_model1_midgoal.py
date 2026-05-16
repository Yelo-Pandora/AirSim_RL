#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np
import torch

MODEL4_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL4_DIR))
MODEL1_DIR = os.path.join(PROJECT_ROOT, "Network", "Model1")

if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)
if MODEL4_DIR not in sys.path:
    sys.path.insert(0, MODEL4_DIR)

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from reinforcement_network import CustomCombinedExtractor
from model1_midgoal_env import Model1MidGoalEnv


def has_tensorboard():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


class ConsecutiveArrivalSaveCallback(BaseCallback):
    def __init__(self, save_dir, formal_prefix="td3_model4_arrived10", resume_name="td3_model4_resume_latest", verbose=1):
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
                print(f"\n[Resume Save] Saved resume checkpoint: {resume_path}.zip")

            if streak >= 10 and self.num_timesteps != self.last_saved_step:
                model_path = os.path.join(self.save_dir, f"{self.formal_prefix}_{self.num_timesteps}_steps")
                self.model.save(model_path)
                self.last_saved_step = self.num_timesteps
                self.training_env.env_method("set_consecutive_arrivals", 0)
                if self.verbose > 0:
                    print(f"\n[Checkpoint] Saved stable checkpoint: {model_path}.zip")

        return True


def main():
    parser = argparse.ArgumentParser(description="Train Model1 with Model4 mid-range AirSim tasks")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of Model4 task cache before training")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensorboard_enabled = has_tensorboard()

    env = Model1MidGoalEnv(force_rebuild=args.rebuild)
    log_dir = os.path.join(script_dir, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    vec_env = DummyVecEnv([lambda: env])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(),
        net_arch=[512, 512, 512, 512, 256, 128],
        activation_fn=torch.nn.ReLU,
    )

    save_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    resume_path = os.path.join(save_dir, "td3_model4_resume_latest.zip")
    callback = ConsecutiveArrivalSaveCallback(save_dir=save_dir)

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
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=1e-4,
            buffer_size=2 ** 18,
            learning_starts=5000,
            batch_size=256,
            gamma=0.986,
            tau=0.005,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir if tensorboard_enabled else None,
            device=device,
            verbose=0,
        )

    print("Starting Model4 -> Model1 training...")
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name="TD3_Model4_MidGoal" if tensorboard_enabled else None,
        callback=callback,
        reset_num_timesteps=not os.path.exists(resume_path),
    )


if __name__ == "__main__":
    main()
