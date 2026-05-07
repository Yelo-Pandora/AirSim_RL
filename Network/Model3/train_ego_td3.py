"""
Training script for EGO-Planner + TD3.
Trains a TD3 agent that outputs high-level velocity commands,
while EGO-Planner handles low-level collision-free trajectory generation.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from env.ego_uav_env import EGOUAVEnv
from network.td3_network import CustomCombinedExtractor
from ego_planner.ego_planner import EGOPlanner
from config import (
    TOTAL_TIMESTEPS, SAVE_FREQ, BUFFER_SIZE, BATCH_SIZE,
    LEARNING_RATE, POLICY_NOISE, NOISE_CLIP, POLICY_FREQ,
)


def make_env(seed: int = 0, replan_every: int = 5):
    """Create a new environment instance with planner attached."""
    env = EGOUAVEnv(replan_every=replan_every)
    env.planner = EGOPlanner(env)
    return env


def train():
    print("=" * 60)
    print("EGO-Planner + TD3 Training")
    print("=" * 60)

    set_random_seed(42)

    # Create environment
    env = make_env()

    # Action noise for TD3 exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=0.0, sigma=0.1 * np.ones(n_actions))

    # Checkpoint callback
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=log_dir,
        name_prefix="ego_td3",
    )

    # Create TD3 model
    model = TD3(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=10000,
        batch_size=BATCH_SIZE,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        action_noise=action_noise,
        policy_kwargs=dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={"features_dim": 78},
            net_arch=dict(pi=[512, 512, 256, 128], qf=[512, 512, 256, 128]),
        ),
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        policy_freq=POLICY_FREQ,
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    model.save(os.path.join(log_dir, "ego_td3_initial"))

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
        )
        model.save(os.path.join(log_dir, "ego_td3_final"))
        print("\nTraining complete! Model saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save(os.path.join(log_dir, "ego_td3_interrupted"))
    finally:
        env.close()


if __name__ == "__main__":
    train()
