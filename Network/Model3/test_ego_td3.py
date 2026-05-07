"""
Evaluation script for trained EGO-Planner + TD3 model.
Runs episodes with the trained policy and reports statistics.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from stable_baselines3.common.utils import set_random_seed

from env.ego_uav_env import EGOUAVEnv
from ego_planner.ego_planner import EGOPlanner
from network.td3_network import CustomCombinedExtractor
from config import (
    MAX_STEPS, ARRIVAL_RADIUS, V_MAX,
)


def evaluate(model_path: str, num_episodes: int = 10):
    """
    Evaluate a trained model over multiple episodes.

    Args:
        model_path: Path to the saved TD3 model
        num_episodes: Number of evaluation episodes
    """
    print("=" * 60)
    print(f"EGO-Planner + TD3 Evaluation")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    set_random_seed(42)

    stats = {
        "arrivals": 0,
        "collisions": 0,
        "timeouts": 0,
        "total_reward": [],
        "steps_per_episode": [],
        "final_distances": [],
    }

    for ep in range(num_episodes):
        env = EGOUAVEnv()
        env.planner = EGOPlanner(env)

        model = TD3.load(
            model_path,
            env=env,
            custom_objects={"features_extractor_class": CustomCombinedExtractor},
        )

        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"  Start: ({env.current_start_pos})")
        print(f"  Goal:  ({env.current_target})")

        step = 0
        while not done and step < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        # Record stats
        stats["total_reward"].append(episode_reward)
        stats["steps_per_episode"].append(step)
        stats["final_distances"].append(info.get('dis2goal', 0.0))

        if info.get('arrived', False):
            stats["arrivals"] += 1
            print(f"  -> ARRIVED in {step} steps, reward: {episode_reward:.1f}")
        elif info.get('collision', False):
            stats["collisions"] += 1
            print(f"  -> CRASHED after {step} steps, reward: {episode_reward:.1f}")
        else:
            stats["timeouts"] += 1
            print(f"  -> TIMEOUT after {step} steps, reward: {episode_reward:.1f}")

        env.close()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes:            {num_episodes}")
    print(f"  Arrivals:            {stats['arrivals']} ({stats['arrivals']/num_episodes*100:.0f}%)")
    print(f"  Collisions:          {stats['collisions']} ({stats['collisions']/num_episodes*100:.0f}%)")
    print(f"  Timeouts:            {stats['timeouts']} ({stats['timeouts']/num_episodes*100:.0f}%)")
    print(f"  Avg reward:          {np.mean(stats['total_reward']):.2f} +/- {np.std(stats['total_reward']):.2f}")
    print(f"  Avg steps:           {np.mean(stats['steps_per_episode']):.1f}")
    print(f"  Avg final distance:  {np.mean(stats['final_distances']):.2f}m")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate EGO-Planner + TD3 model")
    parser.add_argument("--model", type=str, default="logs/ego_td3_final.zip",
                        help="Path to saved TD3 model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(args.model, args.episodes)
