#!/usr/bin/env python

import argparse
import os
import sys

import torch
from stable_baselines3 import TD3


MODEL4_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL4_DIR))
MODEL1_DIR = os.path.join(PROJECT_ROOT, "Network", "Model1")

if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)
if MODEL4_DIR not in sys.path:
    sys.path.insert(0, MODEL4_DIR)

from model1_midgoal_env import Model1MidGoalEnv


def evaluate(model_path, episodes=5, force_rebuild=False):
    env = Model1MidGoalEnv(force_rebuild=force_rebuild)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    model = TD3.load(model_path, env=env, device=device)

    for ep in range(episodes):
        obs, info = env.reset()
        print(
            f"Episode {ep + 1}/{episodes} | "
            f"Task: {info.get('start')} -> {info.get('target')} | "
            f"Region: {info.get('region')} | "
            f"Route: 1/{info.get('route_total')}"
        )
        done = False
        episode_reward = 0.0
        step_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if step_info.get("subgoal_reached"):
                print(
                    f"  subgoal reached -> advancing to "
                    f"{step_info.get('route_index') + 1}/{step_info.get('route_total')}"
                )

        print(
            f"Episode {ep + 1} end | Reward: {episode_reward:.2f} | "
            f"Reason: {step_info.get('collision') and 'collision' or ('arrived' if step_info.get('arrived') else 'timeout')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run a trained Model1 policy on Model4 mid-range tasks")
    parser.add_argument("model_path", type=str, help="Path to Model1 TD3 checkpoint (.zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of Model4 task cache")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    evaluate(args.model_path, episodes=args.episodes, force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
