#!/usr/bin/env python
"""
train_rspg.py — Training script for RLoPlanner (RSPG + EGO-Planner)

Trains the Recurrent Soft Policy Gradient agent to generate local targets,
which are then tracked by the EGO-Planner for smooth, collision-free navigation.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure Model3 is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from network.rspg_network import RSPGAgent
from uav_env.uav_env import UAVRLOEnv


def generate_random_obstacles(env, n_obstacles=10, area=30, height_range=(-10, -2)):
    """Generate random cylindrical obstacles for training."""
    env.clear_sim_obstacles()
    for _ in range(n_obstacles):
        pos = [
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(height_range[0], height_range[1]),
        ]
        radius = np.random.uniform(0.5, 2.0)
        env.add_sim_obstacle(pos, radius)


def train(args):
    # Set random seed
    np.random.seed(config.TRAIN_SEED)
    torch.manual_seed(config.TRAIN_SEED)

    # Create save directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # Initialize environment
    env = UAVRLOEnv()

    # Initialize agent
    agent = RSPGAgent(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        config=config,
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        agent.load(args.resume)
        start_episode = int(args.resume.split("_ep")[-1].split(".")[0]) + 1

    print(f"Starting training: {config.TRAIN_EPISODES} episodes, "
          f"obs_dim={config.OBS_DIM}, action_dim={config.ACTION_DIM}")
    print(f"Device: {agent.device}")

    episode_rewards = []
    success_count = 0
    reason_counts = {}

    for episode in range(start_episode, config.TRAIN_EPISODES):
        # Generate random obstacles for this episode
        generate_random_obstacles(env, n_obstacles=np.random.randint(5, 15))

        # Reset environment
        obs = env.reset()
        episode_reward = 0.0
        reward_component_sums = {}
        hx = None
        step = 0
        info = {"reason": "not_started"}

        while step < config.TRAIN_MAX_STEPS:
            # Select action
            action, hx = agent.select_action(obs, hx=hx)

            # Execute step
            next_obs, reward, done, info = env.step(action)
            executed_action = info.get("executed_action", action)

            # Store in replay buffer
            agent.replay_buffer.push(obs, executed_action, reward, next_obs, done)

            episode_reward += reward
            for name, value in info.get("reward_components", {}).items():
                reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(value)
            obs = next_obs
            step += 1

            # Update agent every 100 time steps per paper Algorithm 1
            if step % 100 == 0:
                agent.update()

            if done:
                break

        # Episode summary
        episode_rewards.append(episode_reward)
        reason = info.get("reason", "unknown") or "running"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason == "arrived":
            success_count += 1

        # Logging
        if (episode + 1) % config.TRAIN_LOG_INTERVAL == 0:
            success_rate = success_count / (episode + 1) * 100
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            component_log = " ".join(
                f"{key}:{value:.1f}" for key, value in sorted(reward_component_sums.items())
            )
            print(f"Episode {episode + 1}/{config.TRAIN_EPISODES} | "
                  f"Reward: {episode_reward:.2f} | Avg(50): {avg_reward:.2f} | "
                  f"Success: {success_rate:.1f}% | "
                  f"Steps: {step} | "
                  f"Reason: {reason} | "
                  f"Dist: {info.get('dist_to_goal', float('nan')):.2f}m | "
                  f"Alt: {info.get('altitude', float('nan')):.2f}m | "
                  f"Collisions: {info.get('collision_count', 0)} | "
                  f"Alpha: {agent.alpha.item():.4f} | "
                  f"R[{component_log}]")

        # Save checkpoint
        if (episode + 1) % config.TRAIN_SAVE_INTERVAL == 0:
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"rspg_ep{episode + 1}.pth")
            agent.save(save_path)
            print(f"  Saved model to {save_path}")

    # Final save
    final_path = os.path.join(config.MODEL_SAVE_DIR, "rspg_final.pth")
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Overall success rate: {success_count / config.TRAIN_EPISODES * 100:.1f}%")
    print(f"Termination reasons: {reason_counts}")


def main():
    parser = argparse.ArgumentParser(description="Train RLoPlanner RSPG agent")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    args = parser.parse_args()

    if args.episodes:
        config.TRAIN_EPISODES = args.episodes

    train(args)


if __name__ == "__main__":
    main()
