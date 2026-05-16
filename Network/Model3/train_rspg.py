#!/usr/bin/env python
"""
Training script for RLoPlanner (RSPG + EGO-Planner).

Tightened to match the paper's protocol more closely:
- global training budget driven by timesteps
- replay updates every 100 global timesteps
- rolling reward statistics over 250 episodes
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from network.rspg_network import RSPGAgent
from uav_env.uav_env import UAVRLOEnv


def generate_random_obstacles(env, n_obstacles=10, area=30, height_range=(-10, -2)):
    """Generate random cylindrical obstacles for offline simulation training."""
    env.clear_sim_obstacles()
    for _ in range(n_obstacles):
        pos = [
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(height_range[0], height_range[1]),
        ]
        radius = np.random.uniform(0.5, 2.0)
        env.add_sim_obstacle(pos, radius)


def append_metrics_row(path, row, write_header=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    np.random.seed(config.TRAIN_SEED)
    torch.manual_seed(config.TRAIN_SEED)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    env = UAVRLOEnv(planner_mode=args.planner_mode)
    agent = RSPGAgent(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        config=config,
    )

    start_episode = 0
    global_steps = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        agent.load(args.resume)
        base = os.path.basename(args.resume)
        if "_ep" in base:
            start_episode = int(base.split("_ep")[-1].split(".")[0]) + 1

    total_timestep_budget = args.timesteps if args.timesteps is not None else config.TRAIN_TOTAL_TIMESTEPS
    max_episodes = args.episodes if args.episodes is not None else config.TRAIN_EPISODES

    print(
        f"Starting training: timestep_budget={total_timestep_budget}, "
        f"max_episodes={max_episodes}, obs_dim={config.OBS_DIM}, action_dim={config.ACTION_DIM}"
    )
    print(f"Device: {agent.device}")
    print(f"AirSim mode: {env.use_airsim}")
    print(f"Planner mode: {env.planner_mode}")

    episode_rewards = []
    success_count = 0
    reason_counts = {}

    for episode in range(start_episode, max_episodes):
        if global_steps >= total_timestep_budget:
            break

        if not env.use_airsim:
            generate_random_obstacles(env, n_obstacles=np.random.randint(5, 15))

        env.set_current_episode(episode + 1)
        env.set_total_train_steps(global_steps)
        obs = env.reset()
        episode_reward = 0.0
        reward_component_sums = {}
        hx = None
        step = 0
        info = {"reason": "not_started"}

        while step < config.TRAIN_MAX_STEPS and global_steps < total_timestep_budget:
            action, hx = agent.select_action(obs, hx=hx)
            next_obs, reward, done, info = env.step(action)
            executed_action = info.get("executed_action", action)

            # Store in replay buffer
            agent.replay_buffer.push(obs, executed_action, reward, next_obs, done)

            episode_reward += reward
            for name, value in info.get("reward_components", {}).items():
                reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(value)
            obs = next_obs
            step += 1
            global_steps += 1
            env.set_total_train_steps(global_steps)

            if global_steps % config.TRAIN_UPDATE_INTERVAL == 0:
                agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)
        reason = info.get("reason", "unknown") or "running"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason == "arrived":
            success_count += 1
        elif reason == "collision":
            collision_count += 1
        elif reason == "timeout":
            trapped_count += 1

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

            print(
                f"Episode {episode + 1}/{max_episodes} | "
                f"GlobalSteps: {global_steps}/{total_timestep_budget} | "
                f"Reward: {episode_reward:.2f} | Avg({config.TRAIN_REWARD_WINDOW}): {avg_reward:.2f} | "
                f"Success: {success_rate:.1f}% | Collision: {collision_rate:.1f}% | Trapped: {trapped_rate:.1f}% | "
                f"Steps: {step} | Reason: {reason} | "
                f"StartRegion: {info.get('start_region')} -> GoalRegion: {info.get('goal_region')} | "
                f"Alpha: {agent.alpha.item():.4f} | SPS: {steps_per_sec:.2f}"
            )

            metrics_row = {
                "episode": episode + 1,
                "global_steps": global_steps,
                "episode_reward": round(float(episode_reward), 6),
                f"avg_reward_{config.TRAIN_REWARD_WINDOW}": round(avg_reward, 6),
                "success_rate": round(success_rate, 4),
                "collision_rate": round(collision_rate, 4),
                "trapped_rate": round(trapped_rate, 4),
                "episode_steps": step,
                "reason": reason,
                "start_region": info.get("start_region"),
                "goal_region": info.get("goal_region"),
                "alpha": round(float(agent.alpha.item()), 6),
                "steps_per_sec": round(float(steps_per_sec), 6),
            }
            append_metrics_row(
                config.TRAIN_METRICS_CSV,
                metrics_row,
                write_header=not metrics_header_written,
            )
            metrics_header_written = True

        if (episode + 1) % config.TRAIN_SAVE_INTERVAL == 0 or global_steps >= total_timestep_budget:
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"rspg_ep{episode + 1}.pth")
            agent.save(save_path)
            print(f"Saved model to {save_path}")

    final_path = os.path.join(config.MODEL_SAVE_DIR, "rspg_final.pth")
    agent.save(final_path)
    total_episodes_run = max(len(episode_rewards), 1)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Overall success rate: {success_count / config.TRAIN_EPISODES * 100:.1f}%")
    print(f"Termination reasons: {reason_counts}")


def main():
    parser = argparse.ArgumentParser(description="Train RLoPlanner RSPG agent")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--episodes", type=int, default=None, help="Override max number of episodes")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timestep budget")
    parser.add_argument(
        "--planner-mode",
        type=str,
        default=config.PLANNER_MODE,
        choices=["ego", "straight"],
        help="Low-level execution mode",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
