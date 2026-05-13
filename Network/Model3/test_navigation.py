#!/usr/bin/env python
"""
test_navigation.py — Evaluation script for trained RLoPlanner model.

Runs evaluation episodes and computes metrics per paper Section V-B:
- Average reward
- Success rate
- Trapped rate
- Collision rate
- Path efficiency (straight-line / actual path length)
- Average acceleration and jerk
"""

import os
import sys
import argparse
import numpy as np

# Ensure Model3 is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from network.rspg_network import RSPGAgent
from uav_env.uav_env import UAVRLOEnv


def generate_test_obstacles(env, n_obstacles=10, area=30, height_range=(-10, -2)):
    """Generate test obstacles."""
    env.clear_sim_obstacles()
    for _ in range(n_obstacles):
        pos = [
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(height_range[0], height_range[1]),
        ]
        radius = np.random.uniform(0.5, 2.0)
        env.add_sim_obstacle(pos, radius)


def evaluate(model_path, n_episodes=250, verbose=True):
    """Evaluate trained model."""
    # Initialize
    agent = RSPGAgent(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        config=config,
    )
    agent.load(model_path)
    agent.actor.eval()

    env = UAVRLOEnv()

    # Metrics
    successes = 0
    collisions = 0
    trapped = 0
    total_reward = 0.0
    total_steps = 0
    path_lengths = []
    straight_dists = []
    accelerations = []
    jerks = []

    for ep in range(n_episodes):
        generate_test_obstacles(env, n_obstacles=np.random.randint(8, 15))
        obs = env.reset()
        hx = None
        episode_reward = 0.0
        positions = [env.current_pos.copy()]
        velocities = [env.current_vel.copy()]
        start_pos = env.current_pos.copy()
        done = False

        while not done:
            action, hx = agent.select_action(obs, hx=hx, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            positions.append(env.current_pos.copy())
            velocities.append(env.current_vel.copy())

        total_reward += episode_reward
        total_steps += env.step_count

        reason = info.get("reason", "")
        if reason == "arrived":
            successes += 1
        elif reason == "collision":
            collisions += 1
        else:
            trapped += 1

        # Path length
        path_len = sum(
            np.linalg.norm(positions[i] - positions[i - 1])
            for i in range(1, len(positions))
        )
        path_lengths.append(path_len)

        # Straight-line distance from start to goal
        straight_dist = np.linalg.norm(start_pos - env.goal)
        straight_dists.append(straight_dist)

        # Acceleration and jerk
        positions_arr = np.array(positions)
        if len(positions_arr) > 2:
            dt = config.PLANNER_DT
            velocities_arr = np.diff(positions_arr, axis=0) / dt
            if len(velocities_arr) > 1:
                acc = np.diff(velocities_arr, axis=0) / dt
                accelerations.append(np.mean(np.linalg.norm(acc, axis=1)))
                if len(acc) > 1:
                    jerk = np.diff(acc, axis=0) / dt
                    jerks.append(np.mean(np.linalg.norm(jerk, axis=1)))

        if verbose:
            status = "SUCCESS" if reason == "arrived" else f"FAILED({reason})"
            pe = straight_dist / path_len if path_len > 0 else 0.0
            print(f"  Episode {ep + 1}/{n_episodes}: {status} | "
                  f"Steps: {env.step_count} | Reward: {episode_reward:.2f} | "
                  f"Path: {path_len:.2f}m | Efficiency: {pe:.3f}")

    # Summary
    success_rate = successes / n_episodes * 100
    collision_rate = collisions / n_episodes * 100
    trapped_rate = trapped / n_episodes * 100
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes

    # Path efficiency: straight-line / actual path (only for successful episodes)
    path_efficiencies = []
    for i in range(n_episodes):
        if path_lengths[i] > 0 and straight_dists[i] > 0:
            path_efficiencies.append(straight_dists[i] / path_lengths[i])
    avg_path_efficiency = np.mean(path_efficiencies) if path_efficiencies else 0.0

    avg_acc = np.mean(accelerations) if accelerations else 0.0
    avg_jerk = np.mean(jerks) if jerks else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:            {n_episodes}")
    print(f"Average reward:      {avg_reward:.2f}")
    print(f"Success rate:        {success_rate:.1f}%")
    print(f"Collision rate:      {collision_rate:.1f}%")
    print(f"Trapped rate:        {trapped_rate:.1f}%")
    print(f"Average steps:       {avg_steps:.1f}")
    print(f"Path efficiency:     {avg_path_efficiency:.3f}")
    print(f"Average acceleration:{avg_acc:.3f} m/s²")
    print(f"Average jerk:        {avg_jerk:.3f} m/s³")
    print("=" * 50)

    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "trapped_rate": trapped_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_path_efficiency": avg_path_efficiency,
        "avg_acceleration": avg_acc,
        "avg_jerk": avg_jerk,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLoPlanner model")
    parser.add_argument("model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=250, help="Number of evaluation episodes (default 250 per paper)")
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    evaluate(args.model_path, n_episodes=args.episodes, verbose=not args.quiet)


if __name__ == "__main__":
    main()
