#!/usr/bin/env python
"""
Evaluation script for trained RLoPlanner model.

Metrics are aligned with the paper's Section V-B:
- Average reward over navigation tasks
- Success rate
- Trapped rate (timeout only)
- Collision rate
- Path efficiency over successful runs only
- Average acceleration and jerk
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from network.rspg_network import RSPGAgent
from uav_env.uav_env import UAVRLOEnv


def generate_test_obstacles(env, n_obstacles=10, area=30, height_range=(-10, -2)):
    """Generate test obstacles for offline simulation mode."""
    env.clear_sim_obstacles()
    for _ in range(n_obstacles):
        pos = [
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(-area / 2, area / 2),
            np.random.uniform(height_range[0], height_range[1]),
        ]
        radius = np.random.uniform(0.5, 2.0)
        env.add_sim_obstacle(pos, radius)


def evaluate(model_path, n_episodes=config.EVAL_EPISODES, verbose=True, planner_mode=config.PLANNER_MODE):
    agent = RSPGAgent(
        obs_dim=config.OBS_DIM,
        action_dim=config.ACTION_DIM,
        config=config,
    )
    agent.load(model_path)
    agent.actor.eval()

    env = UAVRLOEnv(planner_mode=planner_mode)

    successes = 0
    collisions = 0
    trapped = 0
    other_failures = 0
    total_reward = 0.0
    total_steps = 0
    successful_path_efficiencies = []
    successful_times = []
    accelerations = []
    jerks = []
    failure_reasons = {}

    for ep in range(n_episodes):
        if not env.use_airsim:
            generate_test_obstacles(env, n_obstacles=np.random.randint(8, 15))

        obs = env.reset()
        hx = None
        episode_reward = 0.0
        positions = [env.current_pos.copy()]
        start_pos = env.current_pos.copy()
        done = False

        while not done:
            action, hx = agent.select_action(obs, hx=hx, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            positions.append(env.current_pos.copy())

        total_reward += episode_reward
        total_steps += env.step_count
        reason = info.get("reason", "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        path_len = sum(
            np.linalg.norm(positions[i] - positions[i - 1])
            for i in range(1, len(positions))
        )
        straight_dist = float(np.linalg.norm(start_pos - env.goal))

        positions_arr = np.array(positions, dtype=np.float32)
        if len(positions_arr) > 2:
            dt = config.PLANNER_DT
            velocities_arr = np.diff(positions_arr, axis=0) / dt
            if len(velocities_arr) > 1:
                acc = np.diff(velocities_arr, axis=0) / dt
                if len(acc) > 0:
                    accelerations.append(float(np.mean(np.linalg.norm(acc, axis=1))))
                if len(acc) > 1:
                    jerk = np.diff(acc, axis=0) / dt
                    jerks.append(float(np.mean(np.linalg.norm(jerk, axis=1))))

        if reason == "arrived":
            successes += 1
            if path_len > 1e-6 and straight_dist > 1e-6:
                successful_path_efficiencies.append(straight_dist / path_len)
            successful_times.append(env.step_count * config.PLANNER_DT)
        elif reason == "collision":
            collisions += 1
        elif reason == "timeout":
            trapped += 1
        else:
            other_failures += 1

        if verbose:
            status = "SUCCESS" if reason == "arrived" else f"FAILED({reason})"
            efficiency = straight_dist / path_len if path_len > 1e-6 else 0.0
            print(
                f"Episode {ep + 1}/{n_episodes}: {status} | "
                f"Steps: {env.step_count} | Reward: {episode_reward:.2f} | "
                f"Path: {path_len:.2f}m | Efficiency: {efficiency:.3f} | "
                f"Regions: {env.start_region}->{env.goal_region} | "
                f"PlannerMode: {env.planner_mode}"
            )

    success_rate = successes / n_episodes * 100.0
    collision_rate = collisions / n_episodes * 100.0
    trapped_rate = trapped / n_episodes * 100.0
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    avg_path_efficiency = float(np.mean(successful_path_efficiencies)) if successful_path_efficiencies else 0.0
    avg_acc = float(np.mean(accelerations)) if accelerations else 0.0
    avg_jerk = float(np.mean(jerks)) if jerks else 0.0
    avg_time_to_goal = float(np.mean(successful_times)) if successful_times else 0.0

    results = {
        "episodes": n_episodes,
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "trapped_rate": trapped_rate,
        "other_failure_rate": other_failures / n_episodes * 100.0,
        "avg_steps": avg_steps,
        "avg_time_to_goal_success_only": avg_time_to_goal,
        "avg_path_efficiency_success_only": avg_path_efficiency,
        "avg_acceleration": avg_acc,
        "avg_jerk": avg_jerk,
        "failure_reasons": failure_reasons,
    }

    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(config.EVAL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:                    {n_episodes}")
    print(f"Average reward:              {avg_reward:.2f}")
    print(f"Success rate:                {success_rate:.1f}%")
    print(f"Collision rate:              {collision_rate:.1f}%")
    print(f"Trapped rate:                {trapped_rate:.1f}%")
    print(f"Other failure rate:          {other_failures / n_episodes * 100.0:.1f}%")
    print(f"Average steps:               {avg_steps:.1f}")
    print(f"Average time to goal(success){avg_time_to_goal:.2f}s")
    print(f"Path efficiency(success):    {avg_path_efficiency:.3f}")
    print(f"Average acceleration:        {avg_acc:.3f} m/s^2")
    print(f"Average jerk:                {avg_jerk:.3f} m/s^3")
    print(f"Saved summary:               {config.EVAL_RESULTS_JSON}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLoPlanner model")
    parser.add_argument("model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.EVAL_EPISODES,
        help="Number of evaluation episodes (paper metrics use 200 tasks)",
    )
    parser.add_argument(
        "--planner-mode",
        type=str,
        default=config.PLANNER_MODE,
        choices=["ego", "straight"],
        help="Low-level execution mode",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    evaluate(
        args.model_path,
        n_episodes=args.episodes,
        verbose=not args.quiet,
        planner_mode=args.planner_mode,
    )


if __name__ == "__main__":
    main()
