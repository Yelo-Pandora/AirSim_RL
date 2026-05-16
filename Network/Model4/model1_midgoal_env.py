import os
import sys

import numpy as np

MODEL4_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL4_DIR))
MODEL1_DIR = os.path.join(PROJECT_ROOT, "Network", "Model1")

if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)
if MODEL4_DIR not in sys.path:
    sys.path.insert(0, MODEL4_DIR)

from reinforcement_network import AirSimUAVEnv
from task_source import Model4TaskSource


class Model1MidGoalEnv(AirSimUAVEnv):
    """
    Reuse Model1's low-level env, but drive it with Model4's long-range route
    decomposition and mid-goal handoff.
    """

    def __init__(self, force_rebuild=False):
        super().__init__()
        self.task_source = Model4TaskSource(
            client=self.client,
            vehicle_name=self.vehicle_name,
            force_rebuild=force_rebuild,
        )
        self.route_subgoals = []
        self.route_goal = None
        self.route_index = 0
        self.route_regions = ""

    def reset(self, seed=None, options=None):
        options = options or {}
        if "start_pos" not in options or "target" not in options:
            task = self.task_source.sample_task()
            options = dict(options)
            options["start_pos"] = task["start"].tolist()
            options["target"] = task["target"].tolist()
            options["region"] = f"{task['start_region']}->{task['goal_region']}"
            self.route_subgoals = [np.array(p, dtype=np.float32).copy() for p in task["route_subgoals"]]
            self.route_goal = np.array(task["global_goal"], dtype=np.float32).copy()
            self.route_regions = options["region"]
            self.route_index = 0
        else:
            self.route_subgoals = [np.array(options["target"], dtype=np.float32).copy()]
            self.route_goal = np.array(options["target"], dtype=np.float32).copy()
            self.route_regions = str(options.get("region", "manual"))
            self.route_index = 0

        obs, info = super().reset(seed=seed, options=options)
        info["route_index"] = self.route_index
        info["route_total"] = len(self.route_subgoals)
        info["route_goal"] = self.route_goal.copy() if self.route_goal is not None else None
        info["route_regions"] = self.route_regions
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["route_index"] = self.route_index
        info["route_total"] = len(self.route_subgoals)
        info["route_goal"] = self.route_goal.copy() if self.route_goal is not None else None
        info["route_regions"] = self.route_regions

        if terminated and info.get("arrived", False):
            if self.route_index < len(self.route_subgoals) - 1:
                self._advance_to_next_subgoal()
                terminated = False
                info["arrived"] = False
                info["subgoal_reached"] = True
                info["route_index"] = self.route_index
                info["route_total"] = len(self.route_subgoals)
            else:
                info["final_goal_reached"] = True

        return obs, reward, terminated, truncated, info

    def _advance_to_next_subgoal(self):
        drone_pos = self.current_target - self.current_rel_pos
        self.route_index += 1
        next_target = self.route_subgoals[self.route_index].copy()
        self.current_start_pos = np.array(drone_pos, dtype=np.float32).copy()
        self.current_target = next_target.copy()
        self.current_region = self.route_regions
        self.start_rel_pos = self.current_target - self.current_start_pos
        self.start_dist = float(np.linalg.norm(self.start_rel_pos))
        if self.start_dist < 1e-5:
            self.start_dist = 1.0
        self.last_dis2goal = self.start_dist
        self._refresh_waypoints_from_current_segment()
        self._draw_goal_marker()

    def _refresh_waypoints_from_current_segment(self):
        self.waypoints = []
        for i in range(1, 17):
            ratio = i / 16.0
            wp = self.current_start_pos + (self.current_target - self.current_start_pos) * ratio
            self.waypoints.append(wp)
        self.passed_waypoints_mask = np.zeros(16, dtype=bool)
