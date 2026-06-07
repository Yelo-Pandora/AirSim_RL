import os
import sys
import time
import subprocess
import math
import numpy as np
import airsim

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Late import for torch and SB3 to avoid immediate failure if missing
try:
    import torch
    from stable_baselines3 import TD3
except ImportError:
    torch = None
    TD3 = None

# Add TD3_base and Astar_planner to path
MODEL1_DIR = os.path.join(REPO_ROOT, "Network", "TD3_base")
MODEL6_DIR = os.path.join(REPO_ROOT, "Network", "Astar_planner")
if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)
if MODEL6_DIR not in sys.path:
    sys.path.insert(0, MODEL6_DIR)

from reinforcement_network import AirSimUAVEnv
from graph_planner import WaypointGraphPlanner
from local_target_utils import (
    format_local_target_altitudes,
    randomize_intermediate_target_altitudes,
)
from occupancy_planner import OccupancyAStarPlanner
from td3_executor import TD3SegmentExecutor
import config as model6_config

class NavPipeline:
    def __init__(self, model_path=None, planner_mode=None):
        self.model_path = model_path
        self.planner_mode = planner_mode or model6_config.UPPER_PLANNER
        self.executor = None
        self.planner = None
        self.airsim_process = None
        
    def launch_airsim(self):
        """Launch the AirSim executable."""
        exe_path = os.path.join(REPO_ROOT, "AirSimDemo", "Windows", "BlocksV2.exe")
        if not os.path.exists(exe_path):
            return False, f"Executable not found at {exe_path}"
        
        try:
            # Check if AirSim is already running
            try:
                import psutil
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] == 'BlocksV2.exe':
                        return True, "AirSim is already running."
            except ImportError:
                pass
            
            self.airsim_process = subprocess.Popen([exe_path])
            return True, "AirSim launch command sent."
        except Exception as e:
            return False, f"Failed to launch AirSim: {e}"

    def init_navigation(self):
        """Initialize environment and load model using Astar_planner hierarchical components."""
        if torch is None or TD3 is None:
            return False, "Dependencies missing: torch or stable-baselines3 not found."
            
        try:
            # Initialize Astar_planner executor
            self.executor = TD3SegmentExecutor(model_path=self.model_path)

            safety = None
            needs_airsim_safety = model6_config.VALIDATE_WAYPOINTS_WITH_AIRSIM
            if needs_airsim_safety:
                from waypoint_safety import AirSimWaypointSafety
                safety = AirSimWaypointSafety(self.executor.env.client)

            if self.planner_mode == "occupancy":
                self.planner = OccupancyAStarPlanner(
                    client=self.executor.env.client,
                )
            elif model6_config.VALIDATE_WAYPOINTS_WITH_AIRSIM:
                self.planner = WaypointGraphPlanner(waypoint_filter=safety.is_safe)
            else:
                self.planner = WaypointGraphPlanner()
            
            return True, f"Hierarchical navigation initialized (Astar_planner, planner={self.planner_mode})."
        except Exception as e:
            return False, f"Failed to initialize: {e}"

    def plan_path(self, start_pos, end_pos):
        """Run the configured Astar_planner upper planner."""
        if self.planner is None:
            return None, "Planner not initialized."
            
        try:
            plan = self.planner.plan(start_pos, end_pos)
            plan = randomize_intermediate_target_altitudes(plan)
            waypoints = plan["points"]
            return waypoints, (
                f"Path planned via {plan.get('planner', self.planner_mode)} "
                f"(Points: {len(waypoints)}, Dist: {plan['path_length']:.1f}m). "
                f"Local target z/alt: {format_local_target_altitudes(waypoints)}"
            )
        except Exception as e:
            return None, f"Planning error: {e}"

    def run_navigation(self, waypoints, status_callback=None):
        """Execute navigation through waypoints using Lower TD3 Executor (Astar_planner)."""
        if self.executor is None:
            return False, "Executor not initialized."
        
        try:
            if status_callback:
                status_callback(f"Starting hierarchical navigation for {len(waypoints)} points...")
            
            summaries = self.executor.execute_path(waypoints)
            
            # Check results
            arrived_segments = sum(1 for item in summaries if item["arrived"])
            total_segments = len(waypoints) - 1
            
            if arrived_segments == total_segments:
                return True, f"Success: Reached destination ({arrived_segments}/{total_segments} segments)."
            else:
                failed_idx = len(summaries) - 1
                reason = summaries[-1]["end_reason"]
                return False, f"Failed at segment {failed_idx}: {reason} ({arrived_segments}/{total_segments} reached)."
                
        except Exception as e:
            return False, f"Navigation error: {e}"

    def run_batch_test(self, task_list, status_callback=None):
        """
        Run multiple navigation tasks and calculate arrival rate.
        task_list: list of dicts [{'start': [x,y,z], 'end': [x,y,z]}, ...]
        """
        results = []
        total = len(task_list)
        arrived_count = 0
        
        if status_callback:
            status_callback(f"Starting batch test for {total} tasks...")
            
        for idx, task in enumerate(task_list):
            start = np.array(task['start'])
            end = np.array(task['end'])
            
            if status_callback:
                status_callback(f"\n--- Task {idx+1}/{total} ---")
                status_callback(f"From {start} to {end}")
            
            # 1. Plan
            waypoints, msg = self.plan_path(start, end)
            if not waypoints:
                if status_callback:
                    status_callback(f"Task {idx+1} failed at planning: {msg}")
                results.append({'task_id': idx+1, 'status': 'planning_failed', 'msg': msg})
                continue
            if status_callback:
                status_callback(msg)
            
            # 2. Navigate
            success, nav_msg = self.run_navigation(waypoints, status_callback=None) # Keep quiet during batch
            
            if success:
                arrived_count += 1
                status = 'success'
            else:
                status = 'failed'
                
            if status_callback:
                status_callback(f"Task {idx+1} result: {status} ({nav_msg})")
                
            results.append({
                'task_id': idx+1,
                'start': start.tolist(),
                'end': end.tolist(),
                'status': status,
                'msg': nav_msg
            })
            
        arrival_rate = (arrived_count / total) * 100 if total > 0 else 0
        summary = {
            'total': total,
            'arrived': arrived_count,
            'arrival_rate': f"{arrival_rate:.2f}%",
            'details': results
        }
        
        return summary
