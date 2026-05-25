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

# Add Model1 to path for TD3 components
MODEL1_DIR = os.path.join(REPO_ROOT, "Network", "Model1")
if MODEL1_DIR not in sys.path:
    sys.path.insert(0, MODEL1_DIR)

# Add Model5 to path for A* components
MODEL5_DIR = os.path.join(REPO_ROOT, "Network", "Model5")
if MODEL5_DIR not in sys.path:
    sys.path.insert(0, MODEL5_DIR)

from reinforcement_network import AirSimUAVEnv
from astar import astar, simplify_path
from grid_map import OccupancyGrid
from segmentation_rules import SegmentationRules

class NavPipeline:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(MODEL1_DIR, "checkpoints", "td3_arrived10_308804_steps.zip")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else "cpu"
        self.env = None
        self.model = None
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
        """Initialize environment and load model."""
        if torch is None or TD3 is None:
            return False, "Dependencies missing: torch or stable-baselines3 not found."
            
        try:
            if self.env is None:
                self.env = AirSimUAVEnv()
            
            if self.model is None:
                print(f"Loading model from {self.model_path}...")
                self.model = TD3.load(self.model_path, env=self.env, device=self.device)
            return True, "Navigation initialized."
        except Exception as e:
            return False, f"Failed to initialize: {e}"

    def plan_path(self, start_pos, end_pos):
        """Run A* to generate waypoints with fallback inflation levels."""
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            segmentation = SegmentationRules(client)
            segmentation.apply()
            
            import config
            
            # Collect obstacles
            obstacle_names = segmentation.obstacle_objects()
            obstacle_centers = []
            for name in obstacle_names:
                try:
                    pose = client.simGetObjectPose(name)
                    if math.isfinite(pose.position.x_val) and (pose.position.x_val != 0 or pose.position.y_val != 0):
                        obstacle_centers.append((pose.position.x_val, pose.position.y_val))
                except:
                    continue
            
            # Try multiple inflation levels from config
            for radius, margin in config.PLANNING_RADIUS_SCHEDULE:
                grid = OccupancyGrid(
                    config.MAP_X_MIN, config.MAP_X_MAX,
                    config.MAP_Y_MIN, config.MAP_Y_MAX,
                    config.MAP_RESOLUTION
                )
                
                for x, y in obstacle_centers:
                    grid.mark_disc((x, y), radius)
                
                inflated = grid.inflated_copy(margin)
                
                # Snap start/end to grid
                start_cell = inflated.world_to_cell(start_pos)
                end_cell = inflated.world_to_cell(end_pos)
                
                # If start/end is occupied, find nearest free cell
                start_cell = inflated.nearest_free(start_cell, config.NEAREST_FREE_SEARCH_RADIUS)
                end_cell = inflated.nearest_free(end_cell, config.NEAREST_FREE_SEARCH_RADIUS)
                    
                if start_cell is None or end_cell is None:
                    continue # Try next inflation level

                cells = astar(inflated, start_cell, end_cell)
                if cells is not None:
                    cells = simplify_path(cells, grid=inflated)
                    waypoints = [inflated.cell_to_world(c, z=start_pos[2]) for c in cells]
                    waypoints[-1] = end_pos.astype(np.float32)
                    return waypoints, f"Path planned successfully (Radius: {radius}m)."
            
            return None, "A* failed to find a path even with minimum inflation."
            
        except Exception as e:
            return None, f"Planning error: {e}"

    def run_navigation(self, waypoints, status_callback=None):
        """Execute navigation through waypoints using TD3 with waypoint interpolation."""
        if torch is None or TD3 is None:
            return False, "Dependencies missing: torch or stable-baselines3 not found."
            
        if self.model is None or self.env is None:
            return False, "Navigation not initialized."
        
        try:
            # Step 1: Interpolate waypoints if they are too far apart
            dense_waypoints = [waypoints[0]]
            max_dist = 10.0 # Maximum 10 meters per segment
            for i in range(len(waypoints) - 1):
                p1 = waypoints[i]
                p2 = waypoints[i+1]
                dist = np.linalg.norm(p2 - p1)
                if dist > max_dist:
                    num_steps = int(np.ceil(dist / max_dist))
                    for s in range(1, num_steps):
                        interp_p = p1 + (p2 - p1) * (s / num_steps)
                        dense_waypoints.append(interp_p)
                dense_waypoints.append(p2)
            
            if status_callback:
                status_callback(f"Path densified: {len(waypoints)} -> {len(dense_waypoints)} points")
                status_callback("Resetting environment and stabilizing...")
            
            obs, _ = self.env.reset(options={
                "start_pos": dense_waypoints[0],
                "target": dense_waypoints[1]
            })
            
            # Step 2: Sequential navigation
            for i in range(1, len(dense_waypoints)):
                target = dense_waypoints[i]
                if status_callback:
                    status_callback(f"Navigating to point {i}/{len(dense_waypoints)-1}: {np.round(target, 1)}")
                
                # Update environment internal state for this segment
                self.env.current_target = np.array(target, dtype=np.float32)
                self.env.current_start_pos = np.array(dense_waypoints[i-1], dtype=np.float32)
                self.env.start_dist = float(np.linalg.norm(self.env.current_target - self.env.current_start_pos))
                self.env.last_dis2goal = self.env.start_dist
                self.env.step_count = 0 # Reset step counter to avoid timeout
                
                # Re-generate internal waypoints for TD3's potential logic
                self.env.waypoints = []
                for w_idx in range(1, 17):
                    self.env.waypoints.append(self.env.current_start_pos + (self.env.current_target - self.env.current_start_pos) * (w_idx / 16.0))
                self.env.passed_waypoints_mask = np.zeros(16, dtype=bool)
                
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # Smooth handoff
                    is_last_wp = (i == len(dense_waypoints) - 1)
                    arrival_threshold = 1.5 if is_last_wp else 3.0
                    
                    if info.get('xy_dist', 100) < arrival_threshold:
                        break
                    
                    if info.get('collision', False):
                        return False, f"Collision detected at segment {i}!"
                    
                    if terminated or truncated:
                        if not info.get('arrived', False):
                            return False, f"Failed at segment {i}: {info.get('end_reason', 'unknown')}"
                        break
                
            return True, "Destination reached successfully!"
        except Exception as e:
            return False, f"Navigation error: {e}"
