# Model7: Hierarchical A* + External PPO Navigation

Model7 is the comparison baseline for Model6.  It keeps Model6's upper
occupancy-grid A* planner, including clearance and waypoint-uniformity logic,
but replaces the lower Model1 TD3 executor with the pretrained PPO image policy
from:

<https://github.com/bilalkabas/PPO-based-Autonomous-Navigation-for-Quadcopters>

The external repository is kept under `external/` and its pretrained checkpoint
is loaded from `saved_policy/ppo_navigation_policy.zip`.

## Key Difference From Model6

- Upper layer: same Model6 A* local target planner.
- Lower layer: external PPO `CnnPolicy` with 50x50 RGB camera observations.
- Segment execution: before each local segment, the UAV yaw is aligned toward
  the next local target so PPO's body-frame forward action roughly follows the
  segment direction.

The PPO model is not goal-conditioned; it was trained for a corridor/hole task.
Model7 therefore measures how that pretrained visual policy transfers under the
same upper planner, rather than claiming PPO has the same target-aware input as
our TD3 lower controller.

## Usage

Plan only:

```powershell
python Network\Model7\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --plan-only
```

Run full AirSim navigation:

```powershell
python Network\Model7\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5
```

Use a specific PPO checkpoint:

```powershell
python Network\Model7\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --ppo-model external\PPO-based-Autonomous-Navigation-for-Quadcopters\saved_policy\ppo_navigation_policy.zip
```

## Requirements

Model7 expects `stable_baselines3`, `torch`, `gym`, and `airsim` to be
available.  The referenced PPO project was built around `stable-baselines3==1.2.0`
and `gym==0.21.0`, so checkpoint compatibility may depend on the local package
versions.

If current SB3 asks for `shimmy`, Model7 includes a small local compatibility
shim under `Network/Model7/shimmy/` that converts the old Gym action and
observation spaces when loading the pretrained checkpoint.
