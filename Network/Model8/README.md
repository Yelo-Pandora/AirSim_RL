# Model8: Hierarchical A* + External DDPG Navigation

Model8 is another comparison baseline for Model6.  It keeps Model6's upper
occupancy-grid A* planner, including clearance and waypoint-uniformity logic,
but replaces the lower Model1 TD3 executor with the pretrained lidar DDPG policy
from:

<https://github.com/nishantpandey4/Drone-navigation-and-obstacle-avoidance-using-DDPG>

The external repository is kept under `external/`.  The default checkpoint is:

```text
external/Drone-navigation-and-obstacle-avoidance-using-DDPG/Lidar/model/lidar_model.zip
```

## Key Difference From Model6

- Upper layer: same Model6 A* local target planner.
- Lower layer: external DDPG `MlpPolicy` trained on lidar, or lidar + depth features.
- Segment execution: each local target becomes the DDPG segment goal; Model8
  detects the checkpoint observation shape and builds either the `1x5` lidar
  observation or the `1x10` lidar + depth observation.

The external DDPG was trained in a different AirSim setup, so Model8 measures
transfer under the same upper planner rather than assuming identical training
conditions.

## Usage

Plan only:

```powershell
python Network\Model8\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --plan-only
```

Run full AirSim navigation:

```powershell
python Network\Model8\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5
```

Use a specific DDPG checkpoint:

```powershell
python Network\Model8\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --ddpg-model external\Drone-navigation-and-obstacle-avoidance-using-DDPG\Lidar\model\lidar_model.zip
python Network\Model8\navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --ddpg-model external\Drone-navigation-and-obstacle-avoidance-using-DDPG\Lidar+Depth\model\drone_depth_lidar.zip
```

Observation adaptation is automatic: `(1, 5)` uses
`[front_lidar, left_lidar, right_lidar, distance_to_goal, vertical_offset]`;
`(1, 10)` prepends five depth-image sector features before state and lidar.

## Requirements

Model8 expects `stable_baselines3`, `torch`, `gym`, and `airsim` to be
available.  The referenced DDPG project was built around `stable-baselines3==1.7.0`
and `gym==0.21.0`, so checkpoint compatibility may depend on local package
versions.

If current SB3 asks for `shimmy`, Model8 includes a small local compatibility
shim under `Network/Model8/shimmy/` that converts old Gym spaces when loading
the pretrained checkpoint.  The DDPG loader also overrides old serialized
schedule objects via `custom_objects` for compatibility with the current SB3
runtime.
