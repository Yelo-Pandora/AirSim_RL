# Model6: Hierarchical A* + TD3 Navigation

Model6 keeps Model3's hierarchical idea, but replaces the learned upper local
target generator with an A* planner and uses Model1's mature TD3 policy as the
lower action controller.

## Architecture

1. Upper decision network: occupancy-grid A*.
   - Input: global start and global goal coordinates.
   - Map source: prebuilt top-down occupancy map in `topdown_depth/`.
   - Output: an automatically generated local target sequence from the A* path.

2. Lower action network: Model1 TD3.
   - Input per segment: local target `n` and local target `n+1`.
   - Execution: reset Model1's AirSim environment for that segment and let TD3
     control the UAV until it reaches the next local target.

3. Full navigation repeats local TD3 segments until the global goal is reached
   or a segment fails.

`dataset/relative_coordinates_export.csv` is no longer the default source of
intermediate targets.  It remains available only as a comparison fallback via
`--planner csv`.

## Usage

Plan only:

```powershell
python navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --plan-only
```

Compare against the old CSV waypoint-graph planner:

```powershell
python navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --plan-only --planner csv
```

Run full AirSim navigation:

```powershell
python navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5
```

Use a specific Model1 TD3 checkpoint:

```powershell
python navigate_hierarchical.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5 --td3-model ..\Model1\checkpoints\td3_arrived10_418754_steps.zip
```

Intermediate local targets use a relaxed arrival rule: each xyz-axis error must
be within `0.8m`.  The final goal keeps a stricter final-segment rule.

When AirSim validation is enabled, candidate local targets that are too close to
segmentation obstacle objects or locally surrounded by obstacle centers are
removed before A* builds the upper graph.

The default occupancy planner does not depend on those candidate CSV points.  It
builds a 2D AirSim occupancy grid, runs A*, then extracts local targets from
turning points and spacing along the grid path.

Offline occupancy-map construction and review scripts live in
`occupancy_tools/`. They are not called by the runtime navigation pipeline.
