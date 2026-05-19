# Model5: A* AirSim Navigation Baseline

Model5 is a non-learning baseline for the same AirSim + UE5 scene used by the
other models.  It uses Model2's segmentation convention:

- ID 1: traversable road-like surfaces (`street`, `sidewalk`, `curb`)
- ID 2: blocked objects (`building`, furniture, signs, lamps, plants, etc.)
- ID 0: unknown/background

The navigator builds a 2D occupancy grid in AirSim world coordinates.  It asks
AirSim for scene objects, reads each object's segmentation ID, projects obstacle
objects into the grid, inflates them by the UAV safety margin, then runs A*.
The final path is simplified to turning points and followed with world-frame
velocity commands.

Run from `Network/Model5`:

```powershell
python navigate_astar.py
```

Or provide a manual task:

```powershell
python navigate_astar.py --start -51.3 1.48 -5 --goal -22.1 110.98 -5
```

This baseline is deliberately conservative.  A single front segmentation image
is not enough for global A*, so segmentation IDs and scene-object poses are used
to build the global map; image segmentation remains useful for future local
replanning and debugging.

