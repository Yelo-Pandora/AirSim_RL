import numpy as np

import config


def randomize_intermediate_target_altitudes(plan):
    """Randomize intermediate local target altitude for safer cruise."""
    if not config.LOCAL_TARGET_RANDOMIZE_INTERMEDIATE_ALTITUDE:
        return plan

    points = plan.get("points", [])
    if len(points) <= 2:
        return plan

    min_altitude, max_altitude = config.LOCAL_TARGET_INTERMEDIATE_ALTITUDE_RANGE
    min_altitude = float(min_altitude)
    max_altitude = float(max_altitude)
    if min_altitude > max_altitude:
        min_altitude, max_altitude = max_altitude, min_altitude

    adjusted_points = [
        np.array(point, dtype=np.float32).copy()
        for point in points
    ]
    rng = np.random.default_rng()
    altitudes = rng.uniform(
        min_altitude,
        max_altitude,
        size=len(adjusted_points) - 2,
    )
    for point, altitude in zip(adjusted_points[1:-1], altitudes):
        point[2] = float(config.OCCUPANCY_GROUND_Z - altitude)

    adjusted_plan = dict(plan)
    adjusted_plan["points"] = adjusted_points
    adjusted_plan["path_length"] = path_length(adjusted_points)
    return adjusted_plan


def format_local_target_altitudes(points):
    """Return a short readable summary of local target z/altitude values."""
    rows = []
    for index, point in enumerate(points):
        z = float(point[2])
        altitude = float(config.OCCUPANCY_GROUND_Z - z)
        rows.append(f"{index}:{z:.2f}({altitude:.2f}m)")
    return ", ".join(rows)


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            np.linalg.norm(points[index] - points[index - 1])
            for index in range(1, len(points))
        )
    )
