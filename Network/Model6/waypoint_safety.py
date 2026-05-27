import math

import numpy as np

import config


def _matches_any(name, patterns):
    lower = name.lower()
    return any(pattern in lower for pattern in patterns)


class AirSimWaypointSafety:
    """Filter upper-layer waypoints that are too close to segmented obstacles."""

    def __init__(self, client):
        self.client = client
        self.obstacle_centers = self._collect_obstacle_centers()
        print(f"[Model6] Waypoint safety: collected {len(self.obstacle_centers)} obstacle centers")

    def is_safe(self, point):
        point = np.asarray(point, dtype=np.float32)
        if len(self.obstacle_centers) == 0:
            return True

        deltas = self.obstacle_centers[:, :2] - point[:2]
        xy_distances = np.linalg.norm(deltas, axis=1)
        z_distances = np.abs(self.obstacle_centers[:, 2] - point[2])

        near_mask = (xy_distances <= config.WAYPOINT_OBSTACLE_CLEARANCE) & (
            z_distances <= max(config.GRAPH_MAX_Z_DIFF, config.WAYPOINT_OBSTACLE_CLEARANCE)
        )
        if np.any(near_mask):
            return False

        return not self._is_surrounded(point, xy_distances, deltas)

    def _is_surrounded(self, point, xy_distances, deltas):
        local = deltas[xy_distances <= config.WAYPOINT_SURROUNDED_RADIUS]
        if len(local) == 0:
            return False

        free_directions = 0
        for ray_index in range(config.WAYPOINT_SURROUNDED_RAYS):
            theta = 2.0 * math.pi * ray_index / config.WAYPOINT_SURROUNDED_RAYS
            direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            projections = local @ direction
            lateral = np.abs(local[:, 0] * direction[1] - local[:, 1] * direction[0])
            blocked = np.any(
                (projections > 0.0)
                & (projections <= config.WAYPOINT_SURROUNDED_RADIUS)
                & (lateral <= config.WAYPOINT_OBSTACLE_CLEARANCE)
            )
            if not blocked:
                free_directions += 1

        return free_directions < config.WAYPOINT_SURROUNDED_MIN_FREE_DIRECTIONS

    def _collect_obstacle_centers(self):
        centers = []
        names = self._candidate_obstacle_names()
        for name in names:
            try:
                pose = self.client.simGetObjectPose(name)
            except Exception:
                continue
            x = float(pose.position.x_val)
            y = float(pose.position.y_val)
            z = float(pose.position.z_val)
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            if x == 0.0 and y == 0.0 and z == 0.0:
                continue
            centers.append([x, y, z])

        if not centers:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(centers, dtype=np.float32)

    def _candidate_obstacle_names(self):
        names = set()
        for pattern in config.OBSTACLE_OBJECT_PATTERNS:
            try:
                names.update(self.client.simListSceneObjects(f".*{pattern}.*"))
            except Exception:
                continue

        if names:
            return sorted(names)

        try:
            all_names = self.client.simListSceneObjects(".*")
        except Exception:
            return []

        blocked = []
        for name in all_names:
            if _matches_any(name, config.OBSTACLE_OBJECT_PATTERNS):
                blocked.append(name)
                continue
            try:
                if int(self.client.simGetSegmentationObjectID(name)) == 2:
                    blocked.append(name)
            except Exception:
                continue
        return sorted(blocked)
