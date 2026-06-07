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

        # Phase 1: quick center-based rejection (increased clearance)
        clearance = config.WAYPOINT_OBSTACLE_CLEARANCE
        near_mask = (xy_distances <= clearance) & (
            z_distances <= max(config.GRAPH_MAX_Z_DIFF, clearance)
        )
        if np.any(near_mask):
            return False

        # Phase 2: surrounded check using obstacle centers
        if self._is_surrounded(point, xy_distances, deltas):
            return False

        # Phase 3: ray-cast clearance validation
        # Cast real collision rays from the waypoint to verify actual free space
        return self._ray_cast_clearance(point)

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

    def _ray_cast_clearance(self, point, min_free_dist=None):
        """
        Use AirSim's simTestLineOfSightBetweenPoints to verify free space
        around the candidate waypoint in 8 directions.

        We test from the waypoint to a point at min_free_dist in each direction.
        If ANY direction is blocked, the waypoint is unsafe.

        This replaces the old simCastRay approach (which was not available in
        the AirSim Python API and silently swallowed by the except clause).
        """
        if min_free_dist is None:
            min_free_dist = config.WAYPOINT_RAY_CLEARANCE_DIST

        import airsim

        num_rays = config.WAYPOINT_SURROUNDED_RAYS
        px, py, pz = float(point[0]), float(point[1]), float(point[2])

        for ray_index in range(num_rays):
            theta = 2.0 * math.pi * ray_index / num_rays
            test_point = airsim.GeoPoint(
                latitude=px + math.cos(theta) * min_free_dist,
                longitude=py + math.sin(theta) * min_free_dist,
                altitude=pz,
            )
            try:
                has_los = self.client.simTestLineOfSightToPoint(test_point)
                if not has_los:
                    return False
            except Exception:
                # If the API call fails, conservatively reject
                return False
        return True

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

        try:
            all_names = self.client.simListSceneObjects(".*")
        except Exception:
            return sorted(names)

        for name in all_names:
            try:
                if int(self.client.simGetSegmentationObjectID(name)) == int(config.OBSTACLE_SEGMENTATION_ID):
                    names.add(name)
                    continue
            except Exception:
                pass
            if _matches_any(name, config.OBSTACLE_OBJECT_PATTERNS):
                names.add(name)
        return sorted(names)
