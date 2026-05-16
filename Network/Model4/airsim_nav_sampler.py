import io
import json
import math
import os
import time

import numpy as np
from PIL import Image

import config


def get_ray_directions(hfov=90.0, vfov=60.0, rays_h=5, rays_v=5):
    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)
    h_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, rays_h)
    v_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, rays_v)

    directions = []
    for h in h_angles:
        for v in v_angles:
            d = np.array(
                [
                    np.cos(h) * np.cos(v),
                    np.sin(h) * np.cos(v),
                    np.sin(v),
                ],
                dtype=np.float32,
            )
            directions.append(d / max(np.linalg.norm(d), 1e-6))
    return np.array(directions, dtype=np.float32)


class AirSimNavigationSampler:
    """
    Sample feasible mid-range tasks directly from the live AirSim scene.

    This class is intentionally independent from Model3 so Model4 can feed
    start/goal tasks into Model1 without depending on RLoPlanner code.
    """

    def __init__(self, client, vehicle_name="Drone1"):
        self.client = client
        self.vehicle_name = vehicle_name
        self.points = []
        self.bounds = None
        self.allowed_surface_colors, self.blocked_surface_colors = self._collect_surface_reference_colors()
        self.ray_directions = get_ray_directions(
            hfov=config.RANGE_HFOV,
            vfov=config.RANGE_VFOV,
            rays_h=config.RANGE_RAYS_H,
            rays_v=config.RANGE_RAYS_V,
        )

    def load_or_build(self, force_rebuild=False):
        force_rebuild = force_rebuild or getattr(config, "AIRSIM_NAV_FORCE_REBUILD", False)
        if not force_rebuild and self._load_cache():
            return self.points

        self.build_cache()
        self._save_cache()
        return self.points

    def build_cache(self):
        import airsim

        self.points = []
        self.bounds = self._infer_bounds()
        try:
            original_pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        except Exception:
            original_pose = None

        try:
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        except Exception:
            pass

        max_attempts = int(getattr(config, "AIRSIM_NAV_MAX_ATTEMPTS", 4000))
        target_count = int(getattr(config, "AIRSIM_NAV_CACHE_POINTS", 160))
        min_points = int(getattr(config, "AIRSIM_NAV_MIN_POINTS", 60))

        for attempt in range(max_attempts):
            if len(self.points) >= target_count:
                break

            candidate = self._sample_candidate_point()
            ok, meta = self._probe_point(candidate)
            if not ok:
                continue

            if self._is_too_close_to_existing(meta["point"]):
                continue

            self.points.append(meta)
            if len(self.points) % 10 == 0:
                print(
                    f"[MODEL4] Probed {attempt + 1}/{max_attempts}, "
                    f"accepted {len(self.points)}/{target_count}"
                )

        if original_pose is not None:
            try:
                self.client.simSetVehiclePose(original_pose, True, vehicle_name=self.vehicle_name)
            except Exception:
                pass

        if len(self.points) < min_points:
            raise RuntimeError(
                f"Only found {len(self.points)} feasible points; "
                f"need at least {min_points}."
            )

        print(
            f"[MODEL4] Built {len(self.points)} feasible mid-goal points "
            f"within x[{self.bounds['x_min']:.1f}, {self.bounds['x_max']:.1f}] "
            f"y[{self.bounds['y_min']:.1f}, {self.bounds['y_max']:.1f}]"
        )

    def sample_start_goal(self):
        if len(self.points) < 2:
            raise RuntimeError("Point cache has fewer than 2 points.")

        min_dist = float(getattr(config, "AIRSIM_NAV_MIN_PAIR_DISTANCE", 12.0))
        max_dist = float(getattr(config, "AIRSIM_NAV_MAX_PAIR_DISTANCE", 35.0))

        for _ in range(500):
            start = self.points[np.random.randint(len(self.points))]
            start_point = np.array(start["point"], dtype=np.float32)
            candidates = []

            for goal in self.points:
                if goal["region"] == start["region"]:
                    continue
                goal_point = np.array(goal["point"], dtype=np.float32)
                dist = float(np.linalg.norm(goal_point - start_point))
                if min_dist <= dist <= max_dist:
                    candidates.append(goal)

            if candidates:
                goal = candidates[np.random.randint(len(candidates))]
                return {
                    "start": start_point.copy(),
                    "target": np.array(goal["point"], dtype=np.float32).copy(),
                    "start_region": start["region"],
                    "goal_region": goal["region"],
                    "distance": float(np.linalg.norm(np.array(goal["point"]) - start_point)),
                }

        raise RuntimeError("Failed to sample a valid mid-range start/goal pair.")

    def _load_cache(self):
        if not os.path.exists(config.AIRSIM_NAV_CACHE_JSON):
            return False
        try:
            with open(config.AIRSIM_NAV_CACHE_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        points = payload.get("points", [])
        if len(points) < int(getattr(config, "AIRSIM_NAV_MIN_POINTS", 60)):
            return False

        self.points = points
        self.bounds = payload.get("bounds")
        print(f"[MODEL4] Loaded {len(points)} cached points from {config.AIRSIM_NAV_CACHE_JSON}")
        return True

    def _save_cache(self):
        os.makedirs(os.path.dirname(config.AIRSIM_NAV_CACHE_JSON), exist_ok=True)
        payload = {"bounds": self.bounds, "points": self.points}
        with open(config.AIRSIM_NAV_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[MODEL4] Saved cache to {config.AIRSIM_NAV_CACHE_JSON}")

    def _infer_bounds(self):
        object_names = []
        try:
            object_names = self.client.simListSceneObjects(".*")
        except Exception:
            object_names = []

        sample_cap = int(getattr(config, "AIRSIM_NAV_OBJECT_SAMPLE", 800))
        if object_names and len(object_names) > sample_cap:
            sample_idx = np.random.choice(len(object_names), size=sample_cap, replace=False)
            object_names = [object_names[i] for i in sample_idx]

        xs = []
        ys = []
        for name in object_names:
            try:
                pose = self.client.simGetObjectPose(name)
                x = float(pose.position.x_val)
                y = float(pose.position.y_val)
                z = float(pose.position.z_val)
                if not all(math.isfinite(v) for v in (x, y, z)):
                    continue
                if abs(x) + abs(y) + abs(z) < 1e-6:
                    continue
                xs.append(x)
                ys.append(y)
            except Exception:
                continue

        if len(xs) >= 20 and len(ys) >= 20:
            low = float(getattr(config, "AIRSIM_NAV_BOUNDS_PERCENTILE_LOW", 2.0))
            high = float(getattr(config, "AIRSIM_NAV_BOUNDS_PERCENTILE_HIGH", 98.0))
            margin = float(getattr(config, "AIRSIM_NAV_BOUNDS_MARGIN", 8.0))
            return {
                "x_min": float(np.percentile(xs, low) - margin),
                "x_max": float(np.percentile(xs, high) + margin),
                "y_min": float(np.percentile(ys, low) - margin),
                "y_max": float(np.percentile(ys, high) + margin),
            }

        try:
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            center_x = float(pose.position.x_val)
            center_y = float(pose.position.y_val)
        except Exception:
            center_x = 0.0
            center_y = 0.0

        radius = float(getattr(config, "AIRSIM_NAV_FALLBACK_RADIUS", 60.0))
        return {
            "x_min": center_x - radius,
            "x_max": center_x + radius,
            "y_min": center_y - radius,
            "y_max": center_y + radius,
        }

    def _sample_candidate_point(self):
        x = np.random.uniform(self.bounds["x_min"], self.bounds["x_max"])
        y = np.random.uniform(self.bounds["y_min"], self.bounds["y_max"])
        z = float(np.random.choice(getattr(config, "AIRSIM_NAV_ALTITUDES", (-5.0,))))
        return np.array([x, y, z], dtype=np.float32)

    def _probe_point(self, point):
        import airsim

        pose = airsim.Pose(
            airsim.Vector3r(float(point[0]), float(point[1]), float(point[2])),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        try:
            self.client.simSetVehiclePose(pose, True, vehicle_name=self.vehicle_name)
            time.sleep(float(getattr(config, "AIRSIM_NAV_PROBE_SETTLE_SEC", 0.08)))
        except Exception:
            return False, None

        try:
            if self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided:
                return False, None
        except Exception:
            return False, None

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        actual_point = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

        surface_label, surface_color = self._probe_surface_label()
        if surface_label not in {"street", "sidewalk", "curb"}:
            return False, None

        clearance = self._probe_clearance()
        if clearance is None or clearance < float(getattr(config, "AIRSIM_NAV_POINT_CLEARANCE", 3.0)):
            return False, None

        bottom_distance = self._probe_bottom_distance()
        if bottom_distance is None:
            return False, None

        if not (
            float(getattr(config, "AIRSIM_NAV_BOTTOM_MIN", 2.0))
            <= bottom_distance
            <= float(getattr(config, "AIRSIM_NAV_BOTTOM_MAX", 12.0))
        ):
            return False, None

        return True, {
            "point": [float(actual_point[0]), float(actual_point[1]), float(actual_point[2])],
            "region": self._region_label(actual_point),
            "surface": surface_label,
            "surface_color": list(surface_color) if surface_color is not None else None,
            "clearance": float(clearance),
            "bottom_distance": float(bottom_distance),
        }

    def _collect_surface_reference_colors(self):
        """
        Learn which TopDown segmentation colors correspond to road-like surfaces
        versus building surfaces using scene-object names as weak supervision.
        """
        sample_cap = 24
        allowed = {}
        blocked = set()

        try:
            object_names = self.client.simListSceneObjects(".*")
        except Exception:
            object_names = []

        allowed_specs = [
            ("street", [name for name in object_names if "street" in name.lower()]),
            ("sidewalk", [name for name in object_names if "sidewalk" in name.lower()]),
            ("curb", [name for name in object_names if "curb" in name.lower()]),
        ]
        building_names = [name for name in object_names if "building" in name.lower()]

        for label, names in allowed_specs:
            np.random.shuffle(names)
            for name in names[:sample_cap]:
                color = self._sample_surface_color_from_object(name)
                if color is None:
                    continue
                allowed.setdefault(color, set()).add(label)

        np.random.shuffle(building_names)
        for name in building_names[:sample_cap]:
            color = self._sample_surface_color_from_object(name)
            if color is not None:
                blocked.add(color)

        print(
            f"[MODEL4] Learned {len(allowed)} allowed top-down colors and "
            f"{len(blocked)} blocked building colors"
        )
        return allowed, blocked

    def _sample_surface_color_from_object(self, object_name):
        import airsim

        try:
            pose = self.client.simGetObjectPose(object_name)
        except Exception:
            return None

        x = float(pose.position.x_val)
        y = float(pose.position.y_val)
        z = float(pose.position.z_val)
        if not all(math.isfinite(v) for v in (x, y, z)):
            return None

        probe_pose = airsim.Pose(
            airsim.Vector3r(x, y, -5.0),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        try:
            self.client.simSetVehiclePose(probe_pose, True, vehicle_name=self.vehicle_name)
            time.sleep(float(getattr(config, "AIRSIM_NAV_PROBE_SETTLE_SEC", 0.08)))
        except Exception:
            return None

        return self._get_topdown_center_color()

    def _get_topdown_center_color(self):
        import airsim

        try:
            response = self.client.simGetImages(
                [airsim.ImageRequest("TopDown", airsim.ImageType.Segmentation, False, True)],
                vehicle_name=self.vehicle_name,
            )[0]
            if not response.image_data_uint8:
                return None

            image = np.array(Image.open(io.BytesIO(response.image_data_uint8)))
            if image.ndim != 3:
                return None

            center = image[image.shape[0] // 2, image.shape[1] // 2]
            color = tuple(int(v) for v in center[:3].tolist())
            if color == (0, 0, 0):
                return None
            return color
        except Exception:
            return None

    def _probe_surface_label(self):
        color = self._get_topdown_center_color()
        if color is None:
            return "unknown", None
        if color in self.blocked_surface_colors:
            return "building", color

        labels = self.allowed_surface_colors.get(color)
        if not labels:
            return "unknown", color
        if "street" in labels:
            return "street", color
        if "sidewalk" in labels:
            return "sidewalk", color
        if "curb" in labels:
            return "curb", color
        return "unknown", color

    def _probe_clearance(self):
        lidar_data = None
        lidar_names = (config.PRIMARY_LIDAR_NAME,) + tuple(config.FALLBACK_LIDAR_NAMES)
        for lidar_name in lidar_names:
            try:
                lidar_data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=self.vehicle_name)
                if lidar_data and len(lidar_data.point_cloud) >= 3:
                    break
                lidar_data = None
            except Exception:
                lidar_data = None

        if lidar_data is None:
            return None

        distances = self._project_lidar_to_rays(lidar_data.point_cloud)
        if len(distances) == 0:
            return None
        return float(np.min(distances))

    def _project_lidar_to_rays(self, point_cloud):
        distances = np.full(len(self.ray_directions), config.RANGE_MAX, dtype=np.float32)
        points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return distances

        h_step = np.radians(config.RANGE_HFOV) / max(config.RANGE_RAYS_H - 1, 1)
        v_step = np.radians(config.RANGE_VFOV) / max(config.RANGE_RAYS_V - 1, 1)
        half_diag = np.sqrt((0.5 * h_step) ** 2 + (0.5 * v_step) ** 2) * config.RANGE_MATCH_MARGIN
        cosine_threshold = np.cos(half_diag)

        ray_dirs = self.ray_directions / np.maximum(
            np.linalg.norm(self.ray_directions, axis=1, keepdims=True),
            1e-6,
        )
        for pt in points:
            dist = float(np.linalg.norm(pt))
            if dist < 1e-3 or dist > config.RANGE_MAX:
                continue
            direction = pt / dist
            dots = ray_dirs @ direction
            best_idx = int(np.argmax(dots))
            if dots[best_idx] >= cosine_threshold and dist < distances[best_idx]:
                distances[best_idx] = dist
        return distances

    def _probe_bottom_distance(self):
        for sensor_name in ("DistanceBottom", "DistanceBottom1"):
            try:
                data = self.client.getDistanceSensorData(
                    distance_sensor_name=sensor_name,
                    vehicle_name=self.vehicle_name,
                )
                distance = float(data.distance)
                if math.isfinite(distance) and distance > 0.0:
                    return distance
            except Exception:
                continue
        return None

    def _is_too_close_to_existing(self, point):
        point = np.array(point, dtype=np.float32)
        min_sep = float(getattr(config, "AIRSIM_NAV_MIN_POINT_SEPARATION", 8.0))
        for item in self.points:
            existing = np.array(item["point"], dtype=np.float32)
            if np.linalg.norm(existing - point) < min_sep:
                return True
        return False

    def _region_label(self, point):
        bucket = float(getattr(config, "AIRSIM_NAV_REGION_BUCKET", 20.0))
        ix = int(np.floor(float(point[0]) / bucket))
        iy = int(np.floor(float(point[1]) / bucket))
        iz = int(np.round(float(point[2])))
        return f"cell_{ix}_{iy}_{iz}"
