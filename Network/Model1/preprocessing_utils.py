import numpy as np
import torch
import torch.nn.functional as F
import cv2


LIDAR_EMPTY_VALUE = 20.0
LIDAR_FRONT_INDICES = list(range(45, 135, 2))
LIDAR_SIDE_LEFT_INDICES = list(range(0, 45, 3))
LIDAR_SIDE_RIGHT_INDICES = list(range(135, 180, 3))
LIDAR_REAR_INDICES = list(range(180, 360, 6))
LIDAR_105_INDICES = (
    LIDAR_FRONT_INDICES
    + LIDAR_SIDE_LEFT_INDICES
    + LIDAR_SIDE_RIGHT_INDICES
    + LIDAR_REAR_INDICES
)


def decode_depth_planar(response):
    depth_arr = np.array(response.image_data_float, dtype=np.float32)
    expected_size = response.height * response.width
    if depth_arr.size != expected_size:
        raise ValueError(
            f"Depth 数据长度不匹配: got={depth_arr.size}, expected={expected_size}"
        )
    return depth_arr.reshape(response.height, response.width)


def resize_depth_for_ldtd3(depth_img):
    return cv2.resize(depth_img, (256, 144), interpolation=cv2.INTER_AREA)


def downsample_depth_minpool(depth_img_256x144):
    depth_tensor = torch.from_numpy(depth_img_256x144).unsqueeze(0).unsqueeze(0).float()
    downsampled = -F.max_pool2d(-depth_tensor, kernel_size=16, stride=16)
    return downsampled


def lidar_points_to_360(point_cloud, empty_value=LIDAR_EMPTY_VALUE):
    if point_cloud is None or len(point_cloud) < 3:
        return np.ones(360, dtype=np.float32) * empty_value

    points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.ones(360, dtype=np.float32) * empty_value

    dists = np.linalg.norm(points, axis=1)
    angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    angles = (angles + 90.0) % 360.0

    lidar_360 = np.ones(360, dtype=np.float32) * empty_value
    for dist, angle in zip(dists, angles):
        angle_idx = int(angle) % 360
        if dist < lidar_360[angle_idx]:
            lidar_360[angle_idx] = dist
    return lidar_360


def downsample_lidar_105(lidar_360):
    downsampled = lidar_360[LIDAR_105_INDICES]
    return torch.from_numpy(downsampled).float()


def action_to_target_velocity(current_velocity, action, dt=0.5, max_v=10.0):
    target_vel = np.asarray(current_velocity, dtype=np.float32) + np.asarray(action, dtype=np.float32) * dt
    v_mag = float(np.linalg.norm(target_vel))
    if v_mag > max_v and v_mag > 1e-6:
        target_vel = (target_vel / v_mag) * max_v
    return target_vel.astype(np.float32)
