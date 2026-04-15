import airsim
import numpy as np
import time

def test_lidar():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    vehicles = client.listVehicles()
    vehicle_name = vehicles[0] if vehicles else ""

    print("Fetching Lidar1...")
    lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name=vehicle_name)
    print("Point cloud length:", len(lidar_data.point_cloud))
    
    if len(lidar_data.point_cloud) > 0:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        print("First 5 points:", points[:5])
        dists = np.linalg.norm(points, axis=1)
        print("First 5 dists:", dists[:5])
        print("Min dist:", np.min(dists), "Max dist:", np.max(dists))

if __name__ == "__main__":
    test_lidar()