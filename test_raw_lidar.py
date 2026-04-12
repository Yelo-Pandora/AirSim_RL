import airsim

def test_raw_lidar():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    vehicles = client.listVehicles()
    vehicle_name = vehicles[0] if vehicles else ""

    try:
        raw_data = client.client.call('getLidarData', "Lidar1", vehicle_name)
        print("Raw LiDAR data types:")
        for i, item in enumerate(raw_data):
            print(f"Index {i}: type={type(item)}")
            if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, bytes):
                print(f"  Length: {len(item)}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_raw_lidar()