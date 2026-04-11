import airsim
import time

def test_sensors():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    vehicles = client.listVehicles()
    vehicle_name = vehicles[0] if vehicles else ""
    print(f"Using vehicle: '{vehicle_name}'")

    print("\n--- Before Takeoff ---")
    bottom = client.getDistanceSensorData(distance_sensor_name="DistanceBottom", vehicle_name=vehicle_name)
    top = client.getDistanceSensorData(distance_sensor_name="DistanceTop", vehicle_name=vehicle_name)
    print(f"DistanceBottom: {bottom.distance:.4f} meters")
    print(f"DistanceTop: {top.distance:.4f} meters")

    print("\nTaking off to get some altitude...")
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    
    time.sleep(1)

    print("\n--- After Takeoff ---")
    bottom = client.getDistanceSensorData(distance_sensor_name="DistanceBottom", vehicle_name=vehicle_name)
    top = client.getDistanceSensorData(distance_sensor_name="DistanceTop", vehicle_name=vehicle_name)
    print(f"DistanceBottom: {bottom.distance:.4f} meters")
    print(f"DistanceTop: {top.distance:.4f} meters")
        
    client.reset()
    client.enableApiControl(False, vehicle_name)

if __name__ == "__main__":
    test_sensors()