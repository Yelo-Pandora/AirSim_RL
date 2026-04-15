from airsim_client.airsim_bridge import AirSimBridge
import time

bridge = AirSimBridge()

print("Enable control")
bridge.client.enableApiControl(True)

print("Arm")
bridge.client.armDisarm(True)

print("Takeoff")
bridge.client.takeoffAsync().join()

print("Move up")
bridge.client.moveByVelocityAsync(0, 0, -1, 3).join()

print("Done")