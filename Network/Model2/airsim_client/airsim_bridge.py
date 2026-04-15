import airsim
import numpy as np
import time

class AirSimBridge:
    def __init__(self, ip="127.0.0.1"):
        self.client = airsim.MultirotorClient(ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.set_segmentation()
    
    def takeoff(self, height=3.0):
        print("Taking off...")
        self.client.takeoffAsync().join()
        print("Sleeping 2s...")
        time.sleep(2)

        print("Manual move up...")
        self.client.moveByVelocityAsync(0, 0, -1, 2).join()

        print("Takeoff done")

    def get_state(self):
        """
        返回无人机状态，包括位置、速度、相机图像等
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        return {
            "position": np.array([pos.x_val, pos.y_val, pos.z_val]),
            "velocity": np.array([vel.x_val, vel.y_val, vel.z_val])
        }

    def get_image(self, camera_name="0", image_type=airsim.ImageType.Scene):
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, image_type, False, False)
        ])
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)
        return img

    # def move_by_action(self, action):
    #     """
    #     action 可以是前进/后退/上/下/左/右，简单封装
    #     """
    #     x, y, z = action
    #     self.client.moveByVelocityAsync(x, y, z, duration=0.3).join()
    def move_by_action(self, action):
        """
        action = (forward, vz, yaw_rate)
        forward: 前进速度
        vz: 上下
        yaw_rate: 转向速度
        """

        # ===== 1. 初始化平滑变量 =====
        if not hasattr(self, "prev_action"):
            self.prev_action = np.zeros(3)

        action = np.array(action, dtype=np.float32)

        # ===== 2. 动作平滑（关键）=====
        alpha = 0.7
        action = alpha * self.prev_action + (1 - alpha) * action
        self.prev_action = action

        forward, vz, yaw_rate = action

        # ===== 3. 控制参数（可调）=====
        duration = 0.3   # 控制频率（越小越丝滑）
        max_forward = 3.0
        max_vz = 2.0
        max_yaw = 60.0   # deg/s

        # ===== 4. 归一化动作 =====
        vx = float(forward * max_forward)
        vz = float(vz * max_vz)
        yaw_rate = float(yaw_rate * max_yaw)

        # ===== 5. 使用机体坐标系控制（关键🔥）=====
        self.client.moveByVelocityBodyFrameAsync(
            vx=vx,
            vy=0.0,   # 不允许横向漂移（更稳定）
            vz=vz,
            duration=duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )

        # ===== 6. 轻微sleep（保证控制频率）=====
        import time
        time.sleep(duration)

    def get_segmentation(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
        ])

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)

        return img
    
    def set_segmentation(self):
        print("Setting segmentation IDs...")

        
        self.client.simSetSegmentationObjectID(".*", 0, True)

        
        self.client.simSetSegmentationObjectID("SM_AM_vol8_sidewalk.*", 1, True)
        self.client.simSetSegmentationObjectID(".*sidewalk.*", 1, True)
        self.client.simSetSegmentationObjectID("SM_AM_vol8_street.*", 1, True)
        self.client.simSetSegmentationObjectID(".*street.*", 1, True)
        self.client.simSetSegmentationObjectID("SM_AM_vol8_curb.*", 1, True)
        self.client.simSetSegmentationObjectID(".*curb.*", 1, True)

        
        self.client.simSetSegmentationObjectID("AM_vol8_building.*", 2, True)
        self.client.simSetSegmentationObjectID(".*building.*", 2, True)

        self.client.simSetSegmentationObjectID(".*ad_column.*", 2, True)
        self.client.simSetSegmentationObjectID(".*adstand.*", 2, True)
        self.client.simSetSegmentationObjectID(".*ad_stand.*", 2, True)
        self.client.simSetSegmentationObjectID(".*air_conditioner.*", 2, True)
        self.client.simSetSegmentationObjectID(".*awning.*", 2, True)
        self.client.simSetSegmentationObjectID(".*bench.*", 2, True)
        self.client.simSetSegmentationObjectID(".*bicycle.*", 2, True)
        self.client.simSetSegmentationObjectID(".*bike_stand.*", 2, True)
        self.client.simSetSegmentationObjectID(".*blanket.*", 2, True)
        self.client.simSetSegmentationObjectID(".*bus_stop.*", 2, True)
        self.client.simSetSegmentationObjectID(".*chair.*", 2, True)
        self.client.simSetSegmentationObjectID(".*cusion.*", 2, True)
        self.client.simSetSegmentationObjectID(".*cutlery.*", 2, True)
        self.client.simSetSegmentationObjectID(".*dec_glass.*", 2, True)
        self.client.simSetSegmentationObjectID(".*drainage.*", 2, True)
        self.client.simSetSegmentationObjectID(".*drape.*", 2, True)
        self.client.simSetSegmentationObjectID(".*flower.*", 2, True)
        self.client.simSetSegmentationObjectID(".*flower_pot.*", 2, True)
        self.client.simSetSegmentationObjectID(".*food_cart.*", 2, True)
        self.client.simSetSegmentationObjectID(".*food_stand.*", 2, True)
        self.client.simSetSegmentationObjectID(".*fork.*", 2, True)
        self.client.simSetSegmentationObjectID(".*frame_menu.*", 2, True)
        self.client.simSetSegmentationObjectID(".*glass.*", 2, True)
        self.client.simSetSegmentationObjectID(".*hydrant.*", 2, True)
        self.client.simSetSegmentationObjectID(".*infobox.*", 2, True)
        self.client.simSetSegmentationObjectID(".*kiosk.*", 2, True)
        self.client.simSetSegmentationObjectID(".*knife.*", 2, True)
        self.client.simSetSegmentationObjectID(".*lamp.*", 2, True)
        self.client.simSetSegmentationObjectID(".*lamp_facade.*", 2, True)
        self.client.simSetSegmentationObjectID(".*leafdry.*", 2, True)
        self.client.simSetSegmentationObjectID(".*logo.*", 2, True)
        self.client.simSetSegmentationObjectID(".*mailbox.*", 2, True)
        self.client.simSetSegmentationObjectID(".*manhole.*", 2, True)
        self.client.simSetSegmentationObjectID(".*menu.*", 2, True)
        self.client.simSetSegmentationObjectID(".*metro_entrance.*", 2, True)
        self.client.simSetSegmentationObjectID(".*napkin.*", 2, True)
        self.client.simSetSegmentationObjectID(".*newspaper.*", 2, True)
        self.client.simSetSegmentationObjectID(".*parking_meter.*", 2, True)
        self.client.simSetSegmentationObjectID(".*pepper.*", 2, True)
        self.client.simSetSegmentationObjectID(".*plant.*", 2, True)
        self.client.simSetSegmentationObjectID(".*plant_bush.*", 2, True)
        self.client.simSetSegmentationObjectID(".*plate.*", 2, True)
        self.client.simSetSegmentationObjectID(".*pot.*", 2, True)
        self.client.simSetSegmentationObjectID(".*railing.*", 2, True)
        self.client.simSetSegmentationObjectID(".*scaffolding.*", 2, True)
        self.client.simSetSegmentationObjectID(".*sign.*", 2, True)
        self.client.simSetSegmentationObjectID(".*singpost.*", 2, True)
        self.client.simSetSegmentationObjectID(".*spice_maker.*", 2, True)
        self.client.simSetSegmentationObjectID(".*street_barier.*", 2, True)   # 对应 vol8 和 vol9，已合并
        self.client.simSetSegmentationObjectID(".*street_clock.*", 2, True)
        self.client.simSetSegmentationObjectID(".*street_lamp.*", 2, True)
        self.client.simSetSegmentationObjectID(".*street_pole.*", 2, True)
        self.client.simSetSegmentationObjectID(".*street_sign.*", 2, True)
        self.client.simSetSegmentationObjectID(".*sunshade.*", 2, True)
        self.client.simSetSegmentationObjectID(".*table.*", 2, True)
        self.client.simSetSegmentationObjectID(".*tablecloth.*", 2, True)
        self.client.simSetSegmentationObjectID(".*table_set.*", 2, True)
        self.client.simSetSegmentationObjectID(".*thyme.*", 2, True)
        self.client.simSetSegmentationObjectID(".*ticket_machine.*", 2, True)
        self.client.simSetSegmentationObjectID(".*traffic_lights.*", 2, True)
        self.client.simSetSegmentationObjectID(".*transformator.*", 2, True)
        self.client.simSetSegmentationObjectID(".*trashcan.*", 2, True)
        self.client.simSetSegmentationObjectID(".*tree_stand.*", 2, True)
        self.client.simSetSegmentationObjectID(".*wheel_stopper.*", 2, True)
        success = self.client.simSetSegmentationObjectID(".*building.*", 2, True) 
        print("Ground matched:", success)
        print("Segmentation IDs set.")