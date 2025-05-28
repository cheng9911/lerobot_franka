# import pickle
# import time
# from dataclasses import dataclass, field, replace
# from pathlib import Path

# import numpy as np
# import torch

# from lerobot.common.robot_devices.cameras.utils import Camera
# from lerobot.common.robot_devices.motors.dynamixel import (
#     OperatingMode,
#     TorqueMode,
#     convert_degrees_to_steps,
# )
# from lerobot.common.robot_devices.motors.utils import MotorsBus
# from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


# @dataclass
# class camera_Config:
    
   
#     cameras: dict[str, Camera] = field(default_factory=lambda: {})
# class camera_manager:
#     def __init__(
#                 self,
#                 config: camera_Config| None = None,
#                 **kwargs,
#             ):
#         if config is None:
#             config = camera_Config()
#         self.config = replace(config, **kwargs)
#         self.cameras = self.config.cameras
#         self.is_connected = False
#         self.logs = {}
#     def connect(self):
#         for name in self.cameras:
#             self.cameras[name].connect()
#         self.is_connected = True
#     def  disconnect(self):
#         for name in self.cameras:
#             self.cameras[name].disconnect()
#         self.is_connected = False
#     def _test(self):
#         images = {}
#         for name in self.cameras:
#             before_camread_t = time.perf_counter()
#             images[name] = self.cameras[name].async_read()

#             self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
#             self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t



# if __name__ == "__main__":
    
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Test Image", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
