import time
import threading
from threading import Thread
import numpy as np
import cv2

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
from perception.cameras.mvs_cam import MVSCamera

class MVSWrapperCamera:
    """Wrapper that adapts Hikrobot MVS Camera to LeRobot 2.0 framework compatibility"""
    def __init__(self, camera_index: int = 0, serial: str = "DA8942731", exposure: int = 20000, fps: int = 30, width: int = 1536, height: int = 1536):
        self.camera_index = camera_index
        self.serial = serial
        self.exposure = exposure
        self.fps = fps
        self.width = width
        self.height = height
        self.channels = 3
        
        self.is_connected = False
        self.camera = None
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"MVSCamera({self.serial}) is already connected.")
            
        # Initialize MVS sdk with user dict format
        self.camera = MVSCamera({"serial": self.serial, "exposure": self.exposure})
        self.is_connected = True

    def read(self, temporary_color_mode=None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"MVSCamera({self.serial}) is not connected.")
            
        start_time = time.perf_counter()
        
        ret, frame, capture_ts = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from MVSCamera {self.serial}.")

        # MVS returns BGR by cvtColor, convert to LeRobot RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply resize with black padding strictly mirroring old _02_combine_... preprocessing
        target_w, target_h = self.width, self.height
        h, w = frame.shape[:2]
        if h != target_h or w != target_w:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            frame = canvas

        self.color_image = frame
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        return frame

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                self.read()
            except Exception as e:
                import traceback
                traceback.print_exc()

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"MVSCamera({self.serial}) is not connected.")

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while True:
            if self.color_image is not None:
                return self.color_image

            time.sleep(1 / self.fps)
            num_tries += 1
            if num_tries > self.fps * 2:
                raise TimeoutError("Timed out waiting for async_read() to start.")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(f"MVSCamera({self.serial}) is not connected.")

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        if hasattr(self.camera, 'release'):
            self.camera.release()
        self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
