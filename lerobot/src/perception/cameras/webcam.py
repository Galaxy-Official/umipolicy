import logging

import cv2
from PIL import Image
import numpy as np
from .base_camera import BaseCamera

logger = logging.getLogger(__name__)


@BaseCamera.register_camera("webcamTactile")
class WebCamTactile(BaseCamera):
    def __init__(self, 
                camera_id = None, 
                v4l_path = None,
                fps=20,
                resolution=(640, 480)
                ):
        self.camera_id = camera_id
        self.v4l_path = v4l_path
        
        if self.camera_id is None and v4l_path is None:
            raise ValueError("Either camera_id or v4l_path must be provided")
        
        if self.camera_id is not None:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera with id {self.camera_id}")
        if self.v4l_path is not None:
            self.cap = cv2.VideoCapture(self.v4l_path, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera")
            
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        w, h = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def get_data(self):
        assert self.cap.isOpened(), "Failed to open webcam."
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam.") 
            return None, None
        
        color_image = frame
        height, width = color_image.shape[:2]

        crop_width = int(width * 3 / 5)
        crop_height = int(height * 3 / 5)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        color_image = color_image[start_y:start_y+crop_height, 
                                    start_x:start_x+crop_width]
        color_image = cv2.resize(color_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        return color_image, None

    def __del__(self):
        self.cap.release()


@BaseCamera.register_camera("webcam")
class WebCam(BaseCamera):
    def __init__(self, 
                camera_id = None, 
                v4l_path = None,
                fps=20,
                resolution=(640, 480)):
        self.camera_id = camera_id
        self.v4l_path = v4l_path
        
        if self.camera_id is None and v4l_path is None:
            raise ValueError("Either camera_id or v4l_path must be provided")
        
        if self.camera_id is not None:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera with id {self.camera_id}")
        if self.v4l_path is not None:
            self.cap = cv2.VideoCapture(self.v4l_path, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera")
            
        w, h = resolution
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_data(self):
        assert self.cap.isOpened(), "Failed to open webcam."
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam.")
            return None, None
        color_image = frame
        
        return color_image, None

    def __del__(self):
        self.cap.release()
