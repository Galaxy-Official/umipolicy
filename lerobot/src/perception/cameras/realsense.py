import logging

import cv2
import numpy as np

from PIL import Image

from .base_camera import BaseCamera

logger = logging.getLogger(__name__)


@BaseCamera.register_camera("realsense")
class RealSense(BaseCamera):
    def __init__(self, camera_sn: str):
        import pyrealsense2 as rs
        self.camera_sn = camera_sn
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        self.config.enable_device(self.camera_sn)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30
        )
        # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(self.device)
        self.mtx = self.getIntrinsics()
        logger.info(self.mtx)
        self.hole_filling = rs.hole_filling_filter()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # cam init
        logger.info("cam init ...")
        i = 60
        while i > 0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            i -= 1
        logger.info("cam init done.")

    def get_data(self, hole_filling=False, return_type: str = "PIL"):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            # depth_frame = aligned_frames.get_depth_frame()
            # if hole_filling:
            #     depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue
            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            break
        if return_type == "PIL":
            return Image.fromarray(color_image), None
        elif return_type == "numpy":
            return color_image, None

    def inpaint(self, img, missing_value=0):
        """
        pip opencv-python == 3.4.8.29
        :param image:
        :param roi: [x0,y0,x1,y1]
        :param missing_value:
        :return:
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        # if scale < 1e-3:
        #     pdb.set_trace()
        img = (
            img.astype(np.float32) / scale
        )  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img

    def getXYZRGB(
        self, color, depth, robot_pose, camee_pose, camIntrinsics, inpaint=True
    ):
        """

        :param color:
        :param depth:
        :param robot_pose: array 4*4
        :param camee_pose: array 4*4
        :param camIntrinsics: array 3*3
        :param inpaint: bool
        :return: xyzrgb
        """
        import open3d as o3d

        if inpaint:
            depth = self.inpaint(depth)
        color_image = o3d.geometry.Image(color)
        depth_image = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )

        fx, fy, cx, cy = (
            camIntrinsics[0, 0],
            camIntrinsics[1, 1],
            camIntrinsics[0, 2],
            camIntrinsics[1, 2],
        )
        width, height = color.shape[1], color.shape[0]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )

        world_pose = np.dot(robot_pose, camee_pose)
        pcd.transform(world_pose)

        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)

        xyzrgb = np.hstack((xyz, rgb))
        return xyzrgb

    def getleft(self, obj1):
        index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
        index = np.bitwise_and(obj1[:, 1] < 0.5, index)
        index = np.bitwise_and(obj1[:, 1] > -0.5, index)
        # index = np.bitwise_and(obj1[:, 2] > -0.1, index)
        index = np.bitwise_and(obj1[:, 2] > 0.35, index)
        index = np.bitwise_and(obj1[:, 2] < 0.7, index)
        return obj1[index]

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = (
            color_frame.get_profile()
            .as_video_stream_profile()
            .get_intrinsics()
        )
        print("intr", intrinsics)
        mtx = [
            intrinsics.width,
            intrinsics.height,
            intrinsics.ppx,
            intrinsics.ppy,
            intrinsics.fx,
            intrinsics.fy,
        ]
        camIntrinsics = np.array(
            [[mtx[4], 0, mtx[2]], [0, mtx[5], mtx[3]], [0, 0, 1.0]]
        )
        return camIntrinsics

    def __del__(self):
        self.pipeline.stop()
