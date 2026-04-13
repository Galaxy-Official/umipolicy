import logging
import os

import cv2
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return np.flip(color_image, -1).copy()


class CameraD400:
    def __init__(self, camera_id=0):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        if camera_id >= len(devices):
            raise ValueError("Camera ID is out of range")
        self.config.enable_device(
            devices[camera_id].get_info(rs.camera_info.serial_number)
        )

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30
        )
        # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        rs.rs400_advanced_mode(self.device)
        self.mtx = self.getIntrinsics()
        logger.info(self.mtx)
        # with open(r"config/d435_high_accuracy.json", 'r') as file:
        #    json_text = file.read().strip()
        # advanced_mode.load_json(json_text)

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
            np.asanyarray(depth_frame.get_data())
            np.asanyarray(color_frame.get_data())
            i -= 1
        logger.info("cam init done.")

    def get_data(self, hole_filling=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image

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


def cap_img(
    cam: CameraD400,
    tag: str = "default",
    img_id: int = 0,
    save_flag: bool = False,
    video_path: str = None,
) -> int:
    img, _ = cam.get_data()
    img = img.copy()  # why copy?
    img_id += 1
    if save_flag:
        assert video_path is not None, "video_path is None"
        filename = os.path.join(video_path, f"{img_id:05}.jpg")
        cv2.imwrite(filename, img)
    # Show Image
    cv2.imshow(f"{tag} view", img)
    cv2.waitKey(1)
    return img_id


# def get_video_path(task_id, sub_task_id, video_id, tag):
#     video_path = f"output/task{task_id}/{sub_task_id}/video{video_id}/{tag}"
#     if os.path.exists(video_path):
#         shutil.rmtree(video_path)
#     os.makedirs(video_path)
#     return video_path


# def capture(args: argparse.Namespace):
#     robot_ip = "192.168.2.100"
#     local_ip = "192.168.2.103"
#     cam_bird = CameraD400(0)  # bird view
#     cam_ego = CameraD400(1)  # ego view
#     img_bird_id = 0
#     img_ego_id = 0

#     while True:
#         # 实时显示图像，但不保存
#         cap_img(cam_bird, "bird", img_bird_id, save_flag=False)
#         cap_img(cam_ego, "ego", img_ego_id, save_flag=False)

#         # 按 "s" 开始捕获图像
#         if cv2.waitKey(100) & 0xFF == ord("s"):
#             print("Start capturing images...")
#             break
#     video_ego_path = get_video_path(
#         args.task_id, args.sub_task_id, args.video_id, "ego"
#     )
#     video_bird_path = get_video_path(
#         args.task_id, args.sub_task_id, args.video_id, "bird"
#     )
#     while True:
#         try:
#             if cv2.waitKey(100) & 0xFF == ord("e"):
#                 print("Ending...")
#                 break
#             # 尝试同时捕获bird和ego视角的图像
#             img_bird_id = cap_img(
#                 cam=cam_bird,
#                 tag="bird",
#                 img_id=img_bird_id,
#                 save_flag=True,
#                 video_path=video_bird_path,
#             )
#             img_ego_id = cap_img(
#                 cam=cam_ego,
#                 tag="ego",
#                 img_id=img_ego_id,
#                 save_flag=True,
#                 video_path=video_ego_path,
#             )
#             print(
#                 f"Saved {img_bird_id} Bird Eye View Image and {img_ego_id} Ego Eye View Image"
#             )
#         except KeyboardInterrupt as e:
#             print("KeyboradInterrupt")
#             break
#     cv2.destroyAllWindows()
