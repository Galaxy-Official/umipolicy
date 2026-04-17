import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time
import os
import pickle

class CameraD400(object):
    def __init__(self, camera_id="233722071807", width=1280, height=720):
        target_serial_number = camera_id  # 替换为你要使用的设备的序列号

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(target_serial_number)

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        # self.config.enable_stream(rs.stream.color, c, rs.format.bgr8, 10)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        # self.mtx = self.getIntrinsics()
        # print(self.mtx)
        # with open(r"config/d435_high_accuracy.json", 'r') as file:
        #    json_text = file.read().strip()
        # advanced_mode.load_json(json_text)

        self.hole_filling = rs.hole_filling_filter()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # cam init
        print("cam init ...")
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
        print("cam init done.")

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
        if scale < 1e-3:
            pdb.set_trace()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img

    def getXYZRGB(self,color, depth, robot_pose,camee_pose,camIntrinsics,inpaint=True,depth_scale=None):
        '''
        :param color:
        :param depth:
        :param robot_pose: array 4*4
        :param camee_pose: array 4*4
        :param camIntrinsics: array 3*3
        :param inpaint: bool
        :param depth_scale: float, change to meter unit
        :return: xyzrgb
        '''
        heightIMG, widthIMG, _ = color.shape
        # pdb.set_trace()
        # heightIMG = 720
        # widthIMG = 1280
        # depthImg = depth / 10000.
        assert depth_scale is not None
        depthImg = depth * depth_scale
        # depthImg = depth
        if inpaint:
            depthImg = self.inpaint(depthImg)
        robot_pose = np.dot(robot_pose, camee_pose)

        [pixX, pixY] = np.meshgrid(np.arange(widthIMG), np.arange(heightIMG))
        camX = (pixX - camIntrinsics[0][2]) * depthImg / camIntrinsics[0][0]
        camY = (pixY - camIntrinsics[1][2]) * depthImg / camIntrinsics[1][1]
        camZ = depthImg

        camPts = [camX.reshape(camX.shape + (1,)), camY.reshape(camY.shape + (1,)), camZ.reshape(camZ.shape + (1,))]
        camPts = np.concatenate(camPts, 2)
        camPts = camPts.reshape((camPts.shape[0] * camPts.shape[1], camPts.shape[2]))  # shape = (heightIMG*widthIMG, 3)
        worldPts = np.dot(robot_pose[:3, :3], camPts.transpose()) + robot_pose[:3, 3].reshape(3,
                                                                                              1)  # shape = (3, heightIMG*widthIMG)
        rgb = color.reshape((-1, 3)) / 255.
        rgb[:, [0, 2]] = rgb[:, [2, 0]]
        xyzrgb = np.hstack((worldPts.T, rgb))
        # xyzrgb = self.getleft(xyzrgb)
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
            color_frame.get_profile().as_video_stream_profile().get_intrinsics()
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


def vis_pc(xyzrgb):
    import open3d as o3d
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pc1.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
    o3d.visualization.draw_geometries([camera_frame, pc1])
if __name__ == "__main__":
    cam = CameraD400("233722071467") # TODO: change SN code
    i = 0
    while True:
        print(f"{i}th")
        color, depth = cam.get_data(hole_filling=False)

        depth_sensor = cam.pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # xyz in meter, rgb in [0, 1]
        xyzrgb = cam.getXYZRGB(color, depth, np.identity(4), np.identity(4), cam.getIntrinsics(), inpaint=False, depth_scale=depth_scale)
        # xyzrgb = xyzrgb[xyzrgb[:, 2] <= 1.5, :]
        print(np.mean(xyzrgb[:, 2]))
        vis_pc(xyzrgb)

        cv2.imshow('color', color)
        while True:
            if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        cmd = input("whether save? (y/n): ")
        if cmd == 'y':
            cv2.imwrite(f"camera_tmp/rgb_{i}.png", color)
            pickle.dump(depth, open(f"camera_tmp/depth_{i}.pkl", "wb"))
            # np.savez(f"xyzrgb_{i}.npz", point_cloud=xyzrgb[:, :3], rgb=xyzrgb[:, 3:])
            i += 1
        elif cmd == 'n':
            cmd = input("whether quit? (y/n): ")
            if cmd == 'y':
                break
            elif cmd == 'n':
                pass
            else:
                raise ValueError
        else:
            raise ValueError