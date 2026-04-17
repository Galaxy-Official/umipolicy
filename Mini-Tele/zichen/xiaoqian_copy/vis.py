import cv2
import os
import json
import h5py
import pickle
import numpy as np
import open3d as o3d
import glob
from utilities.constants import EPS
from geometry_utils import *
path = "/home/rhos/xiaoyang/calib_out/capture_0301/notebook/view_000_pose_003"
W, H = 1280, 720
camera = o3d.camera.PinholeCameraIntrinsic()
mtx = np.array([[910.12329102 ,  0.    ,     649.51123047],
                [  0.    ,     908.83721924, 369.29821777],
                [  0.     ,      0.    ,       1.        ]])
def calculate_rotation_axis(point1, point2):
        """
        计算旋转轴的方向和旋转中心
        输入:
        - point1: 第一个点 (x1, y1, z1)
        - point2: 第二个点 (x2, y2, z2)

        返回:
        - base_joint_direction: 旋转轴的方向 (单位向量)
        - base_joint_base: 旋转中心 (point1)
        """
        # 计算旋转轴的方向
        direction = point2 - point1
        
        # 计算单位化方向向量 (旋转轴方向)
        base_joint_direction = direction / np.linalg.norm(direction)
        
        # 旋转中心就是 point1
        base_joint_base = point1
        
        return base_joint_direction, base_joint_base
camera_int = mtx
camera.set_intrinsics(W, H, camera_int[0,0], camera_int[1,1], camera_int[0,2], camera_int[1,2])

frame_id = 0
## read depth
with h5py.File(os.path.join(path, "depth.h5"), 'r') as file:
    key = str(frame_id).zfill(5) + "_depth.png"
    depth = np.array(file[key][:])

## read rgb
rgb_frame = cv2.imread(os.path.join(path, "rgb.jpg"))
rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

# 读取图像并转换为RGB
numpy_image = rgb_frame
if numpy_image.dtype != np.uint8:
    numpy_image = numpy_image.astype(np.uint8)
rgb_image = numpy_image

# 读取深度数据
numpy_depth = depth
numpy_depth = numpy_depth.astype(np.float32)
depth_image = numpy_depth  # (H, W), original, scale: mm

# 创建 RGBD 图像并生成点云
color_raw = o3d.geometry.Image(rgb_image[:, :, :3])  # (H, W, C)
depth_scene = o3d.geometry.Image(depth_image)        # (H, W)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_scene,
    depth_scale=1000.0,  # 如果深度图像的单位是毫米
    convert_rgb_to_intensity=False)
pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = pcd_scene.points
pcd.colors = pcd_scene.colors

# 假设你有一些点例如 point1, point2 和 point3 (以下是举例)
# point1 = np.array([-0.01, -0.04, 0.54])
# point2 = np.array([0.12, -0.05, 0.53])
# point3 = np.array([0.03, -0.20, 0.53])

point1 = np.array([-0.06, -0.07, 0.71])
point2 = np.array([-0.08, 0.04, 0.54])
point3 = np.array([0.09, 0.00, 0.45])

# 绘制箭头（point2 -> point1）
# joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
joint_rotations,joint_translations = calculate_rotation_axis(point1, point2)
joint_num = 0
joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
rotation = np.zeros((3, 3))
temp2 = np.cross(joint_rotations, np.array([1., 0., 0.]))
if np.linalg.norm(temp2) < EPS:
    temp1 = np.cross(np.array([0., 1., 0.]), joint_rotations)
    temp1 /= np.linalg.norm(temp1)
    temp2 = np.cross(joint_rotations, temp1)
    temp2 /= np.linalg.norm(temp2)
else:
    temp2 /= np.linalg.norm(temp2)
    temp1 = np.cross(temp2, joint_rotations)
    temp1 /= np.linalg.norm(temp1)
rotation[:, 0] = temp1
rotation[:, 1] = temp2
rotation[:, 2] = joint_rotations
joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
joint_axis.translate(joint_translations.reshape((3, 1)))
joint_axis.paint_uniform_color([1, 0, 0])
# vis.add_geometry(joint_axis)
points = np.array(pcd.points)
radius = points[0].max() - points[0].min()
radius = 0.01 * radius
# 计算箭头的旋转矩阵
# rotation_axis = np.cross([0, 0, 1], direction)
# rotation_angle = np.arccos(np.dot([0, 0, 1], direction) / np.linalg.norm(direction))
# rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
# arrow.translate(point2)
# arrow.rotate(rotation_matrix, center=point2)
mesh_arrow = \
    get_thick_line(begin=point1, end=point2, radius=0.005, color=[1, 0, 0])
vis.add_geometry(mesh_arrow)
  # 红色箭头
# vis.add_geometry(joint_axis)

# 绘制 point3 (affordance point)
affordance_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
affordance_point.translate(point3)
affordance_point.paint_uniform_color([0, 1, 0])  # 绿色
vis.add_geometry(affordance_point)
mesh_sphere = draw_point(point1, [1, 0, 0], radius=0.015)
vis.add_geometry(mesh_sphere)

mesh_sphere = draw_point(point2, [1, 0, 0], radius=0.015)
vis.add_geometry(mesh_sphere)
# 显示可视化
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
