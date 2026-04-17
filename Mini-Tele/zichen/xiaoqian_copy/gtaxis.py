import cv2
import os
import json
import h5py
import pickle
import numpy as np
import open3d as o3d
import glob

path = "/home/rhos/xiaoyang/calib_out/capture_0301/notebook/view_000_pose_003"

def pick_point():
    # 创建一个可视化窗口
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(pcd)
    # 开始可视化，允许用户选择点
    print("请在点云上点击关键点。按 'q' 键退出并记录点坐标。")
    vis.run()
    vis.destroy_window()
    # 获取用户选择的点索引
    picked_points = vis.get_picked_points()
    print("用户选择的点索引: ", picked_points)

    picked_points = np.array([picked_point.index for picked_point in picked_points])
    print(picked_points)
    return picked_points


# 设置相机内参
W, H = 1280, 720
camera = o3d.camera.PinholeCameraIntrinsic()
mtx = np.array([[910.12329102 ,  0.    ,     649.51123047],
 [  0.    ,     908.83721924, 369.29821777],
 [  0.     ,      0.    ,       1.        ]])

camera_int = mtx
print(camera_int)
camera.set_intrinsics(W, H, camera_int[0,0], camera_int[1,1], camera_int[0,2], camera_int[1,2])

frame_id = 0
## read depth
with h5py.File(os.path.join(path, "depth.h5"), 'r') as file:
    key = str(frame_id).zfill(5) + "_depth.png"
    depth = np.array(file[key][:])

## read rgb
# cap = cv2.VideoCapture(os.path.join(path, "rgb.mp4"))
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
# ret, frame = cap.read()
# rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# cap.release()

rgb_frame = cv2.imread(os.path.join(path,"rgb.jpg"))
rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

# 读取图像并转换为RGB
numpy_image = rgb_frame
if numpy_image.dtype != np.uint8:
    numpy_image = numpy_image.astype(np.uint8)
rgb_image = numpy_image


# # 获取去畸变的 RGB 图像
# dist = camera_params["dist"]
# rgb_image = cv2.undistort(rgb_image, camera_int, dist)

numpy_depth = depth
numpy_depth = numpy_depth.astype(np.float32)
depth_image = numpy_depth       # (H, W), original, scale: mm

# 创建 RGBD 图像并生成点云
color_raw = o3d.geometry.Image(rgb_image[:, :, :3])      # (H, W, C)
depth_scene = o3d.geometry.Image(depth_image)            # (H, W)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_scene, 
    depth_scale=1000.0,  # 如果深度图像的单位是毫米
    convert_rgb_to_intensity=False)
pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)
org_pcd = np.concatenate([pcd_scene.points, pcd_scene.colors], axis=-1)

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = pcd_scene.points
pcd.colors = pcd_scene.colors
#vis.add_geometry(pcd)

# # 定义点的位置
# point1 = np.array([-0.07, 0.16, 0.42])  # point1坐标
# point2 = np.array([-0.07, -0.04, 0.63])  # point2坐标
# point3 = np.array([0.09, 0.00, 0.45])  # point3坐标


# # 创建从point2到point1的箭头（可以用LineSet表示）
# arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)

# # 计算箭头的方向向量和旋转
# direction = point1 - point2  # 箭头方向
# arrow.translate(point2)  # 将箭头的位置设置为point2
# arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross([0, 0, 1], direction) * np.arccos(np.dot([0, 0, 1], direction) / np.linalg.norm(direction))), center=point2)


# # 绘制箭头
# arrow.paint_uniform_color([1, 0, 0])  # 将箭头颜色设置为红色

# # 可视化点云和箭头
# o3d.visualization.draw_geometries([pcd, arrow])


axis_keypoints = pick_point()

# [Open3D INFO] Adding point #356517 (0.21, -0.06, 0.70) to selection.
# [Open3D INFO] Adding point #362166 (0.02, -0.06, 0.70) to selection.
# [Open3D INFO] Adding point #694021 (0.12, 0.15, 0.71) to selection.