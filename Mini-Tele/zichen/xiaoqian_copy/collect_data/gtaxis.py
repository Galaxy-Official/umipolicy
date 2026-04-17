import cv2
import os
import json
import h5py
import pickle
import numpy as np
import open3d as o3d
import glob
import configargparse

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str)
    parser.add_argument('--keypoint', type=str)

    args = parser.parse_args()
    return args

args = config_parse()
path = f"{args.root_dir}/{args.global_key}/init_capture"

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

    picked_indices = np.array([picked_point.index for picked_point in picked_points])

    picked_points = np.asarray(pcd.points)[picked_indices]

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

# frame_id = 0
# ## read depth
# with h5py.File(os.path.join(path, "depth.h5"), 'r') as file:
#     key = str(frame_id).zfill(5) + "_depth.png"
#     depth = np.array(file[key][:]) 

os.makedirs(path, exist_ok=True)
depth = pickle.load(open(os.path.join(path, "depth.pkl"), "rb"))
rgb_frame = cv2.imread(os.path.join(path,"rgb.jpg"))
rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

os.makedirs(f"{args.root_dir}/{args.global_key}/init_info", exist_ok=True)
# 读取图像并转换为RGB
numpy_image = rgb_frame
if numpy_image.dtype != np.uint8:
    numpy_image = numpy_image.astype(np.uint8)
rgb_image = numpy_image

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

keypoints = pick_point()
print(keypoints)
# axis1, axis2, affordance point
pickle.dump(keypoints, open(f"{args.root_dir}/{args.global_key}/init_info/{args.keypoint}.pkl", "wb"))

point_cloud = np.concatenate([np.array(pcd.points), np.array(pcd.colors)], axis=-1) # (N, 6)
pickle.dump(point_cloud, open(f"{args.root_dir}/{args.global_key}/init_info/init_pc.pkl", "wb"))