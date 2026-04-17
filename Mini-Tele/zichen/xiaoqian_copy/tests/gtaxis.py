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

    parser.add_argument('--root_dir', type=str, default="pred_keypoints/capture_0301/door/view_001_pose_000")
    # parser.add_argument('--global_key', type=str)
    parser.add_argument('--keypoint', type=str)
    parser.add_argument('--filename', type=str, default="gt_keypoints.json")
    args = parser.parse_args()
    return args

args = config_parse()
# path = f"{args.root_dir}/{args.global_key}/init_capture"

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

## 修改成点云直接读取
point_cloud = pickle.load(open(f"{args.root_dir}/point_cloud.pkl", "rb"))
cam_pc = point_cloud["points"]
pc_rgb = point_cloud["colors"]

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cam_pc)
pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
vis.add_geometry(pcd)


keypoints = pick_point()[0].tolist()
print(keypoints)
# kp_axis1, kp_axis2, kp_contact

# Handle corrupted JSON file gracefully
if os.path.exists(f"{args.root_dir}/{args.filename}"):
    try:
        with open(f"{args.root_dir}/{args.filename}", "r") as f:
            keypoints_dict = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Corrupted JSON file detected ({e}). Creating new file.")
        keypoints_dict = {}
else:
    keypoints_dict = {}

keypoints_dict.update({args.keypoint: keypoints})

# Write the updated dictionary back to file
with open(f"{args.root_dir}/{args.filename}", "w") as f:
    json.dump(keypoints_dict, f, indent=2)

vis.run()
vis.destroy_window()
