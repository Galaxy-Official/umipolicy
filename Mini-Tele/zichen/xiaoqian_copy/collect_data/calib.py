import open3d as o3d
import pickle
from utils.transformation import *
# pc = pickle.load(open("/home/rhos/xiaoyang/collect_data/data/2025-04-02_17:16:01/init_info/init_pc.pkl", "rb"))
# points, colors = pc[..., :3], pc[..., 3:]

# # 可视化点云
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# vis.add_geometry(pcd)

# frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# vis.add_geometry(frame_o3d)

# vis.run()
# vis.destroy_window()

import json
import cv2
import numpy as np

INTRINSICS = {
    # wrist
    "233722071807": np.array([[910.12329102,   0.,         649.51123047, 0],
                            [  0.,         908.83721924, 369.29821777,  0],
                            [  0. ,          0.,           1.        ,  0]]), 
    # top
    "001622071104": np.array([[914.21118164,   0. ,        641.01922607, 0],
                            [  0.,         913.02062988, 364.07424927,  0],
                            [  0.,           0.,           1.        ,  0]])
}
dist_coeffs = np.array([0, 0, 0, 0])  # 假设无畸变

devices = []
extrinsics = {}

calib_timestamp = "2025-06-05_22_17"

for camera_id in INTRINSICS.keys() :
    print(camera_id)
    camera_matrix = INTRINSICS[camera_id][:, :3]
    image = cv2.imread(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}_{camera_id}/rgb.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 设置ArUco标记字典（可以选择不同类型的字典）
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    # 检测标记并获得相应的信息
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 如果检测到标记
    if len(corners) > 0:
        # 绘制标记的边界
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # 定义标记尺寸（单位：米，假设每个标记的边长为 0.1 米）
        marker_length = 0.15

        # 估计标记的位姿（相机坐标系中的位置和朝向）
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        print("rvecs", rvecs)
        print("tvecs", tvecs)

        # 绘制每个标记的位姿
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.15)  # 绘制坐标轴，长度为 0.1 米

        # 显示图像
        cv2.imshow("Detected ArUco Markers", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("没有检测到ArUco标记")

    # xyzrot6d = np.concatenate([rvecs[0,0], tvecs[0,0]], axis=0)
    
    # ext_mat = xyz_rot_transform(xyzrot6d, from_rep="axis_angle", to_rep="matrix")

    rotation_matrix, _ = cv2.Rodrigues(rvecs[0, 0])
    ext_mat = np.zeros((4, 4))
    ext_mat[:3, :3] = rotation_matrix
    ext_mat[:3, 3] = tvecs[0, 0]
    ext_mat[3, 3] = 1

    extrinsics[camera_id] = ext_mat
    print("ext_mat", ext_mat)

    devices.append(camera_id)

np.save(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}/devices.npy",
        np.array(devices)
        )
np.save(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}/extrinsics.npy",
        extrinsics)
np.save(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}/intrinsics.npy",
        INTRINSICS)

tcp_mat = json.load(open(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}_{devices[0]}/robot_pose.json", "r"))
print("tcp_mat", tcp_mat)
tcp_quaternion = xyz_rot_transform(tcp_mat, from_rep="matrix", to_rep="quaternion")
np.save(f"/home/rhos/xiaoyang/collect_data/data/calib/{calib_timestamp}/tcp.npy",
        tcp_quaternion)
print("tcp_quaternion", tcp_quaternion)