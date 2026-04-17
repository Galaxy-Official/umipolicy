# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
from utils.transformation import xyz_rot_transform

# ==================== 配置参数 ====================
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
marker_length = 0.15  # ArUco标记尺寸

# 要使用的时间戳列表
target_timestamps = [
    "2025-06-05_21_57", "2025-06-05_21_58", "2025-06-05_22_01", 
    "2025-06-05_22_02", "2025-06-05_22_09", "2025-06-05_22_11",
    "2025-06-05_22_12", "2025-06-05_22_13", "2025-06-05_22_15", 
    "2025-06-05_22_17"
]

# 输出时间戳
output_timestamp = "2025-06-05_22_17_optimized"

# 路径配置
RAW_DATA_PATH = "data/calib/raw_data"
OUTPUT_PATH = "data/calib"

def detect_aruco_markers(image_path, camera_matrix):
    """检测ArUco标记，返回外参矩阵（完全复制原始代码逻辑）"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if len(corners) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        
        # 完全按照原始代码的方式构建外参矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvecs[0, 0])
        ext_mat = np.eye(4)
        ext_mat[:3, :3] = rotation_matrix
        ext_mat[:3, 3] = tvecs[0, 0]
        
        return ext_mat
    
    return None

def collect_all_extrinsics():
    """收集所有时间戳的外参矩阵"""
    all_extrinsics = {}  # {camera_id: [ext_mat1, ext_mat2, ...]}
    all_tcp_poses = []
    
    # 初始化
    for camera_id in INTRINSICS.keys():
        all_extrinsics[camera_id] = []
    
    print("\n=== 收集所有时间戳的外参数据 ===")
    
    for timestamp in target_timestamps:
        print("处理时间戳: {}".format(timestamp))
        timestamp_tcp = None
        
        for camera_id in INTRINSICS.keys():
            camera_matrix = INTRINSICS[camera_id][:, :3]
            
            image_path = "{}/{}_{}/rgb.jpg".format(RAW_DATA_PATH, timestamp, camera_id)
            robot_pose_path = "{}/{}_{}/robot_pose.json".format(RAW_DATA_PATH, timestamp, camera_id)
            
            if os.path.exists(image_path) and os.path.exists(robot_pose_path):
                # 检测ArUco并获取外参
                ext_mat = detect_aruco_markers(image_path, camera_matrix)
                
                if ext_mat is not None:
                    all_extrinsics[camera_id].append(ext_mat)
                    
                    # 读取TCP位置
                    with open(robot_pose_path, 'r') as f:
                        tcp_mat = json.load(f)
                    
                    if timestamp_tcp is None:
                        timestamp_tcp = tcp_mat
                    
                    print("  ✓ 相机 {}: 成功检测".format(camera_id))
                    print("    外参平移: [{:.3f}, {:.3f}, {:.3f}]".format(
                        ext_mat[0,3], ext_mat[1,3], ext_mat[2,3]))
                else:
                    print("  ✗ 相机 {}: 检测失败".format(camera_id))
            else:
                print("  ✗ 相机 {}: 文件不存在".format(camera_id))
        
        if timestamp_tcp is not None:
            all_tcp_poses.append(timestamp_tcp)
    
    return all_extrinsics, all_tcp_poses

def average_extrinsics_simple(extrinsics_list):
    """简单平均外参矩阵"""
    if len(extrinsics_list) == 0:
        return None
    
    if len(extrinsics_list) == 1:
        return extrinsics_list[0]
    
    # 分离平移和旋转
    translations = []
    rotations = []
    
    for ext_mat in extrinsics_list:
        translations.append(ext_mat[:3, 3])
        rotations.append(ext_mat[:3, :3])
    
    # 平均平移
    avg_translation = np.mean(translations, axis=0)
    
    # 平均旋转（转换为轴角再平均）
    axis_angles = []
    for R in rotations:
        rvec, _ = cv2.Rodrigues(R)
        axis_angles.append(rvec.flatten())
    
    avg_axis_angle = np.mean(axis_angles, axis=0)
    avg_rotation, _ = cv2.Rodrigues(avg_axis_angle)
    
    # 构建平均外参
    avg_ext = np.eye(4)
    avg_ext[:3, :3] = avg_rotation
    avg_ext[:3, 3] = avg_translation
    
    return avg_ext

def average_extrinsics_robust(extrinsics_list):
    """鲁棒平均外参矩阵（去掉离群值）"""
    if len(extrinsics_list) <= 3:
        return average_extrinsics_simple(extrinsics_list)
    
    # 先计算简单平均作为参考
    simple_avg = average_extrinsics_simple(extrinsics_list)
    ref_translation = simple_avg[:3, 3]
    
    # 计算每个外参与参考的距离
    distances = []
    translations = []
    for ext_mat in extrinsics_list:
        trans = ext_mat[:3, 3]
        dist = np.linalg.norm(trans - ref_translation)
        distances.append(dist)
        translations.append(trans)
    
    # 去掉距离最大的2个
    distances = np.array(distances)
    farthest_indices = np.argsort(distances)[-2:]
    
    print("  去掉离群值:")
    for idx in farthest_indices:
        trans = translations[idx]
        print("    #{}: [{:.3f}, {:.3f}, {:.3f}] (距离: {:.3f})".format(
            idx+1, trans[0], trans[1], trans[2], distances[idx]))
    
    # 使用剩余的外参重新计算平均
    valid_extrinsics = [extrinsics_list[i] for i in range(len(extrinsics_list)) 
                       if i not in farthest_indices]
    
    print("  使用 {}/{} 个外参计算平均值".format(len(valid_extrinsics), len(extrinsics_list)))
    
    return average_extrinsics_simple(valid_extrinsics)

def optimize_all_extrinsics(all_extrinsics):
    """优化所有相机的外参"""
    print("\n=== 优化外参 ===")
    
    final_extrinsics = {}
    
    for camera_id in INTRINSICS.keys():
        extrinsics_list = all_extrinsics[camera_id]
        
        if len(extrinsics_list) == 0:
            print("相机 {}: 无有效数据".format(camera_id))
            continue
        
        print("相机 {}: 使用 {} 个外参进行优化".format(camera_id, len(extrinsics_list)))
        
        # 显示所有候选外参的平移部分
        print("  所有候选外参平移:")
        for i, ext_mat in enumerate(extrinsics_list):
            trans = ext_mat[:3, 3]
            print("    #{}: [{:.3f}, {:.3f}, {:.3f}]".format(i+1, trans[0], trans[1], trans[2]))
        
        # 根据数据量选择优化方法
        if len(extrinsics_list) == 1:
            final_ext = extrinsics_list[0]
            print("  只有一个外参，直接使用")
        elif len(extrinsics_list) <= 3:
            final_ext = average_extrinsics_simple(extrinsics_list)
            print("  数据较少，使用简单平均")
        else:
            final_ext = average_extrinsics_robust(extrinsics_list)
        
        final_extrinsics[camera_id] = final_ext
        
        final_trans = final_ext[:3, 3]
        print("  最终外参平移: [{:.3f}, {:.3f}, {:.3f}]".format(
            final_trans[0], final_trans[1], final_trans[2]))
        
        # 计算误差
        if len(extrinsics_list) > 1:
            errors = []
            for ext_mat in extrinsics_list:
                error = np.linalg.norm(ext_mat[:3, 3] - final_trans)
                errors.append(error)
            
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            print("  平均误差: {:.4f}m, 最大误差: {:.4f}m".format(avg_error, max_error))
    
    return final_extrinsics

def save_results(final_extrinsics, all_tcp_poses):
    """保存最终结果"""
    devices = list(final_extrinsics.keys())
    
    # 创建输出目录
    output_dir = "{}/{}".format(OUTPUT_PATH, output_timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存外参
    np.save("{}/devices.npy".format(output_dir), np.array(devices))
    np.save("{}/extrinsics.npy".format(output_dir), final_extrinsics)
    np.save("{}/intrinsics.npy".format(output_dir), INTRINSICS)
    
    # 保存TCP（使用第一个TCP位姿）
    if len(all_tcp_poses) > 0:
        tcp_mat = all_tcp_poses[0]
        tcp_quaternion = xyz_rot_transform(tcp_mat, from_rep="matrix", to_rep="quaternion")
        np.save("{}/tcp.npy".format(output_dir), tcp_quaternion)
    
    # 保存报告
    report = {
        'method': 'Multi-timestamp Averaging',
        'timestamps_used': target_timestamps,
        'total_measurements': sum(len(all_extrinsics[cam]) for cam in devices),
        'final_extrinsics': {cam_id: ext_mat.tolist() for cam_id, ext_mat in final_extrinsics.items()}
    }
    
    with open("{}/calibration_report.json".format(output_dir), 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n=== 最终结果 ===")
    print("标定完成！结果保存到: {}".format(output_dir))
    print("最终外参平移部分:")
    for cam_id, ext_mat in final_extrinsics.items():
        trans = ext_mat[:3, 3]
        print("  相机 {}: [{:.3f}, {:.3f}, {:.3f}]".format(cam_id, trans[0], trans[1], trans[2]))
    
    return output_dir

def main():
    print("简化多时间戳相机标定工具")
    print("="*50)
    print("方法：收集多个时间戳的外参，去掉离群值后平均")
    print("时间戳: {}".format(target_timestamps))
    print("输出: {}".format(output_timestamp))
    
    # 收集所有外参
    all_extrinsics, all_tcp_poses = collect_all_extrinsics()
    
    # 检查数据
    total_measurements = sum(len(all_extrinsics[cam]) for cam in INTRINSICS.keys())
    if total_measurements == 0:
        print("错误：没有收集到任何有效数据")
        return
    
    print("总共收集到 {} 个外参测量值".format(total_measurements))
    for cam_id in INTRINSICS.keys():
        print("  相机 {}: {} 个测量值".format(cam_id, len(all_extrinsics[cam])))
    
    # 优化外参
    final_extrinsics = optimize_all_extrinsics(all_extrinsics)
    
    # 保存结果
    output_dir = save_results(final_extrinsics, all_tcp_poses)
    
    print("\n可以使用以下命令查看结果:")
    print("python calib_jierui.py")

if __name__ == "__main__":
    main() 