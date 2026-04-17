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

# 标定参数
MARKER_LENGTH = 0.15  # ArUco标记尺寸（米）
USE_SUBPIXEL_REFINEMENT = True  # 使用亚像素精度优化
USE_POSE_REFINEMENT = True  # 使用位姿优化
DISTORTION_COEFFS = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # 5个畸变系数

# 要标定的时间戳
calib_timestamp = "2025-06-05_22_17"

# 输出路径
output_dir = f"data/calib/{calib_timestamp}_optimized_single"

def create_optimized_detector_params():
    """创建优化的ArUco检测参数"""
    parameters = cv2.aruco.DetectorParameters()
    
    # 优化检测参数
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    
    # 轮廓检测参数
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    
    # 角点检测参数
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    
    # 精度参数
    parameters.markerBorderBits = 1
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # 错误检测参数
    parameters.maxErroneousBitsInBorderRate = 0.35
    parameters.errorCorrectionRate = 0.6
    
    return parameters

def detect_aruco_with_refinement(image, camera_matrix, dist_coeffs, parameters):
    """使用亚像素精度检测ArUco标记"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 图像预处理 - 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 检测ArUco标记
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if len(corners) == 0:
        return None, None, None, None
    
    # 亚像素角点优化
    if USE_SUBPIXEL_REFINEMENT:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), criteria)
    
    # 估计位姿
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
    
    # 位姿精化（如果启用）
    if USE_POSE_REFINEMENT and len(corners) > 0:
        # 使用solvePnPRefineLM进行更精确的位姿估计
        object_points = np.array([
            [-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
            [MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
            [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
            [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
        ], dtype=np.float32)
        
        refined_rvecs = []
        refined_tvecs = []
        
        for i, corner in enumerate(corners):
            # 重新排列角点顺序
            image_points = corner[0].astype(np.float32)
            
            # 使用LM算法优化位姿
            success, rvec_refined, tvec_refined = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs,
                rvecs[i], tvecs[i], useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            if success:
                refined_rvecs.append(rvec_refined)
                refined_tvecs.append(tvec_refined)
            else:
                refined_rvecs.append(rvecs[i])
                refined_tvecs.append(tvecs[i])
        
        rvecs = np.array(refined_rvecs)
        tvecs = np.array(refined_tvecs)
    
    return corners, ids, rvecs, tvecs

def analyze_detection_quality(corners, camera_matrix):
    """分析检测质量"""
    if corners is None or len(corners) == 0:
        return {"quality": "failed", "score": 0}
    
    corner = corners[0][0]  # 第一个标记的角点
    
    # 计算标记面积
    area = cv2.contourArea(corner)
    
    # 计算角点到图像中心的距离
    image_center = np.array([camera_matrix[0,2], camera_matrix[1,2]])
    marker_center = np.mean(corner, axis=0)
    center_distance = np.linalg.norm(marker_center - image_center)
    
    # 计算标记的规则性（矩形程度）
    rect = cv2.minAreaRect(corner)
    rect_area = rect[1][0] * rect[1][1]
    rectangularity = area / rect_area if rect_area > 0 else 0
    
    # 综合质量评分
    area_score = min(area / 10000, 1.0)  # 面积越大越好
    center_score = max(0, 1.0 - center_distance / 500)  # 距离中心越近越好
    rect_score = rectangularity  # 越接近矩形越好
    
    total_score = (area_score * 0.4 + center_score * 0.3 + rect_score * 0.3)
    
    if total_score > 0.8:
        quality = "excellent"
    elif total_score > 0.6:
        quality = "good"
    elif total_score > 0.4:
        quality = "fair"
    else:
        quality = "poor"
    
    return {
        "quality": quality,
        "score": total_score,
        "area": area,
        "center_distance": center_distance,
        "rectangularity": rectangularity
    }

def calibrate_camera_optimized(camera_id, calib_timestamp):
    """优化的相机标定"""
    print(f"\n=== 优化标定相机 {camera_id} ===")
    
    camera_matrix = INTRINSICS[camera_id][:3, :3]
    image_path = f"data/calib/raw_data/{calib_timestamp}_{camera_id}/rgb.jpg"
    robot_pose_path = f"data/calib/raw_data/{calib_timestamp}_{camera_id}/robot_pose.json"
    
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在 {image_path}")
        return None, None
    
    if not os.path.exists(robot_pose_path):
        print(f"错误：位姿文件不存在 {robot_pose_path}")
        return None, None
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None, None
    
    print(f"图像尺寸: {image.shape}")
    
    # 创建优化的检测参数
    parameters = create_optimized_detector_params()
    
    # 检测ArUco标记
    corners, ids, rvecs, tvecs = detect_aruco_with_refinement(
        image, camera_matrix, DISTORTION_COEFFS, parameters)
    
    if corners is None:
        print("未检测到ArUco标记")
        return None, None
    
    print(f"检测到 {len(corners)} 个标记")
    
    # 分析检测质量
    quality_info = analyze_detection_quality(corners, camera_matrix)
    print(f"检测质量: {quality_info['quality']} (评分: {quality_info['score']:.3f})")
    print(f"  标记面积: {quality_info['area']:.1f} 像素²")
    print(f"  距离中心: {quality_info['center_distance']:.1f} 像素")
    print(f"  矩形度: {quality_info['rectangularity']:.3f}")
    
    # 获取位姿数据
    rvec = rvecs[0]
    tvec = tvecs[0]
    
    print(f"旋转向量: {rvec.flatten()}")
    print(f"平移向量: {tvec.flatten()}")
    
    # 计算重投影误差
    object_points = np.array([
        [-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
        [MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
        [MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
        [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
    ], dtype=np.float32)
    
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, DISTORTION_COEFFS)
    
    reprojection_error = cv2.norm(corners[0][0], projected_points.reshape(-1, 2), cv2.NORM_L2) / 4
    print(f"重投影误差: {reprojection_error:.3f} 像素")
    
    # 转换为4x4外参矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = rotation_matrix
    ext_mat[:3, 3] = tvec.flatten()
    
    # 读取机器人位姿
    with open(robot_pose_path, 'r') as f:
        robot_pose = json.load(f)
    
    # 保存检测可视化图像（无窗口模式）
    vis_image = image.copy()
    cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
    cv2.drawFrameAxes(vis_image, camera_matrix, DISTORTION_COEFFS, rvec, tvec, 0.1)
    
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/detection_{camera_id}.jpg", vis_image)
    print(f"检测可视化已保存: {output_dir}/detection_{camera_id}.jpg")
    
    return ext_mat, robot_pose

def main():
    print("优化单组相机标定工具")
    print("="*50)
    print(f"标定时间戳: {calib_timestamp}")
    print(f"ArUco标记尺寸: {MARKER_LENGTH}m")
    print(f"亚像素优化: {'启用' if USE_SUBPIXEL_REFINEMENT else '禁用'}")
    print(f"位姿精化: {'启用' if USE_POSE_REFINEMENT else '禁用'}")
    print(f"输出目录: {output_dir}")
    
    devices = []
    extrinsics = {}
    robot_poses = {}
    
    # 标定每个相机
    for camera_id in INTRINSICS.keys():
        ext_mat, robot_pose = calibrate_camera_optimized(camera_id, calib_timestamp)
        
        if ext_mat is not None:
            extrinsics[camera_id] = ext_mat
            robot_poses[camera_id] = robot_pose
            devices.append(camera_id)
            print(f"✓ 相机 {camera_id} 标定成功")
        else:
            print(f"✗ 相机 {camera_id} 标定失败")
    
    if len(devices) == 0:
        print("错误：没有成功标定的相机")
        return
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{output_dir}/devices.npy", np.array(devices))
    np.save(f"{output_dir}/extrinsics.npy", extrinsics)
    np.save(f"{output_dir}/intrinsics.npy", INTRINSICS)
    
    # 使用第一个相机的机器人位姿
    first_camera = devices[0]
    tcp_mat = robot_poses[first_camera]
    tcp_quaternion = xyz_rot_transform(tcp_mat, from_rep="matrix", to_rep="quaternion")
    np.save(f"{output_dir}/tcp.npy", tcp_quaternion)
    
    # 保存标定报告
    report = {
        "timestamp": calib_timestamp,
        "marker_length": MARKER_LENGTH,
        "subpixel_refinement": USE_SUBPIXEL_REFINEMENT,
        "pose_refinement": USE_POSE_REFINEMENT,
        "cameras_calibrated": devices,
        "extrinsics": {cam_id: ext_mat.tolist() for cam_id, ext_mat in extrinsics.items()}
    }
    
    with open(f"{output_dir}/calibration_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== 标定完成 ===")
    print(f"成功标定 {len(devices)} 个相机")
    print(f"结果已保存到: {output_dir}")
    print(f"\n可以将以下时间戳添加到可视化比较中:")
    print(f'timestamps.append("{calib_timestamp}_optimized_single")')

if __name__ == "__main__":
    main() 