import open3d as o3d
import numpy as np
import os
from PIL import Image
from scipy.spatial.transform import Rotation

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m
    
def angle_between_vectors(v1, v2):
    # 计算两个向量之间的夹角
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def calculate_perpendicular_vector(line_keypoint1, line_keypoint2, point_keypoint, return_two_points=False):
    # Line vector
    line_vec = line_keypoint2 - line_keypoint1
    
    # Vector from line_keypoint1 to point_keypoint
    point_vec = point_keypoint - line_keypoint1
    
    # Project point_vec onto line_vec to find the nearest point on the line
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unitvec)
    nearest_point_on_line = line_keypoint1 + projection_length * line_unitvec
    
    if return_two_points:
        return nearest_point_on_line, point_keypoint
    
    # Perpendicular vector from point to the nearest point on the line
    perpendicular_vector = point_keypoint - nearest_point_on_line
    
    return perpendicular_vector


def calc_angles_two_points_and_line(line_keypoint1, line_keypoint2, point_keypoint1, point_keypoint2):
    # 计算点A point_keypoint1 和B point_keypoint2 到直线(line_keypoint1, line_keypoint2)的垂线
    perpendicular_A = calculate_perpendicular_vector(line_keypoint1, line_keypoint2, point_keypoint1)
    perpendicular_B = calculate_perpendicular_vector(line_keypoint1, line_keypoint2, point_keypoint2)
    
    # 计算A点和B点垂线的夹角
    angle = angle_between_vectors(perpendicular_A, perpendicular_B)
    
    return angle


def rotate_point_around_line(p, o, v, theta):
    """
    Rotate point p around a line defined by direction vector v by angle theta (in radians).
    
    :param p: numpy array (3,), point to rotate.
    :param v: numpy array (3,), direction vector of the line.
    :param theta: float, rotation angle in radians.
    :return: numpy array (3,), rotated point.
    """
    # Step 1: Translate the point and axis so that the axis passes through the origin
    p_prime = p - o  # Translate point p to p'
    
    # Step 2: Normalize the rotation axis vector v
    v = v / np.linalg.norm(v)
    
    # Step 3: Compute the Rodrigues rotation formula
    p_prime_rot = (p_prime * np.cos(theta) + np.cross(v, p_prime) * np.sin(theta) + v * np.dot(v, p_prime) * (1 - np.cos(theta)))
    
    # Step 4: Translate back to the original position
    p_rot = p_prime_rot + o

    return p_rot

## visualize
def draw_point(position, color, radius=0.01):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    mesh_sphere.translate(position)
    mesh_sphere.paint_uniform_color(color)
    mesh_sphere.compute_vertex_normals()
    return mesh_sphere


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
 
 
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 
def get_line(begin=[0, 0, 0], end=[0, 0, 1], radius=0.01, color=[1, 0, 1]):
    # 计算起点到终点的向量
    vec_Arr = np.array(end) - np.array(begin)
    
    # 创建一个LineSet对象，表示一条直线
    line = o3d.geometry.LineSet()
    
    # 定义线的两个端点
    line.points = o3d.utility.Vector3dVector([begin, end])
    
    # 定义线的连接方式（0 -> 1）
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    # 设置线的颜色
    line.paint_uniform_color(color)
    
    return line

def get_thick_line(begin=[0, 0, 0], end=[0, 0, 1], radius=0.01, color=[1, 0, 1]):
    # 计算起点到终点的向量
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    vec_Arr = vec_Arr / vec_len
    
    # 创建一个粗线条的圆柱体，圆柱体的半径即为线条的粗细
    mesh_line = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=vec_len)
    mesh_line.paint_uniform_color(color)
    mesh_line.compute_vertex_normals()
    
    # 计算旋转矩阵将圆柱体对准向量
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_line.rotate(rot_mat, center=np.array([0, 0, 0]))
    
    # 将圆柱体从起点位置平移到正确的位置
    mesh_line.translate(0.5*(np.array(end) - np.array(begin)) + np.array(begin))
    
    return mesh_line

# 计算旋转矩阵，使得圆柱体对准向量
def caculate_align_mat(vec_Arr):
    z_axis = np.array([0, 0, 1])
    # 计算旋转轴（两个向量的叉积）
    axis = np.cross(z_axis, vec_Arr)
    axis_len = np.linalg.norm(axis)
    
    # 如果轴的长度为零（即两个向量已经对齐），则不需要旋转
    if axis_len < 1e-6:
        return np.eye(3)
    
    axis = axis / axis_len  # 单位化旋转轴
    
    # 计算旋转角度
    cos_angle = np.dot(z_axis, vec_Arr)
    sin_angle = axis_len
    
    # 使用Rodrigues' rotation formula计算旋转矩阵
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + K * sin_angle + np.dot(K, K) * (1 - cos_angle)
    return R

def get_arrow(begin=[0,0,0],end=[0,0,1],radius=0.01, color=[1,0,1]):
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    vec_Arr = vec_Arr / vec_len
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
 
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height= radius*5, # 0.2 * 1
        cone_radius=radius * 3 / 2, # 0.06 * 1
        cylinder_height=radius*20, # 0.8 * 1
        cylinder_radius=radius # 0.04 * 1
    )
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.compute_vertex_normals()
 
    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20) # 0.001
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color(color)
    mesh_sphere_begin.compute_vertex_normals()
 
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow, mesh_sphere_begin

def compute_angle_and_distance(P1, d1, P2, d2):
    # 计算角度差
    dot_product = np.dot(d1, d2)
    magnitude_d1 = np.linalg.norm(d1)
    magnitude_d2 = np.linalg.norm(d2)
    cos_theta = dot_product / (magnitude_d1 * magnitude_d2)
    # cos_theta = np.abs(cos_theta)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 夹角theta，单位为弧度
    angle = angle * (180 / np.pi)
    angle = min(angle, 180 - angle)

    # 计算距离差
    v = P2 - P1
    cross_product = np.cross(d1, d2)
    distance = np.abs(np.dot(v, cross_product)) / np.linalg.norm(cross_product)

    return angle, distance

def axis_dir_error(pred_joint, gt_joint):
    pred_joint = pred_joint / np.linalg.norm(pred_joint)
    gt_joint = gt_joint / np.linalg.norm(gt_joint)
    cos_theta = np.dot(pred_joint, gt_joint)

    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 夹角theta，单位为弧度
    angle = angle * (180 / np.pi)
    angle = min(angle, 180 - angle)

    return angle

def point_to_plane_distance(point_A, point_B, normal_vector):
    # 计算点A到点B的向量
    point_to_plane_vector = point_A - point_B  # (bz, 3)
    
    # 计算点到平面垂线方向的投影
    projection_length = np.dot(point_to_plane_vector, normal_vector)  # (bz, 1)
    
    # 计算点到平面的距离
    distance = np.abs(projection_length) / np.linalg.norm(normal_vector)  # (bz, 1)
    
    return distance