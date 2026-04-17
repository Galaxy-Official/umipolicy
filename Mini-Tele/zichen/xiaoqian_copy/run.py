import os
import time
import tqdm
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import fpsample
import numpy as np

def calculate_rotation_axis(point1, point2):
    direction = point2 - point1
    base_joint_direction = direction / np.linalg.norm(direction)
    base_joint_base = point1
    return base_joint_direction, base_joint_base
def rotate_vector(vec, axis='z'):
    # 定义旋转矩阵
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
    elif axis == 'y':
        rotation_matrix = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])
    elif axis == 'z':
        rotation_matrix = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    # 将旋转矩阵应用到方向向量
    rotated_vec = np.dot(rotation_matrix, vec)
    return rotated_vec

def transform_vector_and_point(camera_vector, camera_point, T_camera2robot):
    """
    将相机坐标系下的旋转方向向量和旋转基点转换为机器人坐标系下的对应值。
    
    :param camera_vector: 相机坐标系下的旋转方向向量 (3,)
    :param camera_point: 相机坐标系下的旋转基点 (3,)
    :param T_camera2robot: 从相机坐标系到机器人坐标系的变换矩阵 (4,4)
    :return: robot_vector, robot_point
        robot_vector: 机器人坐标系下的旋转方向向量 (3,)
        robot_point: 机器人坐标系下的旋转基点 (3,)
    """
    
    # 旋转方向向量转换 (旋转矩阵部分)
    rotation_matrix = T_camera2robot[:3, :3]  # 提取旋转矩阵部分
    robot_vector = np.dot(rotation_matrix, camera_vector)  # 转换方向向量
    
    # 旋转基点转换 (齐次坐标转换)
    camera_point_homogeneous = np.append(camera_point, 1)  # 将基点转换为齐次坐标
    robot_point_homogeneous = np.dot(T_camera2robot, camera_point_homogeneous)  # 转换位置
    robot_point = robot_point_homogeneous[:3]  # 提取转换后的3D坐标
    
    return robot_vector, robot_point
if __name__ == '__main__':
    camera_loaded = False
    robot_loaded = False
    task = -1     #closing as 1, open as -1
    joint_type = 1 # rotate as 0
    joint_re = 45
    time_steps = 10
    vis = True
    temp_observation_path = "./temp_data/observation.npz"
    temp_service_path = "./temp_data_server/service.npz"
    temp_flag_path = "./temp_data_server/flag.npy"
    remote_repo_path = "/home/rhos/xiaoyang"                   # TODO: set your own remote repo path
    remote_observation_path = f"{remote_repo_path}/temp_data/observation.npz"
    remote_service_path = f"{remote_repo_path}/temp_data/service.npz"
    remote_flag_path = f"{remote_repo_path}/temp_data/flag.npy"
   
    HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
    
    HOME_POSE_VERTICAL=np.array([[-0.686003 ,  -0.05910128 , 0.7251944,   0.33998513],
                                [-0.09462285,  0.99547792, -0.00838056, -0.08529349],
                                [-0.72141971, -0.07436905 ,-0.68849317 , 0.84215456],
                                [ 0.      ,    0.  ,        0.     ,     1.        ]])
        
    
    point1 = np.array([0.44, -0.13, 0.30])##Attention!! Flexiv's coordinate
    point2 = np.array([0.66, -0.13, 0.30])
    #base_joint_direction, base_joint_base = calculate_rotation_axis(point1,point2)
    try:
        print("===> initializing camera")
        start_time = time.time()
        camera = CameraD400()
        camera_loaded = True
        end_time = time.time()
        print("===> camera initialized", end_time - start_time)

        print("===> initializing robot")
        start_time = time.time()
        robot = Flexiv()
        robot_loaded = True
        end_time = time.time()
        print("===> robot initialized", end_time - start_time)
        # robot.movePosePrimitive(HOME_POSE)
        time.sleep(1.0)
        print("===> getting observation")
        start_time = time.time()
        color, depth = camera.get_data(hole_filling=False)
        depth_sensor = camera.pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        xyzrgb = camera.getXYZRGB(color, depth, np.identity(4), np.identity(4), camera.getIntrinsics(), inpaint=False, depth_scale=depth_scale)
        # xyzrgb = xyzrgb[xyzrgb[:, 2] <= 1.5, :]
        xyzrgb = xyzrgb[xyzrgb[:, 2] > 0.05, :]
        cam_pc = xyzrgb[:, 0:3]
        pc_color = xyzrgb[:, 3:6]
        end_time = time.time()
        print("===> observation got", end_time - start_time)
        points = cam_pc
        colors = pc_color
        num_points = 70000
        print(points.shape)
        # while points.shape[0] < num_points:
        #     points = np.concatenate([points, points], 0)
        #     colors = np.concatenate([colors, colors], 0)
        # fps_idx = fpsample.fps_npdu_kdtree_sampling(points, num_points)
        # cam_pc = points[fps_idx]
        # pc_color = colors[fps_idx]


        print("===> preprocessing observation")
        start_time = time.time()
        #EE2robot = robot.readPose()
        cam2robot = robot.readCamPose()
        robot2cam = np.linalg.inv(cam2robot)
        base_pc = transform_pc(cam_pc, cam2robot)
        space_mask_x = np.logical_and(base_pc[:, 0] > -0.13, base_pc[:, 0] < 1.5)
        space_mask_y = np.logical_and(base_pc[:, 1] > -0.29, base_pc[:, 1] < 0.55)
        space_mask_z = base_pc[:, 2] > 0.42
        # # space_mask_z = base_pc[:, 2] > 0.55             # microwave: pad + safe (rotate)
        # # space_mask_z = base_pc[:, 2] > 0.52             # refrigerator: storagefurniture
        # # space_mask_z = base_pc[:, 2] > 0.4              # safe: pad + microwave
        # # space_mask_z = base_pc[:, 2] > 0.27             # storagefurniture: microwave
        # # space_mask_z = base_pc[:, 2] > 0.27             # drawer: microwave
        #space_mask_z = base_pc[:, 2] > 0.4              # washingmachine: pad + microwave
        space_mask = np.logical_and(np.logical_and(space_mask_x, space_mask_y), space_mask_z)
        base_pc_space = base_pc[space_mask, :]
        pc_color_space = pc_color[space_mask, :]
        cam_pc_space = transform_pc(base_pc_space, robot2cam)
        end_time = time.time()
        print("===> observation preprocessed", end_time - start_time)
        

        print("===> sending request")
        start_time = time.time()
        # np.savez(temp_observation_path, point_cloud=cam_pc_space, rgb=pc_color_space)
        # time.sleep(0.5)
        # while not (os.path.isfile(temp_observation_path) and os.access(temp_observation_path, os.R_OK)):
        #     time.sleep(0.1)
        # send(temp_observation_path, remote_observation_path, 
        #      remote_ip=remote_ip, port=port, username=username, key_filename=key_filename)
        # time.sleep(0.5)
        # os.remove(temp_observation_path)
        # np.savez("/home/rhos/xiaoyang/temp_data/observation.npz", point_cloud=cam_pc_space, rgb=pc_color_space)
        if vis:
            visualize(cam_pc_space, pc_color_space, whether_frame=True, whether_bbox=True, window_name="observation")
        end_time = time.time()
        print("===> request sent", end_time - start_time)

        print("===> reading response")
        start_time = time.time()
        while True:
            got_service = np.load(temp_flag_path).item()
            print("got_service",got_service)
            if got_service:
                # os.remove(temp_flag_path)
                break
            else:
                time.sleep(0.5)
        time.sleep(0.5)
        service = np.load(temp_service_path, allow_pickle=True)
        num_grasps = service['num_grasps']
        print("num_grasps",num_grasps)
        if num_grasps == 0:
            print("no grasps detected")
        else:
            cam_joint_base = service['joint_base']
            cam_joint_direction = service['joint_direction']
            cam_affordable_position = service['affordable_position']
            joint_type = service['joint_type']
            joint_re = service['joint_re']
            grasp_score = service['grasp_score']
            grasp_width = service['grasp_width']
            grasp_depth = service['grasp_depth']
            grasp_affordance = service['grasp_affordance']
            cam_grasp_translation = service['grasp_translation']
            cam_grasp_rotation = service['grasp_rotation']
            cam_grasp_pose = np.eye(4)
            cam_grasp_pose[:3, 3] = cam_grasp_translation
            cam_grasp_pose[:3, :3] = cam_grasp_rotation
            base_joint_base = transform_pc(cam_joint_base[None, :], cam2robot)[0]
            base_joint_direction = transform_dir(cam_joint_direction[None, :], cam2robot)[0]
            base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
            base_grasp_pose = cam2robot @ cam_grasp_pose
            # base_grasp_pose = cam2robot @ cam_grasp_pose
            base_joint_direction, base_joint_base = transform_vector_and_point(cam_joint_direction, cam_joint_base, cam2robot)
            print("########",base_joint_base)
            print("$$$$$$$$",base_joint_direction)

            base_grasp_pose[:3, 3] += (grasp_depth - 0.05) * base_grasp_pose[:3, 0] # TODO: hardcode to avoid collision
            print("base_grasp_pose",base_grasp_pose)
            if joint_type == 0:
                # TODO: only for horizontal grasp to avoid singular robot state
                flip = np.arccos(np.dot(base_grasp_pose[:3, 2], np.array([0., 0., 1.]))) / np.pi * 180.0 < 45
                if flip:
                    print("flipped")
                    base_grasp_pose[:3, 1] = -base_grasp_pose[:3, 1]
                    base_grasp_pose[:3, 2] = -base_grasp_pose[:3, 2]
                rotate = base_grasp_pose[:3, 0][2] > 0
                if rotate:
                    print("rotated")
                    target_x_axis = base_grasp_pose[:3, 0].copy()
                    target_x_axis[2] = -target_x_axis[2]
                    rotation_angle = np.arccos(np.dot(base_grasp_pose[:3, 0], target_x_axis))
                    rotation_direction = np.array([base_grasp_pose[:3, 0][0], base_grasp_pose[:3, 0][1]])
                    rotation_direction /= np.linalg.norm(rotation_direction)
                    rotation_direction = np.array([-rotation_direction[1], rotation_direction[0], 0.])
                    rotation_matrix = tf.rotation_matrix(angle=rotation_angle, direction=rotation_direction, point=base_grasp_pose[:3, 3])
                    base_grasp_pose = rotation_matrix @ base_grasp_pose
            elif joint_type == 1:
                horizontal = np.arccos(np.dot(base_grasp_pose[:3, 0], np.array([1., 0., 0.]))) / np.pi * 180.0 < 45
                if horizontal:
                    print("horizontal")
                else:
                    print("vertical")
            else:
                raise ValueError
            base_pre_grasp_pose = base_grasp_pose.copy()
            base_pre_grasp_pose[:3, 3] -= 0.05 * base_pre_grasp_pose[:3, 0]
            g2g = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
            base_gripper_pose = np.eye(4)
            base_gripper_pose[:3, :3] = base_grasp_pose[:3, :3] @ g2g
            base_gripper_pose[:3, 3] = base_grasp_pose[:3, 3]
            base_pre_gripper_pose = np.eye(4)
            base_pre_gripper_pose[:3, :3] = base_pre_grasp_pose[:3, :3] @ g2g
            base_pre_gripper_pose[:3, 3] = base_pre_grasp_pose[:3, 3]
        time.sleep(0.5)
        # os.remove(temp_service_path)
        end_time = time.time()
        print("===> response read", end_time - start_time)
        if vis:
            if num_grasps != 0:
                visualize(base_pc_space, pc_color_space, 
                          joint_translations=base_joint_base[None, :], joint_rotations=base_joint_direction[None, :], affordable_positions=base_affordable_position[None, :], 
                          grasp_poses=base_grasp_pose[None, ...], grasp_widths=np.array([grasp_width]), grasp_depths=np.array([0.]), grasp_affordances=np.array([grasp_affordance]), 
                          whether_frame=True, whether_bbox=True, window_name="prediction")
                

        
        GRASP_POSE_PRE = np.array([[ 0.14552466 , 0.85055408 , 0.50535169 , 0.64934784],
 [ 0.15323511 , 0.4852481 , -0.86084452, -0.04201242],
 [-0.97741576 , 0.20271173, -0.05971919 , 0.42870945],
 [ 0.  ,        0.      ,    0.      ,    1.        ]])
        GRASP_POSE = np.array([[-0.16322684  ,0.59797017  ,0.78472204 , 0.68635893],
 [ 0.1441217  , 0.80130895 ,-0.58063147 ,-0.18894374],
 [-0.97600509 , 0.01832084, -0.2169756  , 0.42943752],
 [ 0.      ,    0.   ,       0.    ,      1.        ]])

        print("===> starting manipulation")
        start_time = time.time()
        if num_grasps == 0:
            exit(1)
        else:

            # robot.movePosePrimitive(GRASP_POSE_PRE)
            # time.sleep(1)

            robot.movePosePrimitive(GRASP_POSE)
            time.sleep(1)
            robot.send_gripper_state(0.01 ,0.1, 20)
            time.sleep(1)
            robot.movePosePrimitive(GRASP_POSE_PRE)
            time.sleep(1)

            # robot.send_gripper_state(0.01 ,0.1, 20)
            
            # move
            # for time_step in tqdm.trange(13):
            #     current_EE2robot = robot.readPose()
            #     if joint_type == 0:
            #         rotation_angle = 10 * 1 * 1 / 180.0 * np.pi
            #         delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
            #     elif joint_type == 1:
            #         translation_distance = -5.0 * task / 100.0
            #         delta_pose = tf.translation_matrix(base_joint_direction * translation_distance)
            #     else:
            #         raise ValueError
            #     target_EE2robot = delta_pose @ current_EE2robot
            #     robot.movePosePrimitive(target_EE2robot)
            #     time.sleep(0.3)
        end_time = time.time()
        print("===> manipulation done", end_time - start_time)
        # robot.send_gripper_state(0.8 ,0.1, 20)
        # robot.homing()
    except Exception as e:
        print(e)
        if camera_loaded:
            del camera
        if robot_loaded:
            del robot