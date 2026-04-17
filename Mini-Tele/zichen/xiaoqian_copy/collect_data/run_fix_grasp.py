import os
import time
import tqdm
import configargparse
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import fpsample
import numpy as np
import threading
import cv2
import pickle

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str)
    # record config
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episode', type=int, default=0)

    args = parser.parse_args()
    return args

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


def record(args, robot, camera):

    if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}"):
        os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")
    else:
        #print("NOT RECORDED")
        #return
        os.system(f"rm -r {args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")

    for save_key in ["tcp", "joint_position", "gripper_width", "cam_top_rgb", "cam_top_depth"]:
        if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}"):
            os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}")

    timestamp = 0
    while True:
        
        ## robot state
        tcp = robot.get_tcp()
        gripper_width = robot.get_gripper_width()
        joint_position = robot.get_joint_positions()
        print("tcp", tcp, "joint_position", joint_position, "gripper_width", gripper_width)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/tcp/{str(timestamp).zfill(10)}.npy", tcp)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/joint_position/{str(timestamp).zfill(10)}.npy", joint_position)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/gripper_width/{str(timestamp).zfill(10)}.npy", gripper_width)

        ## camera
        color, depth = camera.get_data(hole_filling=False)
        cv2.imwrite(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_rgb/{str(timestamp).zfill(10)}.png", color)
        pickle.dump(depth, open(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_depth/{str(timestamp).zfill(10)}.pkl", "wb"))

        time.sleep(1/30) # 30 Hz
        timestamp += 1


if __name__ == '__main__':

    args = config_parse()

    camera_loaded = False
    robot_loaded = False
    task = -1     #closing as 1, open as -1
    joint_type = 1 # rotate as 0
    joint_re = 45
    time_steps = 10
    vis = True

    temp_service_path = f"{args.root_dir}/{args.global_key}/init_info/init_grasp.npz"
    
    HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
    
    
    try:
        print("===> initializing camera")
        start_time = time.time()
        camera = CameraD400(camera_id="001622071104") # top-view: 001622071104
        # camera = CameraD400(camera_id="233722071807")
        camera_loaded = True
        end_time = time.time()
        print("===> camera initialized", end_time - start_time)

        print("===> initializing robot")
        start_time = time.time()
        robot = Flexiv()
        robot_loaded = True
        robot.movePosePrimitive(HOME_POSE)
        end_time = time.time()
        print("===> robot initialized", end_time - start_time)

        # assert 0

        print("===> reading response")
        start_time = time.time()
        
        # cam2robot = robot.readCamPose()
        # print("pose",cam2robot)
        cam2robot = np.array([[ 0.05262318, -0.99199495 , 0.11479036 , 0.56598359],
                            [-0.99846966, -0.05030948 , 0.02296276 ,-0.01270892],
                            [-0.0170039 , -0.11582306 ,-0.99312431,  0.65807974],
                            [ 0.   ,       0.    ,      0.        ,  1.        ]])
        # cam2robot = np.load(f"{args.root_dir}/{args.global_key}/init_capture/cam2robot.npz")
        robot2cam = np.linalg.inv(cam2robot)

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
            # print("########",base_joint_base)
            # print("$$$$$$$$",base_joint_direction)

    
        print("===> starting manipulation")
        if args.record:
            thread = threading.Thread(
                                    target=record, 
                                    kwargs={"args":args, "robot":robot, "camera":camera},
                                    daemon=True)
            thread.start()

        base_gripper_pose = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0],
                             [-0.05031023 , 0.99846963 , 0.02296254 , 0],
                             [-0.11582299,  0.0170036 , -0.99312432  , 0],
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
        base_gripper_pose[:3, 3] += base_affordable_position # offset
        base_gripper_pose[:3, 3] += np.array([0.065, 0.07, -0.02])

        start_time = time.time()
        if num_grasps == 0:
            exit(1)
        else:

            print("done pre pose")
            # robot.movePosePrimitive(base_gripper_pose)
            
            from utilities.transformation import rotation_transform, xyz_rot_transform
            tcp_pose = xyz_rot_transform(
                                    base_gripper_pose,
                                    from_rep="matrix", 
                                    to_rep="quaternion", 
                                ) # quat wxyz
            print("base_gripper_pose", base_gripper_pose)
            print("tcp_pose", tcp_pose)
            robot.send_tcp_pose(tcp_pose)
            time.sleep(1)
            robot.send_gripper_state(0.005 ,0.1, 20)
            time.sleep(1)

            # # move
            # for time_step in tqdm.trange(10):
            #     current_EE2robot = robot.readPose()
            #     if joint_type == 0:
            #         rotation_angle = 5 * 1 * 1 / 180.0 * np.pi
            #         delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
            #     elif joint_type == 1:
            #         translation_distance = -5.0 * task / 100.0
            #         delta_pose = tf.translation_matrix(base_joint_direction * translation_distance)
            #     else:
            #         raise ValueError
            #     target_EE2robot = delta_pose @ current_EE2robot
            #     # robot.movePosePrimitive(target_EE2robot)
            #     tcp_pose = xyz_rot_transform(
            #                         target_EE2robot,
            #                         from_rep="matrix", 
            #                         to_rep="quaternion", 
            #                     ) # quat wxyz
            #     robot.send_tcp_pose(tcp_pose)
            #     time.sleep(1)
            #     # time.sleep(0.1)
        end_time = time.time()
        robot.send_gripper_state(0.5 ,0.1, 20)
        print("===> manipulation done", end_time - start_time)
        # robot.send_gripper_state(0.8 ,0.1, 20)
        # robot.homing()
    except Exception as e:
        print(e)
        if camera_loaded:
            del camera
        if robot_loaded:
            del robot
