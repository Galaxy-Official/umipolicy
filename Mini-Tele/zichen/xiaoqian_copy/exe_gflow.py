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


if __name__ == '__main__':
    HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
    
    HOME_POSE_VERTICAL=np.array([[-0.686003 ,  -0.05910128 , 0.7251944,   0.33998513],
                                [-0.09462285,  0.99547792, -0.00838056, -0.08529349],
                                [-0.72141971, -0.07436905 ,-0.68849317 , 0.84215456],
                                [ 0.      ,    0.  ,        0.     ,     1.        ]])
    camera_loaded = False
    robot_loaded = False
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
        num_points = 50000
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
        space_mask_z = base_pc[:, 2] > 0.05
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
        # np.savez("/home/rhos/xiaoyang/temp_data/observation.npz", point_cloud=cam_pc_space, rgb=pc_color_space)
        
        service_flow = np.load("/home/rhos/xiaoyang/general_flow/demo/output/bucket_0_hand/robot_exec.pkl", allow_pickle=True)
        motion_plan_SE3 = service_flow['motion_plan_SE3']
        gripper_3D_pos = service_flow['gripper_3d_pos']
        print("gripper_3D_pos",gripper_3D_pos)

        temp_service_path = "./temp_data_server/service.npz"
        temp_flag_path = "./temp_data_server/flag.npy"

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
        # print("###############################")
        cam_affordable_position = gripper_3D_pos
        base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
        base_grasp_pose = cam2robot @ cam_grasp_pose
        # base_grasp_pose = cam2robot @ cam_grasp_pose
        print("contact_point",base_affordable_position)


        ####################TEST#########################
        position = np.array([cam_affordable_position[0], cam_affordable_position[1], cam_affordable_position[2], 1])
        grasp_cam =  motion_plan_SE3[0] @ cam_grasp_pose
        print("next_grasp_pose",cam2robot @ grasp_cam)
        for motion_SE in motion_plan_SE3:
            transformed_point = position @ motion_SE
            position = np.array([transformed_point[0], transformed_point[1], transformed_point[2]])
            trans = transform_pc(position[None, :], cam2robot)[0]
            print("transed_pos",trans)
            position = np.array([transformed_point[0], transformed_point[1], transformed_point[2], 1])
        ####################TEST#########################


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
        visualize(base_pc_space, pc_color_space, 
                          joint_translations=base_joint_base[None, :], joint_rotations=base_joint_direction[None, :], affordable_positions=base_affordable_position[None, :], 
                          grasp_poses=base_grasp_pose[None, ...], grasp_widths=np.array([grasp_width]), grasp_depths=np.array([0.]), grasp_affordances=np.array([grasp_affordance]), 
                          whether_frame=True, whether_bbox=True, window_name="prediction")
        

        print("base_grasp_pose",base_grasp_pose)
        # exit(0)
        GRASP_POSE_PRE = np.array([[ 0.20746573 , 0.08061335, -0.97491513 , 0.58246665],
                                    [ 0.92332988 ,-0.34534126 , 0.16793276, -0.24290056],
                                    [-0.32314078, -0.93500858 ,-0.14607918 , 0.18465904],
                                    [ 0.   ,       0. ,         0.      ,    1.        ]])
        

        GRASP_POSE = np.array([[-0.99108104 ,-0.11308203 , 0.0705041  , 0.71214777],
                                [-0.00565147 , 0.56426011 , 0.82557773 ,-0.16355672],
                                [-0.13314065 , 0.81781598 ,-0.55986658 , 0.16117968],
                                [ 0.   ,       0.      ,    0.       ,   1.        ]])
        

        g2g_x = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
        
        g2g_y = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]])
        
        g2g_z = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
        
        GRASP_POSE_PRE[:3, :3] = GRASP_POSE_PRE[:3, :3] @ g2g_x
        GRASP_POSE_PRE[:3, :3] = GRASP_POSE_PRE[:3, :3] @ g2g_y
        GRASP_POSE_PRE[:3, :3] = GRASP_POSE_PRE[:3, :3] @ g2g_z
        exit(0)
        print("===> starting manipulation")
        if num_grasps == 0:
            exit(1)
        else:

            robot.movePosePrimitive(GRASP_POSE_PRE)
            time.sleep(1)
            # exit(0)
            # robot.movePosePrimitive(GRASP_POSE)
            # time.sleep(1)

            robot.send_gripper_state(0.01 ,0.1, 20)
            
            # move
            for motion_SE3 in motion_plan_SE3:
                current_pose = robot.readPose()
                print("current",current_pose)
                delta = robot2cam @ motion_SE3
                print("delta",delta)
                target_EE2robot = current_pose @ delta
                print("target",target_EE2robot)
                # robot.movePosePrimitive(delta)
                # time.sleep(1)
        print("===> manipulation done", end_time - start_time)
        robot.send_gripper_state(0.8 ,0.1, 20)
        
    except Exception as e:
        print(e)
        if camera_loaded:
            del camera
        if robot_loaded:
            del robot
