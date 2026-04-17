import os
import configargparse
import time
import numpy as np
import torch
import pickle
import open3d as o3d

from utilities.env_utils import setup_seed
from utilities.data_utils import transform_pc, transform_dir
from utilities.metrics_utils import invaffordance_metrics, invaffordances2affordance
from utilities.constants import seed, max_grasp_width


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # data config
    parser.add_argument('--cat', type=str, default='Microwave', help='the category of the object')
    # grasp config
    # parser.add_argument('--graspnet', action='store_true', help='whether call graspnet')
    parser.add_argument('--gsnet_weight_path', type=str, default='anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='the path to graspnet weight')
    parser.add_argument('--max_grasp_width', type=float, default=max_grasp_width, help='the max width of the gripper')
    # task config
    parser.add_argument('--selected_part', type=int, default=0, help='the selected part of the object')
    # others
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str)
    parser.add_argument('--selected_grasp_idx', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = config_parse()
    print("please clear temporary data directory")

    def calculate_rotation_axis(point1, point2):
        """
        计算旋转轴的方向和旋转中心
        输入:
        - point1: 第一个点 (x1, y1, z1)
        - point2: 第二个点 (x2, y2, z2)

        返回:
        - base_joint_direction: 旋转轴的方向 (单位向量)
        - base_joint_base: 旋转中心 (point1)
        """
        # 计算旋转轴的方向
        direction = point2 - point1
        
        # 计算单位化方向向量 (旋转轴方向)
        base_joint_direction = direction / np.linalg.norm(direction)
        
        # 旋转中心就是 point1
        base_joint_base = point1
        
        return base_joint_direction, base_joint_base

    # pred_selected_affordable_position = np.array([0.04, 0.08, 0.49])
    # point1 = np.array([0.16, -0.13, 0.61])
    # point2 = np.array([-0.10, -0.12, 0.61])

    # init_keypoints = pickle.load(open(f"{args.root_dir}/{args.global_key}")) # tTODO mp
    # point1, point2, pred_selected_affordable_position = init_keypoints

    pred_selected_affordable_position = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_contact.pkl", "rb"))[0]
    point1 = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_axis1.pkl", "rb"))[0]
    point2 = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_axis2.pkl", "rb"))[0]
    print("pred_selected_affordable_position", pred_selected_affordable_position)
    print("point1", point1)
    print("point2", point2)

    pred_selected_joint_direction,pred_selected_joint_base = calculate_rotation_axis(point1, point2)


    setup_seed(args.seed)
    # temp_request_path = 'temp_data/observation.npz'
    # temp_response_path = 'temp_data_server/service.npz'
    # temp_flag_path = 'temp_data_server/flag.npy'
    if args.cat == "Microwave":
        joint_types = [0]
        joint_res = [-1]
    elif args.cat == "Refrigerator":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "Safe":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "StorageFurniture":
        joint_types = [1, 0]
        joint_res = [0, -1]
    elif args.cat == "Drawer":
        joint_types = [1, 1, 1]
        joint_res = [0, 0, 0]
    elif args.cat == "WashingMachine":
        joint_types = [0]
        joint_res = [-1]
    else:
        raise ValueError(f"Unknown category {args.cat}")

    # if args.graspnet:
    print("===> loading graspnet")
    start_time = time.time()
    from munch import DefaultMunch
    from gsnet import AnyGrasp
    grasp_detector_cfg = {
        'checkpoint_path': args.gsnet_weight_path, 
        'max_gripper_width': args.max_grasp_width, 
        'gripper_height': 0.03, 
        'top_down_grasp': False, 
        'add_vdistance': True, 
        'debug': True
    }
    grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
    grasp_detector = AnyGrasp(grasp_detector_cfg)
    grasp_detector.load_net()
    
    end_time = time.time()
    print(f"===> loaded graspnet {end_time - start_time}")
    
    # while True:
    # print("===> listening to request")
    start_time = time.time()
    serviced = np.array(False)
    # np.save(temp_flag_path, serviced)
    got_request = False
    # while not got_request:
    #     got_request = os.path.exists(temp_request_path)
    #     print(got_request)
    #     time.sleep(5.0)             # NOTE: hardcode to be longer than the writing time
    #     if got_request:
    #         while not (os.path.isfile(temp_request_path) and os.access(temp_request_path, os.R_OK)):
    #             time.sleep(0.1)
    #     else:
    #         time.sleep(0.1)

    point_cloud = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/init_pc.pkl", "rb"))
    cam_pc = point_cloud[:, :3]
    pc_rgb = point_cloud[:, 3:]

    import fpsample
    num_points = 5000
    fps_idx = fpsample.fps_npdu_kdtree_sampling(cam_pc, num_points)
    cam_pc = cam_pc[fps_idx]
    pc_rgb = pc_rgb[fps_idx]
    print("cam_pc", cam_pc.shape)
    print("pc_rgb", pc_rgb.shape)

    # # 创建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cam_pc)
    # pcd.colors = o3d.utility.Vector3dVector(pc_rgb)

    # # 调用可视化窗口显示点云
    # o3d.visualization.draw_geometries([pcd])

    # observation = np.load(temp_request_path, allow_pickle=True)
    # cam_pc = observation['point_cloud']
    # print(cam_pc)
    # pc_rgb = observation['rgb']
    c2c = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    cam_pc_model = transform_pc(cam_pc, c2c)
    time.sleep(0.5)
    #os.remove(temp_request_path)
    end_time = time.time()
    print(f"===> got request {end_time - start_time}")
    
    print("===> detecting grasps")
    # if args.graspnet:
    start_time = time.time()
    # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=None, lims=[
    #     np.floor(np.min(pcd_grasp[:, 0])) - 0.1, np.ceil(np.max(pcd_grasp[:, 0])) + 0.1, 
    #     np.floor(np.min(pcd_grasp[:, 1])) - 0.1, np.ceil(np.max(pcd_grasp[:, 1])) + 0.1, 
    #     np.floor(np.min(pcd_grasp[:, 2])) - 0.1, np.ceil(np.max(pcd_grasp[:, 2])) + 0.1])
    # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=[-float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf')])
    # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=None, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    try:
        # gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='fast')
        gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, apply_object_mask=False, dense_grasp=True, collision_detection=True)
    except:
        gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='slow')
    # print("gg_grasp", gg_grasp)
    if gg_grasp is None:
        gg_grasp = []
    else:
        if len(gg_grasp) != 2:
            gg_grasp = []
        else:
            gg_grasp, pcd_o3d = gg_grasp
            gg_grasp = gg_grasp.nms().sort_by_score()
            grippers_o3d = gg_grasp.to_open3d_geometry_list()
            # frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])
    # if len(gg_grasp) == 0:
    #     np.savez(temp_response_path, 
    #                 joint_type=joint_types[args.selected_part], 
    #                 joint_re=joint_res[args.selected_part], 
    #                 num_grasps=0)
    #     while not (os.path.isfile(temp_response_path) and os.access(temp_response_path, os.R_OK)):
    #         time.sleep(0.1)
    #     serviced = np.array(True)
    #     np.save(temp_flag_path, serviced)
    #     while True:
    #         while not (os.path.isfile(temp_flag_path) and os.access(temp_flag_path, os.R_OK)):
    #             time.sleep(0.1)
    #         serviced = np.load(temp_flag_path).item()
    #         if not serviced:
    #             # os.remove(temp_flag_path)
    #             # os.remove(temp_response_path)
    #             break
    #         else:
    #             time.sleep(0.1)
    #     continue
    grasp_scores, grasp_widths, grasp_depths, grasp_translations, grasp_rotations, grasp_invaffordances = [], [], [], [], [], []
    for g_idx, g_grasp in enumerate(gg_grasp):
        grasp_score = g_grasp.score
        grasp_scores.append(grasp_score)
        grasp_width = g_grasp.width
        grasp_widths.append(grasp_width)
        grasp_depth = g_grasp.depth
        grasp_depths.append(grasp_depth)
        grasp_translation = g_grasp.translation
        grasp_translations.append(grasp_translation)
        grasp_rotation = g_grasp.rotation_matrix
        grasp_rotations.append(grasp_rotation)
        grasp_invaffordance = invaffordance_metrics(grasp_translation, grasp_rotation, grasp_score, pred_selected_affordable_position, 
                                                    pred_selected_joint_base, pred_selected_joint_direction, joint_types[args.selected_part])
        grasp_invaffordances.append(grasp_invaffordance)
    grasp_affordances = invaffordances2affordance(grasp_invaffordances)
    
    selected_grasp_idx = np.argsort(grasp_affordances)[::-1][args.selected_grasp_idx]
    # selected_grasp_idx = np.argmax(grasp_affordances)
    # selected_grasp_idx_second = np.argsort(grasp_affordances)[::-1][1]
    # selected_grasp_idx = np.argsort(grasp_affordances)[::-1][0] # 3
    # grippers_o3d = gg_grasp[selected_grasp_idx].to_open3d_geometry_list()
    # frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])

    selected_grasp_score = grasp_scores[selected_grasp_idx]
    selected_grasp_width = grasp_widths[selected_grasp_idx]
    selected_grasp_width = max(min(selected_grasp_width * 1.5, args.max_grasp_width), 0.0)
    selected_grasp_depth = grasp_depths[selected_grasp_idx]
    selected_grasp_translation = grasp_translations[selected_grasp_idx]
    selected_grasp_rotation = grasp_rotations[selected_grasp_idx]
    selected_grasp_affordance = grasp_affordances[selected_grasp_idx]
    end_time = time.time()
    print(f"===> anygrasp detected {end_time - start_time} {len(gg_grasp)}")
    
    print("===> sending response")
    start_time = time.time()
    temp_response_path = f"{args.root_dir}/{args.global_key}/init_info/init_grasp.npz"
    print("temp_response_path", temp_response_path )
    np.savez(temp_response_path,
                joint_base=pred_selected_joint_base, 
                joint_direction=pred_selected_joint_direction, 
                affordable_position=pred_selected_affordable_position, 
                joint_type=joint_types[args.selected_part], 
                joint_re=joint_res[args.selected_part], 
                num_grasps=len(gg_grasp), 
                grasp_score=selected_grasp_score, 
                grasp_width=selected_grasp_width, 
                grasp_depth=selected_grasp_depth, 
                grasp_translation=selected_grasp_translation, 
                grasp_rotation=selected_grasp_rotation, 
                grasp_affordance=selected_grasp_affordance)
    pickle.dump(gg_grasp[selected_grasp_idx:selected_grasp_idx+1], open(f"{args.root_dir}/{args.global_key}/init_info/init_grasp_orig.pkl", "wb"))
