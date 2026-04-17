import os
# Try to import configargparse, fallback to argparse if not available
try:
    import configargparse as argparse
except ImportError:
    import argparse as argparse
import time
import numpy as np
import torch
import pickle
import open3d as o3d
from geometry_utils import *

def config_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = config_parse()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ## vis point cloud
    point_cloud = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/init_pc.pkl", "rb"))
    cam_pc = point_cloud[:, :3]
    pc_rgb = point_cloud[:, 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd)

    ## vis grasp
    gg_grasp = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/init_grasp_orig.pkl", "rb"))
    grippers_o3d = gg_grasp.to_open3d_geometry_list()
    frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    for gripper in grippers_o3d:
        vis.add_geometry(gripper)

    ## vis axis
    pred_selected_affordable_position = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_contact.pkl", "rb"))[0]
    point1 = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_axis1.pkl", "rb"))[0]
    point2 = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_info/kp_axis2.pkl", "rb"))[0]

    points = np.array(pcd.points)
    radius = points[0].max() - points[0].min()
    radius = 0.01 * radius

    mesh_sphere = draw_point(pred_selected_affordable_position, [1, 0, 0], radius=radius)
    vis.add_geometry(mesh_sphere)
    mesh_sphere = draw_point(point1, [0, 1, 0], radius=radius)
    vis.add_geometry(mesh_sphere)
    mesh_sphere = draw_point(point2, [0, 1, 1], radius=radius)
    vis.add_geometry(mesh_sphere)


    vis.run()
    vis.destroy_window()