import os
import pickle
import configargparse
import time
import numpy as np
import torch
import open3d as o3d
from geometry_utils import *

# p2m : F-A, F-M, F-G
# known m: F-A, F-G

gsnet_weight_path = "log/checkpoint_detection.tar"
max_grasp_width = 0.08

print("===> loading graspnet")
start_time = time.time()
from munch import DefaultMunch
from gsnet import AnyGrasp
grasp_detector_cfg = {
    'checkpoint_path': gsnet_weight_path, 
    'max_gripper_width': max_grasp_width, 
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

root_name = "capture_0301"
object_name = "folder"
sample_id = "view_002_pose_004"

'''
failure case:
bucket view_002_pose_001
'''

point_cloud = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/point_cloud.pkl", "rb"))
cam_pc = point_cloud["points"]
pc_rgb = point_cloud["colors"]

pred_keypoints = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/pred_keypoints.pkl", "rb"))
pred_joint_directions = pred_keypoints["pred_joint_directions"]
pred_anchor_points = pred_keypoints["pred_anchor_points"]
pred_part_points = pred_keypoints["pred_part_points"]

start_time = time.time()
try:
    gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, apply_object_mask=False, dense_grasp=True, collision_detection=True)
except:
    gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='slow')
gg_grasp, pcd_o3d = gg_grasp
gg_grasp = gg_grasp.nms().sort_by_score()
# gg_grasp = gg_grasp[:5]

orig_scores = gg_grasp.scores
orig_scores = (orig_scores - np.mean(orig_scores)) / np.std(orig_scores)

part_angles = []
for translation in gg_grasp.translations:
    part_angles.append(
                    calc_angles_two_points_and_line(
                    pred_anchor_points, 
                    pred_anchor_points + pred_joint_directions, 
                    pred_part_points, 
                    translation
                ))
part_angles = np.array(part_angles)
part_angles_scores = (180-part_angles) / 180
part_angles_scores = (part_angles_scores - np.mean(part_angles_scores)) / np.std(part_angles_scores)

to_axis_distance = []
for translation in gg_grasp.translations:
    perp_vector = calculate_perpendicular_vector(pred_anchor_points, 
                                                pred_anchor_points+pred_joint_directions, 
                                                translation)
    to_axis_distance.append(
        np.linalg.norm(perp_vector)
    )
to_axis_distance = np.array(to_axis_distance)
to_axis_distance_scores = to_axis_distance
to_axis_distance_scores = (to_axis_distance_scores - np.mean(to_axis_distance_scores)) / np.std(to_axis_distance_scores)

new_scores = 0.25 * orig_scores + part_angles_scores + 0.5 * to_axis_distance_scores # 1:4:2
# for i in range(len(gg_grasp.scores)):
#     print(orig_scores[i], part_angles_scores[i], to_axis_distance_scores[i], new_scores[i])

gg_grasp.scores = new_scores
gg_grasp = gg_grasp.sort_by_score()
gg_grasp = gg_grasp[:2]

## vis
vis = o3d.visualization.Visualizer()
vis.create_window()

grippers_o3d = gg_grasp.to_open3d_geometry_list()
frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

for gripper in grippers_o3d:
    vis.add_geometry(gripper)
vis.add_geometry(pcd_o3d)
vis.add_geometry(frame_o3d)

points = np.array(pcd_o3d.points)
radius = points[0].max() - points[0].min()
radius = 0.01 * radius

# pred_joint_directions = pred_joint_directions / np.linalg.norm(pred_joint_directions)
# end = pred_anchor_points + pred_joint_directions
# mesh_arrow, mesh_sphere_begin = \
#     get_arrow(begin=pred_anchor_points, end=end, radius=radius, color=[0, 0, 1])
# vis.add_geometry(mesh_arrow)
# vis.add_geometry(mesh_sphere_begin)
# nearest_point_on_line, point_keypoint = calculate_perpendicular_vector(
#                                     pred_anchor_points, 
#                                     end, 
#                                     pred_part_points, 
#                                     return_two_points=True)
# mesh_arrow, mesh_sphere_begin = \
#     get_arrow(begin=nearest_point_on_line, end=point_keypoint, radius=radius, color=[1, 0, 1])
# vis.add_geometry(mesh_arrow)
# vis.add_geometry(mesh_sphere_begin)

## draw plane

mesh_sphere = draw_point(pred_anchor_points + (-5) * pred_joint_directions, [1, 0, 0], radius=radius)
vis.add_geometry(mesh_sphere)
mesh_sphere = draw_point(pred_anchor_points + 5 * pred_joint_directions, [1, 0, 0], radius=radius)
vis.add_geometry(mesh_sphere)
mesh_sphere = draw_point(pred_part_points, [0, 1, 0], radius=radius)
vis.add_geometry(mesh_sphere)

all_keypoints = np.array([pred_anchor_points + (-5) * pred_joint_directions, 
                        pred_anchor_points + 5 * pred_joint_directions, 
                        pred_part_points
                        ])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(all_keypoints)
mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
mesh.paint_uniform_color([0.8, 0.2, 0.2])  # 颜色为红色 (RGB)
vis.add_geometry(mesh)


vis.update_renderer()
vis.poll_events()
vis.run()
vis.destroy_window()