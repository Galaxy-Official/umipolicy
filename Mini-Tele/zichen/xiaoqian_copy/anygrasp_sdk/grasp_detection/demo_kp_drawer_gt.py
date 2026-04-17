import os
import pickle
import configargparse
import time
import numpy as np
import torch
import open3d as o3d
from geometry_utils import *
import json

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
object_name = "drawer"
sample_id = "view_000_pose_001"
# failure case: drawer view_000_pose_001

point_cloud = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/point_cloud.pkl", "rb"))
cam_pc = point_cloud["points"]
pc_rgb = point_cloud["colors"]

# pred_keypoints = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/pred_keypoints.pkl", "rb"))
# pred_joint_directions = pred_keypoints["pred_joint_directions"]
# pred_part_points = pred_keypoints["pred_part_points"]
gt_keypoints = json.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/gt_keypoints.json", "rb"))
kp_axis, kp_contact = gt_keypoints["kp_axis"], gt_keypoints["kp_contact"]
pred_joint_directions = kp_axis - kp_contact
pred_anchor_points = kp_contact
pred_part_points = kp_contact

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

# part_angles = []
# for translation in gg_grasp.translations:
#     part_angles.append(
#                     calc_angles_two_points_and_line(
#                     pred_anchor_points, 
#                     pred_anchor_points + pred_joint_directions, 
#                     pred_part_points, 
#                     translation
#                 ))
# part_angles = np.array(part_angles)
# part_angles_scores = (180-part_angles) / 180
# part_angles_scores = (part_angles_scores - np.mean(part_angles_scores)) / np.std(part_angles_scores)

to_part_distance = []
for translation in gg_grasp.translations:
    part_distance = point_to_plane_distance(
                            translation, 
                            pred_part_points,
                            pred_joint_directions, 
                        )
    to_part_distance.append(-part_distance)
to_part_distance = np.array(to_part_distance)
to_part_distance_scores = to_part_distance
to_part_distance_scores = (to_part_distance_scores - np.mean(to_part_distance_scores)) / np.std(to_part_distance_scores)

new_scores = 0.5 * orig_scores + to_part_distance_scores
# # for i in range(len(gg_grasp.scores)):
# #     print(orig_scores[i], part_angles_scores[i], to_axis_distance_scores[i], new_scores[i])

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
radius = 0.05 * radius

pred_joint_directions = pred_joint_directions / np.linalg.norm(pred_joint_directions)
end = pred_part_points + (-0.2) * pred_joint_directions
# , mesh_sphere_begin 
mesh_arrow = \
    get_thick_line(begin=pred_part_points, end=end, radius=radius/4., color=[1, 0, 0])
vis.add_geometry(mesh_arrow)
# vis.add_geometry(mesh_sphere_begin)

mesh_sphere = draw_point(end, [1, 0, 0], radius=radius/2.)
vis.add_geometry(mesh_sphere)

mesh_sphere = draw_point(pred_part_points, [0, 1, 0], radius=radius/2.)
vis.add_geometry(mesh_sphere)

vis.update_renderer()
vis.poll_events()
vis.run()
vis.destroy_window()