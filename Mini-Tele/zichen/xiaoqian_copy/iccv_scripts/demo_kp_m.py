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
# document_box  view_002_pose_004

point_cloud = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/point_cloud.pkl", "rb"))
cam_pc = point_cloud["points"]
pc_rgb = point_cloud["colors"]

pred_keypoints = pickle.load(open(f"../../pred_keypoints/{root_name}/{object_name}/{sample_id}/pred_keypoints_m.pkl", "rb"))
pred_joint_directions = pred_keypoints["pred_joint_directions"]
pred_anchor_points = pred_keypoints["pred_anchor_points"]
pred_manip_points = pred_keypoints["pred_manip_points"]

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

to_manip_distance = []
for translation in gg_grasp.translations:
    dist = 0
    for pred_manip_point in pred_manip_points:
        dist += np.sqrt(((pred_manip_point - translation) ** 2).sum())
    dist = dist / 5.
    to_manip_distance.append(
        - dist
    )
to_manip_distance = np.array(to_manip_distance)

to_manip_distance_scores = to_manip_distance
to_manip_distance_scores = (to_manip_distance_scores - np.mean(to_manip_distance_scores)) / np.std(to_manip_distance_scores)

new_scores = 0.25 * orig_scores + to_manip_distance_scores
# for i in range(len(gg_grasp.scores)):
#     print(orig_scores[i], to_manip_distance_scores[i], new_scores[i])

gg_grasp.scores = new_scores
gg_grasp = gg_grasp.sort_by_score()
gg_grasp = gg_grasp[:2]

## vis
vis = o3d.visualization.Visualizer()
vis.create_window()

grippers_o3d = gg_grasp.to_open3d_geometry_list()
frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])

for gripper in grippers_o3d:
    vis.add_geometry(gripper)
vis.add_geometry(pcd_o3d)
vis.add_geometry(frame_o3d)

points = np.array(pcd_o3d.points)
radius = points[0].max() - points[0].min()
radius = 0.01 * radius

pred_joint_directions = pred_joint_directions / np.linalg.norm(pred_joint_directions)
end = pred_anchor_points + 0.2 * pred_joint_directions
# mesh_arrow, mesh_sphere_begin = \
#     get_arrow(begin=pred_anchor_points, end=end, radius=radius, color=[0, 0, 1])
mesh_arrow = \
    get_thick_line(begin=pred_anchor_points, end=end, radius=radius, color=[1, 0, 0])
vis.add_geometry(mesh_arrow)
# vis.add_geometry(mesh_sphere_begin)

mesh_sphere = draw_point(end, [1, 0, 0], radius=radius*3)
vis.add_geometry(mesh_sphere)

mesh_sphere = draw_point(pred_anchor_points, [1, 0, 0], radius=radius*3)
vis.add_geometry(mesh_sphere)

for pred_manip_point in pred_manip_points:
    mesh_sphere = draw_point(pred_manip_point, [0, 0, 1], radius=radius*3)
    vis.add_geometry(mesh_sphere)

vis.update_renderer()
vis.poll_events()
vis.run()
vis.destroy_window()