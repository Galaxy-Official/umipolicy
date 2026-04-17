import open3d as o3d
import pickle
from utils.transformation import *
pc = pickle.load(open("/home/rhos/xiaoyang/collect_data/data/2025-04-02_17:16:01/init_info/init_pc.pkl", "rb"))
points, colors = pc[..., :3], pc[..., 3:]

# 可视化点云
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd)

frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis.add_geometry(frame_o3d)

vis.run()
vis.destroy_window()