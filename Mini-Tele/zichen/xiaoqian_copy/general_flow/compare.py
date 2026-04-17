import open3d as o3d
import pickle

vis = o3d.visualization.Visualizer()
vis.create_window()

# point_cloud_demo = pickle.load(open("demo/output/safe_0_hand/robot_exec.pkl", "rb"))["pcd"]
# print(point_cloud_demo.shape)
# pcd_scene = o3d.geometry.PointCloud()
# pcd_scene.points = o3d.utility.Vector3dVector(point_cloud_demo[:, :3])  # 转换为 Open3D 的向量格式
# pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud_demo[:, 3:])  # 转换为 Open3D 的向量格式
# vis.add_geometry(pcd_scene)

point_cloud_test = pickle.load(open("demo/output/box_0_hand/robot_exec.pkl", "rb"))["pcd"]
print(point_cloud_test.shape)
pcd_scene = o3d.geometry.PointCloud()
pcd_scene.points = o3d.utility.Vector3dVector(point_cloud_test[:, :3])  # 转换为 Open3D 的向量格式
pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud_test[:, 3:])  # 转换为 Open3D 的向量格式
vis.add_geometry(pcd_scene)

# point_cloud_demo = pickle.load(open("demo/output/book_0_hand/robot_exec.pkl", "rb"))["pcd"]
# print(point_cloud_demo.shape)
# pcd_scene = o3d.geometry.PointCloud()
# pcd_scene.points = o3d.utility.Vector3dVector(point_cloud_demo[:, :3])  # 转换为 Open3D 的向量格式
# pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud_demo[:, 3:])  # 转换为 Open3D 的向量格式
# vis.add_geometry(pcd_scene)


#################
# point_cloud = pickle.load(open("demo/output/safe_0_hand/input_data.pkl", "rb"))
# print(point_cloud["x"].shape)
# pcd_scene = o3d.geometry.PointCloud()
# pcd_scene.points = o3d.utility.Vector3dVector(point_cloud["pos"])  # 转换为 Open3D 的向量格式
# pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud["x"])  # 转换为 Open3D 的向量格式
# vis.add_geometry(pcd_scene)

point_cloud = pickle.load(open("demo/output/box_0_hand/input_data.pkl", "rb"))
print(point_cloud["x"].shape)
pcd_scene = o3d.geometry.PointCloud()
pcd_scene.points = o3d.utility.Vector3dVector(point_cloud["pos"])  # 转换为 Open3D 的向量格式
pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud["x"])  # 转换为 Open3D 的向量格式
vis.add_geometry(pcd_scene)

# point_cloud = pickle.load(open("demo/output/book_0_hand/input_data.pkl", "rb"))
# print(point_cloud["x"].shape)
# pcd_scene = o3d.geometry.PointCloud()
# pcd_scene.points = o3d.utility.Vector3dVector(point_cloud["pos"])  # 转换为 Open3D 的向量格式
# pcd_scene.colors = o3d.utility.Vector3dVector(point_cloud["x"])  # 转换为 Open3D 的向量格式
# vis.add_geometry(pcd_scene)

vis.update_renderer()
vis.poll_events()
vis.run()
vis.destroy_window()