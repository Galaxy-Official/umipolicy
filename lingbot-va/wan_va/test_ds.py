import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    class Args: pass
    args = Args()
    args.dataset_path = "./Data/handcap_lingbot"
    ds = LeRobotDataset(args.dataset_path)
    print("Episodes keys:", ds.episode_data_index.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
