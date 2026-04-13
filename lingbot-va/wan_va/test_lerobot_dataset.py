import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("./Data/handcap_lingbot")
    print("LeRobotDataset initialized!", len(ds))
    meta = ds.meta
    ep_map = {}
    meta_iterable = meta.episodes.values() if isinstance(meta.episodes, dict) else meta.episodes
    for val in meta_iterable:
        ep_map[val['episode_index']] = val
    print(len(ep_map), "episodes parsed from meta")
except Exception as e:
    import traceback
    traceback.print_exc()
