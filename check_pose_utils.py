import sys
sys.path.append("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src")
from lerobot.datasets.pose_utils import pose_to_mat
print(pose_to_mat.__code__.co_varnames)
with open("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src/lerobot/datasets/pose_utils.py") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "def pose_to_mat" in line:
            print("".join(lines[i:i+15]))
            break
