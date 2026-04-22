import sys
sys.path.append("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src")
with open("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src/lerobot/datasets/pose_utils.py") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "def convert_pose_mat_rep" in line:
            print("".join(lines[i:i+30]))
            break
