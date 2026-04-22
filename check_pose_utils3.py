import sys
sys.path.append("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src")
with open("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src/lerobot/datasets/pose_utils.py") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "def mat_to_rot6d" in line:
            print("".join(lines[i:i+10]))
            break
