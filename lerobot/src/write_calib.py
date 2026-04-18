import os
import json
from pathlib import Path

calib_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "koch"
calib_dir.mkdir(parents=True, exist_ok=True)
calib_file = calib_dir / "main.json"

data = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -34, "range_min": 0, "range_max": 4095},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -107, "range_min": 2033, "range_max": 2118},
    "elbow_flex": {"id": 3, "drive_mode": 1, "homing_offset": 195, "range_min": 1618, "range_max": 1677},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -42, "range_min": 2044, "range_max": 2147},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -965, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -92, "range_min": 1985, "range_max": 2207}
}

with open(calib_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ 标定文件已霸王硬上弓式写入本地存档: {calib_file}")
print("后续运行 LeRobot 3.0 都会判定你【已经标定过】，将直接进入 EE 解算遥控阶段！不会再跳出让你标定的提示了！")
