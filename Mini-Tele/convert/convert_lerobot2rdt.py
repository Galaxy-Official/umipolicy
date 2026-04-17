"""
Convert LeRobot data to RDT format
example usage:
``` bash
python -m convert.convert_lerobot2rdt --input_path $HOME/DATA/OCL4Rob/pour_water --output_path $HOME/DATA/RDT/pour_water
```
"""
import os
import argparse
from pathlib import Path


import h5py
import numpy as np
from pyarrow import parquet as pq
from PIL import Image
from tqdm import tqdm


DATA_PATH = "lerobot/test/data/chunk-000"
IMAGES_PATH = "lerobot/test/images"
META_PATH = "lerobot/test/meta"
VIDEOS_PATH = "lerobot/test/videos"

def parse_data(data_str: str) -> tuple[int]:
    return tuple(map(int, data_str.split("-")))

def sort_path(path_list: list[Path]) -> list[Path]:
    sorted_list = sorted(
        path_list,
        key=lambda p: (parse_data(p.parts[-6]), int(p.stem.split('_')[1]))
    )
    return sorted_list

def load_images(images_path: Path) -> np.ndarray:
    image_files = sorted(images_path.glob("frame_*.png"))
    images = [np.array(Image.open(img_file)) for img_file in image_files]
    return np.stack(images)

def get_image_path(data_path: Path, key) -> Path:
    base_dir = data_path.parent.parent.parent
    filename = data_path.name
    _, num_str = filename.split('_')
    num_int = int(num_str.split('.')[0])
    episode_str = f"episode_{num_int:06d}"
    image_path = base_dir / "images" / f"observation.images.{key}" / episode_str
    return image_path

def convert_lerobot2rdt(
    input_path: str,
    output_path: str,
) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_count = 0
    data_path_list = []
    images_path_dict = {
        "wrist": [],
        "head": [],
        "side": [],
    }
    # meta_path_list = []
    # videos_path_list = []
    data_num = 0
    for child in input_path.iterdir():
        if child.is_dir():
            data_path = child / DATA_PATH
            for sub_dir in data_path.iterdir():
                images_path_dict["head"].append(get_image_path(sub_dir, "head"))
                images_path_dict["wrist"].append(get_image_path(sub_dir, "wrist"))
                images_path_dict["side"].append(get_image_path(sub_dir, "side"))
                data_path_list.append(sub_dir)
                data_num += 1
    print(f"=== data number: {data_num} ===")
    data_path_list = sort_path(data_path_list)
    for key, paths in images_path_dict.items():
        images_path_dict[key] = sort_path(paths)
    assert len(data_path_list) == len(images_path_dict["wrist"]), f"{len(data_path_list)} != {len(images_path_dict['wrist'])}"
    for idx, data_path in tqdm(enumerate(data_path_list), total=len(data_path_list), desc="Converting episodes"):
        cam_right_wrist_path = images_path_dict["wrist"][idx]
        cam_high_path = images_path_dict["head"][idx]
        cam_left_wrist_path = images_path_dict["side"][idx]
        cam_right_wrist = load_images(cam_right_wrist_path)
        cam_high = load_images(cam_high_path)
        cam_left_wrist = load_images(cam_left_wrist_path)
        table = pq.read_table(data_path)
        df = table.to_pandas()
        states = np.stack(df['observation.state'].values)
        actions = np.stack(df['action'].values)
        traj_len = states.shape[0]
        qpos = []
        target = []
        for state, action in zip(states, actions):
            gripper = state[7]
            if gripper > 0.1:
                gripper_width = 0.1
            elif gripper <= 0.1 - 1e-5:
                gripper_width = 0.01
            state[7] = gripper_width
            qpos.append(state)
            target.append(action)
        hdf5_path = os.path.join(output_path, f"episode_{save_count}.hdf5")
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('observations/qpos', data=np.array(qpos))
            f.create_dataset('observations/images/cam_high', data=cam_high)
            f.create_dataset('observations/images/cam_right_wrist', data=cam_right_wrist)
            f.create_dataset('observations/images/cam_left_wrist', data=cam_left_wrist)
            f.create_dataset('actions', data=np.array(target))
        save_count += 1
        


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(
        description="Convert LeRobot data to RDT format."
    )
    arg_parse.add_argument(
        "--input_path",
        help="LeRobot data path.",
        required=True,
    )
    arg_parse.add_argument(
        "--output_path",
        help="RDT data path for finetuning.",
        required=True,
    )
    args = arg_parse.parse_args()
    convert_lerobot2rdt(
        Path(args.input_path),
        Path(args.output_path),
    )