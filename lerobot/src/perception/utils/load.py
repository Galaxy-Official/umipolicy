import os
from pathlib import Path
from typing import List, Tuple

import cv2
import requests
import toml
import torch
from PIL import Image


def load_image(image_str: str) -> Image.Image:
    """
    Load an image from a file path or URL and convert it to RGB format.
    Args:
        image_str (str): The file path or URL of the image.
    Returns:
        Image.Image: The loaded image in RGB format.
    """
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert(
            "RGB"
        )
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def load_video(video_str: str) -> List[Image.Image]:
    """
    Load a video from a file or URL and return a list of frames as PIL Image objects.
    Args:
        video_str (str): The path to the video file or URL.
    Returns:
        List[Image.Image]: A list of frames as PIL Image objects.
    """
    if video_str.startswith("http"):
        cap = cv2.VideoCapture(video_str)
    else:
        cap = cv2.VideoCapture(video_str)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    return frames


def load_frames_path_list(video_path: str) -> List[str]:
    frame_names = [
        p
        for p in os.listdir(video_path)
        if os.path.splitext(p)[-1]
        in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def load_frames_path_list_startswith(video_path: str, starts_with: str = "frame_") -> list[str]:
    # 仅选择以 "frame_" 开头的图片文件
    frame_names = [
        p for p in os.listdir(video_path)
        if p.startswith("frame_") and os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    # 根据文件名中下划线后的数字排序（例如：frame_000043.png）
    frame_names.sort(key=lambda p: int(p.split('_')[1].split('.')[0]))
    return frame_names


def load_frames_path_dict_from_path_dict(
    video_path_dict: dict[str, Path],
) -> tuple[dict[str, list[str]], int]:
    """
    Load frame paths from multiple video directories.
    Args:
        video_path_list (dict[str, Path]): A dictionary of video directories.
    Returns:
        List[List[str]]: A list of lists of frame paths.
    """
    frame_names_dict = {}
    frame_length = None
    for key, video_path in video_path_dict.items():
        frame_names: list[Path] = [
            p.name
            for p in Path(video_path).iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(Path(p).stem))
        if frame_length is None:
            frame_length = len(frame_names)
        else:
            assert frame_length == len(frame_names), f"Frame length mismatch. {frame_length} != {len(frame_names)}"
        frame_names_dict[key] = frame_names
    return frame_names_dict, frame_length


# TODO deprecated function
# def load_alpha_clip(
#     clip_model_path: str,
#     alpha_vision_ckpt_path: str,
#     device: torch.device = None,
# ):
#     model, preprocess = alpha_clip.load(
#         name=clip_model_path,
#         alpha_vision_ckpt_pth=alpha_vision_ckpt_path,
#         device=device,
#     )
#     print(f"Loading model from {clip_model_path}")
#     mask_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Resize(
#                 (224, 224)
#             ),  # change to (336,336) when using ViT-L/14@336px
#             transforms.Normalize(0.5, 0.26),
#         ]
#     )
#     return model, preprocess, mask_transform


# def load_roi_clip(
#     clip_model_path: str,
#     clip_ckpt_path: Optional[str] = None,
#     device: torch.device = None,
# ) -> Tuple[torch.nn.Module, Any, Any]:
#     clip_model_path_split = clip_model_path.split("/")
#     cache_dir = "/".join(clip_model_path_split[:-2])
#     model_id = "/".join(clip_model_path_split[-2:])
#     print(f"Loading model from {clip_model_path}")
#     clip_model, clip_processor = open_clip.create_model_from_pretrained(
#         model_name=f"hf-hub:{model_id}",
#         cache_dir=cache_dir,
#     )
#     clip_model.to(device)
#     print(f"Loading tokenizer from {clip_model_path}")

#     clip_tokenizer = open_clip.get_tokenizer(
#         model_name=f"hf-hub:{model_id}",
#         cache_dir=cache_dir,
#     )
#     if clip_ckpt_path is not None:
#         print(f"Loading CLIP checkpoint from {clip_ckpt_path}")
#         clip_ckpt = torch.load(clip_ckpt_path, map_location="cpu")
#         msg = clip_model.load_state_dict(clip_ckpt, strict=False)
#         print(msg)
#     roi_model = OpenClipWithROI(clip_model=clip_model)
#     return roi_model, clip_processor, clip_tokenizer


def load_word_by_line_txt(path: str) -> List[str]:
    results = []
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if len(line) > 0:
                results.append(line)
    return results


def get_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # if device.type == "cuda":
    #     # use bfloat16 for the entire notebook
    #     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     if torch.cuda.get_device_properties(0).major >= 8:
    #         torch.backends.cuda.matmul.allow_tf32 = True
    #         torch.backends.cudnn.allow_tf32 = True
    # elif device.type == "mps":
    #     print(
    #         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
    #         "give numerically different outputs and sometimes degraded performance on MPS. "
    #         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    #     )
    return device


def read_causal_relation_from_txt(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()

    attributes = []
    affordances = []
    edges = {
        "pos": [],
        "neg": [],
    }

    for line in lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue  # Skip comments and empty lines
        if "1" in line:
            source, target = line.split("1")
            edges["pos"].append((source.strip(), target.strip()))
        elif "0" in line:
            source, target = line.split("0")
            edges["neg"].append((source.strip(), target.strip()))
        else:
            if line.islower():
                attributes.append(line)
            else:
                affordances.append(line.lower())

    return attributes, affordances, edges


def read_causal_relation_from_toml(
    file_path: str,
):
    with open(file_path, "r") as file:
        config = toml.load(file)
    attributes = config["attributes"]
    affordances = config["affordances"]
    logic_chains = config["logic_chains"]
    return attributes, affordances, logic_chains

def read_task_config_from_toml(
    file_path: str,
):
    with open(file_path, "r") as file:
        config = toml.load(file)
    task_discription = config["task_discription"]
    return task_discription

def get_video_and_output_path_dict(
    task_id: int,
    sub_task_id: int,
    video_id: int,
    video_type: str,
    ckpt_type: str = "none",
) -> tuple[dict[str, Path], dict[str, Path]]:

    video_father_dir = Path(
        f"./demo_dataset/{video_type}/task{task_id}/{sub_task_id}/video{video_id}"
    )
    output_father_dir = Path(
        f"./demo_output/{video_type}/task{task_id}/{sub_task_id}/video{video_id}_{ckpt_type}"
    )  # /{bird_ego_key}_ReLU"
    video_path_dict = {
        sub_dir.name: sub_dir
        for sub_dir in video_father_dir.iterdir()
        if sub_dir.is_dir()
    }

    output_path_dict = {
        sub_dir.name: output_father_dir / sub_dir.name
        for sub_dir in video_father_dir.iterdir()
        if sub_dir.is_dir()
    }

    for output_path in output_path_dict.values():
        os.makedirs(output_path, exist_ok=True)

    return video_path_dict, output_path_dict

def get_lerobot_video_and_output_path_dict(
    lerobot_dir: Path,
    config: dict
) -> tuple[dict[str, Path], dict[str, Path]]:
    VIEW_DICT = {
        "third": "side",
        "bird": "head",
        "ego": "wrist"
    }
    video_path_dict = {}
    output_path_dict = {}
    task_name = config["task_name"]
    time_chunk = config["time_chunk"]
    video_id = config["video_id"]
    video_parent_path = lerobot_dir / task_name / time_chunk / "lerobot/test/videos/chunk-000"
    for view_type, lerobot_view_type in VIEW_DICT.items():
        video_path_dict[view_type] = video_parent_path / f"observation.images.{lerobot_view_type}" / f"episode_{video_id:06d}.mp4"
    output_father_dir = Path(
        f"./demo_output/lerobot/{task_name}/{time_chunk}/{video_id}"
    )
    for view_type, lerobot_view_type in VIEW_DICT.items():
        output_path_dict[view_type] = output_father_dir / view_type
    for output_path in output_path_dict.values():
        os.makedirs(output_path, exist_ok=True)
    return video_path_dict, output_path_dict
    

def get_video_and_output_path(
    task_id: int,
    sub_task_id: int,
    video_id: int,
    video_type: str,
    ckpt_type: str = "none",
    bird_ego_key: str = "ego",
) -> Tuple[str, str]:
    if video_type == "realsense":
        video_path = f"./demo_dataset/realsense/task{task_id}/{sub_task_id}/video{video_id}/{bird_ego_key}"
        output_dir = f"./demo_output/realsense/task{task_id}/{sub_task_id}/video{video_id}_{ckpt_type}/{bird_ego_key}_ReLU"
    elif video_type == "real_time":
        video_path = None
        output_dir = f"./demo_output/real_time/task{task_id}/{sub_task_id}/video{video_id}_{ckpt_type}/{bird_ego_key}"
    elif video_type == "rgb_camera":
        video_path = f"./demo_dataset/rgb_camera/task{task_id}/{sub_task_id}/video{video_id}"
        output_dir = f"./demo_output/rgb_camera/task{task_id}/{sub_task_id}/video{video_id}_{ckpt_type}"
    elif video_type == "iphone_camera":  # shoot by iphone camera
        video_path = (
            f"./demo_dataset/iphone_camera/video{sub_task_id}_{video_id}"
        )
        output_dir = f"./demo_output/iphone_camera/video{sub_task_id}_{video_id}/{ckpt_type}"
    else:
        raise ValueError(f"Invalid video type: {video_type}")
    os.makedirs(output_dir, exist_ok=True)
    return video_path, output_dir
