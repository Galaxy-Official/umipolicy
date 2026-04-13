import base64
import os
import tempfile
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


def encode_video_to_base64(video_path: str | list[str], fps: int = 30) -> str:
    """
    Encode a video file or a collection of image frames to base64 format.
    
    Args:
        video_path: Either a path to a video file or a list of paths to image files representing frames
        fps: Frames per second when creating video from images (default: 30)
    
    Returns:
        Base64 encoded video as a data URL
    """
    if isinstance(video_path, str):
        with open(video_path, "rb") as video_file:
            result = base64.b64encode(video_file.read()).decode("utf-8")
        return f"data:video/mp4;base64,{result}"
    elif isinstance(video_path, list) and all(isinstance(path, str) for path in video_path):
        # List of image paths representing frames
        if not video_path:
            raise ValueError("Empty list of image paths provided")
            
        # Read first image to get dimensions
        first_image = cv2.imread(video_path[0])
        if first_image is None:
            raise ValueError(f"Could not read image from {video_path[0]}")
            
        height, width, _ = first_image.shape
        
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # Add each frame to the video
            for image_path in video_path:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not read image from {image_path}")
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                video_writer.write(img)
            
            video_writer.release()
            
            # Read the temporary video file and encode it
            with open(temp_path, "rb") as video_file:
                result = base64.b64encode(video_file.read()).decode("utf-8")
                
            return f"data:video/mp4;base64,{result}"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    else:
        raise TypeError("video_path must be either a string or a list of strings")

def decode_base64_to_video(base64_string, output_path="output.mp4"):
    with open(output_path, "wb") as video_file:
        video_file.write(base64.b64decode(base64_string))


def sample_frames_from_video(
    frames: npt.NDArray,
    num_frames: int,
) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(
    path: str,
    num_frames: int = -1,
) -> list[Image.Image]:
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]

@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1

    @property
    def pil_images(self) -> list[Image.Image]:
        # video_path = download_video_asset(self.name)
        ret = video_to_pil_images_list(self.name, self.num_frames)
        return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
        # video_path = download_video_asset(self.name)
        ret = video_to_ndarrays(self.name, self.num_frames)
        return ret