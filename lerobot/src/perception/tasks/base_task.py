# Decision Making File
# OCL4Robotics

import argparse
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from pathlib import Path


import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from loguru import logger
from perception.cameras import BaseCamera
from perception.models import BaseModel
from perception.pipelines import ObjectDetectionPipeline, OclPipeline, Sam2Pipeline
from perception.utils.load import (  # load_frames_path_list,
    get_device,
    get_video_and_output_path_dict,
    get_lerobot_video_and_output_path_dict,
    load_frames_path_dict_from_path_dict,
    read_causal_relation_from_toml,
)


class TaskStatus(Enum):
    STOP = "STOP"
    CONTINUE = "CONTINUE"
    RETURN = "RETURN"


class ViewType(Enum):
    BIRD = "bird"
    EGO = "ego"
    THIRD = "third"


class BaseTask(ABC):
    task_registry = {}

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        device = get_device()
        self.task_id = self.config["dataset"]["task"]
        self.attribute_buffer_dict = {}
        self.affordance_buffer_dict = {}
        if args.video_type == "camera":
            video_type = "real_time"
            for key in ["bird", "ego", "third"]:
                self.attribute_buffer_dict[key] = deque(maxlen=args.buffer_length)
                self.affordance_buffer_dict[key] = deque(maxlen=args.buffer_length)
        elif args.video_type == "video":
            video_type = "realsense"
            if args.data_format == "lerobot":
                assert args.lerobot_dir is not None, "`lerobot_dir` shouldn't be none when you use LeRobot data format."
                lerobot_config = config.get("lerobot", None)
                assert lerobot_config is not None, "lerobot config should be exist in tome file."
                self.video_path_dict, self.output_path_dict = get_lerobot_video_and_output_path_dict(
                    lerobot_dir=Path(args.lerobot_dir),
                    config=lerobot_config
                )
            elif args.data_format == "images":
                self.video_path_dict, self.output_path_dict = (
                    get_video_and_output_path_dict(
                        task_id=self.task_id,
                        sub_task_id=self.config["dataset"]["sub_task"],
                        video_id=self.config["dataset"]["video"],
                        video_type=video_type,
                        ckpt_type=self.config["clip_model"].get("ckpt_type", "none"),
                    )
                )
            else:
                logger.error("NotImplementedError, not such data_format")
                raise NotImplementedError
            for key in self.video_path_dict.keys():
                # TODO deque dubug
                self.attribute_buffer_dict[key] = deque(maxlen=args.buffer_length)
                self.affordance_buffer_dict[key] = deque(maxlen=args.buffer_length)
        else:
            raise ValueError(f"Unknown video type {args.video_type}")
        self.show_image = args.show_image
        causal_relation_path = f"./causal_relations/task{self.task_id}.toml"
        self.key_attr, self.key_aff, self.logic_chains = (
            read_causal_relation_from_toml(
                causal_relation_path,
            )
        )
        self.object_detector = ObjectDetectionPipeline(
            detector_path=self.config["detector"]["path"],
            device=device,
        )
        self.sam2_predictor_dict = {
            view.value: Sam2Pipeline(
                model_path=self.config["sam2"]["path"],
                model_cfg=self.config["sam2"]["cfg"],
                device=device,
                video_type=args.video_type,
            )
            for view in ViewType
        }  # Three View Sam2Predictor
        logger.info(self.config["clip_model"]["ckpt_type"])
        self.ocl_pipeline = OclPipeline(
            model=BaseModel.create_model(
                model_name=self.config["clip_model"]["name"],
                device=device,
                ckpt_type=self.config["clip_model"]["ckpt_type"],
            )
        )
        # Get causal relations
        self.target_id_dict = self.config["target_id"]
        self.confunding_target_id_dict = self.config.get("confunding_target_id", None)
        demo_path = "demo_test"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.demo_subdir = os.path.join(demo_path, timestamp)
        os.makedirs(self.demo_subdir, exist_ok=True)

    def camera_input(self, view_names: list[str]) -> dict[str, Image.Image]:
        if not hasattr(self, "frame_index"):
            self.frame_index = {}
        
        # 为每个视图单独管理帧索引
        for view in view_names:
            if view not in self.frame_index:
                self.frame_index[view] = 0
            else:
                self.frame_index[view] += 1
        
        # 检查是否已经初始化了视频处理
        if not hasattr(self, "camera_video_caps"):
            self.camera_video_caps = {}
            self.camera_video_frame_counts = {}
            self.camera_video_paths = {}
            
            # 获取视频路径配置
            camera_config = self.config.get("camera_videos", "camera_videos_example.json")
            if not camera_config:
                raise ValueError("No camera_videos configuration found in config file")
            
            # 为每个视图初始化视频
            for view in view_names:
                if view in camera_config:
                    video_path = camera_config[view]
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    
                    # 打开视频文件
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise ValueError(f"Failed to open video file: {video_path}")
                    
                    self.camera_video_caps[view] = cap
                    self.camera_video_frame_counts[view] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.camera_video_paths[view] = video_path
                    logger.info(f"Loaded video for {view} view: {video_path} ({self.camera_video_frame_counts[view]} frames)")
                else:
                    raise ValueError(f"No video path configured for view: {view}")
        
        # 读取每个视图的当前帧
        img_dict = {}
        for view in view_names:
            if view in self.camera_video_caps:
                # 检查是否超出视频帧数
                if self.frame_index[view] >= self.camera_video_frame_counts[view]:
                    raise StopIteration(f"No more frames in {view} video.")
                
                # 设置当前帧位置
                self.camera_video_caps[view].set(cv2.CAP_PROP_POS_FRAMES, self.frame_index[view])
                ret, frame = self.camera_video_caps[view].read()
                
                if not ret:
                    raise StopIteration(f"Failed to read frame from {view} video.")
                
                # 转换为PIL图像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_dict[view] = Image.fromarray(frame_rgb)
        
        return img_dict

    def video_input(self, is_path=False) -> dict[str, Image.Image | str]:
        # Similar to streaming_input, but reads from a static video or image sequence.
        if not hasattr(self, "frame_index"):
            self.frame_index = {view.value: 0 for view in ViewType}
        else:
            for key in self.frame_index:
                self.frame_index[key] += self.config["sam2"]["frame_interval"]
        
        # 检查是否已经初始化了视频处理
        if not hasattr(self, "is_video_mp4"):
            self.is_video_mp4 = {}
            self.video_caps = {}
            self.video_frame_counts = {}
            
            # 检查每个视频路径是否为MP4文件
            for key, path in self.video_path_dict.items():
                if str(path).lower().endswith('.mp4'):
                    self.is_video_mp4[key] = True
                    # 打开视频文件
                    self.video_caps[key] = cv2.VideoCapture(str(path))
                    # 获取视频总帧数
                    self.video_frame_counts[key] = int(self.video_caps[key].get(cv2.CAP_PROP_FRAME_COUNT))
                else:
                    self.is_video_mp4[key] = False
                    # 对于非MP4文件，使用原有的图片序列处理方式
                    if not hasattr(self, "frame_names"):
                        self.frame_names_dict, self.frame_length = (
                            load_frames_path_dict_from_path_dict(self.video_path_dict)
                        )
        
        # 处理MP4视频
        if any(self.is_video_mp4.values()):
            img_dict = {}
            for key, is_mp4 in self.is_video_mp4.items():
                if is_mp4:
                    # 检查是否超出视频帧数
                    if self.frame_index[key] >= self.video_frame_counts[key]:
                        raise StopIteration("No more frames in the video.")
                    
                    # 设置当前帧位置
                    self.video_caps[key].set(cv2.CAP_PROP_POS_FRAMES, self.frame_index[key])
                    ret, frame = self.video_caps[key].read()
                    
                    if not ret:
                        raise StopIteration("Failed to read frame from video.")
                    
                    # 转换为PIL图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    if is_path:
                        # 对于MP4，我们无法返回路径，所以返回None
                        img_dict[key] = None
                    else:
                        img_dict[key] = img
                else:
                    # 处理非MP4文件（图片序列）
                    if self.frame_index[key] < self.frame_length:
                        frame_path = os.path.join(
                            self.video_path_dict[key], self.frame_names_dict[key][self.frame_index[key]]
                        )
                        img = Image.open(frame_path)
                        if is_path:
                            img_dict[key] = frame_path
                        else:
                            img_dict[key] = img
                    else:
                        raise StopIteration("No more frames in the video.")
            
            return img_dict
        
        # 处理图片序列（原有逻辑）
        if any(self.frame_index[key] < self.frame_length for key in self.frame_index):
            img_dict = {}
            for key, frame_names in self.frame_names_dict.items():
                frame_path = os.path.join(
                    self.video_path_dict[key], frame_names[self.frame_index[key]]
                )
                img = Image.open(frame_path)
                if is_path:
                    img_dict[key] = frame_path
                else:
                    img_dict[key] = img
            return img_dict
        else:
            raise StopIteration("No more frames in the video.")

    @abstractmethod
    def decision_making(self, video_type: str = "camera") -> TaskStatus:
        pass

    @abstractmethod
    def object_selection(self, video_type: str = "camera") -> tuple[str, str]:
        pass

    # Other methods using TaskStatus
    @abstractmethod
    def run(self, video_type: str = "camera"):
        pass

    @classmethod
    def register_task(cls, task_name):
        def wapper(subclass):
            cls.task_registry[task_name] = subclass
            return subclass

        return wapper

    @classmethod
    def create_task(cls, task_name, *args, **kwargs):
        if task_name not in cls.task_registry:
            raise ValueError(f"Task with key '{task_name}' is not registered.")
        task_class = cls.task_registry[task_name]
        return task_class(*args, **kwargs)

    @staticmethod
    def caculate_average_probs(attr_aff_probs_lists):
        # 用defaultlist初始化每个list对应的key的值
        total_values = defaultdict(lambda: defaultdict(float))
        # 记录每个key出现的次数
        key_counts = defaultdict(lambda: defaultdict(int))

        for attr_aff_prob in attr_aff_probs_lists:
            for outer_key, inner_dict in attr_aff_prob.items():
                for inner_key, values in inner_dict.items():
                    for value in values:
                        total_values[outer_key][inner_key] += value
                        key_counts[outer_key][inner_key] += 1

        average_probs = {
            outer_key: {
                inner_key: total_values[outer_key][inner_key]
                / key_counts[outer_key][inner_key]
                for inner_key in total_values[outer_key]
            }
            for outer_key in total_values
        }

        return average_probs

    def draw_causal_graph(
        self, attr_prob_dict, aff_prob_dict, affordance_name: str = "pour"
    ):

        related_edges = [
            edge for edge in self.logic_chains if edge["affordance"] == affordance_name
        ]
        pos_attribute_names = [
            edge["attribute"]
            for edge in related_edges
            if edge["is_positive_affect"]
        ]
        neg_attribute_names = [
            edge["attribute"]
            for edge in related_edges
            if not edge["is_positive_affect"]
        ]

        fig = go.Figure()
        ATTRIBUTE_UPPERBOUND = 8.6  # const variable
        ATTRIBUTE_LOWBOUND = 5
        AFFORDANCE_UPPERBOUND = 180
        AFFORDANCE_LOWBOUND = 20

        pos_attributes = {
            name: attr_prob_dict[name] for name in pos_attribute_names
        }
        neg_attributes = {
            name: attr_prob_dict[name] for name in neg_attribute_names
        }
        affordance_value = {affordance_name: aff_prob_dict[affordance_name]}

        pos_attribute_size = {
            name: max(
                ATTRIBUTE_LOWBOUND,
                min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
            )
            for name, value in pos_attributes.items()
        }
        neg_attribute_size = {
            name: max(
                ATTRIBUTE_LOWBOUND,
                min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
            )
            for name, value in neg_attributes.items()
        }
        size = max(
            AFFORDANCE_LOWBOUND,
            min(
                math.log(affordance_value[affordance_name] + 1) * 50,
                AFFORDANCE_UPPERBOUND,
            ),
        )

        avg_pos = (
            sum(pos_attributes.values()) / len(pos_attributes)
            if len(pos_attributes) > 0
            else 0
        )
        avg_neg = (
            sum(neg_attributes.values()) / len(neg_attributes)
            if len(neg_attributes) > 0
            else 0
        )

        y_aff = (
            0.9 * avg_pos
            - 0.9 * avg_neg
            + 1.1 * sum(affordance_value.values())
        ) * 5
        prob = max(0, min(100, (y_aff + 2) * 10))
        y_neg = -5
        x_mid_pos = 0

        y_pos = 5 + y_aff

        if y_aff > 0.2:
            y_neg = -5 + (y_aff - 0.2)

        # affordance气球
        fig.add_trace(
            go.Scatter(
                x=[x_mid_pos],
                y=[y_aff],
                mode="markers+text",
                marker=dict(size=24, color="#EFB6C8", symbol="square"),
                text=affordance_name + f" (able):{prob:.2f}%",
                textposition="middle right",
                textfont=dict(color="black", size=10),
                zorder=10,
            ),
        )

        # 加减号
        if pos_attribute_names:
            if len(pos_attribute_names) % 2 == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x_mid_pos],
                        y=(
                            [(y_pos + y_aff + 1) / 2]
                            if (y_pos + y_aff + 1) / 2 < 8.5
                            else [8.5]
                        ),
                        mode="text",
                        text=["+"],
                        textposition="middle center",
                        textfont=dict(color="#FFB200", size=15, weight="bold"),
                    ),
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[x_mid_pos + 0.03],
                        y=(
                            [(y_pos + y_aff + 1) / 2]
                            if (y_pos + y_aff + 1) / 2 < 8.5
                            else [8.5]
                        ),
                        mode="text",
                        text=["+"],
                        textposition="middle right",
                        textfont=dict(color="#FFB200", size=15, weight="bold"),
                    ),
                )
        if neg_attribute_names:
            if len(neg_attribute_names) % 2 == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x_mid_pos],
                        y=(
                            [(y_neg + y_aff - 1) / 2]
                            if (y_neg + y_aff - 1) / 2 > -8.5
                            else [-8.5]
                        ),
                        mode="text",
                        text=["-"],
                        textposition="middle center",
                        textfont=dict(color="#EB5B00", size=15, weight="bold"),
                    ),
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[x_mid_pos + 0.04],
                        y=(
                            [(y_neg + y_aff - 1) / 2]
                            if (y_neg + y_aff - 1) / 2 > -8.5
                            else [-8.5]
                        ),
                        mode="text",
                        text=["-"],
                        textposition="middle right",
                        textfont=dict(color="#EB5B00", size=15, weight="bold"),
                    ),
                )

        x_pos = -(len(pos_attributes) - 1) * 0.3 / 2
        balloon_image_path = "resources/balloon.png"
        balloon_image = Image.open(balloon_image_path)
        for attr in pos_attributes.keys():
            if y_pos + (pos_attribute_size[attr] - 5) * 0.54 > 9:
                y_pos_edited = 9 - (pos_attribute_size[attr] - 5) * 0.54
            else:
                y_pos_edited = y_pos
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, x_mid_pos],
                    y=[y_pos_edited, y_aff],
                    mode="lines",
                    line=dict(color="#FFB200", width=2),
                    zorder=2,
                ),
            )
            fig.add_layout_image(
                dict(
                    source=balloon_image,
                    xref="x",
                    yref="y",
                    x=x_pos,
                    y=y_pos_edited + (pos_attribute_size[attr] - 5) * 0.27,
                    sizex=pos_attribute_size[attr] * 1.1,
                    sizey=pos_attribute_size[attr] * 1.1,
                    xanchor="center",
                    yanchor="middle",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[
                        y_pos_edited
                        + (pos_attribute_size[attr] - 5) * 0.54
                        + 0.75
                    ],
                    mode="text",
                    text= "full" if attr == "empty" else "nearly full",
                    textposition="top center",
                    textfont=dict(color="black", size=10),
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_pos, (x_pos + x_mid_pos) / 2, x_mid_pos],
                    y=[y_pos_edited, (y_pos_edited + y_aff) / 2, y_aff],
                    mode="markers",
                    marker=dict(
                        symbol="arrow-down",
                        size=6,
                        color="#FFB200",
                        angleref="previous",
                    ),
                    zorder=2,
                ),
            )
            x_pos += 0.3

        # arrow lines and balloons of neg attr
        x_neg = -(len(neg_attributes) - 1) * 0.3 / 2
        weight_image_path = "resources/weight.png"
        weight_image = Image.open(weight_image_path)
        neg_line_width = (y_aff - y_neg) / 3
        for attr in neg_attributes.keys():
            fig.add_trace(
                go.Scatter(
                    x=[x_neg, x_mid_pos],
                    y=[y_neg, y_aff],
                    mode="lines",
                    line=dict(color="#EB5B00", width=neg_line_width),
                    zorder=2,
                ),
            )
            fig.add_layout_image(
                dict(
                    source=weight_image,
                    xref="x",
                    yref="y",
                    x=x_neg,
                    y=y_neg + (neg_attribute_size[attr] - 5) / 5.5,
                    sizex=neg_attribute_size[attr] / 2.5,
                    sizey=neg_attribute_size[attr] / 2.5,
                    xanchor="center",
                    yanchor="middle",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_neg],
                    y=[
                        y_neg
                        + (neg_attribute_size[attr] - 5) / 5.5
                        - neg_attribute_size[attr] / 5
                    ],
                    mode="text",
                    text="empty",
                    textposition="bottom center",
                    textfont=dict(color="black", size=10),
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_neg, (x_neg + x_mid_pos) / 2, x_mid_pos],
                    y=[y_neg, (y_neg + y_aff) / 2, y_aff],
                    mode="markers",
                    marker=dict(
                        symbol="arrow-down",
                        size=6,
                        color="#EB5B00",
                        angleref="previous",
                    ),
                    zorder=2,
                ),
            )
            x_neg += 0.3

        fig.update_xaxes(range=[-1, 1])
        fig.update_yaxes(range=[-11.5, 11.5])
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=True),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        # Convert Plotly figure to image array with higher resolution
        img = fig.to_image(format="png", scale=3)
        img_array = np.frombuffer(img, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Crop the image to half its width, centered
        height, width, _ = img_cv.shape
        start_x = width // 5
        start_y = height // 15
        end_y = height - int(height / 3.8)
        end_x = start_x + (width // 5) * 3
        cropped_img = img_cv[start_y:end_y, start_x:end_x]

        # Encode the cropped image back to bytes
        _, img_encoded = cv2.imencode(".png", cropped_img)
        return img_encoded.tobytes()

    def process_attributes_affordances(self, attr_prob_dict, aff_prob_dict):
        """
        Combine attributes and affordances from multiple perspectives.
        """
        aggregated_attrs = {key: [] for key in self.key_attr}
        aggregated_affs = {key: [] for key in self.key_aff}
        # Simple average example
        for key, obj_attr_prob in attr_prob_dict.items():
            for attr, prob in obj_attr_prob[self.target_id_dict[key]].items():
                aggregated_attrs[attr].append(prob)
        for key, obj_aff_prob in aff_prob_dict.items():
            for aff, prob in obj_aff_prob[self.target_id_dict[key]].items():
                aggregated_affs[aff].append(prob)

        attr_bool_dict = {}
        aff_bool_dict = {}
        for key, prob_list in aggregated_attrs.items():
            attr_bool_dict[key] = self.average_pool(prob_list)
        for key, prob_list in aggregated_affs.items():
            aff_bool_dict[key] = self.average_pool(prob_list)
        return attr_bool_dict, aff_bool_dict

    @staticmethod
    def average_pool(array: list[float]) -> float:
        score = sum(array) / len(array)
        return score

    @staticmethod
    def max_pool(array: list[float]) -> float:
        score = max(array)
        return score

    @staticmethod
    def vote_pool(array: list[float]) -> bool:
        count = len(array)
        positive_count = 0
        for value in array:
            if value >= 0.5:
                positive_count += 1
        return positive_count * 2 >= count
    
    def cleanup_video_resources(self):
        """释放视频资源"""
        # 释放camera视频资源
        if hasattr(self, "camera_video_caps"):
            for cap in self.camera_video_caps.values():
                cap.release()
            logger.info("Released camera video resources")
        
        # 释放video视频资源
        if hasattr(self, "video_caps"):
            for cap in self.video_caps.values():
                cap.release()
            logger.info("Released video resources")
