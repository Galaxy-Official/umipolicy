# Decision Making File
# OCL4Robotics

import argparse
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from perception.utils.load import (
    get_device,
    get_video_and_output_path_dict,
    load_frames_path_dict_from_path_dict,
    read_causal_relation_from_toml,
)
from perception.models.vlm.base_vlm import BaseVLM
from ..base_task import BaseTask, ViewType, TaskStatus


@BaseTask.register_task("base_vlm_task")
class BaseVLMTask(BaseTask):

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        device = get_device()
        self.task_id = self.config["dataset"]["task"]
        if args.video_type == "camera":
            video_type = "real_time"
        elif args.video_type == "video":
            video_type = "realsense"
        else:
            raise ValueError(f"Unknown video type {args.video_type}")
        self.show_image = args.show_image
        causal_relation_path = f"./causal_relations/task{self.task_id}.toml"
        self.key_attr, self.key_aff, self.logic_chains = read_causal_relation_from_toml(
            causal_relation_path,
        )
        # self.affordance_wo_buffer = deque(maxlen=args.buffer_length)  # affordance without logic chain
        self.video_path_dict, self.output_path_dict = get_video_and_output_path_dict(
            task_id=self.task_id,
            sub_task_id=self.config["dataset"]["sub_task"],
            video_id=self.config["dataset"]["video"],
            video_type=video_type,
            ckpt_type=self.config["clip_model"].get("ckpt_type", "none"),
        )
        # self.attribute_buffer_dict = defaultdict(
        #     lambda: defaultdict(
        #         lambda: defaultdict(lambda: deque(maxlen=args.buffer_length))
        #     )
        # )
        # for view_type in self.video_path_dict.keys():
        #     for obj_id in range(self.config["detector"]["num_objects"]):
        #         for attr_name in self.key_attr:
        #             self.attribute_buffer_dict[view_type][obj_id][attr_name]
        self.attribute_buffer_dict = {}
        self.affordance_buffer_dict = {}
        for key in self.video_path_dict.keys():
            # TODO deque dubug
            self.attribute_buffer_dict[key] = deque(maxlen=args.buffer_length)
            self.affordance_buffer_dict[key] = deque(maxlen=args.buffer_length)
        # Get causal relations
        self.target_id_dict = self.config["target_id"]
        self.confunding_target_id_dict = self.config.get("confunding_target_id", None)
        self.vlm: BaseVLM = BaseVLM.create_model(
            model_name=self.config["clip_model"]["name"],
        )
        demo_path = "demo_test"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.demo_subdir = os.path.join(demo_path, timestamp)
        os.makedirs(self.demo_subdir, exist_ok=True)

    def run(self, video_type="camera"):
        assert video_type in [
            "camera",
            "video",
        ], f"Invalid image input type {video_type}."
        while True:
            status = self.decision_making(video_type)
            if status == TaskStatus.STOP:
                logger.info("Task stopped.")
                break
            elif status == TaskStatus.CONTINUE:
                logger.info("Task continues.")
            else:
                logger.error(f"Unknown status. {status}")
                break
    
    def decision_making(self, video_type = "camera"):
        pass