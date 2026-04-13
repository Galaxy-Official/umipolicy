from loguru import logger
import os
from typing import Any, Dict, List, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


from perception.utils.draw import show_box, show_mask, show_points, show_text
from perception.utils.result import DetectionResult

matplotlib.set_loglevel("warning")
# logger = logging.getLogger(__name__)


class Sam2Pipeline:
    """
    Pipeline for SAM2.
    """

    def __init__(
        self,
        model_path: str,
        model_cfg: str,
        device: torch.device,
        video_type: str = "video",
    ):
        assert video_type in [
            "video",
            "camera",
        ], f"Invalid video type {video_type}."
        self.video_type = video_type
        # 将单一的初始化标志改为字典，支持每个视角单独初始化
        self.view_init_status = {}  # 字典，键为视角名称，值为布尔值
        logger.info(f"Loading SAM2 {video_type} model from {model_path}")
        # if video_type == "video":
        #     from sam2.build_sam import build_sam2_video_predictor
        #     from sam2.sam2_video_predictor import SAM2VideoPredictor
        #     self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        #         model_cfg,
        #         model_path,
        #         device=device,
        #     )
        # else:
        from sam2.build_sam import build_sam2_camera_predictor
        from sam2.sam2_camera_predictor import SAM2CameraPredictor
        self.predictor: SAM2CameraPredictor = build_sam2_camera_predictor(
            model_cfg,
            model_path,
            device=device,
        )
        # 将检测框数据也按视角分开存储
        self.bbox_data_dict = {}

    def reset_tracker(self, view_name="default"):
        """
        重置特定视角的跟踪器状态，使其可以在下一帧重新进行初始化
        如果检测到 mask 丢失，可以调用此方法重置特定视角的跟踪器

        Args:
            view_name (str): 视角名称，用于标识要重置哪个视角的跟踪器
        """
        # 将该视角标记为已初始化但需要重新检测
        self.view_init_status[view_name] = True
        
        # 根据不同的预测器类型进行不同的重置操作
        assert hasattr(self.predictor, 'reset_state')
        # 如果预测器有 reset 方法，直接调用
        self.predictor.reset_state()
        logger.info(f"SAM2 {view_name} 视角跟踪器已重置")

    def streaming_sam_and_tracking(
        self,
        image: Image,
        bbox_data: List[DetectionResult] = None,
        points_data: List[Dict[str, List[int]]] = None,
        output_dir: str = None,
        frame_idx: int = 0,
        save_img: bool = True,
        view_name: str = "default",  # 添加视角名称参数
    ):
        # assert (
        #     self.video_type == "camera"
        # ), "This function is only for camera input."
        out_obj_ids = []
        vis_frame_stride = 20  # 每20帧保存一次图像

        frame = np.array(image.convert("RGB"))
        height, width = frame.shape[:2]

        # 检查该视角是否已初始化
        if view_name not in self.view_init_status:
            self.view_init_status[view_name] = False
            logger.info(f"初始化 {view_name} 视角的跟踪状态")

        # 如果该视角尚未初始化，需要bbox来初始化第一次跟踪
        if not self.view_init_status[view_name] or frame_idx == 0:
            logger.info(f"正在使用检测框初始化 {view_name} 视角的SAM2")
            
            # 检查是否有检测框数据
            if not bbox_data:
                # 没有检测结果，无法初始化该视角
                raise ValueError(f"无法检测到初始化 {view_name} 视角所需的目标")

            # 存储该视角的检测框数据
            self.bbox_data_dict[view_name] = bbox_data
            if view_name == "ego":
                print()
            # 初始化跟踪器
            self.predictor.load_first_frame(frame)
            
            # 逐个添加目标检测框进行初始化追踪
            for i, detection in enumerate(bbox_data):
                ann_obj_id = i  # 为每个目标分配唯一ID
                box = np.array(detection.box.xyxy, dtype=np.float32)
                # 基于检测框添加新提示
                _, out_obj_ids, out_mask_logits = (
                    self.predictor.add_new_prompt(
                        frame_idx=frame_idx,
                        obj_id=ann_obj_id,
                        bbox=box,
                    )
                )
            
            # 更新该视角的初始化状态
            self.view_init_status[view_name] = True

            if save_img:
                logger.info(f"保存 {view_name} 视角的第一帧检测框图像")
                logger.info(
                    f"检测框数量: {len(out_obj_ids)} {len(bbox_data)} {len(out_mask_logits)}"
                )
                plt.figure(figsize=(9, 6))
                plt.title(f"{view_name} frame {frame_idx}")
                plt.imshow(image)
                for i, out_obj_id in enumerate(out_obj_ids):
                    show_text(
                        bbox_data[i].label,
                        np.array(bbox_data[i].box.xyxy, dtype=np.float32),
                        plt.gca(),
                    )
                    show_box(
                        np.array(bbox_data[i].box.xyxy, dtype=np.float32),
                        plt.gca(),
                    )
                    show_mask(
                        (out_mask_logits[i] > 0.0).cpu().numpy(),
                        plt.gca(),
                        obj_id=out_obj_id,
                    )
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(output_dir, f"{view_name}_frame_{frame_idx}_bbox.png")
                )
        else:
            # 非首帧，直接进行tracking
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
        
        if save_img:
            # 每隔一定帧数保存分割结果
            vis_frame_stride = 1
            plt.close("all")
            if frame_idx % vis_frame_stride == 0:
                plt.figure(figsize=(6, 4))
                plt.title(f"{view_name} frame {frame_idx}")
                plt.imshow(image)
                for i, (out_obj_id, out_mask) in enumerate(zip(out_obj_ids, out_mask_logits)):
                    show_mask(
                        (out_mask > 0.0).cpu().numpy(),
                        plt.gca(),
                        obj_id=out_obj_id,
                    )
                    # 显示标签文本在mask的右上角
                    if self.bbox_data_dict.get(view_name) and i < len(self.bbox_data_dict.get(view_name, [])):
                        label = self.bbox_data_dict[view_name][i].label
                        mask_coords = np.where((out_mask > 0.0).cpu().numpy())
                        if len(mask_coords[0]) > 0 and len(mask_coords[1]) > 0:
                            # 找到mask的右上角位置
                            top_y = mask_coords[0].min()
                            right_x = mask_coords[1].max()
                            plt.text(right_x, top_y, label, fontsize=8, color='white', 
                                    bbox=dict(facecolor='red', alpha=0.7), ha='right', va='top')
                plt.savefig(os.path.join(output_dir, f"{view_name}_frame_{frame_idx}.png"))
        
        # 按照指定方式整理输出，包含label信息
        video_segment = {
            out_obj_id: {
                'mask': (out_mask_logits[i] > 0.0).cpu().numpy(),
                'label': bbox_data[i].label if bbox_data and i < len(bbox_data) else self.bbox_data_dict.get(view_name, [{}])[i].label if self.bbox_data_dict.get(view_name) and i < len(self.bbox_data_dict.get(view_name, [])) else f'object_{i}'
            }
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        # 返回该视角检测到的目标数量和视频分割结果
        detected_obj_count = len(self.bbox_data_dict.get(view_name, [])) 
        return detected_obj_count, video_segment

    def video_sam_and_tracking(
        self,
        video_path: str,
        frame_names: List[str],
        output_dir: str,
        prompts_data: Union[List[Dict[str, List[int]]], List[DetectionResult]],
        prompt_type: str = None,
        save_img: bool = True,
    ):
        """
        Combine `sam_and_tracking_with_points` and `sam_and_tracking_with_bbox` into a single function.
        Set `prompt_type` to "points" or "bbox" depending on the input.
        Static SAM2 pipeline for video segmentation and tracking.

        Args:
            video_path (str): The path to the video file.
            frame_names (List[str]): The list of frame names.
            output_dir (str): The path to the output directory.
            prompts_data (Union[List[Dict[str, List[int]]], List[DetectionResult]]): The list of prompts.
            prompt_type (str): The type of prompt. Either "points" or "bbox".
            save_img (bool): Whether to save the output images.
        """
        assert prompt_type in [
            "points",
            "bbox",
        ], "prompt_type must be 'points' or 'bbox'."
        assert (
            self.video_type == "video"
        ), "This function is only for video input."
        inference_state = self.predictor.init_state(video_path=video_path)
        ann_frame_idx = 0  # the frame index we interact with
        prompts = {}
        obj_id_len = len(prompts_data)
        if prompt_type == "points":
            for i, point_labels in enumerate(prompts_data):
                ann_obj_id = i
                points = np.array(point_labels["points"], dtype=np.float32)
                labels = np.array(point_labels["labels"], dtype=np.int32)
                prompts[ann_obj_id] = (points, labels)
                _, out_obj_ids, out_mask_logits = (
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )
                )
        elif prompt_type == "bbox":
            for i, detection in enumerate(prompts_data):
                ann_obj_id = i
                box = np.array(detection.box.xyxy, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = (
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )
                )

        if save_img:
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(
                Image.open(
                    os.path.join(video_path, frame_names[ann_frame_idx])
                )
            )
            for i, out_obj_id in enumerate(out_obj_ids):
                if prompt_type == "points":
                    show_points(*prompts[out_obj_id], plt.gca())
                elif prompt_type == "bbox":
                    show_text(
                        prompts_data[i].label,
                        np.array(prompts_data[i].box.xyxy, dtype=np.float32),
                        plt.gca(),
                    )
                    show_box(
                        np.array(prompts_data[i].box.xyxy, dtype=np.float32),
                        plt.gca(),
                    )
                show_mask(
                    (out_mask_logits[i] > 0.0).cpu().numpy(),
                    plt.gca(),
                    obj_id=out_obj_id,
                )
            plt.savefig(
                os.path.join(
                    output_dir, f"frame_{ann_frame_idx}_{prompt_type}.png"
                )
            )
            plt.close()

        video_segments = self.compute_video_segment(
            inference_state=inference_state,
            video_path=video_path,
            frame_names=frame_names,
            output_dir=output_dir,
            save_img=save_img,
            prompts_data=prompts_data,
            prompt_type=prompt_type,
        )
        return video_segments, obj_id_len

    def compute_video_segment(
        self,
        inference_state: Dict[str, Any],
        video_path: str,
        frame_names: List[str],
        output_dir: str,
        save_img: bool = True,
        prompts_data: Union[List[Dict[str, List[int]]], List[DetectionResult]] = None,
        prompt_type: str = None,
    ):
        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(
            inference_state=inference_state,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: {
                    'mask': (out_mask_logits[i] > 0.0).cpu().numpy(),
                    'label': prompts_data[i].label if prompt_type == 'bbox' and prompts_data and i < len(prompts_data) else f'object_{i}'
                }
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        if save_img:
            # render the segmentation results every few frames
            vis_frame_stride = 100
            plt.close("all")
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(
                    Image.open(
                        os.path.join(video_path, frame_names[out_frame_idx])
                    )
                )
                for out_obj_id, segment_data in video_segments[
                    out_frame_idx
                ].items():
                    out_mask = segment_data['mask']
                    label = segment_data['label']
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                    # 显示标签文本在mask的右上角
                    mask_coords = np.where(out_mask)
                    if len(mask_coords[0]) > 0 and len(mask_coords[1]) > 0:
                        # 找到mask的右上角位置
                        top_y = mask_coords[0].min()
                        right_x = mask_coords[1].max()
                        plt.text(right_x, top_y, label, fontsize=8, color='white', 
                                bbox=dict(facecolor='red', alpha=0.7), ha='right', va='top')
                plt.savefig(
                    os.path.join(output_dir, f"frame_{out_frame_idx}.png")
                )

        return video_segments
