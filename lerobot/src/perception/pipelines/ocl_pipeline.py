from loguru import logger
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from perception.models import BaseClipModel
from perception.utils.common import mask_to_bounding_box
from perception.utils.function import clamped_value, float_ReLU
from perception.utils.load import load_word_by_line_txt
from perception.utils.result import BoundingBox, DetectionResult

# logger = logging.getLogger(__name__)


class OclPipeline:
    ANN_PATHS = {
        "attributes": "./demo_dataset/attr_list.txt",
        "affordances": "./demo_dataset/aff_list.txt",
    }
    GUASSIAN_SIGMA = 1.0

    def __init__(
        self,
        model: BaseClipModel,
    ):
        self.model = model
        self.load_attr_aff_list()

    def load_attr_aff_list(self):
        self.attribute_list = load_word_by_line_txt(
            self.ANN_PATHS["attributes"]
        )
        logger.info(f"attribute number: {len(self.attribute_list)}")
        self.affordance_list = load_word_by_line_txt(
            self.ANN_PATHS["affordances"]
        )
        logger.info(f"affordance number: {len(self.affordance_list)}")

        self.attr2idx = {
            attr: idx for idx, attr in enumerate(self.attribute_list)
        }
        self.aff2idx = {
            aff: idx for idx, aff in enumerate(self.affordance_list)
        }

    def calculate_new_aff(
        self,
        attr_value: float,
        aff_value: float,
        is_pos: bool,
        weight: float = 0.5,
        bias: float = 0.5,
    ):
        """
        Calculate the new affordance value based on the attribute value and the current affordance value.

        Args:
        - attr_value: float
        - aff_value: float
        - is_pos: bool

        Returns:
        - float
        """
        if is_pos:
            return aff_value + float_ReLU(attr_value - bias) * weight
        else:
            return aff_value - float_ReLU(attr_value - bias) * weight

    def sigle_frame_attr_aff_probabilities(
        self,
        frame_result,
        key_attr,
        key_aff,
        use_logic_chain,
        logic_chains,
    ):
        attr_probs = {}
        aff_probs = {}
        for obj_id, obj_result in frame_result.items():
            for attr in key_attr:
                attr_probs[obj_id][attr].append(
                    obj_result["attr"][self.attr2idx[attr]]
                )
            for aff in key_aff:
                aff_value = obj_result["aff"][self.aff2idx[aff]]
                if use_logic_chain:
                    for logic_chain in logic_chains:
                        if logic_chain["affordance"] == aff:
                            aff_value = self.calculate_new_aff(
                                attr_value=obj_result["attr"][
                                    self.attr2idx[logic_chain["attribute"]]
                                ],
                                aff_value=aff_value,
                                is_pos=logic_chain["is_positive_affect"],
                                weight=logic_chain["weight"],
                                bias=logic_chain["bias"],
                            )
                    aff_value = clamped_value(0, 1, aff_value)
                aff_probs[obj_id][aff].append(aff_value)
        return attr_probs, aff_probs

    def causal_inference_result(
        self,
        results,
        key_attr: List[str],
        key_aff: List[str],
        logic_chains: List[Dict[str, Any]],
        use_logic_chain: bool = False,
        use_filter: bool = True,
        sigma: float = 1.0,
    ) -> Tuple[
        Dict[int, Dict[str, List[float]]],
        Dict[int, Dict[str, List[float]]],
        Dict[int, Dict[str, List[float]]],
        Dict[int, Dict[str, List[float]]],
    ]:
        obj_ids = results[0].keys()
        attr_probs = {}
        aff_probs = {}
        for obj_id in obj_ids:
            attr_probs[obj_id] = {attr: [] for attr in key_attr}
            aff_probs[obj_id] = {aff: [] for aff in key_aff}

        for frame_result in results.values():
            for obj_id, obj_result in frame_result.items():
                for attr in key_attr:
                    attr_probs[obj_id][attr].append(
                        obj_result["attr"][self.attr2idx[attr]]
                    )
                for aff in key_aff:
                    aff_value = obj_result["aff"][self.aff2idx[aff]]
                    if use_logic_chain:
                        for logic_chain in logic_chains:
                            if logic_chain["affordance"] == aff:
                                aff_value = self.calculate_new_aff(
                                    attr_value=obj_result["attr"][
                                        self.attr2idx[logic_chain["attribute"]]
                                    ],
                                    aff_value=aff_value,
                                    is_pos=logic_chain["is_positive_affect"],
                                    weight=logic_chain["weight"],
                                    bias=logic_chain["bias"],
                                )
                        aff_value = clamped_value(0, 1, aff_value)
                    aff_probs[obj_id][aff].append(aff_value)
        if use_filter:
            # Generate smoothed versions without altering the original dictionaries
            smoothed_attr_probs = {}
            smoothed_aff_probs = {}
            for obj_id in attr_probs:
                smoothed_attr_probs[obj_id] = {}
                for attr in attr_probs[obj_id]:
                    smoothed_attr_probs[obj_id][attr] = gaussian_filter1d(
                        np.array(attr_probs[obj_id][attr]), sigma=sigma
                    ).tolist()
            for obj_id in aff_probs:
                smoothed_aff_probs[obj_id] = {}
                for aff in aff_probs[obj_id]:
                    smoothed_aff_probs[obj_id][aff] = gaussian_filter1d(
                        np.array(aff_probs[obj_id][aff]), sigma=sigma
                    ).tolist()
            assert len(attr_probs) == len(smoothed_attr_probs)
            return (
                attr_probs,
                aff_probs,
                smoothed_attr_probs,
                smoothed_aff_probs,
            )
        else:
            return attr_probs, aff_probs, None, None

    def save(self, task_id: int, output_dir: str):
        data_items = [
            ("results", self.results),
            ("attr", self.attr_probs),
            ("attr_smoothed", self.smoothed_attr_probs),
            ("aff", self.aff_probs),
            ("aff_smoothed", self.smoothed_aff_probs),
            ("aff_new", self.aff_probs_new),
            ("aff_smoothed_new", self.smoothed_aff_probs_new),
        ]
        for name, data in data_items:
            file_path = os.path.join(output_dir, f"task{task_id}_{name}.pkl")
            with open(file_path, "wb") as file:
                pickle.dump(data, file)

    def boundingbox_calculate_text_probabilities(
        self,
        frame: Image,
        detections: List[DetectionResult],
        labels: List[str] = None,
        key_attr: List[str] = None,
        key_aff: List[str] = None,
        logic_chains: List[Dict[str, Any]] = None,
    ):
        results = {}
        result_dict = {}
        obj_id_len = len(detections)
        label_list, attr_probs, aff_probs = (
            self.model.calculate_text_probabilities(
                frame,
                detections,
                labels,
                self.attribute_list,
                self.affordance_list,
                obj_id_len,
            )
        )
        for detection, attr_prob, aff_prob in zip(
            detections,
            attr_probs.to("cpu").numpy(),
            aff_probs.to("cpu").numpy(),
        ):
            result_dict[detection.label] = {
                "detection": detection,
                "attr": attr_prob,
                "aff": aff_prob,
            }
        results[0] = result_dict

        # w/o edges
        (
            self.attr_probs,
            self.aff_probs,
            _,
            _,
        ) = self.causal_inference_result(
            results,
            key_attr,
            key_aff,
            logic_chains,
            use_logic_chain=False,
            use_filter=False,
        )
        # w/ edges
        _, self.aff_probs_new, _, _ = self.causal_inference_result(
            results,
            key_attr,
            key_aff,
            logic_chains,
            use_logic_chain=True,
            use_filter=False,
        )
        return self.attr_probs, self.aff_probs, self.aff_probs_new

    def streaming_calculate_text_probabilities(
        self,
        frame: Image,
        video_segment,
        obj_id_len,
        key_attr,
        key_aff,
        logic_chains,
    ):
        """Calculate the attribute and affordance probabilities of the objects in the frame.

        Args:
            frame (Image): The frame to process.
            video_segment (dict): _description_
            obj_id_len (_type_): _description_
            key_attr (_type_): _description_
            key_aff (_type_): _description_
            logic_chains (_type_): _description_
        """
        results = {}
        detections = []
        result_dict = {}
        for out_obj_id, out_mask in video_segment.items():
            detections.append(
                DetectionResult(
                    score=0,
                    label=out_mask["label"],
                    box=BoundingBox.from_xyxy(
                        mask_to_bounding_box(out_mask["mask"][0])
                    ),  # TODO
                    mask=out_mask["mask"][0],
                )
            )
        label_list, attr_probs, aff_probs = (
            self.model.calculate_text_probabilities(
                frame,
                detections,
                attribute_list=self.attribute_list,
                affordance_list=self.affordance_list,
                obj_id_len=obj_id_len,
            )
        )
        for detection, attr_prob, aff_prob in zip(
            detections,
            attr_probs.to("cpu").numpy(),
            aff_probs.to("cpu").numpy(),
        ):
            result_dict[detection.label] = {
                "detection": detection,
                "attr": attr_prob,
                "aff": aff_prob,
            }
        results[0] = result_dict

        # w/o edges
        (
            self.attr_probs,
            self.aff_probs,
            _,
            _,
        ) = self.causal_inference_result(
            results,
            key_attr,
            key_aff,
            logic_chains,
            use_logic_chain=False,
            use_filter=False,
        )
        # w/ edges
        _, self.aff_probs_new, _, _ = self.causal_inference_result(
            results,
            key_attr,
            key_aff,
            logic_chains,
            use_logic_chain=True,
            use_filter=False,
        )
        return self.attr_probs, self.aff_probs, self.aff_probs_new

    def calculate_text_probabilities(
        self,
        video_path: str,
        frame_names: list[str],
        vis_frame_stride: int,
        video_segments,
        obj_id_len,
        key_attr,
        key_aff,
        logic_chains,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the attribute and affordance probabilities of the objects in the frame.

        Args:
        - frame: Any
        - detections: List[Any]
        - attribute_list: List[str]
        - affordance_list: List[str]
        - obj_id_len: int

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]
        """
        results = {}
        logger.info("Calculate Text Probabilities:")
        for out_frame_idx in tqdm(
            range(0, len(frame_names), vis_frame_stride)
        ):
            frame = Image.open(
                os.path.join(video_path, frame_names[out_frame_idx])
            )
            # plt.imshow(frame)
            detections = []
            result_dict = {}
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                detections.append(
                    DetectionResult(
                        score=0,
                        label=out_obj_id,
                        box=BoundingBox.from_xyxy(
                            mask_to_bounding_box(out_mask[0])
                        ),  # TODO
                        mask=out_mask[0],
                    )
                )
            label_list, attr_probs, aff_probs = (
                self.model.calculate_text_probabilities(
                    frame,
                    detections,
                    self.attribute_list,
                    self.affordance_list,
                    obj_id_len,
                )
            )
            for detection, attr_prob, aff_prob in zip(
                detections,
                attr_probs.to("cpu").numpy(),
                aff_probs.to("cpu").numpy(),
            ):
                result_dict[detection.label] = {
                    "detection": detection,
                    "attr": attr_prob,
                    "aff": aff_prob,
                }
            results[out_frame_idx] = result_dict

        # w/o edges
        (
            self.attr_probs,
            self.aff_probs,
            self.smoothed_attr_probs,
            self.smoothed_aff_probs,
        ) = self.causal_inference_result(
            results,
            key_attr,
            key_aff,
            logic_chains,
            use_logic_chain=False,
            sigma=self.GUASSIAN_SIGMA,
        )
        # w/ edges
        _, self.aff_probs_new, _, self.smoothed_aff_probs_new = (
            self.causal_inference_result(
                results,
                key_attr,
                key_aff,
                logic_chains,
                use_logic_chain=True,
                sigma=self.GUASSIAN_SIGMA,
            )
        )
        self.results = results
