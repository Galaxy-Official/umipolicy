# import logging

import torch
from PIL import Image
from transformers import pipeline
from loguru import logger
from perception.utils.result import DetectionResult

# logger = logging.getLogger(__name__)


class ObjectDetectionPipeline:
    """
    Pipeline for object detection. (Grounding DINO)
    """

    def __init__(
        self,
        detector_path: str,
        device: torch.device,
    ) -> None:
        detector_path = (
            detector_path
            if detector_path is not None
            else "IDEA-Research/grounding-dino-base"
        )
        logger.info(f"Loading object detector from {detector_path}")
        self.object_detector = pipeline(
            model=detector_path,
            task="zero-shot-object-detection",
            device=device,
        )

    def object_detection(
        self,
        image: Image.Image,
        labels: list,
        threshold: float = 0.25,
        select_highest_score: bool = True,
    ) -> list:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        
        Args:
            image: Input image for detection
            labels: List of labels to detect
            threshold: Detection confidence threshold
            select_highest_score: If True, select only the highest scoring detection per object type
        
        Returns:
            List of DetectionResult objects
        """
        labels = [
            label if label.endswith(".") else label + "." for label in labels
        ]
        results = self.object_detector(
            image, candidate_labels=labels, threshold=threshold
        )
        logger.debug(f"Detection results: {results}")
        results = [DetectionResult.from_dict(result) for result in results]
        
        if select_highest_score:
            results = self._select_highest_score_per_object(results)
        return results
    
    def _select_highest_score_per_object(self, results: list) -> list:
        """
        Select the highest scoring detection for each unique object type.
        
        Args:
            results: List of DetectionResult objects
            
        Returns:
            List of DetectionResult objects with highest score per object type
        """
        if not results:
            return results
            
        # Group detections by label
        label_groups = {}
        for result in results:
            label = result.label
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(result)
        
        # Select highest scoring detection for each label
        highest_score_results = []
        for label, detections in label_groups.items():
            highest_score_detection = max(detections, key=lambda x: x.score)
            highest_score_results.append(highest_score_detection)
        
        # Sort by score in descending order
        highest_score_results.sort(key=lambda x: x.score, reverse=True)
        
        return highest_score_results
