import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @classmethod
    def from_xyxy(cls, xyxy: List[float]) -> "BoundingBox":
        return cls(xmin=xyxy[0], ymin=xyxy[1], xmax=xyxy[2], ymax=xyxy[3])

    def to_json(self) -> Dict:
        return json.dumps(
            {
                "xmin": int(self.xmin),
                "ymin": int(self.ymin),
                "xmax": int(self.xmax),
                "ymax": int(self.ymax),
            }
        )


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )

    def to_json(self) -> Dict:
        detection_dict = {
            "score": self.score,
            "label": self.label,
            "box": json.loads(self.box.to_json()),
        }
        if self.mask is not None:
            detection_dict["mask"] = self.mask.tolist()
        return json.dumps(detection_dict)
