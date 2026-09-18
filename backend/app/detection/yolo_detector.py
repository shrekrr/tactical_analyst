"""
YOLOv8 detector implementation using the `ultralytics` library.
Automatically selects CUDA if available, falls back to CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from app.detection.base import Detection, ObjectDetector

# COCO class IDs relevant to football
_COCO_PERSON_ID = 0
_COCO_SPORTS_BALL_ID = 32

# Map from COCO class id → our label
_CLASS_MAP = {
    _COCO_PERSON_ID: "player",
    _COCO_SPORTS_BALL_ID: "ball",
}


class YOLODetector(ObjectDetector):
    """
    YOLOv8/v11 detector via ultralytics.

    If a football-specific model is provided the class mapping from that
    model's names will be used.  Otherwise, COCO weights are used and
    ``person`` is remapped to ``player``.

    Parameters
    ----------
    model_path : str | Path
        Path to a .pt weight file, or a model name auto-downloaded by
        ultralytics (e.g. ``"yolov8n.pt"``).
    confidence : float
        Detection confidence threshold.
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
    classes : list[int] | None
        Restrict detection to specific COCO class IDs.
        Defaults to person + sports ball.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence: float = 0.25,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading YOLO model: {} on {}", model_path, device)
        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.confidence = confidence

        # Default to person + ball classes (COCO)
        self.classes = classes or [_COCO_PERSON_ID, _COCO_SPORTS_BALL_ID]

        # Detect whether this is a custom football model by checking names
        self._custom_names: dict[int, str] = {}
        if hasattr(self.model, "names") and self.model.names:
            for cid, name in self.model.names.items():
                lname = name.lower()
                if any(k in lname for k in ("player", "ball", "referee", "goalkeeper")):
                    self._custom_names[cid] = lname

        logger.info(
            "YOLO ready — device={} conf={} custom_names={}",
            device,
            confidence,
            bool(self._custom_names),
        )

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """Run YOLO inference on a single BGR frame."""
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=self.classes if not self._custom_names else None,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                # Determine class label
                if self._custom_names:
                    label = self._custom_names.get(cls_id, "unknown")
                else:
                    label = _CLASS_MAP.get(cls_id, "unknown")

                if label == "unknown":
                    continue

                detections.append(
                    Detection(
                        frame=frame_idx,
                        class_name=label,
                        confidence=conf,
                        bbox=xyxy,
                        class_id=cls_id,
                    )
                )

        return detections
