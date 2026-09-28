from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Detection:
    cls: int
    conf: float
    box_xyxy: Tuple[float, float, float, float]


class YOLODetector:
    """Wrapper around Ultralytics YOLO for football player/ball detection."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLODetector") from exc
        self._model = YOLO(model_path)

    def detect(self, frame) -> List[Detection]:
        """Detect only people (0) and ball (32) in a frame."""
        results = self._model.predict(frame, verbose=False, classes=[0, 32])
        return self._to_detections(results)

    @staticmethod
    def _to_detections(results: Iterable) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(cls=cls, conf=conf, box_xyxy=(x1, y1, x2, y2)))
        return detections
