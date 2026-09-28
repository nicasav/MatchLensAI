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

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        person_class_id: int | None = None,
        ball_class_id: int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLODetector") from exc
        self._model = YOLO(model_path)
        self._person_class_id = self._resolve_class_id(person_class_id, "person", default=0)
        self._ball_class_id = self._resolve_class_id(ball_class_id, "sports ball", default=32)
        self._classes = list({self._person_class_id, self._ball_class_id})

    def detect(self, frame) -> List[Detection]:
        """Detect person and sports-ball classes in a frame."""
        results = self._model.predict(frame, verbose=False, classes=self._classes)
        return self._to_detections(results)

    def _resolve_class_id(self, configured_id: int | None, label: str, default: int) -> int:
        if configured_id is not None:
            return int(configured_id)
        names = self._model.names
        if isinstance(names, dict):
            for idx, name in names.items():
                if str(name).strip().lower() == label:
                    return int(idx)
        elif isinstance(names, list):
            for idx, name in enumerate(names):
                if str(name).strip().lower() == label:
                    return int(idx)
        return default

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
