from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class TrackedObject:
    track_id: int
    cls: int
    conf: float
    box_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]


class YOLOTracker:
    """Uses YOLO track mode (ByteTrack/BoT-SORT) to persist IDs across frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        person_class_id: int | None = None,
        ball_class_id: int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLOTracker") from exc
        self._model = YOLO(model_path)
        self._tracker = tracker
        self._conf = conf
        self._person_class_id = self._resolve_class_id(person_class_id, "person", default=0)
        self._ball_class_id = self._resolve_class_id(ball_class_id, "sports ball", default=32)
        self._classes = list({self._person_class_id, self._ball_class_id})

    def track_frame(self, frame) -> List[TrackedObject]:
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            classes=self._classes,
            conf=self._conf,
            tracker=self._tracker,
        )
        return self._to_tracked_objects(results)

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
    def _to_tracked_objects(results: Iterable) -> List[TrackedObject]:
        tracked: List[TrackedObject] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for box, track_id in zip(boxes, boxes.id):
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tracked.append(
                    TrackedObject(
                        track_id=int(track_id.item()),
                        cls=cls,
                        conf=conf,
                        box_xyxy=(x1, y1, x2, y2),
                        center_xy=(cx, cy),
                    )
                )
        return tracked
