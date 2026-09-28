from __future__ import annotations

import cv2

from src.tracking import TrackedObject


def draw_annotations(frame, tracked_objects: list[TrackedObject]):
    rendered = frame.copy()
    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(v) for v in obj.box_xyxy]
        color = (0, 255, 0) if obj.cls == 0 else (0, 165, 255)
        label = f"ID {obj.track_id} {'player' if obj.cls == 0 else 'ball'}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rendered, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return rendered
