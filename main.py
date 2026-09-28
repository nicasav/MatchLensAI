from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Dict

import cv2

from src.homography import HomographyProjector
from src.metrics import PlayerMetricsEngine
from src.tracking import YOLOTracker
from src.ui import draw_annotations


DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": "yolov8n.pt",
    "tracker": "bytetrack.yaml",
    "confidence": 0.25,
    "fps": 30.0,
    "homography": {
        "src_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "dst_points": [[0, 0], [105, 0], [105, 68], [0, 68]],
    },
}


def load_config(config_path: str | None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)
    cfg.update({k: v for k, v in file_cfg.items() if k != "homography"})
    if "homography" in file_cfg:
        cfg["homography"] = {**cfg["homography"], **file_cfg["homography"]}
    return cfg


def process_video(video_path: str, output_path: str | None, config: Dict[str, Any]):
    tracker = YOLOTracker(
        model_path=config["model_path"],
        tracker=config["tracker"],
        conf=float(config["confidence"]),
    )
    projector = HomographyProjector(
        src_points=config["homography"]["src_points"],
        dst_points=config["homography"]["dst_points"],
    )
    metrics = PlayerMetricsEngine(fps=float(config["fps"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = capture_fps if math.isfinite(capture_fps) and capture_fps > 0 else float(config["fps"])
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video file: {output_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracked = tracker.track_frame(frame)
        for obj in tracked:
            if obj.cls == 0:
                projected = projector.project_points([obj.center_xy])
                if len(projected):
                    metrics.update(obj.track_id, tuple(projected[0]))

        annotated = draw_annotations(frame, tracked)
        if writer is not None:
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()

    return metrics.summarize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatchLens AI video analysis")
    parser.add_argument("--video", required=True, help="Path to input match video (.mp4)")
    parser.add_argument("--output", help="Optional output path for annotated video")
    parser.add_argument("--config", help="Optional JSON config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not video_path.is_file():
        raise ValueError(f"Input path is not a file: {args.video}")

    metrics_summary = process_video(args.video, args.output, config)
    for track_id, values in metrics_summary.items():
        print(f"Player {track_id}: {values}")


if __name__ == "__main__":
    main()
