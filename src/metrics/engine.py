from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PlayerTrackState:
    total_distance: float = 0.0
    last_position: Tuple[float, float] | None = None
    speeds: List[float] = field(default_factory=list)


class PlayerMetricsEngine:
    """Computes basic per-player movement metrics from projected points."""

    def __init__(self, fps: float = 30.0, meters_per_unit: float = 1.0) -> None:
        self.fps = fps
        self.meters_per_unit = meters_per_unit
        self._state: Dict[int, PlayerTrackState] = {}

    def update(self, track_id: int, position_xy: Tuple[float, float]) -> None:
        state = self._state.setdefault(track_id, PlayerTrackState())
        if state.last_position is not None:
            distance_units = float(np.linalg.norm(np.array(position_xy) - np.array(state.last_position)))
            distance_m = distance_units * self.meters_per_unit
            state.total_distance += distance_m
            state.speeds.append(distance_m * self.fps)
        state.last_position = position_xy

    def summarize(self) -> Dict[int, Dict[str, float | str]]:
        summary: Dict[int, Dict[str, float | str]] = {}
        for track_id, state in self._state.items():
            avg_speed = float(np.mean(state.speeds)) if state.speeds else 0.0
            summary[track_id] = {
                "total_distance_m": round(state.total_distance, 2),
                "avg_speed_mps": round(avg_speed, 2),
                "heatmap": "placeholder",
                "pass_tracking": "placeholder",
            }
        return summary
