from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


class HomographyProjector:
    """Projects image coordinates onto a top-down pitch plane."""

    def __init__(
        self,
        src_points: Sequence[Sequence[float]],
        dst_points: Sequence[Sequence[float]],
    ) -> None:
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        if self.src_points.shape != (4, 2) or self.dst_points.shape != (4, 2):
            raise ValueError("src_points and dst_points must each be shape (4, 2)")
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def project_points(self, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        pts = np.array(list(points), dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.matrix)
        return projected.reshape(-1, 2)

    def warp_pitch(self, frame, output_size: Tuple[int, int]):
        return cv2.warpPerspective(frame, self.matrix, output_size)
