"""GUI annotation utilities for the pose-based repetition counter.

This module draws skeleton lines, joint circles, elbow-angle arcs,
left/right range-of-motion progress bars, and a repetition dashboard
onto each video frame using OpenCV.
"""

import cv2
import numpy as np
from math import pi
from cv2.typing import MatLike
from shared_constants import UPPER_THRESHOLD, LOWER_THRESHOLD

# ---------------------------------------------------------------------------
# Module-level aesthetic constants (defined once, not per frame)
# ---------------------------------------------------------------------------
_COLOR_CYAN = (255, 255, 0)
_COLOR_MAGENTA = (255, 0, 255)
_COLOR_WHITE = (250, 250, 250)
_COLOR_BLACK = (15, 15, 15)
_FONT = cv2.FONT_HERSHEY_DUPLEX

_SIDE_COLOR: dict[str, tuple[int, int, int]] = {
    "left": _COLOR_CYAN,
    "right": _COLOR_MAGENTA,
}


def draw_short_angle_arc(
    frame: MatLike,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Draw the inner (shortest) angle arc between three points.

    Args:
        frame: The BGR image to draw on (modified in place).
        p1: Coordinates [x, y] of the first outer point (e.g. shoulder).
        p2: Coordinates [x, y] of the vertex point (e.g. elbow).
        p3: Coordinates [x, y] of the second outer point (e.g. wrist).
        color: BGR colour tuple for the arc fill and border.
    """
    # absolute segment angles from the horizon
    ang_a = np.degrees(np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
    ang_b = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]))

    # calculate cyclic difference to ensure shortest path (<180°)
    diff = (ang_b - ang_a + 180) % 360 - 180
    ang_final = ang_a + diff  # to know where the angle finishes

    center = (int(p2[0]), int(p2[1]))
    axes = (45, 45)  # ellipse's radius

    # drawing with transparency
    overlay = frame.copy()
    cv2.ellipse(overlay, center, axes, 0, ang_a, ang_final, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # arc's border
    cv2.ellipse(frame, center, axes, 0, ang_a, ang_final, color, 3, cv2.LINE_AA)


def annotate_metrics(
    frame: MatLike,
    keypoints: dict[str, np.ndarray],
    angles: dict[str, float],
    num_flexions: int,
    resolution: tuple[int, int],
    label: str = "REPS",
) -> MatLike:
    """Overlay pose metrics onto a video frame.

    Draws per-arm skeleton lines, joint circles, elbow-angle arcs,
    range-of-motion progress bars, and a repetition count dashboard.

    Args:
        frame: Source BGR frame (not modified; a copy is returned).
        keypoints: Dict mapping ``"left"`` / ``"right"`` to an (N, 2) array
            of [x, y] joint coordinates (shoulder, elbow, wrist).
        angles: Dict mapping ``"left"`` / ``"right"`` to the elbow angle in
            radians (or 0 when the angle is unavailable).
        num_flexions: Current repetition count to display.
        resolution: ``(width, height)`` of the frame in pixels.
        label: Text shown above the repetition counter. Defaults to ``"REPS"``.

    Returns:
        A new annotated BGR frame.
    """
    W, H = resolution
    annotated_frame = frame.copy()

    # --- arms processing ---
    for side in ["left", "right"]:
        if side not in keypoints or side not in angles:
            continue

        color = _SIDE_COLOR[side]
        # p1: shoulder, p2: elbow, p3: wrist
        pts = keypoints[side]
        if len(pts) < 3:
            continue
        p1, p2, p3 = pts[0], pts[1], pts[2]

        # draw arc and skeleton
        draw_short_angle_arc(annotated_frame, p1, p2, p3, color)

        cv2.line(
            annotated_frame,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            _COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            annotated_frame,
            (int(p2[0]), int(p2[1])),
            (int(p3[0]), int(p3[1])),
            _COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        # points with glow effect
        for pt in [p1, p2, p3]:
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(
                annotated_frame, center, 10, _COLOR_BLACK, -1, cv2.LINE_AA
            )  # border
            cv2.circle(annotated_frame, center, 6, color, -1, cv2.LINE_AA)  # center

        # angle text (in place)
        angle_deg = angles[side] * 180 / pi
        txt_pos = (int(p2[0]) + 20, int(p2[1]) - 20)
        # shadow for legibility
        cv2.putText(
            annotated_frame,
            f"{angle_deg:.0f}",
            txt_pos,
            _FONT,
            0.9,
            _COLOR_BLACK,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"{angle_deg:.0f}",
            txt_pos,
            _FONT,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        # progress range of motion bars
        # clip ensures the results persists inside the range (0, 1). angle normalized
        progress = np.clip(
            (angles[side] - LOWER_THRESHOLD) / (UPPER_THRESHOLD - LOWER_THRESHOLD), 0, 1
        )

        bar_w, bar_h = 25, 220
        bar_x = 40 if side == "left" else W - 65
        bar_y = H // 2 - 110

        # bar container
        cv2.rectangle(
            annotated_frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            _COLOR_BLACK,
            -1,
        )
        cv2.rectangle(
            annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, 1
        )

        # progress fill
        fill_h = int(bar_h * progress)
        cv2.rectangle(
            annotated_frame,
            (bar_x, bar_y + bar_h - fill_h),
            (bar_x + bar_w, bar_y + bar_h),
            color,
            -1,
        )

        # L/R label
        cv2.putText(
            annotated_frame,
            side[0].upper(),
            (bar_x + 3, bar_y - 15),
            _FONT,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # --- repetitions dashboard ---
    # transparent rectangle
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (20, 20), (240, 130), _COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

    cv2.rectangle(annotated_frame, (20, 20), (240, 130), _COLOR_CYAN, 1)
    cv2.putText(
        annotated_frame, label, (40, 50), _FONT, 0.6, _COLOR_CYAN, 1, cv2.LINE_AA
    )
    cv2.putText(
        annotated_frame,
        f"{num_flexions:02d}",
        (65, 115),
        _FONT,
        2.2,
        _COLOR_WHITE,
        4,
        cv2.LINE_AA,
    )

    # --- inferior bar (only aesthetics) ---
    cv2.rectangle(annotated_frame, (0, H - 4), (W, H), _COLOR_CYAN, -1)

    return annotated_frame
