"""Per-frame maths utilities for the pose-based repetition counter.

Provides joint-angle calculation and arm-state threshold helpers.
Model export / optimisation utilities live in ``export_model.py``.
"""

from math import pi

import numpy as np


def calculate_joint_angle(
    p_top: np.ndarray, p_mid: np.ndarray, p_bottom: np.ndarray
) -> float:
    """
    Calculates the relative angle at a joint (vertex B) formed by two segments.

    Args:
        p_top (np.ndarray): Coordinates [x, y] of the upper point (e.g., Shoulder).
        p_mid (np.ndarray): Coordinates [x, y] of the vertex point (e.g., Elbow).
        p_bottom (np.ndarray): Coordinates [x, y] of the lower point (e.g., Wrist).

    Returns:
        float: The signed angle in radians between the two segments.
    """

    if len(p_top) != 2 or len(p_mid) != 2 or len(p_bottom) != 2:
        raise ValueError("Points should be of shape (2,). One or more points are not.")
    # Create vectors originating from the joint
    v_upper = p_top - p_mid
    v_lower = p_bottom - p_mid

    # np.arctan2 parameters are (y, x). Calculates angle
    angle_upper = np.arctan2(v_upper[1], v_upper[0])
    angle_lower = np.arctan2(v_lower[1], v_lower[0])

    # The relative angle is the difference between the two orientations
    joint_angle = (angle_upper - angle_lower + pi) % (2 * pi) - pi

    return joint_angle


def is_arm_down(
    left_elbow_angle: float | None,
    right_elbow_angle: float | None,
    lower_threshold: float,
) -> bool:
    """
    Returns True if at least one elbow angle indicates the arm is in the
    "down" position (angle >= lower_threshold).

    If both angles are `None` then it returns False.

    Args:
        left_elbow_angle: Angle of the left elbow in radians, or None if invalid.
        right_elbow_angle: Angle of the right elbow in radians, or None if invalid.
        lower_threshold: Minimum angle (in radians) to consider the arm down.

    Returns:
        True if any valid elbow angle meets or exceeds the threshold.
    """
    return any(
        angle is not None and angle <= lower_threshold
        for angle in (left_elbow_angle, right_elbow_angle)
    )


def is_arm_up(
    left_elbow_angle: float | None,
    right_elbow_angle: float | None,
    upper_threshold: float,
) -> bool:
    """
    Returns True if at least one elbow angle indicates the arm is in the
    "up" position (angle <= upper_threshold).

    If both angles are `None` then it returns False.

    Args:
        left_elbow_angle: Angle of the left elbow in radians, or None if invalid.
        right_elbow_angle: Angle of the right elbow in radians, or None if invalid.
        upper_threshold: Maximum angle (in radians) to consider the arm up.

    Returns:
        True if any valid elbow angle is below or equal to the threshold.
    """
    return any(
        angle is not None and angle >= upper_threshold
        for angle in (left_elbow_angle, right_elbow_angle)
    )
