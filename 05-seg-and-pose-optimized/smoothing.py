"""Keypoint smoothing utilities for the pose-based repetition counter.

Provides a sliding-window moving average for 2-D keypoint coordinates,
used to reduce jitter in YOLO26 pose predictions before angle calculation.
"""

import numpy as np
from collections import deque


# Restricted to 2-D point histories: generalising to arbitrary dimensions
# would reduce clarity without meaningful gain for this use case.
def get_moving_average(
    point_history: np.ndarray | deque[np.ndarray], window_size: int
) -> np.ndarray:
    """Calculates the moving average for a multi-dimensional point history.

    If `window_size` is greater than the history length, the average is
    calculated from all available points.

    Args:
        point_history (NDArray): A 2D array of shape (N, 2), where N is the
            number of points and 2 is the number of dimensions (x, y).
        window_size (int): The number of recent samples to average.
            Must be a positive integer.

    Returns:
        NDArray: A 1D array of shape (2,) representing the smoothed
            multi-dimensional point.

    Raises:
        ValueError: If `point_history` is of different shape than (N, 2) or `window_size` <= 0.
    """

    if isinstance(
        point_history, deque
    ):  # deque is permited because is simpler and more readable
        point_history = np.array(point_history)

    # it doesn't have 2 dimensions there aren't 2 colums per row... error
    if point_history.ndim != 2 or point_history.shape[1] != 2:
        raise ValueError(
            f"Points history should be of shape (N, 2): current shape {point_history.shape}"
        )

    if window_size <= 0:
        raise ValueError("Window size should be greater than 0")

    actual_window = min(len(point_history), window_size)

    recent_points = point_history[-actual_window:]
    return np.mean(recent_points, axis=0)
