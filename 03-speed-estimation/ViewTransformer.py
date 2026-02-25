import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source_view = source.astype(np.float32)
        self.target_view = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(self.source_view, self.target_view)

    def transform_points(self, points: np.ndarray):
        reshaped_points = points.reshape(-1, 1, 2)  # opencv required shape
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
