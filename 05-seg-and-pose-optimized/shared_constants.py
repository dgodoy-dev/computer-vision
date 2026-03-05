"""Shared configuration constants for the pose-based repetition counter.

Angles are in radians. Lower elbow angles correspond to greater elbow flexion; 
higher angles correspond to arm extension.
"""

from math import pi

# Model weights used for YOLO26 pose estimation
MODEL_PATH = "yolo26s-pose.pt"


UPPER_THRESHOLD = 0.8 * pi  # almost 180 degrees
LOWER_THRESHOLD = 0.50 * pi  # exactly 90 degrees with respect to the ground
