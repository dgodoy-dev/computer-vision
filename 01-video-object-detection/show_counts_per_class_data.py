import torch
from collections import Counter

HOME = "01-video-object-detection"

object_count_per_frame = torch.load("outputs/counts/counts_per_class_and_frame.pt")

STEPS = 60
LIMIT = 5


for i, conteo in enumerate(object_count_per_frame[: LIMIT * STEPS : 10]):
    frame = i * STEPS
    if conteo:
        print(f"Frame {frame}: {dict(conteo)}")
    else:
        print(f"Frame {frame}: sin detecciones")
