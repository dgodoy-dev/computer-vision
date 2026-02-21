from collections import Counter
from pathlib import Path
from ultralytics import YOLO
import cv2
import sys
import os
import supervision as sv
from torch import save, tensor

HOME = "01-video-object-detection"
VIDEO_DIR = "./assets/videos"
VIDEO_PATH = VIDEO_DIR + "/traffic_video.mp4"
CLASSES_TO_PREDICT = ["car"]

if not os.path.exists(VIDEO_PATH):
    print(f"{VIDEO_PATH} does not exist.")
    sys.exit(1)

model = YOLO("models/yolo26n.pt")

class_names = model.names

names_inv = {name: id for id, name in class_names.items()}  # inverted, name is the key
classes_to_predict = [
    names_inv[name] for name in CLASSES_TO_PREDICT
]  # translation from names to ids

results = model.predict(VIDEO_PATH, stream=True, classes=[classes_to_predict])

class_count_per_frame: list[int] = []

for r in results:
    if not r.boxes:
        class_count_per_frame.append(-1)
        continue

    # only predicts cars, so the number of predictions is the number of cars
    class_count_per_frame.append(len(r.boxes))

save(class_count_per_frame, f"outputs/counts/car_count_per_frame.pt")
