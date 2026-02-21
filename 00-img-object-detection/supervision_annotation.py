from ultralytics import YOLO
import supervision as sv
import cv2
import sys
import os

IMAGE_PATH = "./assets/images/image.png"

if not os.path.exists(IMAGE_PATH):
    print(f"{IMAGE_PATH} file not found")
    sys.exit(1)
else:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"The image {IMAGE_PATH} exists but could't get decoded (format error).")
        sys.exit(1)

model = YOLO("models/yolo26n.pt")

results = model.predict(image.copy())[0]  # just one image

detections = sv.Detections.from_ultralytics(results[0])
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_frame = box_annotator.annotate(image.copy(), detections=detections)
annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)

cv2.imwrite("outputs/image_detected.png", annotated_frame)
