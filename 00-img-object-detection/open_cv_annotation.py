from ultralytics import YOLO
import cv2
import sys
import os

IMAGE_PATH = "./assets/images/image.png"

# checking existence and proper load
if not os.path.exists(IMAGE_PATH):
    print(f"File {IMAGE_PATH} not found.")
    sys.exit(1)
else:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Image {IMAGE_PATH} exists but could't get decoded (format error)")
        sys.exit(1)

model = YOLO("models/yolo26n.pt")
results = model.predict(image.copy())[0]  # just one image

if results.boxes is None:
    print("There was no prediction.")
    sys.exit(1)

detections = results.boxes.data.cpu().numpy()

for d in detections:
    # data extraction and casting
    x1, y1, x2, y2 = map(int, d[:4])
    conf = float(d[4])
    cls = int(d[5])

    # label config
    class_name = model.names[cls]
    label = f"{class_name}: {conf:.2f}"

    # opencv draw
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.putText(
        image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 5
    )

cv2.imwrite("outputs/image_opencv_detected.png", image)
