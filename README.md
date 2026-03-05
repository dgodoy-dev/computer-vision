# 🧠 Computer Vision

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111827?style=flat)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

A collection of practical computer vision use cases built with different detections models like **YOLO26 (Ultralytics)** and **RF-DETR (Roboflow)**, **OpenCV**, and **Supervision**. This repository is structured as a growing set of self-contained examples, ranging from simple image inference to complex real-time video analytics.

---

## 🖼️ Demo

| Supervision Annotator | OpenCV Annotator |
|---|---|
| ![Supervision demo](docs/demo_supervision.png) | ![OpenCV demo](docs/demo_opencv.png) |

| Pose Estimation Repetition Counter |
|---|
| ![Pose Counter demo](docs/demo_pose_counting.gif) |

---

## 📦 Tech Stack

| Library | Purpose |
|---|---|
| [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) | Object detection & tracking model |
| [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr) | Transformer-based object detection model |
| [OpenCV](https://opencv.org/) | Image/video I/O and drawing |
| [Supervision](https://github.com/roboflow/supervision) | High-level annotation utilities |
| [PyTorch](https://pytorch.org/) | Tensor operations and model I/O |
| [Matplotlib](https://matplotlib.org/) | Data visualization |
| [tqdm](https://github.com/tqdm/tqdm) | Progress bars for video processing |

---

## 🗂️ Use Cases

### `00` — Image Object Detection

> Detect objects in a static image using two different annotation backends.

| Script | Description |
|---|---|
| `open_cv_annotation.py` | Runs YOLO inference on an image and draws bounding boxes + labels using raw **OpenCV** drawing functions |
| `supervision_annotation.py` | Same inference, but uses the **Supervision** `BoxAnnotator` and `LabelAnnotator` for cleaner, high-level overlays |

**Input:** `assets/images/<your_image>.png`  
**Output:** `outputs/<your_image>_opencv_detected.png` / `outputs/<your_image>_detected.png`

---

### `01` — Video Object Detection & Counting

> Run YOLO inference over an entire video stream and extract per-frame object counts for downstream analytics.

| Script | Description |
|---|---|
| `video-detection.py` | Runs detection on a traffic video, accumulates per-class object counts for each frame, and saves results to a `.pt` file |
| `car-count-per-frame.py` | Simplified version: counts only cars per frame and saves the list as a `.pt` tensor |
| `show_car_counts.py` | Loads car count data, prints a statistical summary (max, mean, median, std), and plots a count-per-frame graph |
| `show_counts_per_class_data.py` | Loads multi-class count data and prints sampled frame-by-frame detections |

**Input:** `assets/videos/<your_video>.mp4`  
**Output:** `outputs/counts/counts_per_class_and_frame.pt`, `outputs/counts/car_count_per_frame.pt`, `outputs/car_count_per_frame.png`

---

### `02` — Video Object Tracking *(coming soon)*

> Assign persistent IDs to detected objects across frames using YOLO's built-in tracker.

---

### `03` — Video Speed Estimation

> Estimate the real-world speed of detected vehicles across frames using perspective transformation and ByteTrack.

| Script | Description |
|---|---|
| `main.py` | Runs RF-DETR inference on a traffic video, tracks vehicles with ByteTrack, applies a perspective transform to convert pixel movement into real-world distance, and annotates each vehicle with its estimated speed (km/h) colour-coded by range |
| `ViewTransformer.py` | Utility class that wraps OpenCV's `getPerspectiveTransform` to map pixel coordinates to a top-down metric plane |
| `utils.py` | Argument parsing, EMA speed smoothing, and speed-range colour selector |
| `video_downloader.py` | Helper to download a sample video asset via Supervision |

**Input:** `assets/videos/<your_video>.mp4`  
**Output:** `outputs/<your_video>_processed.mp4`

**Run:**
```bash
python 03-speed-estimation/main.py \
  -s assets/videos/<your_video>.mp4 \
  -t outputs/<your_video>_processed.mp4
```

---

### `04` — Speed Estimation (Source-Agnostic) *(in progress)*

> A variant of use case `03` designed to work with any video source, with an automatic perspective transform determination step.

---

### `05` — Pose-Based Repetition Counter

> Counts push-up repetitions in real-time using YOLO26 pose estimation. Tracks elbow angles for both arms with temporal smoothing to avoid false positives, and overlays a live GUI (angle arcs, progress bars, repetition count) on each frame.
>
> ⚠️ **Designed for a side-view camera perspective.** Angle-based detection relies on a lateral view of the subject; other perspectives (frontal, overhead, etc.) have not been tested and may produce incorrect counts.

| Script | Description |
|---|---|
| `simpler_main.py` | Straightforward implementation: extracts the six key upper-body joints explicitly, applies moving-average smoothing, detects arm state transitions (up/down), and counts repetitions using a rising-edge on smoothed boolean history |
| `pythonic_main.py` | Refactored version of the same logic using dictionaries and loops for keypoint management — more scalable when targeting additional joints or sides |
| `utils.py` | `calculate_joint_angle` (vector-based, returns radians), `is_arm_down` / `is_arm_up` threshold helpers, and `optimize_model` (TensorRT/OpenVINO export) |
| `smoothing.py` | `get_moving_average` — sliding-window mean over a `deque` of 2-D keypoint coordinates |
| `gui.py` | `annotate_metrics` — draws skeleton lines, joint circles, angle arcs (with transparency), left/right progress bars, and a repetition dashboard on the frame |
| `shared_constants.py` | `UPPER_THRESHOLD` / `LOWER_THRESHOLD` angles (in radians) that define the up/down arm states |

**Input:** `assets/videos/<your_video>.mp4`  
**Output:** `outputs/<your_video>_processed.mp4` · live preview window

**Run:**
```bash
# Simpler, explicit version
python 05-seg-and-pose-optimized/simpler_main.py

# More pythonic, loop-driven version
python 05-seg-and-pose-optimized/pythonic_main.py
```

> ⚠️ The video file name is currently hardcoded as `FILE_NAME` inside each script. Update it to match your asset before running.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dgodoy-dev/computer-vision.git
cd computer-vision
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv-cv
source .venv-cv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download or place your YOLO model

The scripts use a pre-trained `yolo26n.pt` model. Place the one downloaded automatically from Ultralytics inside the `models/` folder:

Or swap it for any standard Ultralytics model by updating the model path in each script.

### 5. Add your media files

> ⚠️ Media files are **not included** in this repository. You must supply your own.

The `assets/`, `models/`, and `outputs/` directories are already present in the repo (tracked via `.gitkeep`). Just drop your files in the right place:

- Place input **images** inside `assets/images/` (e.g. `assets/images/<your_image>.png`)
- Place input **videos** inside `assets/videos/` (e.g. `assets/videos/<your_video>.mp4`)

The scripts reference these paths by default, so make sure the file names match or update the `IMAGE_PATH` / `VIDEO_PATH` constants at the top of each script.

**Note:** soon scripts to download media files will be available.

---

## 🏃 Running Examples

```bash
# Image detection — OpenCV annotation
python 00-img-object-detection/open_cv_annotation.py

# Image detection — Supervision annotation
python 00-img-object-detection/supervision_annotation.py

# Video detection — multi-class count per frame
python 01-video-object-detection/video-detection.py

# Video detection — car count per frame
python 01-video-object-detection/car-count-per-frame.py

# Visualize car counts
python 01-video-object-detection/show_car_counts.py

# Inspect multi-class counts per frame
python 01-video-object-detection/show_counts_per_class_data.py

# Vehicle speed estimation
python 03-speed-estimation/main.py \
  -s assets/videos/<your_video>.mp4 \
  -t outputs/<your_video>_processed.mp4

# Pose-based repetition counter — simple version
python 05-seg-and-pose-optimized/simpler_main.py

# Pose-based repetition counter — pythonic version
python 05-seg-and-pose-optimized/pythonic_main.py
```

---

## 📁 Project Structure

```
cv/
├── 00-img-object-detection/        # static image detection
│   ├── open_cv_annotation.py
│   └── supervision_annotation.py
├── 01-video-object-detection/      # video inference & counting
│   ├── video-detection.py
│   ├── car-count-per-frame.py
│   ├── show_car_counts.py
│   └── show_counts_per_class_data.py
├── 02-video-object-tracking/       # coming soon
├── 03-speed-estimation/            # vehicle speed estimation via perspective transform
│   ├── main.py
│   ├── ViewTransformer.py
│   ├── utils.py
│   └── video_downloader.py
├── 04-speed-estimation-with-to-determine/  # source-agnostic speed estimation (in progress)
│   └── main.py
├── 05-seg-and-pose-optimized/      # pose-based repetition counter
│   ├── simpler_main.py
│   ├── pythonic_main.py
│   ├── export_model.py
│   ├── gui.py
│   ├── utils.py
│   ├── smoothing.py
│   └── shared_constants.py
│
├── assets/                         # ⚠ contents gitignored — add your own files
│   ├── images/
│   │   └── image.png
│   └── videos/
│       └── video.mp4
│
├── models/                         # ⚠ contents gitignored — add your own weights
│   └── yolo26n.pt
│
├── outputs/                        # ⚠ contents gitignored — auto-generated at runtime
│   ├── image_opencv_detected.png
│   ├── image_detected.png
│   └── counts/
│       ├── car_count_per_frame.pt
│       └── counts_per_class_and_frame.pt
│
├── runs/                           # YOLO auto-generated inference logs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚗 Roadmap

| # | Use Case | Status |
|---|---|---|
| 00 | Image Object Detection | ✅ Done |
| 01 | Video Object Detection & Counting | ✅ Done |
| 02 | Video Object Tracking | 🔜 Coming soon |
| 03 | Video Speed Estimation | ✅ Done |
| 04 | Speed Estimation (source-agnostic) | 🚧 In progress |
| 05 | Pose-Based Repetition Counter | ✅ Done |

---

## 📌 Notes

- All scripts assume they are run from the **project root** directory.
- The `stream=True` flag in video scripts enables memory-efficient frame-by-frame processing instead of loading the whole video at once.
- `.pt` files (PyTorch tensors/lists) are used to persist detection results for later analysis without re-running inference.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
