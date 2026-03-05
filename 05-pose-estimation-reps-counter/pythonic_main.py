"""Pose-based repetition counter — pythonic implementation.

Same logic as ``simpler_main.py`` but keypoints and joints are managed via
dictionaries and loops, making the code more scalable when adding extra joints
or sides. Intended as a refactoring reference alongside the simpler version.

.. note::
    Designed for a **side-view camera perspective**. Angle-based detection
    relies on a lateral view; other perspectives have not been tested.

Run from the project root::

    python 05-seg-and-pose-optimized/pythonic_main.py
"""

from collections import defaultdict, deque

from ultralytics import YOLO
import numpy as np
from utils import calculate_joint_angle, is_arm_down, is_arm_up
import cv2
import supervision as sv
from tqdm import tqdm
from smoothing import get_moving_average
from gui import annotate_metrics
from shared_constants import UPPER_THRESHOLD, LOWER_THRESHOLD, MODEL_PATH


def main():

    VIDEO_PATH = "assets/videos/"
    FILE_NAME = "flexions_in_black_hoodie"
    FILE_EXTENSION = ".mp4"
    SOURCE_PATH = VIDEO_PATH + FILE_NAME + FILE_EXTENSION
    UPPER_JOINTS_KEYPOINTS = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
    }
    ARM_JOINTS = {
        "left": ("left_shoulder", "left_elbow", "left_wrist"),
        "right": ("right_shoulder", "right_elbow", "right_wrist"),
    }

    # for rising edge calculus (going up from down)
    is_down_curr = False
    num_flexions: int = 0

    # extracting info, defining output video and calculating window size
    video_info = sv.VideoInfo.from_video_path(SOURCE_PATH)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=f"outputs/{FILE_NAME}_processed.mp4",
        fourcc=fourcc,
        fps=video_info.fps,
        frameSize=video_info.resolution_wh,
    )

    window_size = video_info.fps // 10  # 1/10 secs

    # create display window
    cv2.namedWindow("Flexions counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Flexions counter", 800, 600)

    # predictions
    model = YOLO(MODEL_PATH)
    results = model.predict(stream=True, source=SOURCE_PATH, verbose=False)

    points_history: dict[str, deque[np.ndarray]] = defaultdict(
        lambda: deque(maxlen=window_size)
    )
    is_down_history: deque[bool] = deque(maxlen=window_size * 2)

    # for each predicted frame
    for i, result in tqdm(enumerate(results), total=video_info.total_frames):
        annotated_frame = result.orig_img.copy()  # copy because orig_img is read only

        keypoints = result.keypoints

        # no detection, nothing to do
        if keypoints is None or len(keypoints) <= 0:
            continue

        # first detection is the most confident (ultralytics convention)
        detection = keypoints.xy[0].cpu().numpy()

        # important keypoints
        points: dict[str, np.ndarray] = {
            name: detection[idx] for name, idx in UPPER_JOINTS_KEYPOINTS.items()
        }

        # update points_history
        for name, point in points.items():
            points_history[name].append(point)

        # calculate smoothed points
        smoothed_points = {
            name: get_moving_average(point, window_size)
            for name, point in points_history.items()
        }

        angles: dict[str, float | None] = {}

        # for every side (left, right)
        for side, (shoulder, elbow, wrist) in ARM_JOINTS.items():
            # if every point is valid for every joint (is not (0,0))
            if all(np.all(points[joint] > 0) for joint in (shoulder, elbow, wrist)):
                # calculate angles per side
                angles[side] = calculate_joint_angle(
                    smoothed_points[shoulder],
                    smoothed_points[elbow],
                    smoothed_points[wrist],
                )
            # not valid and no angle calculated
            else:
                angles[side] = None

        if is_arm_down(angles["left"], angles["right"], LOWER_THRESHOLD):
            is_down_curr = True
            is_down_history.append(is_down_curr)  # updates only when there is a change
        elif is_arm_up(angles["left"], angles["right"], UPPER_THRESHOLD):
            is_down_curr = False
            is_down_history.append(is_down_curr)  # updates only when there is a change
        # else: values persist between thresholds -> nothing changes

        # this slows down the change in case there is one, avoiding instantaneous rising edges
        is_down_np = np.array(is_down_history)
        is_down_curr_smoothed = np.mean(is_down_np[-window_size:]) > 0.5
        is_down_prev_smoothed = np.mean(is_down_np[:window_size]) > 0.5

        # rising edge (goes up from down) rem avoids counting when the buffer is not sufficiently changed
        if (i % window_size == 0) and (
            not is_down_curr_smoothed and is_down_prev_smoothed
        ):
            num_flexions += 1

        # transform angles so they can be managed in gui
        left_elbow_angle = angles["left"] if angles["left"] is not None else 0
        right_elbow_angle = angles["right"] if angles["right"] is not None else 0

        annotated_frame = annotate_metrics(
            annotated_frame,
            {
                "left": np.array(
                    [smoothed_points[joint] for joint in ARM_JOINTS["left"]]
                ),
                "right": np.array(
                    [smoothed_points[joint] for joint in ARM_JOINTS["right"]]
                ),
            },
            {
                "left": left_elbow_angle,
                "right": right_elbow_angle,
            },
            num_flexions,
            video_info.resolution_wh,
        )

        cv2.imshow("Flexions counter", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break

        out.write(annotated_frame)

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
