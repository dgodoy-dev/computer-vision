"""Pose-based repetition counter — straightforward implementation.

Extracts upper-body keypoints explicitly, smooths them with a moving average,
and counts push-up repetitions using a rising-edge on a smoothed boolean
arm-state history. Intended as a readable, step-by-step reference.

.. note::
    Designed for a **side-view camera perspective**. Angle-based detection
    relies on a lateral view; other perspectives have not been tested.

Run from the project root::

    python 05-seg-and-pose-optimized/simpler_main.py
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

    # for rising edge (when going from down to up)
    is_down_curr = False
    num_flexions: int = 0

    # extracting info, calculating optimum values and defining output video
    video_info = sv.VideoInfo.from_video_path(SOURCE_PATH)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=f"outputs/{FILE_NAME}_processed.mp4",
        fourcc=fourcc,
        fps=video_info.fps,
        frameSize=video_info.resolution_wh,
    )

    window_size = video_info.fps // 10

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

        # first detection is the most confident
        detection = keypoints.xy[0].cpu().numpy()

        # important keypoints
        left_shoulder = detection[5]
        right_shoulder = detection[6]
        left_elbow = detection[7]
        right_elbow = detection[8]
        left_wrist = detection[9]
        right_wrist = detection[10]

        # update points_history
        points_history["left_shoulder"].append(left_shoulder)
        points_history["right_shoulder"].append(right_shoulder)
        points_history["left_elbow"].append(left_elbow)
        points_history["right_elbow"].append(right_elbow)
        points_history["left_wrist"].append(left_wrist)
        points_history["right_wrist"].append(right_wrist)

        s_left_shoulder = get_moving_average(
            points_history["left_shoulder"], window_size
        )
        s_right_shoulder = get_moving_average(
            points_history["right_shoulder"], window_size
        )
        s_left_elbow = get_moving_average(points_history["left_elbow"], window_size)
        s_right_elbow = get_moving_average(points_history["right_elbow"], window_size)
        s_left_wrist = get_moving_average(points_history["left_wrist"], window_size)
        s_right_wrist = get_moving_average(points_history["right_wrist"], window_size)

        left_elbow_angle = None
        right_elbow_angle = None

        # check if keypoints are valid
        # todo: checking for not smoothed keypoints, should check for smoothed?
        if (  # every right keypoint is valid?
            np.all(right_shoulder > 0)
            and np.all(right_elbow > 0)
            and np.all(right_wrist > 0)
        ):
            right_elbow_angle = calculate_joint_angle(
                s_right_shoulder, s_right_elbow, s_right_wrist
            )

        if (  # every left keypoint is valid?
            np.all(left_shoulder > 0)
            and np.all(left_elbow > 0)
            and np.all(left_wrist > 0)
        ):
            left_elbow_angle = calculate_joint_angle(
                s_left_shoulder, s_left_elbow, s_left_wrist
            )

        if is_arm_down(left_elbow_angle, right_elbow_angle, LOWER_THRESHOLD):
            is_down_curr = True
            is_down_history.append(is_down_curr)  # updates only when there is a change
        elif is_arm_up(left_elbow_angle, right_elbow_angle, UPPER_THRESHOLD):
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

        left_elbow_angle = left_elbow_angle if left_elbow_angle is not None else 0
        right_elbow_angle = right_elbow_angle if right_elbow_angle is not None else 0
        annotated_frame = annotate_metrics(
            annotated_frame,
            {
                "left": np.array(
                    [
                        s_left_shoulder,
                        s_left_elbow,
                        s_left_wrist,
                    ]
                ),
                "right": np.array(
                    [
                        s_right_shoulder,
                        s_right_elbow,
                        s_right_wrist,
                    ]
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


if __name__ == "__main__":  # this ensures only main thread executes this function
    main()
