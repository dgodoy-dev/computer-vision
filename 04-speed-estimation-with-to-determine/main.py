from collections import defaultdict, deque
from utils import ema_speed, euclidean_dist, parse_arguments
from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
from trackers import ByteTrackTracker
from tqdm import tqdm

# in case of debuging
IS_DEBUGGING = True
SOURCE_VIDEO_PATH = "assets/videos/part.mp4"
TARGET_VIDEO_PATH = "outputs/processed_part.mp4"

SOURCE_VIEW = np.array([[447, 1080], [742, 1072], [1312, 1470], [-560, 1450]])
TARGET_VIEW = np.array([[0.0, 0.0], [3.9, 0.0], [0.0, 14.0], [3.9, 14.0]])
CLASSES = [0]  # only person


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)  # float necessary for cv2 maths
        self.target = target.astype(np.float32)  # float64 it's overkill and slows
        self.m = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_perspective(self, points: np.ndarray):
        # cv2 requires (N, M, L), where M are channels
        reshaped_coordinates = points.reshape(-1, 1, 2)
        transformed_coordinates = cv2.perspectiveTransform(reshaped_coordinates, self.m)
        return transformed_coordinates.reshape(-1, 2)  # back to 2D


if __name__ == "__main__":
    # values initialization
    if IS_DEBUGGING:
        src_vid = SOURCE_VIDEO_PATH
        tar_vid = TARGET_VIDEO_PATH
    else:
        args = parse_arguments()
        src_vid = args.source_video_path
        tar_vid = args.target_video_path

    model = YOLO("models/yolo26s.pt")

    video_info = sv.VideoInfo.from_video_path(src_vid)

    tracker = ByteTrackTracker(frame_rate=video_info.fps)

    view_transformer = ViewTransformer(SOURCE_VIEW, TARGET_VIEW)

    # optimum values for annotators
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    # annotators
    poligon_zone = sv.PolygonZone(SOURCE_VIEW)
    box_annotator = sv.BoxAnnotator(
        thickness=thickness, color_lookup=sv.ColorLookup.TRACK
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, color_lookup=sv.ColorLookup.TRACK
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        color_lookup=sv.ColorLookup.TRACK,
        position=sv.Position.BOTTOM_CENTER,
    )

    # coordinates for speed calculation
    coordinates_history = defaultdict(lambda: deque(maxlen=video_info.fps))
    last_speed = defaultdict(float)

    # video processing
    results_generator = model.predict(
        src_vid, classes=CLASSES, stream=True, verbose=False
    )

    with sv.VideoSink(tar_vid, video_info) as sink:
        for result in tqdm(results_generator, total=video_info.total_frames):
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[
                poligon_zone.trigger(detections)
            ]  # filter, is it in the poligon zone?
            detections = tracker.update(detections)

            # speed calculus
            points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            labels = []
            if len(points) != 0:  # no detections, nothing to do

                points = view_transformer.transform_perspective(np.array(points))

                for tracker_id, [x, y] in zip(detections.tracker_id, points):
                    coordinates_history[tracker_id].append([x, y])

                    # buffer filling
                    if len(coordinates_history[tracker_id]) < video_info.fps:
                        labels.append("Calculating...")
                        last_speed[tracker_id] = 0

                    # actual speed calculation
                    else:
                        dist = euclidean_dist(
                            *coordinates_history[tracker_id][0],
                            *coordinates_history[tracker_id][-1],
                        )

                        time = len(coordinates_history[tracker_id]) / video_info.fps
                        curr_speed = dist / time
                        smoothed_speed = ema_speed(curr_speed, last_speed[tracker_id])

                        labels.append(f"{int(smoothed_speed)} m/s")

            # annotate video
            annotated_frame = result.orig_img.copy()
            annotated_frame = sv.draw_polygon(annotated_frame, SOURCE_VIEW)
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(
                annotated_frame, detections, labels=labels
            )
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)

            sink.write_frame(annotated_frame)
