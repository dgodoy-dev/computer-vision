from collections import defaultdict, deque
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker
import numpy as np
from tqdm import tqdm

from ViewTransformer import ViewTransformer
from utils import ema_speed, id_selector, parse_arguments

# ====== CONFIGURATION ======
# Defines the colors used for different speed ranges.
# Index 0: White (Calculating), 1: Green, 2: Yellow, 3: Red
SPEED_PALETTE = sv.ColorPalette(
    colors=[
        sv.Color.from_hex("#FFFFFF"),  # Index 0: White
        sv.Color.from_hex("#00FF00"),  # Index 1: Green
        sv.Color.from_hex("#E0AC00"),  # Index 2: Yellow
        sv.Color.from_hex("#FF0000"),  # Index 3: Red
    ]
)

# Defines source and target view/perspective
SOURCE_VIEW = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET_VIEW = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

if __name__ == "__main__":
    args = parse_arguments()

    # Load video metadata (resolution, fps)
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # Load the Object Detection Model (Roboflow)
    model = get_model("rfdetr-small")

    # Initialize the Tracker (ByteTrack algorithm)
    tracker = ByteTrackTracker()

    # ====== VISUAL CONFIGURATION (ANNOTATORS) ======

    # Calculate dynamic line thickness/text size based on video resolution
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    # annotator will select color based on the object's class_id
    box_annotator = sv.BoxAnnotator(thickness=thickness, color=SPEED_PALETTE)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color=SPEED_PALETTE,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,  # Trace lasts for 2 seconds
        position=sv.Position.BOTTOM_CENTER,
        color=SPEED_PALETTE,
    )

    # ====== GEOMETRY & STATE MANAGEMENT ======
    # Define the Polygon Zone (ROI)
    polygon_zone = sv.PolygonZone(SOURCE_VIEW)

    # Initialize the Perspective Transformer (Pixels -> Meters)
    view_transformer = ViewTransformer(SOURCE_VIEW, TARGET_VIEW)

    # Stores last 1 second (video_info.fps) of points per car. speed calculation
    coordinate_history = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Dictionary to store previous speed for smoothing (EMA)
    prev_speeds = defaultdict(float)

    # ====== MAIN PROCESSING LOOP ======
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    # open target video
    with sv.VideoSink(args.target_video_path, video_info) as sink:

        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # --- INFERENCE & TRACKING ---
            result = model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)

            # Filter detections: Only keep those inside the polygon zone
            detections = detections[polygon_zone.trigger(detections)]

            detections = tracker.update(detections=detections)

            # --- COORDINATE TRANSFORMATION ---
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            # --- SPEED CALCULATION ---
            labels = []
            custom_class_ids = []  # To override colors based on speed

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinate_history[tracker_id].append(y)  # Store current Y position

                # Phase 1: Not enough data yet (Buffer filling)
                if len(coordinate_history[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}: Calculating...")
                    prev_speeds[tracker_id] = 0
                    custom_class_ids.append(0)  # Index 0 = White Color

                # Phase 2: Calculate Speed
                else:
                    coordinate_start = coordinate_history[tracker_id][0]
                    coordinate_end = coordinate_history[tracker_id][-1]

                    # Math: Distance / Time * Conversion to km/h
                    distance = abs(coordinate_end - coordinate_start)
                    time = len(coordinate_history[tracker_id]) / video_info.fps
                    curr_speed = distance / time * 3.6

                    # Apply Smoothing (Exponential Moving Average)
                    speed = ema_speed(curr_speed, prev_speeds[tracker_id])
                    prev_speeds[tracker_id] = speed

                    labels.append(f"#{tracker_id} {int(speed)} km/h")

                    # Assign Color Index based on Speed (Green/Yellow/Red)
                    custom_class_ids.append(id_selector(speed))

            # --- APPLY CUSTOM COLORS ---
            # Overwrite the detection's class_id so annotators use Speed Palette
            if len(custom_class_ids) > 0:
                detections.class_id = np.array(custom_class_ids)

            # --- DRAWING & DISPLAY ---
            annotated_frame = frame.copy()

            annotated_frame = sv.draw_polygon(
                annotated_frame, polygon=SOURCE_VIEW, color=sv.Color.GREY
            )

            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(annotated_frame)
