import argparse


SPEED_SMOOTHING = 0.2


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_video_path", help="Path to source video file")
    parser.add_argument(
        "-t", "--target_video_path", help="Path to save processed video file"
    )
    return parser.parse_args()


def id_selector(speed: float):
    if speed < 80:
        return 0
    elif speed < 100:
        return 1
    elif speed < 120:
        return 2
    else:
        return 3


def ema_speed(curr_speed: float, last_speed: float):
    if last_speed == 0:
        return curr_speed
    else:
        return curr_speed * SPEED_SMOOTHING + last_speed * (1 - SPEED_SMOOTHING)
