"""Standalone utility for exporting a YOLO26 model to an optimised runtime format.

Run this script once before inference to produce a TensorRT (.engine) or
OpenVINO (.xml/.bin) serialised model that is fused and hardware-optimised
for your specific device.

Usage::

    python 05-seg-and-pose-optimized/export_model.py
    python 05-seg-and-pose-optimized/export_model.py --format openvino
"""

import argparse

from ultralytics import YOLO
from shared_constants import MODEL_PATH


def export_model(model_path: str, export_format: str = "engine") -> YOLO:
    """Export a YOLO model to an optimised runtime format.

    Fuses model layers and optimises the compute graph for the target hardware.
    FP16 precision is enabled by default for an approximately 2× speedup over
    FP32 on supported devices.

    Args:
        model_path: Path to the source ``.pt`` weights file.
        export_format: Target format, e.g. ``"engine"`` (TensorRT) or
            ``"openvino"``. See Ultralytics docs for the full list.

    Returns:
        A YOLO instance loaded from the newly exported model file.
    """
    model = YOLO(model_path)
    model.export(format=export_format, half=True, dynamic=True)
    exported_path = model_path.replace(".pt", f".{export_format}")
    return YOLO(exported_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a YOLO26 model to an optimised format.")
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to the source .pt weights file (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--format",
        default="engine",
        help='Export format, e.g. "engine" (TensorRT) or "openvino" (default: engine)',
    )
    args = parser.parse_args()
    export_model(args.model, args.format)
