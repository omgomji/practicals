"""Detect cars, buses, and trucks in a video using YOLO and OpenCV."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


WINDOW_NAME = "Vehicle Detection Using YOLO"
VEHICLE_CLASS_IDS = (2, 5, 7)  # COCO: car, bus, truck


def parse_args():
    """Parse command-line options for the YOLO vehicle detector."""
    parser = argparse.ArgumentParser(
        description="Detect cars, buses, and trucks from a video using YOLO."
    )
    parser.add_argument(
        "--source",
        default="traffic.mp4",
        help=(
            "Video file path or webcam index (default: traffic.mp4). "
            "Use --source 0 for the default webcam."
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model weights to use (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    return parser.parse_args()


def get_video_source(source_argument):
    """Return a camera index for numeric input, otherwise return the file path."""
    if source_argument.isdigit():
        return int(source_argument)

    source_path = Path(source_argument).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {source_path.resolve()}\n"
            "Add a video named traffic.mp4 to the project directory, or run "
            "with --source path/to/video.mp4. Use --source 0 for a webcam."
        )
    return str(source_path)


def detect_vehicles(model, frame, confidence):
    """Run YOLO on one frame and return the annotated frame and detection count."""
    result = model.predict(
        source=frame,
        classes=list(VEHICLE_CLASS_IDS),
        conf=confidence,
        verbose=False,
    )[0]
    annotated_frame = result.plot()
    detection_count = len(result.boxes) if result.boxes is not None else 0

    cv2.putText(
        annotated_frame,
        f"Vehicles: {detection_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return annotated_frame


def main():
    args = parse_args()
    video_source = get_video_source(args.source)
    model = YOLO(args.model)
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video source: {args.source}. "
            "Check that the video format is supported and the webcam is available."
        )

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            annotated_frame = detect_vehicles(model, frame, args.confidence)
            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
