"""Real-time face detection using HAAR Cascade and MediaPipe."""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


WINDOW_HAAR = "HAAR Face Detection"
WINDOW_MEDIAPIPE = "MediaPipe Face Detection"
MEDIAPIPE_MODEL = Path(__file__).resolve().parent / "face_landmarker.task"


def load_haar_cascade():
    """Load the frontal-face cascade from common OpenCV installation paths."""
    cascade_name = "haarcascade_frontalface_default.xml"
    candidate_paths = (
        Path(cv2.data.haarcascades) / cascade_name,
        Path("/usr/share/opencv5/haarcascades") / cascade_name,
        Path("/usr/share/opencv4/haarcascades") / cascade_name,
        Path(__file__).resolve().parent / cascade_name,
    )

    for cascade_path in candidate_paths:
        if cascade_path.is_file():
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                return cascade

    searched_paths = ", ".join(str(path) for path in candidate_paths)
    raise RuntimeError(
        "Could not load the HAAR Cascade. Searched: "
        f"{searched_paths}. Install the OpenCV Haar cascade data package."
    )


def detect_with_haar(frame, face_cascade):
    """Return a frame with HAAR face detections drawn on it."""
    output = frame.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for x, y, width, height in faces:
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2,
        )

    cv2.putText(
        output,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )
    return output


def detect_with_mediapipe(frame, face_detection):
    """Return a frame with MediaPipe face detections drawn on it."""
    output = frame.copy()
    rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_detection.detect_for_video(
        mp_image,
        int(time.monotonic() * 1000),
    )
    height, width = output.shape[:2]

    for landmarks in results.face_landmarks:
        points = [
            (
                max(0, min(width - 1, int(landmark.x * width))),
                max(0, min(height - 1, int(landmark.y * height))),
            )
            for landmark in landmarks
        ]
        x_values, y_values = zip(*points)
        cv2.rectangle(
            output,
            (min(x_values), min(y_values)),
            (max(x_values), max(y_values)),
            (0, 255, 0),
            2,
        )

    cv2.putText(
        output,
        f"Faces: {len(results.face_landmarks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect faces with HAAR Cascade and/or MediaPipe."
    )
    parser.add_argument(
        "--method",
        choices=("haar", "mediapipe", "both"),
        default="both",
        help="Detection method to display (default: both).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    face_cascade = None
    if args.method in ("haar", "both"):
        face_cascade = load_haar_cascade()
    if args.method in ("mediapipe", "both") and not MEDIAPIPE_MODEL.is_file():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {MEDIAPIPE_MODEL}. "
            "Place face_landmarker.task in the project directory."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {args.camera}.")

    try:
        if args.method in ("mediapipe", "both"):
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(MEDIAPIPE_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            face_detection = vision.FaceLandmarker.create_from_options(options)
        else:
            face_detection = None

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                if args.method in ("haar", "both"):
                    haar_frame = detect_with_haar(frame, face_cascade)
                    cv2.imshow(WINDOW_HAAR, haar_frame)
                if args.method in ("mediapipe", "both"):
                    mediapipe_frame = detect_with_mediapipe(frame, face_detection)
                    cv2.imshow(WINDOW_MEDIAPIPE, mediapipe_frame)

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        finally:
            if face_detection is not None:
                face_detection.close()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
