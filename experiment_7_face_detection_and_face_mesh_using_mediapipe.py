"""Real-time face detection and facial mesh visualization with MediaPipe Tasks."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("face_landmarker.task")


def point_from_landmark(landmark, width, height):
    """Convert a normalized MediaPipe landmark to an OpenCV pixel coordinate."""
    return (
        max(0, min(width - 1, int(landmark.x * width))),
        max(0, min(height - 1, int(landmark.y * height))),
    )


def draw_face_mesh(frame, landmarks):
    """Draw landmark points and a Delaunay mesh over one detected face."""
    height, width = frame.shape[:2]
    points = [point_from_landmark(landmark, width, height) for landmark in landmarks]

    # Face Landmarker supplies the landmarks; Delaunay triangles visualize their mesh.
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for point in points:
        subdiv.insert(point)

    for triangle in subdiv.getTriangleList():
        vertices = np.array(
            [
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[2]), int(triangle[3])),
                (int(triangle[4]), int(triangle[5])),
            ],
            dtype=np.int32,
        )
        if np.all((vertices[:, 0] >= 0) & (vertices[:, 0] < width)) and np.all(
            (vertices[:, 1] >= 0) & (vertices[:, 1] < height)
        ):
            cv2.polylines(frame, [vertices], True, (255, 255, 0), 1)

    for point in points:
        cv2.circle(frame, point, 1, (0, 0, 255), -1)

    # The extrema of all facial landmarks form the detected face bounding box.
    x_values, y_values = zip(*points)
    cv2.rectangle(
        frame,
        (min(x_values), min(y_values)),
        (max(x_values), max(y_values)),
        (0, 255, 0),
        2,
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Download the MediaPipe Face Landmarker model first."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect_for_video(
                mp_image, int(time.monotonic() * 1000)
            )

            for face_landmarks in results.face_landmarks:
                draw_face_mesh(frame, face_landmarks)

            cv2.imshow("Face Detection and Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
