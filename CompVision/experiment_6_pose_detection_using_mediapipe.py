import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Download the model once and save it in this folder as pose_landmarker.task.
model_path = "pose_landmarker.task"

# Connections between MediaPipe's 33 pose landmarks.
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28),
    (28, 30), (28, 32),
]

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
)

cap = cv2.VideoCapture(0)

with vision.PoseLandmarker.create_from_options(options) as pose:
    while True:
        success, img = cap.read()
        if not success:
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = pose.detect_for_video(mp_img, int(time.monotonic() * 1000))

        height, width = img.shape[:2]
        for landmarks in results.pose_landmarks:
            for start, end in connections:
                p1 = (int(landmarks[start].x * width), int(landmarks[start].y * height))
                p2 = (int(landmarks[end].x * width), int(landmarks[end].y * height))
                cv2.line(img, p1, p2, (0, 255, 0), 2)

            for landmark in landmarks:
                point = (int(landmark.x * width), int(landmark.y * height))
                cv2.circle(img, point, 4, (0, 0, 255), cv2.FILLED)

        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
