import cv2
import mediapipe as mp


MODEL_PATH = "hand_landmarker.task"


def main():
    # --------------------------------------------------
    # Initialize MediaPipe Hand Landmarker
    # --------------------------------------------------
    base_options = mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )

    # --------------------------------------------------
    # Start webcam
    # --------------------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        detector.close()
        return

    timestamp_ms = 0

    while True:

        # Read frame
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Mirror the webcam feed
        frame = cv2.flip(frame, 1)

        # --------------------------------------------------
        # Convert BGR → RGB
        # --------------------------------------------------
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase in VIDEO mode
        timestamp_ms += 1

        # --------------------------------------------------
        # Process frame
        # --------------------------------------------------
        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # --------------------------------------------------
        # Draw landmarks
        # --------------------------------------------------
        height, width, _ = frame.shape

        connections = [
            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            # Index finger
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            # Middle finger
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            # Ring finger
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            # Little finger
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            # Palm
            (0, 17),
        ]

        for hand_landmarks in results.hand_landmarks:

            # Draw landmark points
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for start, end in connections:

                x1 = int(
                    hand_landmarks[start].x * width
                )
                y1 = int(
                    hand_landmarks[start].y * height
                )

                x2 = int(
                    hand_landmarks[end].x * width
                )
                y2 = int(
                    hand_landmarks[end].y * height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        # --------------------------------------------------
        # Display output
        # --------------------------------------------------
        cv2.imshow(
            "Hand Tracking",
            frame
        )

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # Release resources
    # --------------------------------------------------
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()