import cv2


def main() -> None:
    # Update this path to the video file you want to analyze.
    video_path = "/home/omji/Downloads/Shutter Island 2010 1080p BluRay x264 [Hindi NF DDP 5.1 - Eng DTS]-MSubs~(Tom) HDChina.mkv"

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frame Rate (FPS): {fps}")

    cv2.namedWindow("Video with FPS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video with FPS", 1280, 720)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        display_frame = cv2.resize(frame, (1280, 720))

        cv2.putText(
            display_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Video with FPS", display_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
