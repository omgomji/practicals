import cv2


def main():
    video_path = "/home/omji/Downloads/[TopMovies.fans] - Swades 2004 BluRay 1080p Hindi DD 5.1 x264 MSubs.mkv"
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frame Rate (FPS): {fps}")

    cv2.namedWindow("Video display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video display", 1280, 720)

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

        cv2.imshow("Video display", display_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
