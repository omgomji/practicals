import cv2
import numpy as np


WINDOW_NAME = "Input and Warped Output"
INPUT_IMAGE_PATH = "/home/omji/Downloads/1_tQYAE9PU2FIrZV5rKwT7AQ.jpg"


def main() -> None:
    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        print(f"Error: Could not read image file: {INPUT_IMAGE_PATH}")
        return

    height, width = img.shape[:2]

    src_points = np.float32(
        [
            [0.25 * width, 0.25 * height],
            [0.75 * width, 0.25 * height],
            [0.25 * width, 0.75 * height],
            [0.75 * width, 0.75 * height],
        ]
    )

    dst_width = int(0.5 * width)
    dst_height = int(0.5 * height)

    dst_points = np.float32(
        [
            [0, 0],
            [dst_width - 1, 0],
            [0, dst_height - 1],
            [dst_width - 1, dst_height - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, (dst_width, dst_height))

    warped_resized = cv2.resize(warped, (width, height))
    stacked = np.hstack((img, warped_resized))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, stacked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
