import cv2


WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


def show_image(window_name: str, image) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow(window_name, image)


def main() -> None:
    image_path = "/home/omji/Downloads/1_tQYAE9PU2FIrZV5rKwT7AQ.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image file: {image_path}")
        return

    show_image("Original Image", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray)

    resized = cv2.resize(img, (400, 300))
    show_image("Resized Image", resized)

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    show_image("Rotated Image", rotated)

    cv2.imwrite("gray_image.jpg", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()