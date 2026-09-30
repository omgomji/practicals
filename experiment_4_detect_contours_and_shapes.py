import os
import sys

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = "shapes.png"
OUTPUT_IMAGE_PATH = "detected_shapes_output.png"


def generate_sample_shapes(path: str) -> None:
    """Create a sample image containing separate geometric shapes."""

    width, height = 800, 600

    # White background
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # Triangle
    triangle = np.array(
        [
            [80, 450],
            [180, 250],
            [280, 450],
        ],
        dtype=np.int32,
    )

    # Square
    square = np.array(
        [
            [350, 100],
            [500, 100],
            [500, 250],
            [350, 250],
        ],
        dtype=np.int32,
    )

    # Rectangle
    rectangle = np.array(
        [
            [330, 350],
            [530, 350],
            [530, 480],
            [330, 480],
        ],
        dtype=np.int32,
    )

    # Pentagon
    pentagon = np.array(
        [
            [570, 450],
            [590, 300],
            [680, 240],
            [770, 300],
            [790, 450],
        ],
        dtype=np.int32,
    )

    # Circle
    circle_center = (180, 120)
    circle_radius = 70

    # Draw shapes in black
    cv2.fillPoly(image, [triangle], (0, 0, 0))
    cv2.fillPoly(image, [square], (0, 0, 0))
    cv2.fillPoly(image, [rectangle], (0, 0, 0))
    cv2.fillPoly(image, [pentagon], (0, 0, 0))
    cv2.circle(image, circle_center, circle_radius, (0, 0, 0), -1)

    cv2.imwrite(path, image)

    print(f"Sample test image created at: {path}")


def classify_shape(vertices: int, contour: np.ndarray) -> str:
    """
    Classify a contour based on the number of vertices.

    For a quadrilateral, distinguish square and rectangle
    using the aspect ratio of its bounding rectangle.
    """

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / float(h)

        if 0.90 <= aspect_ratio <= 1.10:
            return "Square"
        else:
            return "Rectangle"

    elif vertices == 5:
        return "Pentagon"

    elif vertices > 5:
        return "Circle"

    else:
        return "Unknown"


def detect_shapes(image_path: str) -> None:
    """Detect and classify shapes present in an image."""

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    output = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    # Black shapes become white foreground objects.
    _, thresh = cv2.threshold(
        blur,
        60,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Total contours detected: {len(contours)}")

    shape_count = 0

    for contour in contours:

        # Ignore very small contours/noise
        area = cv2.contourArea(contour)

        if area < 500:
            continue

        # Calculate contour perimeter
        perimeter = cv2.arcLength(contour, True)

        # Approximate contour
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        vertices = len(approx)

        # Classify shape
        shape_name = classify_shape(
            vertices,
            contour
        )

        shape_count += 1

        # Draw approximated contour
        cv2.drawContours(
            output,
            [approx],
            -1,
            (0, 255, 0),
            2
        )

        # Find a suitable position for the label
        x, y, w, h = cv2.boundingRect(approx)

        cv2.putText(
            output,
            shape_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    print(f"Total shapes detected: {shape_count}")

    # Save output
    cv2.imwrite(
        OUTPUT_IMAGE_PATH,
        output
    )

    print(
        f"Detected shapes output saved to: "
        f"{OUTPUT_IMAGE_PATH}"
    )

    # Display if GUI is available
    if os.environ.get("DISPLAY") or os.name == "nt":

        cv2.imshow(
            "Detected Shapes",
            output
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(
            "No display detected; "
            "output image was saved to disk instead."
        )


def main() -> None:

    # Use command-line image if provided,
    # otherwise use shapes.png
    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_IMAGE_PATH
    )

    # Generate sample image if it doesn't exist
    if not os.path.exists(image_path):
        generate_sample_shapes(image_path)

    try:
        detect_shapes(image_path)

    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()