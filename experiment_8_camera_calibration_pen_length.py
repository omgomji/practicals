"""Estimate a pen's real length from a photograph using a ruler reference."""

import argparse
import os
from math import hypot

import cv2


DEFAULT_IMAGE_PATH = "pen.jpg"
DEFAULT_KNOWN_LENGTH_MM = 50.0
DEFAULT_ACTUAL_PEN_LENGTH_MM = 145.0
DEFAULT_RULER_POINT_1 = (100, 200)
DEFAULT_RULER_POINT_2 = (350, 200)
DEFAULT_PEN_POINT_1 = (120, 300)
DEFAULT_PEN_POINT_2 = (620, 300)


def distance_between(point_1: tuple[int, int], point_2: tuple[int, int]) -> float:
    """Return the Euclidean distance between two image points."""

    return hypot(point_2[0] - point_1[0], point_2[1] - point_1[1])


def estimate_pen_length(
    ruler_point_1: tuple[int, int],
    ruler_point_2: tuple[int, int],
    known_ruler_length_mm: float,
    pen_point_1: tuple[int, int],
    pen_point_2: tuple[int, int],
) -> tuple[float, float, float]:
    """Return pixel-per-mm, pen pixel length, and estimated pen length."""

    if known_ruler_length_mm <= 0:
        raise ValueError("The known ruler length must be greater than zero.")

    ruler_pixels = distance_between(ruler_point_1, ruler_point_2)
    if ruler_pixels == 0:
        raise ValueError("The ruler calibration points must be different.")

    pen_pixels = distance_between(pen_point_1, pen_point_2)
    pixels_per_mm = ruler_pixels / known_ruler_length_mm
    pen_length_mm = pen_pixels / pixels_per_mm

    return pixels_per_mm, pen_pixels, pen_length_mm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate pen length using ruler-based camera calibration."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        help="Path to the photograph (default: pen.jpg).",
    )
    parser.add_argument(
        "--known-length-mm",
        type=float,
        default=DEFAULT_KNOWN_LENGTH_MM,
        help="Known distance represented by the ruler points (default: 50).",
    )
    parser.add_argument(
        "--actual-length-mm",
        type=float,
        default=DEFAULT_ACTUAL_PEN_LENGTH_MM,
        help="Measured pen length used as ground truth (default: 145).",
    )
    parser.add_argument(
        "--ruler-points",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_RULER_POINT_1 + DEFAULT_RULER_POINT_2,
        help="Ruler calibration points: X1 Y1 X2 Y2.",
    )
    parser.add_argument(
        "--pen-points",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_PEN_POINT_1 + DEFAULT_PEN_POINT_2,
        help="Pen endpoints: X1 Y1 X2 Y2.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the input photograph after calculating the result.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.actual_length_mm <= 0:
        raise ValueError("The actual pen length must be greater than zero.")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    ruler_point_1 = (args.ruler_points[0], args.ruler_points[1])
    ruler_point_2 = (args.ruler_points[2], args.ruler_points[3])
    pen_point_1 = (args.pen_points[0], args.pen_points[1])
    pen_point_2 = (args.pen_points[2], args.pen_points[3])

    pixels_per_mm, pen_pixels, estimated_length_mm = estimate_pen_length(
        ruler_point_1,
        ruler_point_2,
        args.known_length_mm,
        pen_point_1,
        pen_point_2,
    )
    error_mm = abs(args.actual_length_mm - estimated_length_mm)
    percentage_error = error_mm / args.actual_length_mm * 100

    print(f"Pixel-per-mm scale: {pixels_per_mm:.4f}")
    print(f"Pen length in pixels: {pen_pixels:.2f}")
    print(f"Estimated Pen Length = {estimated_length_mm:.2f} mm")
    print(f"Actual Pen Length = {args.actual_length_mm:.2f} mm")
    print(f"Measurement Error = {error_mm:.2f} mm")
    print(f"Percentage Error = {percentage_error:.2f}%")

    if args.show:
        if not (os.environ.get("DISPLAY") or os.name == "nt"):
            print("No display detected; use the printed result instead.")
            return

        cv2.imshow("Pen Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
