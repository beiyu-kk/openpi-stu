"""The release's coordinate parsing, preserved for output parity."""

import re


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
    boxes = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(value) for value in match.groups()]
        boxes.append(
            {
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            }
        )
    return boxes


def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
    points = []
    for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = int(match.group(1)), int(match.group(2))
        points.append({"x": x / 1000 * image_width, "y": y / 1000 * image_height})
    return points
