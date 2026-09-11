"""The single-image script's rendering, with no inference side effects."""

from PIL import Image, ImageDraw


def draw_predictions(
    image: Image.Image, boxes: list[dict], points: list[dict]
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line_width = max(2, min(result.size) // 300)
    for index, box in enumerate(boxes, start=1):
        xy = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rectangle(xy, outline="red", width=line_width)
        draw.text(
            (box["x1"] + line_width, box["y1"] + line_width), str(index), fill="red"
        )
    radius = max(5, min(result.size) // 100)
    for point in points:
        x, y = point["x"], point["y"]
        draw.line((x - radius, y, x + radius, y), fill="lime", width=line_width)
        draw.line((x, y - radius, x, y + radius), fill="lime", width=line_width)
    return result
