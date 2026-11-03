from math import ceil, floor
from pathlib import Path

import pymupdf
from PIL import Image


def extract_regions(page, detections, image_size, labels={"text"}):
    image_width, image_height = image_size

    pixel_to_rotated_page = ~page.rect.torect(
        pymupdf.Rect(0, 0, image_width, image_height)
    )

    words = page.get_text("words")
    output = []

    for label, xyxy in detections:
        if label not in labels:
            continue

        rect = (
            pymupdf.Rect(xyxy) * pixel_to_rotated_page
        ) * page.derotation_matrix

        lines = {}

        for word in words:
            x0, y0, x1, y1 = word[:4]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
                lines.setdefault((word[5], word[6]), []).append(word)

        ordered_lines = sorted(
            lines.values(),
            key=lambda line: (
                min(w[1] for w in line),
                min(w[0] for w in line)
            )
        )

        text = "\n".join(
            " ".join(w[4] for w in sorted(line, key=lambda w: w[7]))
            for line in ordered_lines
        )

        output.append({
            "label": label,
            "bbox": tuple(rect),
            "text": text
        })

    return output


def save_region_crops(result, png_path, label, num_pix_extra=0, filename=None):
    """Save all table or image detections from a PNG; return their paths."""
    if label not in {"table", "image"}:
        raise ValueError("label must be 'table' or 'image'")

    boxes = [box for box in result.boxes if result.names[int(box.cls.item())] == label]
    if filename is not None and len(boxes) != 1:
        raise ValueError("filename requires exactly one matching box")

    png_path = Path(png_path)
    paths = []
    with Image.open(png_path) as image:
        for number, box in enumerate(boxes, start=1):
            path = Path(filename) if filename is not None else png_path.with_name(
                f"{png_path.stem}_{label}_{number:03}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            crop_region(image, box.xyxy[0].tolist(), num_pix_extra).save(path)
            paths.append(path)

    return paths


def crop_region(image, xyxy, num_pix_extra=0):
    """Crop a PIL image around a detected box, clipped to the image bounds."""
    x0, y0, x1, y1 = xyxy
    bounds = (
        max(0, floor(x0 - num_pix_extra)),
        max(0, floor(y0 - num_pix_extra)),
        min(image.width, ceil(x1 + num_pix_extra)),
        min(image.height, ceil(y1 + num_pix_extra)),
    )
    return image.crop(bounds)
