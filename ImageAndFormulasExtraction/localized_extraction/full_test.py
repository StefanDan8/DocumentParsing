import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from extract_regions import crop_region, extract_regions


IMAGE_LABELS = {"image", "image_caption", "image_footnote"}
TABLE_LABELS = {"table", "table_caption", "table_footnote"}


def process_page(page, model, page_dir, num_pix_extra):
    pixmap = page.get_pixmap(dpi=150)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    pixels = np.asarray(image)[:, :, ::-1].copy()
    result = model.predict(source=pixels, imgsz=1024, conf=0.15, verbose=False)[0]
    detections = [
        (result.names[int(box.cls.item())], box.xyxy[0].tolist())
        for box in result.boxes
    ]
    regions = extract_regions(page, detections, image.size, labels=result.names.values())

    page_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images", "tables"):
        (page_dir / name).mkdir(exist_ok=True)

    output = {"regions": [], "images": [], "tables": []}
    counts = {"image": 0, "table": 0}
    for region, (label, xyxy) in zip(regions, detections):
        if label in IMAGE_LABELS:
            group = "images"
        elif label in TABLE_LABELS:
            group = "tables"
        else:
            group = "regions"
        if label in counts:
            counts[label] += 1
            relative_path = Path(group) / f"{label}_{counts[label]:03}.png"
            crop_region(image, xyxy, num_pix_extra).save(page_dir / relative_path)
            region = {"label": label, "bbox": region["bbox"], "path": relative_path.as_posix()}
        output[group].append(region)

    for name, entries in output.items():
        entries.sort(key=lambda entry: (entry["bbox"][1], entry["bbox"][0]))
        (page_dir / f"{name}.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def run_document(pdf_path, output_dir=None, num_pix_extra=0):
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "output" / pdf_path.stem
    output_dir = Path(output_dir)

    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO",
        filename="universal/DocLayout-YOLO-universal.pt",
    )
    model = YOLO(weights)

    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            process_page(page, model, output_dir / f"page_{number:03}", num_pix_extra)
            print(f"Processed page {number}/{len(document)}", flush=True)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="PDF to process")
    parser.add_argument("--output_dir", help="Folder for per-page results")
    parser.add_argument("--num_pix_extra", type=int, default=0)
    args = parser.parse_args()
    print("Results saved to", run_document(args.pdf_path, args.output_dir, args.num_pix_extra))
