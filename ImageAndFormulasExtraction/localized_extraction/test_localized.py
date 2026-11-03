import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from extract_regions import extract_regions


def run_document(pdf_path):
    pdf_path = Path(pdf_path)
    output_dir = Path(__file__).resolve().parents[1] / "output" / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO",
        filename="universal/DocLayout-YOLO-universal.pt",
    )
    model = YOLO(weights)

    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=150)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3
            )[:, :, ::-1].copy()
            result = model.predict(source=image, imgsz=1024, conf=0.15, verbose=False)[0]
            detections = [
                (model.names[int(box.cls.item())], box.xyxy[0].tolist())
                for box in result.boxes
            ]
            regions = extract_regions(page, detections, (pixmap.width, pixmap.height))
            (output_dir / f"page_{number:03}.json").write_text(
                json.dumps(regions, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="PDF to detect and extract text from")
    args = parser.parse_args()
    print("Page JSON files saved to", run_document(args.pdf_path))
