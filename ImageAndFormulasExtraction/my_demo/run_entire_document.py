
import argparse
from pathlib import Path

import fitz
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from viz import visualize_result


def frames_to_pdf(folder_path, pdf_path):
    """Bundle PNG frames into a PDF in lexicographic filename order."""
    with fitz.open() as document:
        for frame in sorted(Path(folder_path).glob("*.png")):
            with Image.open(frame) as image:
                page = document.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, filename=str(frame))
        document.save(pdf_path, deflate=True)


def run_document(folder_path):
    folder_path = Path(folder_path)
    frames = sorted(folder_path.glob("*.png"))
    if not frames:
        raise ValueError(f"No PNG pages in {folder_path}")

    output_dir = Path(__file__).resolve().parents[1] / "output" / folder_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO",
        filename="universal/DocLayout-YOLO-universal.pt",
    )
    model = YOLO(weights)

    for frame in frames:
        result = model.predict(source=str(frame), imgsz=1024, conf=0.15, device="cuda:0", verbose=False)[0]
        visualize_result(result, output_dir / frame.name)

    pdf_path = output_dir / f"{folder_path.name}.pdf"
    frames_to_pdf(output_dir, pdf_path)
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", help="Folder containing page PNGs")
    args = parser.parse_args()
    print("Annotated PDF saved to", run_document(args.folder_path))
