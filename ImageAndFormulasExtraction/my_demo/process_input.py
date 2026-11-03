from pathlib import Path

import fitz


def pdf_to_pngs(pdf_path, output_path):
    """Render each PDF page into output_path/<pdf name>/page_<number>.png."""
    pdf_path = Path(pdf_path)
    page_dir = Path(output_path) / pdf_path.stem

    with fitz.open(pdf_path) as document:
        page_dir.mkdir(parents=True, exist_ok=True)
        for number, page in enumerate(document, start=1):
            page.get_pixmap(dpi=150).save(page_dir / f"page_{number:03}.png")

    return page_dir


if __name__ == "__main__":
    INPUT_PATH = "ImageAndFormulasExtraction/input/raw_input/MunichRe-Half-Year-Financial-Report-2026-en.pdf"
    OUTPUT_FOLDER = "ImageAndFormulasExtraction/input/processed_input"
    pdf_to_pngs(INPUT_PATH, OUTPUT_FOLDER)