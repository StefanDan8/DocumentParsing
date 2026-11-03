
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from viz import visualize_result

weights = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO",
    filename="universal/DocLayout-YOLO-universal.pt"
)

model = YOLO(weights)

print(model.names)

results = model.predict(
    source="ImageAndFormulasExtraction/input/processed_input/MunichRe-Half-Year-Financial-Report-2026-en/page_58.png",
    imgsz=1024,
    conf=0.15,
    device="cuda:0"
)

for box in results[0].boxes:
    label = model.names[int(box.cls.item())]
    bbox = box.xyxy[0].tolist()
    score = float(box.conf.item())

    print(label, bbox, score)

output_path = Path("ImageAndFormulasExtraction/output") / f"{Path(results[0].path).stem}_viz.png"
print("Visualization saved to", visualize_result(results[0], output_path))
