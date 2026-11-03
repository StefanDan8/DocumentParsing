from pathlib import Path

import cv2


def visualize_result(result, output_path):
    """Save a YOLO result with labeled boxes drawn on its source image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result.plot(line_width=2))
    return output_path
