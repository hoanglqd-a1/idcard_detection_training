"""Detect cards with YOLO OBB and classify them by face-masked template matching.

Images loaded by this module use RGB channel order. Labels are template indices.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from utils.processing import (
    apply_mask,
    auto_canny,
    convert_rec2corners,
    crop_image,
    document_detect,
    draw_lines,
    expand_corners,
    extract_card,
    find_intersections,
    get_lines,
    load_image,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CARD_SIZE = (640, 320)  # OpenCV sizes are (width, height).
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _first_corners(results):
    """Return the first detected OBB's corners, or None when no card is found."""
    if not results or results[0].obb is None or len(results[0].obb) == 0:
        return None
    return results[0].obb.xyxyxyxy[0].cpu().numpy()


def predict_corners(model: 'YOLO', image_dir):
    """Predict corners using the original CPU/320px inference settings."""
    return _first_corners(model.predict(image_dir, imgsz=320, conf=0.5, device='cpu'))


def detect_boundingbox(image, model: 'YOLO'):
    """Predict corners using the model's default inference settings."""
    return _first_corners(model(image))


def remove_face(model: 'YOLO', card: np.ndarray):
    """Black out face boxes, expanded by 30%, before template matching."""
    results = model.predict(card, imgsz=640, conf=0.5, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return card

    mask = np.full(card.shape[:2], 255, dtype=np.uint8)
    for box in results[0].boxes.xyxy:
        corners = convert_rec2corners(box.cpu().numpy())
        corners = expand_corners(card.shape, corners, expand_rate=0.3)
        top_left = tuple(corners[0].astype(int))
        bottom_right = tuple(corners[2].astype(int))
        cv2.rectangle(mask, top_left, bottom_right, 0, -1)
    return apply_mask(card, mask)


def load_templates(template_dir, face_model, card_size=CARD_SIZE):
    """Load and mask template images; list indices are the classification labels.

    Preserve directory listing order to retain the existing label mapping.
    """
    paths = (Path(template_dir) / name for name in os.listdir(template_dir))
    return [
        remove_face(face_model, load_image(path, card_size))
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def match(card, templates):
    """Return one normalized correlation score per template."""
    return np.array([
        cv2.matchTemplate(card, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        for template in templates
    ])


def classify(card, templates, threshold=0.7):
    """Return the best template index, or None if no score exceeds threshold."""
    scores = match(card, templates)
    if scores.size == 0:
        return None
    label = int(np.argmax(scores))
    return label if scores[label] > threshold else None


def detect_card(detect_model, image, templates, face_model, threshold=0.8):
    """Return (RGB card, template index), or (None, None) if cropping fails.

    A successfully cropped card with no template match returns (card, None).
    Model and configuration errors propagate to the caller.
    """
    corners = detect_boundingbox(image, detect_model)
    if corners is None:
        return None, None
    try:
        cropped = crop_image(image, corners)
    except ValueError:
        return None, None
    cropped = cv2.resize(cropped, CARD_SIZE)
    card = document_detect(cropped)
    if card is None:
        return None, None
    card = cv2.resize(card, CARD_SIZE)
    masked_card = remove_face(face_model, card)
    return card, classify(masked_card, templates, threshold)


def main():
    from ultralytics import YOLO

    detect_model = YOLO(str(ROOT / 'model' / 'yolov8s-detect.pt')).eval()
    face_model = YOLO(str(ROOT / 'model' / 'yolov8n-face.pt')).eval()
    templates = load_templates(ROOT / 'template_samples', face_model)
    image = load_image(ROOT / 'test_images' / 'image6.png', CARD_SIZE)
    card, label = detect_card(detect_model, image, templates, face_model)
    print('Detected card label:', label)
    if card is None:
        print('No card could be extracted.')
        return
    output_path = ROOT / 'detected_card.png'
    if not cv2.imwrite(str(output_path), cv2.cvtColor(card, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not save the detected card to {output_path}')


if __name__ == '__main__':
    main()
