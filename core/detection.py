from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

from utils import MODELS_DIR, np_to_pil, logger

@dataclass
class DetectionResult:
    annotated: Image.Image
    boxes: np.ndarray           # (N,4) xyxy
    scores: np.ndarray          # (N,)
    classes: np.ndarray         # (N,)
    class_names: Dict[int, str] # id -> name
    image_size: Tuple[int, int] # (H, W)

class Detector:
    def __init__(self, weights: Path = MODELS_DIR / "yolov8n.pt", conf: float = 0.25, iou: float = 0.45):
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights}. Place yolov8n.pt in models/.")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        logger.info("YOLO model initialized.")

    def _annotate(self, img_bgr: np.ndarray, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, class_names) -> Image.Image:
        out = img_bgr.copy()
        for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
            color = (0, 180, 255)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{class_names[int(c)]}:{s:.2f}"
            cv2.putText(out, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return np_to_pil(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def run(self, image_rgb: np.ndarray) -> DetectionResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input must be RGB image (H, W, 3).")
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        class_names = results.names
        if results.boxes is None or len(results.boxes) == 0:
            H, W = img_bgr.shape[:2]
            return DetectionResult(
                annotated=np_to_pil(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
                boxes=np.zeros((0,4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                classes=np.zeros((0,), dtype=np.int32),
                class_names=class_names,
                image_size=(H, W)
            )
        b = results.boxes.xyxy.cpu().numpy()
        s = results.boxes.conf.cpu().numpy()
        c = results.boxes.cls.cpu().numpy().astype(int)
        annotated = self._annotate(img_bgr, b, c, s, class_names)
        H, W = img_bgr.shape[:2]
        return DetectionResult(annotated, b, s, c, class_names, (H, W))