import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils import fig_to_image, np_to_pil

def shap_summary_plot(feature_names, shap_values: np.ndarray, feature_vector: np.ndarray) -> Image.Image:
    values = shap_values
    idx = np.argsort(np.abs(values))[::-1]
    names = [feature_names[i] for i in idx]
    vals = values[idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2b8a3e" if v < 0 else "#d9480f" for v in vals]
    ax.barh(names[:20][::-1], vals[:20][::-1], color=colors[:20][::-1])
    ax.set_xlabel("SHAP value (impact on output)")
    ax.set_title("Feature contributions")
    fig.tight_layout()
    return fig_to_image(fig)

def detection_heatmap(image_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray, sigma: int = 35) -> Image.Image:
    H, W, _ = image_rgb.shape
    heat = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        radius = max(5, int(0.5 * max(x2 - x1, y2 - y1)))
        cv2.circle(heat, (cx, cy), radius, float(s), thickness=-1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heatmap = (plt.cm.jet(heat)[..., :3] * 255).astype(np.uint8)
    overlay = (0.6 * image_rgb + 0.4 * heatmap).astype(np.uint8)
    return np_to_pil(overlay)