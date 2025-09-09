from typing import Dict, Any
import numpy as np

def _bbox_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

def pairwise_min_dist(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return float("inf")
    dmin = float("inf")
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < dmin:
                dmin = d
    return float(dmin)

def extract_features(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, image_size) -> Dict[str, Any]:
    H, W = image_size
    area = H * W
    feats: Dict[str, Any] = {}

    feats["obj_count"] = int(len(boxes))
    feats["avg_conf"] = float(np.mean(scores)) if len(scores) else 0.0

    if len(boxes):
        wh = (boxes[:, 2:4] - boxes[:, 0:2]).clip(min=0)
        areas = wh[:, 0] * wh[:, 1]
        feats["mean_box_area_pct"] = float(np.mean(areas) / area)
        feats["max_box_area_pct"] = float(np.max(areas) / area)
        feats["min_box_area_pct"] = float(np.min(areas) / area)

        centers = np.stack([_bbox_center(b) for b in boxes], axis=0)
        feats["min_center_dist"] = float(pairwise_min_dist(centers))
        feats["mean_center_x"] = float(np.mean(centers[:, 0] / W))
        feats["mean_center_y"] = float(np.mean(centers[:, 1] / H))
    else:
        feats.update({
            "mean_box_area_pct": 0.0, "max_box_area_pct": 0.0, "min_box_area_pct": 0.0,
            "min_center_dist": float("inf"), "mean_center_x": 0.0, "mean_center_y": 0.0
        })

    # class distribution (top-5)
    hist = {}
    for cid in classes:
        hist[int(cid)] = hist.get(int(cid), 0) + 1
    for i, (cid, cnt) in enumerate(sorted(hist.items(), key=lambda kv: -kv[1])[:5]):
        feats[f"class_{cid}_count"] = int(cnt)

    feats["density_per_mpx"] = float(feats["obj_count"] / (area / 1_000_000.0) if area else 0.0)
    return feats