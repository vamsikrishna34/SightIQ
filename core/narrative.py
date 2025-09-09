from typing import Dict

TEMPLATE = (
    "Detected {obj_count} objects (avg confidence {avg_conf:.2f}). "
    "Scene density {density_per_mpx:.2f} per MPx. "
    "{risk_sentence}"
)

def _risk_level(obj_count: int, avg_conf: float, density: float) -> str:
    score = 0.4 * min(1.0, obj_count / 10.0) + 0.4 * avg_conf + 0.2 * min(1.0, density / 50.0)
    if score >= 0.7:
        return "High-risk: crowding or proximity concerns detected."
    if score >= 0.4:
        return "Moderate concern: monitor spacing and object size."
    return "Low risk: no significant anomalies detected."

def generate_narrative(features: Dict[str, float]) -> str:
    obj_count = int(features.get("obj_count", 0))
    avg_conf = float(features.get("avg_conf", 0.0))
    density = float(features.get("density_per_mpx", 0.0))
    risk_sentence = _risk_level(obj_count, avg_conf, density)
    return TEMPLATE.format(obj_count=obj_count, avg_conf=avg_conf, density_per_mpx=density, risk_sentence=risk_sentence)