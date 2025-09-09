import streamlit as st
import numpy as np
from PIL import Image
from utils import to_rgb, set_session, safe_get_session
from core.detection import Detector

st.title("Detection")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    img = Image.open(uploaded)
    img = to_rgb(img)
    set_session(st, "image", img)
    st.image(img, caption="Input image", use_container_width=True)

img = safe_get_session(st, "image", None)
if img is None:
    st.info("Upload an image to begin.")
    st.stop()

col1, col2 = st.columns([1,1])
conf = col1.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
iou  = col2.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05)

if st.button("Run YOLOv8 Detection"):
    detector = Detector(conf=conf, iou=iou)
    res = detector.run(np.array(img))
    st.session_state["det_result"] = {
        "annotated": res.annotated,
        "boxes": res.boxes,
        "scores": res.scores,
        "classes": res.classes,
        "class_names": res.class_names,
        "image_size": res.image_size
    }

det = safe_get_session(st, "det_result", None)
if det is not None:
    st.image(det["annotated"], caption="Detections", use_container_width=True)
    st.json({
        "num_detections": int(len(det["scores"])),
        "classes": {int(k): v for k, v in det["class_names"].items()}
    })

if st.button("Clear session"):
    for k in ("image", "det_result", "features", "reasoning"):
        if k in st.session_state:
            del st.session_state[k]
    st.success("Session cleared.")