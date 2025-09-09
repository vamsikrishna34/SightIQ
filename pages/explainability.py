import streamlit as st
import numpy as np
from utils import safe_get_session
from core.explainability import shap_summary_plot, detection_heatmap

st.title("Explainability")

img = safe_get_session(st, "image", None)
det = safe_get_session(st, "det_result", None)
reason = safe_get_session(st, "reasoning", None)

if img is None or det is None or reason is None:
    st.warning("Complete Detection, Features, and Reasoning first.")
    st.stop()

st.subheader("Detection-focused heatmap")
overlay = detection_heatmap(np.array(img), det["boxes"], det["scores"])
st.image(overlay, use_container_width=True)

st.subheader("SHAP feature contributions")
shap_img = shap_summary_plot(
    reason["feature_names"],
    np.array(reason["shap_values"], dtype=float),
    np.array(reason["feature_vector"], dtype=float)
)
st.image(shap_img, use_container_width=True)

st.success("Explainability rendered. Proceed to Narrative.")