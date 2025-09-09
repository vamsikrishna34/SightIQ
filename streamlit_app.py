import streamlit as st
from utils import ensure_dirs

st.set_page_config(page_title="SightIQ", page_icon="🔍", layout="wide")
ensure_dirs()

st.title("SightIQ")
st.subheader("Visual Intelligence & Explainable Decision Engine")

st.markdown("""
Use the sidebar to navigate the pipeline:
- Detection: upload image and run YOLOv8
- Features: extract structured metrics
- Reasoning: XGBoost (or heuristic fallback)
- Explainability: SHAP + detection heatmap
- Narrative: plain-English summary

Tip: Start at the Detection page.
""")

st.success("Environment ready. Go to the Detection page to begin.")