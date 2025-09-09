import streamlit as st
from utils import safe_get_session
from core.features import extract_features

st.title("Features")

det = safe_get_session(st, "det_result", None)
if det is None:
    st.warning("Run Detection first.")
    st.stop()

features = extract_features(det["boxes"], det["scores"], det["classes"], det["image_size"])
st.session_state["features"] = features

st.subheader("Extracted features")
st.json(features)
st.success("Features ready. Proceed to Reasoning.")