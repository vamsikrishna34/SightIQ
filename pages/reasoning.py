import streamlit as st
import numpy as np
from utils import safe_get_session
from core.reasoning import Reasoner

st.title("Reasoning")

features = safe_get_session(st, "features", None)
if features is None:
    st.warning("Run Features first.")
    st.stop()

reasoner = Reasoner()
out = reasoner.predict(features)
st.session_state["reasoning"] = {
    "y_pred": out.y_pred,
    "shap_values": out.shap_values.tolist(),
    "expected_value": float(out.expected_value),
    "feature_names": out.feature_names,
    "feature_vector": out.feature_vector.tolist()
}

st.metric("Predicted risk score", f"{out.y_pred:.3f}")
st.caption("0=low risk, 1=high risk (interpretation depends on your training data).")
st.success("Reasoning complete. Proceed to Explainability.")