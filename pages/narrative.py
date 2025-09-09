import streamlit as st
from utils import safe_get_session
from core.narrative import generate_narrative

st.title("Narrative")

features = safe_get_session(st, "features", None)
reason = safe_get_session(st, "reasoning", None)

if features is None or reason is None:
    st.warning("Run through Features and Reasoning first.")
    st.stop()

score = reason["y_pred"]
risk_label = "HIGH" if score >= 0.7 else ("MEDIUM" if score >= 0.4 else "LOW")
text = generate_narrative(features)

st.markdown(f"**Risk level:** {risk_label} (score {score:.3f})")
st.write(text)

if st.button("Copy narrative"):
    st.code(text, language="markdown")