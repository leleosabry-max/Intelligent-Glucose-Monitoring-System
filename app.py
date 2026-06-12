import streamlit as st
import backend
import numpy as np

st.set_page_config(page_title="Glucose AI System", layout="wide")

st.title("🧠 Intelligent Glucose Monitoring System")

# ---------------- INFO ----------------
st.info("System runs on backend import. Click refresh to re-run pipeline.")

if st.button("🔄 Refresh Dashboard", key="refresh"):
    import importlib
    importlib.reload(backend)
    st.rerun()

# ---------------- VALIDATION ----------------
required = ["forecast_values", "total_score", "decision", "condition", "action", "rag_output"]

missing = [x for x in required if not hasattr(backend, x)]

if missing:
    st.error(f"Missing backend outputs: {missing}")
    st.stop()

# ===================== OVERVIEW =====================
st.subheader("📊 System Output")

col1, col2, col3 = st.columns(3)

col1.metric("Condition", backend.condition)
col2.metric("Decision", backend.decision)
col3.metric("Score", f"{float(backend.total_score[-1]):.5f}")

st.divider()

# ===================== FORECAST =====================
st.subheader("📈 Glucose Forecast")

st.line_chart(backend.forecast_values)

st.divider()

# ===================== RISK =====================
st.subheader("⚠️ Risk Indicators")

volatility = np.std(backend.forecast_values)
trend = abs(backend.forecast_values[-1] - backend.forecast_values[0])

c1, c2 = st.columns(2)

c1.metric("Volatility", f"{volatility:.5f}")
c2.metric("Trend", f"{trend:.5f}")

st.divider()

# ===================== ACTION =====================
st.subheader("⚙️ Action Router")

st.write("Action:", backend.action["action"])
st.write("Message:", backend.action["message"])
st.write("Priority:", backend.action["priority"])

st.divider()

# ===================== RAG =====================
st.subheader("📚 Medical Knowledge (RAG)")

rag = backend.rag_output

st.write("Range:", rag["range"])
st.write("Meaning:", rag["meaning"])
st.write("Action:", rag["action"])
st.write("Severity:", rag["severity"])

st.divider()

# ===================== SIMILAR CASES =====================
st.subheader("🔍 Similar Cases")

for c in backend.rag_output.get("similar_cases", []):
    st.write(c)

st.divider()

# ===================== SUMMARY =====================
st.subheader("🧠 System Summary")

st.success("""
Pipeline includes:
- SARIMA forecasting
- SVM classification
- SHAP-based importance scoring
- Risk equation (score-based decision)
- RAG medical reasoning
- Action routing engine
""")