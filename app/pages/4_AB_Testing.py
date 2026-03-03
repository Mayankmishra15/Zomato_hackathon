"""
A/B Test Simulation Dashboard.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="A/B Testing | CSAO Rail", page_icon="🧪", layout="wide")
st.title("🧪 A/B Test Simulation")

RESULTS_PATH = "artifacts/eval_results.json"

# Load results
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        res = json.load(f)
    baseline_p8 = res.get("Baseline", {}).get("Precision@8", 0.28)
    model_p8 = res.get("LightGBM", res.get("XGBoost", {})).get("Precision@8", 0.51)
else:
    baseline_p8 = 0.28
    model_p8 = 0.51

# Two-proportion z-test
split = st.slider("Traffic split (Control %)", 30, 70, 50) / 100
n = st.slider("Sample size per arm", 500, 5000, 1000)
n_control = int(n * split)
n_treatment = int(n * (1 - split))

# Simulated conversions
np.random.seed(42)
ctrl_conversions = np.random.binomial(n_control, baseline_p8, 1)[0]
trt_conversions = np.random.binomial(n_treatment, model_p8, 1)[0]
p_control = ctrl_conversions / n_control
p_treatment = trt_conversions / n_treatment

# Z-test
p_pooled = (ctrl_conversions + trt_conversions) / (n_control + n_treatment)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
z = (p_treatment - p_control) / se if se > 0 else 0
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
significant = p_value < 0.05

st.subheader("Experiment Setup")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Control (Baseline) Sample", n_control)
with c2:
    st.metric("Treatment (Model) Sample", n_treatment)
with c3:
    st.metric("Significance Level", "α = 0.05")

st.subheader("Results Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("Control Acceptance Rate", f"{p_control*100:.2f}%", f"{ctrl_conversions} conversions")
with col2:
    delta = (p_treatment - p_control) * 100
    st.metric("Treatment Acceptance Rate", f"{p_treatment*100:.2f}%", f"{delta:+.2f}pp")

st.subheader("Statistical Significance")
if significant:
    st.success(f"✅ Experiment is **significant** (p-value = {p_value:.4f} < 0.05)")
else:
    st.warning(f"⚠️ Experiment is **not significant** (p-value = {p_value:.4f} >= 0.05)")

fig = go.Figure()
fig.add_trace(go.Bar(name="Control", x=["Control"], y=[p_control*100], marker_color="#95a5a6"))
fig.add_trace(go.Bar(name="Treatment", x=["Treatment"], y=[p_treatment*100], marker_color="#E23744"))
fig.update_layout(title="Acceptance Rate by Arm", template="plotly_white", yaxis_title="Acceptance Rate (%)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Guardrail Monitoring")
st.info("Latency: p99 < 200ms | Cart abandonment: no increase > 2%")
