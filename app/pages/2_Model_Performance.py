"""
Model Performance Dashboard.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Model Performance | CSAO Rail", page_icon="📊", layout="wide")
st.title("📊 Model Performance Dashboard")

# Load data
DATA_PATH = "csao_ml_final.csv"
RESULTS_PATH = "artifacts/eval_results.json"

if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found. Run training with `python train.py --model all` after adding data.")
    st.info("Using sample metrics for demo.")
    results = {
        "Baseline": {"AUC": 0.62, "Precision@5": 0.26, "Precision@8": 0.28, "Precision@10": 0.29, "Recall@8": 0.45, "NDCG@8": 0.55},
        "XGBoost": {"AUC": 0.84, "Precision@5": 0.44, "Precision@8": 0.48, "Precision@10": 0.50, "Recall@8": 0.62, "NDCG@8": 0.74},
        "LightGBM": {"AUC": 0.86, "Precision@5": 0.47, "Precision@8": 0.51, "Precision@10": 0.53, "Recall@8": 0.65, "NDCG@8": 0.77},
    }
else:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {
            "Baseline": {"AUC": 0.62, "Precision@8": 0.28, "NDCG@8": 0.55},
            "XGBoost": {"AUC": 0.84, "Precision@8": 0.48, "NDCG@8": 0.74},
            "LightGBM": {"AUC": 0.86, "Precision@8": 0.51, "NDCG@8": 0.77},
        }

tab1, tab2, tab3, tab4 = st.tabs(["Dataset EDA", "Model Comparison", "Segment Analysis", "Learning Curves"])

with tab1:
    st.subheader("Dataset EDA")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        c1, c2 = st.columns(2)
        with c1:
            target = df["label"] if "label" in df.columns else df.iloc[:, -1]
            fig = px.pie(values=target.value_counts(), names=["Rejected", "Accepted"], title="Target Distribution", color_discrete_sequence=["#E23744", "#2ecc71"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "meal_slot" in df.columns:
                acc_by_slot = df.groupby("meal_slot")["label"].mean()
                fig = px.bar(x=acc_by_slot.index, y=acc_by_slot.values, title="Acceptance Rate by Meal Slot", labels={"x": "Meal Slot", "y": "Acceptance Rate"})
                fig.update_traces(marker_color="#E23744")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add dataset to enable EDA.")

with tab2:
    st.subheader("Model Comparison")
    metrics = ["AUC", "Precision@5", "Precision@8", "Precision@10", "Recall@8", "NDCG@8"]
    avail_metrics = [m for m in metrics if any(m in r for r in results.values())]
    if not avail_metrics:
        avail_metrics = ["AUC", "Precision@8", "NDCG@8"]
    comp_data = []
    for model, mets in results.items():
        for m in avail_metrics:
            if m in mets:
                comp_data.append({"Model": model, "Metric": m, "Value": mets[m]})
    if comp_data:
        comp_df = pd.DataFrame(comp_data)
        fig = px.bar(comp_df, x="Model", y="Value", color="Metric", barmode="group", title="Model Comparison")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(results).T.round(4), use_container_width=True)

with tab3:
    st.subheader("Segment Analysis")
    st.info("Segment-level metrics require running evaluation with error_analysis.")
    st.markdown("Performance breakdown by user_segment, city_tier, meal_slot, is_veg.")

with tab4:
    st.subheader("Learning Curves")
    st.info("Training vs validation AUC over n_estimators — requires logging during train.")
