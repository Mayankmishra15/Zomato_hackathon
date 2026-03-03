"""
SHAP Explainability Dashboard.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Explainability | CSAO Rail", page_icon="🔍", layout="wide")
st.title("🔍 Explainability Dashboard")

SHAP_PATH = "artifacts/shap/shap_values.pkl"

if not os.path.exists(SHAP_PATH):
    st.warning("SHAP values not found. Run: `python train.py --model lgbm --shap`")
    st.info("Using sample feature importance for demo.")
    # Demo data
    feature_names = ["item_relevance_score", "cart_value_at_show", "candidate_popularity", "user_engagement_score", "meal_complete_score_y", "cuisine_match", "candidate_avg_rating", "price_vs_user_avg", "has_drink_in_cart", "missing_dessert"]
    importance = [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04]
    shap_data = {"feature": feature_names, "importance": importance}
else:
    import joblib
    data = joblib.load(SHAP_PATH)
    shap_values = data.get("shap_values")
    X_test = data.get("X_test")
    feature_names = data.get("feature_names", list(X_test.columns) if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_test.shape[1])])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:20]
    shap_data = {"feature": [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx], "importance": mean_abs[idx].tolist()}

st.subheader("Global Feature Importance")
df = pd.DataFrame(shap_data)
fig = px.bar(df, x="importance", y="feature", orientation="h", title="Top 20 Features by |SHAP|")
fig.update_layout(height=500, template="plotly_white", yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Group Analysis")
groups = {"User": ["user_engagement", "rfm", "user_avg", "tenure", "price_sensitivity"], "Cart": ["cart_", "meal_complete", "has_main", "has_side", "has_drink"], "Item": ["candidate_", "item_", "cuisine_match", "veg_match"], "Temporal": ["hour", "day", "meal_slot", "weekend"]}
group_sums = {}
for g, keywords in groups.items():
    s = sum(v for f, v in zip(shap_data["feature"], shap_data["importance"]) if any(k in str(f) for k in keywords))
    group_sums[g] = s
fig = px.pie(values=list(group_sums.values()), names=list(group_sums.keys()), title="Contribution by Feature Group", color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Individual Prediction Explainer")
st.info("Use sliders to construct a candidate and view waterfall. (Placeholder — full implementation requires SHAP waterfall)")
