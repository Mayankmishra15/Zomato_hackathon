"""
Zomato Data Exploration — Accept rate heatmap and cart/value distributions.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Zomato Data Exploration | CSAO Rail",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Zomato Data Exploration")

DATA_PATH = "csao_ml_final.csv"

# Load data or use synthetic for demo
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    # Synthetic data for demo when CSV is missing
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "label": np.random.binomial(1, 0.25 + np.random.rand(n) * 0.2),
        "meal_slot": np.random.randint(0, 5, n),
        "item_category": np.random.randint(0, 6, n),
        "cart_value_at_show": np.clip(np.random.exponential(200, n) + 50, 0, 1000),
        "item_price": np.clip(np.random.exponential(80, n) + 20, 0, 400),
    })

df = load_data()
if not os.path.exists(DATA_PATH):
    st.info("Using synthetic data for demo. Add **csao_ml_final.csv** to the project root for real data.")

# --- View data & EDA option ---
st.subheader("Data & EDA")
view_mode = st.radio("View", ["Charts only", "Data table (100 rows)", "EDA summary"], horizontal=True)
if view_mode == "Data table (100 rows)":
    n_show = min(100, len(df))
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing first {n_show} rows.")
elif view_mode == "EDA summary":
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        if df.isna().any().any():
            st.markdown("**Missing counts**")
            st.dataframe(df.isna().sum().to_frame("missing").query("missing > 0"), use_container_width=True)
    with c2:
        st.markdown("**Column dtypes**")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
    st.markdown("**Numeric summary**")
    st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")
# --- Accept rate heatmap: Meal slot vs Item category ---
st.subheader("Accept rate by Meal slot and Item category")

has_label = "label" in df.columns
has_meal = "meal_slot" in df.columns
has_cat = "item_category" in df.columns

if has_label and has_meal and has_cat:
    pivot = df.groupby(["meal_slot", "item_category"])["label"].mean().reset_index()
    heatmap_df = pivot.pivot(index="meal_slot", columns="item_category", values="label")
    # Full grid: meal_slot 0..4, item_category 0..5 (match screenshot)
    full_idx = range(int(df["meal_slot"].max()) + 1)
    full_cols = range(int(df["item_category"].max()) + 1)
    heatmap_df = heatmap_df.reindex(index=full_idx, columns=full_cols).fillna(0)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale=[[0, "#d73027"], [0.35, "#fc8d59"], [0.5, "#fee08b"], [0.65, "#d9ef8b"], [1, "#1a9850"]],
            colorbar=dict(title="Accept rate"),
            hovertemplate="Meal slot: %{y}<br>Item category: %{x}<br>Accept rate: %{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_layout(
        xaxis_title="Item category",
        yaxis_title="Meal slot",
        template="plotly_dark",
        margin=dict(l=60, r=80, t=40, b=50),
        height=400,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.warning("Dataset must contain 'label', 'meal_slot', and 'item_category' for the heatmap.")

# --- Section 4: Cart value & price distributions ---
st.markdown("---")
st.markdown(
    '<span style="display:inline-flex;align-items:center;background:#1E88E5;color:white;'
    'font-weight:bold;width:28px;height:28px;border-radius:6px;justify-content:center;margin-right:8px;">4</span>'
    ' **Cart value & price distributions**',
    unsafe_allow_html=True,
)
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    if "cart_value_at_show" in df.columns:
        fig_cart = px.histogram(
            df,
            x="cart_value_at_show",
            nbins=50,
            title="Cart value at show",
            labels={"cart_value_at_show": "value", "count": "count"},
        )
        fig_cart.update_traces(marker_color="#2196F3")
        fig_cart.update_layout(template="plotly_dark", height=350, showlegend=True)
        fig_cart.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_cart, use_container_width=True)
    else:
        st.info("Column 'cart_value_at_show' not found.")

with col2:
    if "item_price" in df.columns:
        fig_price = px.histogram(
            df,
            x="item_price",
            nbins=50,
            title="Item price (item_price)",
            labels={"item_price": "value", "count": "count"},
        )
        fig_price.update_traces(marker_color="#9C27B0")
        fig_price.update_layout(template="plotly_dark", height=350, showlegend=True)
        fig_price.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Column 'item_price' not found.")
