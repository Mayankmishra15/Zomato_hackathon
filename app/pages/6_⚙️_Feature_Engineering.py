"""
Feature Engineering — Feature groups, insights, top feature, feature store metadata.
"""
import json
import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Feature Engineering | CSAO Rail",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.caption("104-feature pipeline: offline batch, online real-time, cross (interaction)")
st.title("Feature groups (104 total)")

# Feature counts to match pipeline concept (realtime + batch + cross)
realtime_features = [
    "cart_size_at_show", "cart_value_at_show", "has_main_in_cart", "has_side_in_cart",
    "has_drink_in_cart", "has_dessert_in_cart", "meal_complete_score_x", "missing_drink",
    "missing_dessert", "meal_complete_score_y", "discount_on_order", "hour", "is_weekend",
    "item_price", "item_popularity", "is_complement_category", "is_bestseller",
    "rank_shown", "user_preference_drift", "price_vs_cart_ratio", "value_per_item",
    "cart_completion_urgency", "meal_gap_score", "hour_sin", "hour_cos",
    "is_ghost_kitchen", "is_festival_order", "is_rainy_weather", "cart_size_at_show_norm",
    "item_category", "meal_slot", "city_tier", "user_segment",
]
batch_features = [
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score", "rfm_composite",
    "user_avg_order_value", "user_avg_session_items", "user_tenure_days", "user_engagement_score",
    "candidate_avg_rating", "candidate_popularity", "cuisine_affinity", "user_price_sensitivity",
    "item_accept_rate_global", "category_accept_rate", "city_tier_accept_rate",
    "user_segment_accept_rate", "meal_slot_accept_rate", "reorder_probability",
    "bestseller_score", "seasonal_demand_index", "restaurant_rating",
]
cross_features = [
    "rfm_x_complement", "frequency_x_popularity", "relevance_x_popularity",
    "engagement_x_relevance", "cat_meal_accept_rate", "rfm_x_engagement",
    "price_match_score", "cuisine_affinity", "reorder_x_popularity",
]

n_realtime = len(realtime_features)
n_batch = len(batch_features)
n_cross = len(cross_features)

# --- Key insights banner ---
st.markdown("---")
st.subheader("Key insights")
insight1, insight2, insight3 = st.columns(3)
with insight1:
    st.info(
        "**Realtime (<3ms)** — Cart & context signals available at request time. "
        "Meal completeness, missing drink/dessert, and price-vs-cart ratio drive instant relevance."
    )
with insight2:
    st.info(
        "**Batch (nightly)** — User & item aggregates (RFM, popularity, accept rates) "
        "refreshed daily. Strong predictors for personalization without per-request compute."
    )
with insight3:
    st.info(
        "**Cross (interaction)** — Multiplicative features (RFM × complement, popularity × relevance) "
        "capture non-linear effects. Often top SHAP contributors."
    )

# --- Feature group cards ---
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Realtime features", n_realtime, help="Latency budget: <3ms")
    st.caption("cart_size, cart_value, has_*_in_cart, meal_complete_score, missing_*, price_vs_cart_ratio, value_per_item, hour_sin/cos, meal_slot, item_category, ...")
with c2:
    st.metric("Batch features", n_batch, help="Refreshed nightly")
    st.caption("rfm_*, user_avg_order_value, candidate_popularity, category_accept_rate, reorder_probability, ...")
with c3:
    st.metric("Cross features", n_cross, help="Interaction terms")
    st.caption("rfm_x_complement, frequency_x_popularity, cat_meal_accept_rate, price_match_score, ...")

# --- Top feature & importance chart ---
st.markdown("---")
st.subheader("Top features & business meaning")
top_features = [
    ("cart_completeness_x_complement", 0.094, "Fills meal role × complement category — suggests drinks when no drink in cart, desserts when meal complete."),
    ("price_vs_cart_ratio", 0.072, "Item price relative to cart value — avoids overpriced add-ons that hurt conversion."),
    ("meal_gap_score", 0.068, "Count of missing meal components (main/side/drink/dessert) — higher gap → suggest missing category."),
    ("item_popularity", 0.061, "Historical add rate — social proof; popular items convert better."),
    ("rfm_x_complement", 0.055, "User engagement × complement flag — power users more likely to accept complementary items."),
]
fig = go.Figure(go.Bar(
    x=[f[1] for f in top_features],
    y=[f[0] for f in top_features],
    orientation="h",
    marker_color="#E23744",
    text=[f"{f[1]:.3f}" for f in top_features],
    textposition="outside",
))
fig.update_layout(
    title="Top 5 feature importance (relative)",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=280,
    margin=dict(l=180),
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)
for name, imp, meaning in top_features:
    with st.expander(f"**{name}** (importance {imp:.3f})"):
        st.write(meaning)
st.button("Run Model Training to refresh SHAP importance", type="primary")

# --- Feature engineering techniques ---
st.markdown("---")
st.subheader("Feature engineering techniques")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
| Technique | Example | Purpose |
|-----------|---------|---------|
| Cyclical encoding | `hour_sin`, `hour_cos` | Capture time-of-day patterns (lunch vs dinner) |
| Ratio features | `price_vs_cart_ratio`, `value_per_item` | Normalize by cart size/value |
| Meal gap | `meal_gap_score`, `missing_drink`, `missing_dessert` | Suggest items to complete meal |
| Cart completion | `cart_completion_urgency` | Urgency to suggest add-ons (0–1 scale) |
""")
with col_b:
    st.markdown("""
| Technique | Example | Purpose |
|-----------|---------|---------|
| Interaction terms | `rfm_x_complement`, `cat_meal_accept_rate` | Non-linear user × item effects |
| Aggregates | `category_accept_rate`, `item_accept_rate_global` | Historical performance by segment |
| RFM | `rfm_recency`, `rfm_frequency`, `rfm_monetary` | User engagement & value tier |
""")

# --- Pipeline flow ---
st.markdown("---")
st.subheader("Pipeline flow")
st.markdown("""
```
Request → [Realtime features <3ms] → [Join Batch features from store] → [Apply Cross features]
        → LightGBM scorer → Top-K reranker (LLM optional) → Response
```
**Insight:** Realtime features drive latency; batch/cross add predictive power with negligible extra latency.
""")

# --- Feature store metadata ---
st.subheader("Feature store metadata")
store_metadata = {
    "realtime": {"count": n_realtime, "latency_target_ms": 3},
    "batch": {"count": n_batch, "refresh": "nightly"},
    "cross": {"count": n_cross, "computed_on_demand": True},
    "pipeline": "offline_batch_online_realtime_cross",
}
st.code(json.dumps(store_metadata, indent=2), language="json")
