"""
Business Impact — Live revenue calculator + A/B guardrails.
Aligned with Step 7: metric translation, baseline comparison, A/B design.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Business Impact | CSAO Rail",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Business Impact")
st.caption("Live revenue calculator + A/B guardrails")

st.markdown("---")
st.subheader("Live revenue calculator")

baseline_pct = st.slider("Baseline accept rate (%)", 0.0, 60.0, 34.0, 0.5, help="Current add-on acceptance rate without model")
model_pct = st.slider("Model accept rate (%)", 0.0, 60.0, 40.0, 0.5, help="Expected acceptance rate with recommendation model")

col_in1, col_in2, col_in3 = st.columns(3)
with col_in1:
    avg_addon = st.number_input("Avg add-on price (₹)", min_value=10, max_value=1000, value=150, step=10)
with col_in2:
    avg_cart = st.number_input("Avg cart value (₹)", min_value=100, max_value=5000, value=400, step=50)
with col_in3:
    daily_impressions = st.number_input("Daily impressions", min_value=10000, max_value=50_000_000, value=5_000_000, step=100_000, format="%d")

# Match project_business_impact(): rates as decimals for formulas
baseline_rate = baseline_pct / 100.0
model_rate = model_pct / 100.0
incremental_accepts = (model_rate - baseline_rate) * daily_impressions
daily_incremental_revenue = incremental_accepts * avg_addon
aov_lift_pct = (model_rate * avg_addon) / avg_cart if avg_cart else 0  # add-on contribution to AOV at model rate

st.markdown("---")
st.markdown("**Output metrics**")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Incremental accepts/day", f"{int(incremental_accepts):,}")
with m2:
    st.metric("Daily revenue uplift", f"₹{int(daily_incremental_revenue):,}")
with m3:
    st.metric("Monthly revenue uplift", f"₹{int(daily_incremental_revenue * 30):,}")
with m4:
    st.metric("AOV lift %", f"{aov_lift_pct:.2%}")

# --- A/B Test Design (from Step 7) ---
st.markdown("---")
st.subheader("A/B test design & guardrails")

AB_TEST_DESIGN = {
    "experiment_name": "CSAO_LGBM_v1_vs_Baseline",
    "hypothesis": "LightGBM-based recommendations increase add-on acceptance rate by ≥ 20% relative over current popularity-based system",
    "allocation": {
        "control": "50% of users — existing popularity-based CSAO rail",
        "treatment": "50% of users — LightGBM + LLM reranked CSAO rail",
        "stratification": "Stratify by city_tier, user_segment, meal_slot for balanced groups",
    },
    "primary_metrics": {
        "accept_rate": "% of CSAO impressions resulting in item added to cart (main success metric)",
        "aov_lift": "Difference in average order value between treatment and control",
        "attach_rate": "% of orders that include at least one CSAO item",
    },
    "secondary_metrics": {
        "ctr": "Click-through rate on CSAO rail (engagement signal)",
        "c2o_rate": "Cart-to-order conversion rate (ensure no harm to completion)",
        "items_per_order": "Average number of items per completed order",
    },
    "guardrail_metrics": {
        "cart_abandonment": "MUST NOT increase by > 1% absolute (harm signal)",
        "latency_p95": "MUST stay under 300ms — monitor every 15min during experiment",
        "error_rate": "MUST stay under 0.5% prediction service errors",
        "veg_violation": "MUST be 0% — non-veg items shown to strict-veg users",
    },
    "statistical_design": {
        "test_type": "Two-proportion z-test for accept_rate; t-test for AOV",
        "significance": "α = 0.05 (95% confidence)",
        "power": "β = 0.20 (80% power to detect true effect)",
        "mde": "Minimum Detectable Effect = 15% relative lift in accept_rate",
        "sample_size_calc": "n = 2 * (z_α + z_β)² * p(1-p) / (MDE * p)²",
        "expected_runtime": "~7-14 days to reach statistical significance at 5M daily impressions",
    },
    "rollout_plan": {
        "phase_1": "1% canary — 24h — monitor latency + error rate only",
        "phase_2": "10% — 48h — monitor all guardrail metrics",
        "phase_3": "50% A/B split — 7-14 days — full metric evaluation",
        "phase_4": "Full rollout if: primary metric significant + all guardrails green",
    },
    "segment_monitoring": {
        "report_by": ["user_segment", "meal_slot", "city_tier", "is_new_user"],
        "flag_if": "Any segment shows > 5% negative delta in accept_rate vs control",
    },
}

exp1, exp2, exp3 = st.expander("Hypothesis & allocation"), st.expander("Primary & guardrail metrics"), st.expander("Statistical design & rollout")
with exp1:
    st.markdown(f"**{AB_TEST_DESIGN['experiment_name']}**")
    st.write(AB_TEST_DESIGN["hypothesis"])
    st.json(AB_TEST_DESIGN["allocation"])
with exp2:
    st.markdown("**Primary**")
    st.json(AB_TEST_DESIGN["primary_metrics"])
    st.markdown("**Guardrails**")
    st.json(AB_TEST_DESIGN["guardrail_metrics"])
with exp3:
    st.json(AB_TEST_DESIGN["statistical_design"])
    st.markdown("**Rollout**")
    st.json(AB_TEST_DESIGN["rollout_plan"])
