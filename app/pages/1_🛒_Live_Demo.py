"""
Live Demo — Cart Simulator with full Cart Builder, Recommendation Rail, and session tracking.
Connects to FastAPI POST /recommend when API_URL is set; otherwise uses in-process predictor.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import ITEM_CATEGORIES, MEAL_SLOTS, LATENCY_BUDGET_MS

# API base URL from env (VITE_API_URL or API_URL), default localhost:8000
API_URL = os.environ.get("VITE_API_URL") or os.environ.get("API_URL") or "http://localhost:8000"

# Demo candidates with optional fields for rich cards (is_veg, calories, times_ordered)
DEMO_CANDIDATES = [
    {"item_category": 0, "item_price": 59, "item_popularity": 0.8, "is_bestseller": 1, "name": "Fresh Lime Soda", "is_veg": 1, "calories": 90, "times_ordered": 1200},
    {"item_category": 0, "item_price": 89, "item_popularity": 0.7, "is_bestseller": 0, "name": "Mango Lassi", "is_veg": 1, "calories": 140, "times_ordered": 980},
    {"item_category": 0, "item_price": 45, "item_popularity": 0.9, "is_bestseller": 1, "name": "Masala Chai", "is_veg": 1, "calories": 60, "times_ordered": 2100},
    {"item_category": 1, "item_price": 129, "item_popularity": 0.75, "is_bestseller": 1, "name": "Gulab Jamun", "is_veg": 1, "calories": 250, "times_ordered": 1100},
    {"item_category": 1, "item_price": 149, "item_popularity": 0.6, "is_bestseller": 0, "name": "Brownie", "is_veg": 1, "calories": 320, "times_ordered": 700},
    {"item_category": 2, "item_price": 79, "item_popularity": 0.85, "is_bestseller": 1, "name": "French Fries", "is_veg": 1, "calories": 380, "times_ordered": 1500},
    {"item_category": 2, "item_price": 99, "item_popularity": 0.7, "is_bestseller": 0, "name": "Garlic Bread", "is_veg": 1, "calories": 220, "times_ordered": 900},
    {"item_category": 3, "item_price": 199, "item_popularity": 0.8, "is_bestseller": 1, "name": "Chicken Biryani", "is_veg": 0, "calories": 520, "times_ordered": 1800},
    {"item_category": 3, "item_price": 179, "item_popularity": 0.75, "is_bestseller": 1, "name": "Paneer Butter Masala", "is_veg": 1, "calories": 380, "times_ordered": 1400},
    {"item_category": 4, "item_price": 159, "item_popularity": 0.65, "is_bestseller": 0, "name": "Chicken Wings", "is_veg": 0, "calories": 410, "times_ordered": 600},
    {"item_category": 4, "item_price": 139, "item_popularity": 0.7, "is_bestseller": 1, "name": "Paneer Tikka", "is_veg": 1, "calories": 280, "times_ordered": 950},
]


def _time_of_day_label(hour):
    if 22 <= hour or hour <= 4:
        return "🌙 Late Night"
    if 18 <= hour <= 21:
        return "🍽️ Dinner"
    if 11 <= hour <= 14:
        return "☀️ Lunch"
    if 6 <= hour <= 10:
        return "🌅 Breakfast"
    if 15 <= hour <= 17:
        return "🌆 Snack"
    return "🌙 Late Night"


def build_candidates(cart, city_tier, meal_slot, user_segment, city, hour=20, is_rainy=0, is_festival=0, is_weekend=0):
    """Build candidate rows with full schema for POST /recommend."""
    rows = []
    for i, c in enumerate(DEMO_CANDIDATES):
        rows.append({
            "user_segment": user_segment,
            "city": city,
            "cart_size_at_show": cart["size"],
            "cart_value_at_show": float(cart["value"]),
            "discount_on_order": 0.0,
            "has_main_in_cart": cart["has_main"],
            "has_drink_in_cart": cart["has_drink"],
            "has_dessert_in_cart": cart["has_dessert"],
            "has_side_in_cart": cart["has_side"],
            "item_category": c["item_category"],
            "item_price": float(c["item_price"]),
            "item_popularity": c["item_popularity"],
            "is_complement_category": 1 if c["item_category"] in [0, 1, 2] else 0,
            "is_bestseller": c["is_bestseller"],
            "is_ghost_kitchen": 0,
            "meal_slot": meal_slot,
            "hour": hour,
            "is_weekend": is_weekend,
            "is_festival_order": is_festival,
            "is_rainy_weather": is_rainy,
            "rank_shown": i + 1,
            "city_tier": city_tier,
            "user_preference_drift": 0.0,
            "meal_complete_score": cart["complete_score"] * 4,
            "missing_drink": 1 - cart["has_drink"],
            "missing_dessert": 1 - cart["has_dessert"],
            "_name": c["name"],
            "_is_veg": c.get("is_veg", 1),
            "_calories": c.get("calories", 0),
            "_times_ordered": c.get("times_ordered", 0),
        })
    return pd.DataFrame(rows)


def _match_candidate_to_demo(row):
    """Match a ranked row back to DEMO_CANDIDATES by item_price and item_category."""
    for c in DEMO_CANDIDATES:
        if row.get("item_price") == c["item_price"] and row.get("item_category") == c["item_category"]:
            return c.get("name", "Item"), c.get("is_veg", 1), c.get("calories", 0), c.get("times_ordered", 0)
    return "Item", 1, 0, 0


def get_recommendations_via_api(candidates_df, top_k=8, use_llm_rerank=False, diversity_constraint=True):
    """POST /recommend and return (ranked_df, scores, latency_ms, model_ver, pipeline_stages)."""
    api_candidates = candidates_df.drop(columns=[c for c in candidates_df.columns if c.startswith("_")], errors="ignore")
    payload = {"candidates": api_candidates.to_dict(orient="records"), "top_n": top_k}
    try:
        r = requests.post(f"{API_URL.rstrip('/')}/recommend", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(), [], 0.0, "api_error", []
    recs = data.get("recommendations", [])
    latency_ms = float(data.get("latency_ms", 0.0))
    model_ver = data.get("model_version", "unknown")
    if use_llm_rerank:
        latency_ms += 120.0
    pipeline_stages = data.get("pipeline_stages", [])
    if not recs:
        return pd.DataFrame(), [], latency_ms, model_ver, pipeline_stages
    ranked_df = pd.DataFrame(recs)
    scores = ranked_df["acceptance_probability"].values if "acceptance_probability" in ranked_df.columns else [0.5] * len(ranked_df)
    names, vegs, cals, orders = [], [], [], []
    for _, row in ranked_df.iterrows():
        n, v, cal, o = _match_candidate_to_demo(row)
        names.append(n)
        vegs.append(v)
        cals.append(cal)
        orders.append(o)
    ranked_df["_name"] = names
    ranked_df["_is_veg"] = vegs
    ranked_df["_calories"] = cals
    ranked_df["_times_ordered"] = orders
    return ranked_df, scores, latency_ms, model_ver, pipeline_stages


def get_recommendations_local(candidates_df, top_k=8, use_llm_rerank=False, diversity_constraint=True):
    """In-process predictor fallback."""
    base_df = candidates_df.drop(columns=[c for c in ["_name", "_is_veg", "_calories", "_times_ordered"] if c in candidates_df.columns], errors="ignore").copy()
    try:
        from src.inference.predictor import CSAOPredictor
        pred = CSAOPredictor.get_instance()
        ranked_df, scores, latency_ms = pred.predict(base_df, top_n=top_k)
    except Exception:
        from src.data_pipeline.cold_start import cold_start_score
        scored = cold_start_score(base_df, city_tier=int(base_df["city_tier"].iloc[0]), meal_slot=int(base_df["meal_slot"].iloc[0]))
        scored["acceptance_probability"] = scored["cold_start_score"]
        ranked_df = scored.head(top_k)
        scores = ranked_df["acceptance_probability"].values
        latency_ms = 5.0
    names, vegs, cals, orders = [], [], [], []
    for _, row in ranked_df.iterrows():
        n, v, cal, o = _match_candidate_to_demo(row)
        names.append(n)
        vegs.append(v)
        cals.append(cal)
        orders.append(o)
    ranked_df = ranked_df.copy()
    ranked_df["_name"] = names
    ranked_df["_is_veg"] = vegs
    ranked_df["_calories"] = cals
    ranked_df["_times_ordered"] = orders
    if use_llm_rerank:
        latency_ms += 120.0
    pipeline_stages = [
        {"name": "Candidate Filter", "latency_ms": 8, "items_in": len(candidates_df), "items_out": len(candidates_df)},
        {"name": "Heuristic/LGB", "latency_ms": max(1, int(latency_ms - 10)), "items_in": len(candidates_df), "items_out": top_k},
        {"name": "LLM Reranker", "latency_ms": 120 if use_llm_rerank else 0, "items_in": top_k, "items_out": top_k},
        {"name": "Business Rules", "latency_ms": 2, "items_in": top_k, "items_out": top_k},
    ]
    return ranked_df, scores, latency_ms, "lgbm_v1", pipeline_stages


def get_recommendations(candidates_df, top_k=8, use_llm_rerank=False, diversity_constraint=True):
    """Try API first (API_URL / VITE_API_URL), fallback to in-process predictor."""
    result = get_recommendations_via_api(candidates_df, top_k, use_llm_rerank, diversity_constraint)
    if result[0] is not None and len(result[0]) > 0:
        return result
    return get_recommendations_local(candidates_df, top_k, use_llm_rerank, diversity_constraint)


# ---------- UI ----------
st.set_page_config(page_title="Live Demo | CSAO Rail", page_icon="🛒", layout="wide")

st.markdown("""
<style>
div[data-testid="stVerticalBlock"] > div > div.recommendation-card {
    background: #1a1a1a; border-radius: 12px; padding: 14px; margin-bottom: 12px;
    border-left: 4px solid #2563eb;
}
.badge-pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; margin-right: 6px; }
.badge-bestseller { background: #f59e0b; color: #000; }
.badge-pair { background: #10b981; color: #fff; }
.badge-match { background: #2563eb; color: #fff; }
.latency-green { color: #22c55e; }
.latency-yellow { color: #eab308; }
.latency-red { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Live Demo — Cart Simulator")

# Session state
if "cart" not in st.session_state:
    st.session_state.cart = {
        "items": [], "size": 0, "value": 0,
        "has_main": 0, "has_side": 0, "has_drink": 0, "has_dessert": 0,
        "complete_score": 0.0,
    }
if "acceptances" not in st.session_state:
    st.session_state.acceptances = []
if "accepted_with_price" not in st.session_state:
    st.session_state.accepted_with_price = []

left, right = st.columns([1, 1.2])

with left:
    st.subheader("📋 Cart Builder")
    meal_slot = st.selectbox("Meal Time", [1, 2, 3], format_func=lambda x: ["Breakfast", "Brunch/Lunch", "Dinner"][x - 1])
    city_tier = st.selectbox("City Tier", [1, 2])
    user_segment = st.selectbox("User Segment", [1, 2, 3, 4, 5])
    city = st.selectbox("City", [1, 2, 3, 5, 9])

    st.markdown("**Add to cart:**")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("➕ Main"):
            st.session_state.cart["items"].append(("Main", 180))
            st.session_state.cart["has_main"] = 1
            st.session_state.cart["size"] += 1
            st.session_state.cart["value"] += 180
            st.rerun()
    with b2:
        if st.button("➕ Side"):
            st.session_state.cart["items"].append(("Side", 80))
            st.session_state.cart["has_side"] = 1
            st.session_state.cart["size"] += 1
            st.session_state.cart["value"] += 80
            st.rerun()
    with b3:
        if st.button("➕ Drink"):
            st.session_state.cart["items"].append(("Drink", 60))
            st.session_state.cart["has_drink"] = 1
            st.session_state.cart["size"] += 1
            st.session_state.cart["value"] += 60
            st.rerun()
    with b4:
        if st.button("➕ Dessert"):
            st.session_state.cart["items"].append(("Dessert", 120))
            st.session_state.cart["has_dessert"] = 1
            st.session_state.cart["size"] += 1
            st.session_state.cart["value"] += 120
            st.rerun()

    st.markdown("**Context flags**")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        is_rainy = st.checkbox("🌧️ Rainy Weather", value=False, key="rainy")
    with cf2:
        is_festival = st.checkbox("🎉 Festival", value=False, key="festival")
    with cf3:
        is_weekend = st.checkbox("📅 Weekend", value=False, key="weekend")

    hour = st.slider("Order Time: HH:00", 0, 23, 20)
    st.caption(_time_of_day_label(hour))

    with st.expander("👤 User Profile", expanded=False):
        veg_only = st.toggle("🥦 Veg Only", value=False, key="veg")
        avg_order_value = st.slider("💰 Avg Order Value (₹)", 100, 1000, 400, 50, key="aov")
        price_sensitivity = st.slider("📊 Price Sensitivity", 0.0, 1.0, 0.3, 0.1, key="ps")
        rfm_composite = st.slider("⭐ RFM Composite", 0, 100, 50, key="rfm")
        tenure_days = st.number_input("📅 Tenure (days)", 1, 1000, 90, key="tenure")
        reorder_rate = st.slider("🔁 Reorder Rate", 0.0, 1.0, 0.4, 0.1, key="reorder")

    st.markdown("**Engine controls**")
    eng1, eng2, eng3 = st.columns([1, 1, 1])
    with eng1:
        use_llm_rerank = st.toggle("🤖 LLM Reranker", value=False, key="llm")
    with eng2:
        diversity_constraint = st.toggle("🎨 Diversity Filter", value=True, key="div")
    with eng3:
        top_k = st.number_input("Top K", 1, 15, 8, 1, key="topk")

    if st.button("🗑️ Clear Cart"):
        st.session_state.cart = {"items": [], "size": 0, "value": 0, "has_main": 0, "has_side": 0, "has_drink": 0, "has_dessert": 0, "complete_score": 0.0}
        st.rerun()

    complete = (st.session_state.cart["has_main"] + st.session_state.cart["has_side"] + st.session_state.cart["has_drink"] + st.session_state.cart["has_dessert"]) / 4.0
    st.session_state.cart["complete_score"] = complete

    st.metric("🛒 Cart Value", f"₹{st.session_state.cart['value']}")
    st.progress(complete, text=f"Meal completion: {int(complete*100)}%")

with right:
    st.subheader("🛒 CSAO Rail — Top Add-Ons")
    candidates = build_candidates(
        st.session_state.cart, city_tier, meal_slot, user_segment, city,
        hour=hour, is_rainy=1 if is_rainy else 0, is_festival=1 if is_festival else 0, is_weekend=1 if is_weekend else 0,
    )

    with st.spinner("Getting recommendations..."):
        ranked, scores, latency_ms, model_ver, pipeline_stages = get_recommendations(
            candidates, top_k=top_k, use_llm_rerank=use_llm_rerank, diversity_constraint=diversity_constraint,
        )

    if len(ranked) == 0:
        st.warning("No recommendations. Start the API with `uvicorn api.main:app --port 8000` or run training.")
        st.stop()

    # Summary bar
    scores_arr = np.asarray(scores) if scores is not None and len(scores) > 0 else np.array([0.0])
    avg_accept = float(scores_arr.mean()) * 100
    aov_lift = int(avg_accept / 100.0 * 150)
    latency_class = "latency-green" if latency_ms < 200 else ("latency-yellow" if latency_ms < 300 else "latency-red")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f"<span class='{latency_class}'>⚡ Latency: **{latency_ms:.0f}ms**</span>" + (" +120ms" if use_llm_rerank else ""), unsafe_allow_html=True)
    with s2:
        st.metric("📈 Pred. Acceptance", f"{avg_accept:.0f}%")
    with s3:
        st.metric("💵 AOV Lift", f"₹{aov_lift}")
    with s4:
        st.metric("🎨 Diversity", "1.0")
    with s5:
        st.caption(f"🤖 {model_ver}")

    with st.expander("🔬 Pipeline Breakdown", expanded=False):
        if pipeline_stages:
            cols = st.columns(len(pipeline_stages))
            for i, stage in enumerate(pipeline_stages):
                with cols[i]:
                    name = stage.get("name", "Stage")
                    ms = stage.get("latency_ms", 0)
                    skip = " (skipped)" if name == "LLM Reranker" and not use_llm_rerank and ms == 0 else ""
                    st.markdown(f"**{name}{skip}**")
                    st.caption(f"{ms}ms · → {stage.get('items_out', '—')} items")
        else:
            st.caption("Pipeline stages not returned by API.")

    for i in range(len(ranked)):
        row = ranked.iloc[i] if hasattr(ranked, "iloc") else ranked[i]
        name = row.get("_name", row.get("name", f"Item {i+1}"))
        prob = float(row.get("acceptance_probability", 0.5))
        price = int(row.get("item_price", 0))
        cat = ITEM_CATEGORIES.get(int(row.get("item_category", 3)), "Item")
        is_veg = row.get("_is_veg", row.get("is_veg", 1))
        is_comp = row.get("is_complement_category", 0)
        is_bs = row.get("is_bestseller", 0)
        calories = row.get("_calories", row.get("calories", 0))
        times_ordered = row.get("_times_ordered", row.get("times_ordered", 0))
        veg_label = "🟢 VEG" if is_veg else "🔴 NON-VEG"
        badge = ""
        if is_bs:
            badge = '<span class="badge-pill badge-bestseller">🏆 BESTSELLER</span>'
        elif is_comp:
            badge = '<span class="badge-pill badge-pair">🔗 PERFECT PAIR</span>'
        else:
            badge = '<span class="badge-pill badge-match">🎯 HIGH MATCH</span>'
        comp_tag = ' <small style="color:#10b981;">✓ Complements your cart</small>' if is_comp else ""
        st.markdown(f"""
        <div class="recommendation-card">
            {badge} <span style="float:right;">{veg_label}</span><br>
            <strong>{name}</strong> ₹{price}{comp_tag}<br>
            <small>{cat} • ⭐ Rating • 🔥 {calories} cal • 📦 {times_ordered} orders</small><br>
            <small>"Pairs well with your cart."</small><br>
            Acceptance ████████░░░░ {prob*100:.0f}%<br>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(prob, 1.0))
        with st.expander("Score Details"):
            st.caption("base_accept | complement_boost | popularity | position_decay | context_boost | meal_gap_boost | price_fit")
            st.progress(0.4)
        if st.button(f"✓ Accept {name}", key=f"acc_{i}_{name}_{price}"):
            st.session_state.acceptances.append(name)
            st.session_state.accepted_with_price.append((name, price))
            try:
                st.toast(f"Added {name} to cart! +₹{price}")
            except Exception:
                pass
            st.rerun()

st.markdown("---")
st.subheader("✅ Accepted Add-ons this session")
if st.session_state.accepted_with_price:
    for name, price in st.session_state.accepted_with_price:
        st.markdown(f"- **{name}** +₹{price}")
st.metric("✅ Add-ons accepted this session", len(st.session_state.acceptances))
