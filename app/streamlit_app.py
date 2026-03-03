"""
Main Streamlit entry point with multi-page navigation.
"""
import streamlit as st

st.set_page_config(
    page_title="CSAO Rail — Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Zomato red theme
st.markdown("""
<style>
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #E23744;
        margin-bottom: 12px;
    }
    .stMetric {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #E23744;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛒 CSAO Rail System")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Cart Super Add-On** recommendation engine "
    "using LightGBM ranking + LLM explanations."
)

st.title("🛒 CSAO Rail — AI-Powered Add-On Recommendation System")
st.markdown("""
Welcome to the **Cart Super Add-On (CSAO) Recommendation System** — 
an end-to-end ML system that suggests relevant add-on items to boost 
**Average Order Value** while improving customer experience.

### Navigate using the sidebar pages:
| Page | Description |
|------|-------------|
| 📈 Zomato Data Exploration | Accept rate heatmap (meal slot × item category), cart value & item price distributions |
| 🛒 Live Demo | Interactive cart simulator — add items & see real-time recommendations |
| 📊 Model Performance | EDA, metrics dashboard, segment analysis |
| 🔍 Explainability | SHAP feature importance & individual prediction explanations |
| 🧪 A/B Testing | Simulated A/B test: Model vs Baseline with business metrics |
| 🏗️ System Architecture | Architecture diagram, latency budget, production design |
| ⚙️ Feature Engineering | Feature groups (realtime / batch / cross), top feature, feature store metadata |
| 📐 Problem Formulation | Learning-to-Rank vs classification, loss & sequential constraints |
| 💰 Business Impact | Live revenue calculator + A/B guardrails (accept rate, uplift, AOV lift) |
""")
