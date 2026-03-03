"""
Problem Formulation — Learning-to-Rank vs binary classification framing.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Problem Formulation | CSAO Rail",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Problem Formulation")
st.caption("Learning-to-Rank vs binary classification framing")

st.markdown("---")
st.subheader("Task: Learning-to-Rank (pointwise)")
st.markdown("""
- **Given:** user (u), cart (C = {i₁, i₂, …}), context (ctx), candidate set (S)
- **Goal:** Learn f(u, C, ctx, s) → P(accept | u, C, ctx, s)
- **Output:** Rank candidates by P(accept) and return top-N (e.g. top 5)
""")

st.subheader("Ranking vs classification")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Classification**")
    st.markdown("- Single item: accept/reject")
    st.markdown("- Loss: BCE per impression")
    st.markdown("- No explicit ordering of list")
with col_b:
    st.markdown("**Ranking**")
    st.markdown("- Full list: order matters")
    st.markdown("- NDCG, MRR, P@K")
    st.markdown("- Cart state updates after each add")

st.subheader("Loss & sequential constraint")
st.code("""Loss (pointwise): L = -[y log(p) + (1-y) log(1-p)]

Cart state: C_{t+1} = C_t ∪ {accepted_item} → re-score remaining

Cold start: new_user → popularity × complement_score; new_item → content_similarity""", language=None)
