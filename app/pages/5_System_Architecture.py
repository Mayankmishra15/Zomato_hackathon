"""
System Architecture & Production Design Page.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="System Architecture | CSAO Rail", page_icon="🏗️", layout="wide")
st.title("🏗️ System Architecture & Production Design")

st.subheader("Architecture Diagram")
st.markdown("""
```
[Client App] → [API Gateway / Load Balancer]
             → [FastAPI Inference Service (multiple replicas)]
                 → [Feature Store (Redis cache)]
                 → [LightGBM Model (in-memory)]
                 → [Reranker (diversity + business rules)]
             → [Response: Top-8 items + scores]

Background services:
[Kafka stream] → [Feature Pipeline (Spark/Flink)] → [Feature Store]
[Batch training] → [MLflow experiment tracking] → [Model Registry]
```
""")

st.subheader("Latency Budget Breakdown")
latency_data = {
    "Component": ["API Gateway", "Feature retrieval (Redis)", "Model inference", "Reranking", "Serialization", "Network"],
    "ms": [5, 15, 35, 5, 5, 15],
}
import pandas as pd
df = pd.DataFrame(latency_data)
fig = px.bar(df, x="Component", y="ms", title="Latency Budget (Target: 200ms)", color="ms", color_continuous_scale="Reds")
fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Latency (ms)")
st.plotly_chart(fig, use_container_width=True)
st.metric("Total Budget", f"{df['ms'].sum()}ms", "✅ Within 200ms")

st.subheader("Scalability Design")
st.markdown("""
- **Stateless inference pods** → horizontal scaling
- **Redis feature cache** → sub-ms feature lookup
- **Model in memory** → no disk I/O at inference
- **CDN-cached item metadata**
""")

st.subheader("Cold Start Strategy")
st.markdown("""
| Scenario | Fallback |
|----------|----------|
| New User | Heuristic (city + meal_slot + popularity) |
| New Restaurant | Cuisine-category averages |
| Data sufficient | Full ML model |
""")
