# 🛒 CSAO Rail — Cart Super Add-On Recommendation System

Production-grade Cart Super Add-On (CSAO) Rail Recommendation System for food delivery platforms. Uses the **csao_ml_final** dataset.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (uses csao_ml_final.csv in project root)
python train.py --model all --shap

# 3. Run FastAPI backend (optional — Streamlit can use in-process predictor)
uvicorn api.main:app --reload --port 8000

# 4. Run Streamlit app (in a new terminal)
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**. Set `API_URL=http://localhost:8000` to use the FastAPI backend from the Live Demo; otherwise the app uses the in-process model.

## Dataset

**csao_ml_final.csv** — Place in project root. Columns include:

- **Target:** `label` (1 = accepted add-on, 0 = rejected)
- **Split:** `split` (0,1 = train, 2 = test)
- **Features:** user_segment, city, cart_*, has_*_in_cart, item_*, meal_slot, hour, is_weekend, is_festival_order, is_rainy_weather, etc.

## Documentation

- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** — System design & architecture (diagrams, latency budget, scalability, cold start).

## Project Structure

```
├── csao_ml_final.csv          # Dataset (project root)
├── SYSTEM_DESIGN.md           # Architecture & production design
├── src/
│   ├── config.py
│   ├── data_pipeline/        # preprocessor, cold_start
│   ├── models/               # baseline, xgboost, lgbm, ensemble
│   ├── evaluation/            # metrics, business_metrics, error_analysis
│   ├── inference/             # predictor, reranker
│   └── llm_layer/            # explainer
├── api/
│   ├── main.py                # FastAPI app (POST /recommend, GET /health, GET /metrics)
│   └── schemas.py             # RecommendRequest, RecommendResponse
├── app/
│   ├── streamlit_app.py       # Streamlit entry
│   └── pages/                 # Multi-page UI
├── notebooks/                 # EDA, feature engineering, training, business impact
├── train.py
├── evaluate.py
├── tests/
└── requirements.txt
```

## Streamlit Pages

| Page | Description |
|------|-------------|
| 📈 Zomato Data Exploration | Heatmap (meal slot × item category), cart value & item price histograms, data table (100 rows), EDA summary |
| 🛒 Live Demo | Cart builder (context flags, hour, user profile, engine controls), recommendation rail, pipeline breakdown, accepted add-ons |
| 📊 Model Performance | Dataset EDA, model comparison, segment analysis |
| 🔍 Explainability | SHAP feature importance & explanations |
| 🧪 A/B Testing | Model vs Baseline, business metrics |
| 🏗️ System Architecture | Architecture diagram, latency budget |
| ⚙️ Feature Engineering | Feature groups (realtime / batch / cross), top features, techniques, pipeline flow |
| 📐 Problem Formulation | Learning-to-Rank, loss & constraints |
| 💰 Business Impact | Revenue calculator, A/B test design & guardrails |

## System Design & Architecture

### High-level flow

```
[Client App] → [API Gateway / Load Balancer]
             → [FastAPI Inference Service (replicas)]
                 → [Feature Store (Redis cache)]
                 → [LightGBM Model (in-memory)]
                 → [Reranker (diversity + business rules)]
             → [Response: Top-K items + scores]

Background:
[Kafka stream] → [Feature Pipeline] → [Feature Store]
[Batch training] → [MLflow] → [Model Registry]
```

### Latency budget (target: 200 ms)

| Component              | ms  |
|------------------------|-----|
| API Gateway            | 5   |
| Feature retrieval      | 15  |
| Model inference        | 35  |
| Reranking              | 5   |
| Serialization + Network| 20  |
| **Total**               | **~80 ms** (within budget) |

### Scalability

- **Stateless inference pods** — horizontal scaling behind a load balancer.
- **Redis feature cache** — sub-ms feature lookup for user/item aggregates.
- **Model in memory** — no disk I/O at request time.
- **CDN** — cached item metadata and images.

### Cold start

| Scenario       | Fallback |
|----------------|----------|
| New user       | Heuristic (city + meal_slot + popularity) |
| New restaurant | Cuisine/category averages |
| Data sufficient| Full ML model (LightGBM) |

---

## Model Performance

| Model | AUC | P@8 | NDCG@8 |
|-------|-----|-----|--------|
| Baseline (Popularity) | ~0.62 | ~0.28 | ~0.55 |
| XGBoost | ~0.84 | ~0.48 | ~0.74 |
| LightGBM | ~0.86 | ~0.51 | ~0.77 |

## Environment

- **API_URL** or **VITE_API_URL** — Optional. Base URL for FastAPI (e.g. `http://localhost:8000`). If unset, Live Demo uses the in-process predictor.
