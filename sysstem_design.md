# System Design & Architecture

**CSAO Rail** — Cart Super Add-On Recommendation System. This document describes the high-level architecture, latency budget, scalability, and production design.

---

## 1. Overview

The system recommends add-on items to users at cart view time. It uses a **pointwise Learning-to-Rank** model (LightGBM) trained on the `csao_ml_final` dataset, with optional LLM reranking and diversity constraints. The design targets **&lt;200 ms** end-to-end latency for the recommendation API.

---

## 2. Architecture Diagram

### Request path (real-time)

```
[Client App]
     │
     ▼
[API Gateway / Load Balancer]
     │
     ▼
[FastAPI Inference Service]  ← multiple stateless replicas
     │
     ├──► [Feature Store (Redis)]  ← user/item aggregates, &lt;3ms
     │
     ├──► [LightGBM Model]  ← in-memory, batch score candidates
     │
     ├──► [Reranker]  ← diversity + business rules (veg, price cap)
     │
     └──► [Optional: LLM Reranker]  ← +120ms when enabled
     │
     ▼
[Response: Top-K items + scores + metadata]
```

### Background / offline path

```
[Kafka / Event Stream]
     │
     ▼
[Feature Pipeline (Spark / Flink)]
     │
     ▼
[Feature Store]  ← batch refresh (nightly) + real-time keys

[csao_ml_final.csv] ──► [train.py] ──► [MLflow] ──► [Model Registry]
                              │
                              ▼
                    [Artifacts: LGBM + encoders]
```

---

## 3. Request–Response Flow

1. **Client** sends `POST /recommend` with a list of **candidates** (item + context) and `top_n`.
2. **API** validates the body, optionally enriches from the feature store (user RFM, item popularity).
3. **Preprocessor** engineers features (hour_sin/cos, price_vs_cart_ratio, meal_gap_score, etc.).
4. **LightGBM** scores each candidate; results are sorted and trimmed to Top-K.
5. **Reranker** applies diversity and business rules (e.g. no non-veg for veg-only users).
6. **Optional** LLM reranker re-orders or explains; adds ~120 ms.
7. **Response** returns `recommendations`, `latency_ms`, `model_version`.

---

## 4. Latency Budget (Target: 200 ms)

| Component                    | Budget (ms) | Notes                          |
|-----------------------------|-------------|--------------------------------|
| API Gateway / LB            | 5           | Routing, TLS                   |
| Feature retrieval (Redis)   | 15          | User/item aggregates           |
| Feature engineering         | 10          | In-process transforms          |
| Model inference (LightGBM)   | 35          | In-memory, batched             |
| Reranking                   | 5           | Diversity + rules               |
| Serialization + network     | 20          | JSON, client round-trip        |
| **Total (no LLM)**          | **~90**     | Within 200 ms target           |
| With LLM reranker           | +120        | Optional; show in UI as +120ms |

---

## 5. Scalability & Deployment

| Concern            | Design choice |
|--------------------|----------------|
| **Horizontal scaling** | Stateless FastAPI pods behind a load balancer; scale by replica count. |
| **Feature store**      | Redis (or equivalent) for sub-ms lookup of user/item features. |
| **Model serving**      | Model loaded once per process (singleton); no disk I/O per request. |
| **Item metadata**       | Served via CDN or app backend; not in the critical path. |
| **Config**             | Env vars / secrets; no hardcoded credentials. |

---

## 6. Cold Start Strategy

| Scenario         | Behavior |
|------------------|----------|
| **New user**     | Heuristic: `city_tier` + `meal_slot` + `item_popularity` + complement score. |
| **New restaurant / item** | Cuisine/category-level accept rates; content-based fallback. |
| **Sufficient data** | Full LightGBM model with full feature set. |

Cold-start logic lives in `src/data_pipeline/cold_start.py` and is used when the trained model is unavailable or when candidates lack history.

---

## 7. Feature Pipeline (Summary)

- **Realtime (&lt;3 ms):** Cart and context (cart_value, has_*_in_cart, hour, meal_slot, is_weekend, is_rainy, etc.).
- **Batch (nightly):** RFM, user_avg_order_value, category_accept_rate, reorder_probability, etc.
- **Cross (interaction):** rfm_x_complement, price_match_score, cat_meal_accept_rate, etc.

See **Feature Engineering** in the Streamlit app and `src/data_pipeline/preprocessor.py` for the full list.

---

## 8. Security & Observability

- **API:** CORS configured; auth can be added at the gateway (e.g. API key or JWT).
- **Latency:** Response header `X-Latency-Ms` (if middleware enabled); `latency_ms` in JSON.
- **Health:** `GET /health` for liveness; `GET /metrics` for model version and high-level stats.
- **Logging:** Structured logs (request_id, latency, model_version) recommended in production.

---

## 9. Repository Mapping

| Concept in this doc   | Code / config |
|-----------------------|----------------|
| Inference API         | `api/main.py` |
| Request/response      | `api/schemas.py` |
| Predictor             | `src/inference/predictor.py` |
| Cold start            | `src/data_pipeline/cold_start.py` |
| Feature engineering   | `src/data_pipeline/preprocessor.py` |
| Latency constant      | `src/config.py` → `LATENCY_BUDGET_MS` |

For a visual overview and latency chart, use the **🏗️ System Architecture** page in the Streamlit app.
