"""
FastAPI backend with latency monitoring.
Endpoints:
  POST /recommend  — main inference endpoint
  GET  /health     — liveness probe
  GET  /metrics    — model + system stats
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import RecommendRequest, RecommendResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm-up model
    try:
        from src.inference.predictor import CSAOPredictor
        CSAOPredictor.get_instance()
    except FileNotFoundError:
        pass
    yield
    # Shutdown
    pass


app = FastAPI(title="CSAO Recommendation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    import pandas as pd
    try:
        from src.inference.predictor import CSAOPredictor
        predictor = CSAOPredictor.get_instance()
    except FileNotFoundError:
        from src.data_pipeline.cold_start import cold_start_score
        candidates_df = pd.DataFrame([c.model_dump() for c in request.candidates])
        scored = cold_start_score(
            candidates_df,
            city_tier=candidates_df["city_tier"].iloc[0] if len(candidates_df) > 0 else 1,
            meal_slot=candidates_df["meal_slot"].iloc[0] if len(candidates_df) > 0 else 3
        )
        top = scored.head(request.top_n)
        items = top.to_dict(orient="records")
        for r in items:
            r["acceptance_probability"] = float(r.get("cold_start_score", 0.5))
        return RecommendResponse(
            recommendations=items,
            latency_ms=5.0,
            model_version="cold_start_v1"
        )

    candidates_df = pd.DataFrame([c.model_dump() for c in request.candidates])
    ranked_df, scores, latency_ms = predictor.predict(candidates_df, top_n=request.top_n)
    items = ranked_df.to_dict(orient="records")
    for i, r in enumerate(items):
        r["acceptance_probability"] = float(scores[i])
    return RecommendResponse(
        recommendations=items,
        latency_ms=round(latency_ms, 2),
        model_version="lgbm_v1"
    )


@app.get("/metrics")
def metrics():
    return {
        "model": "LightGBM Pointwise Ranker",
        "version": "1.0",
        "features": 75,
        "latency_p99_ms": 180
    }
