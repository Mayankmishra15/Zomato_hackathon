"""Tests for predictor — csao_ml_final schema."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@pytest.fixture
def sample_candidates():
    n = 20
    np.random.seed(42)
    df = pd.DataFrame({
        "user_segment": 5,
        "city": 9,
        "item_category": np.random.randint(0, 5, n),
        "meal_slot": 2,
        "city_tier": 1,
        "hour": 13,
        "cart_value_at_show": 300,
        "cart_size_at_show": 2,
        "item_price": np.random.uniform(50, 150, n),
        "meal_complete_score": 2,
        "item_popularity": np.random.uniform(0.3, 0.9, n),
        "is_bestseller": np.random.randint(0, 2, n),
        "has_main_in_cart": 1,
        "has_side_in_cart": 0,
        "has_drink_in_cart": 0,
        "has_dessert_in_cart": 0,
        "discount_on_order": 0.0,
        "is_complement_category": 1,
        "is_ghost_kitchen": 0,
        "is_weekend": 1,
        "is_festival_order": 0,
        "is_rainy_weather": 0,
        "rank_shown": 1,
        "user_preference_drift": 0.0,
        "missing_drink": 1,
        "missing_dessert": 1,
    })
    return df


def test_cold_start_score(sample_candidates):
    from src.data_pipeline.cold_start import cold_start_score
    scored = cold_start_score(sample_candidates, city_tier=1, meal_slot=2)
    assert "cold_start_score" in scored.columns
    assert len(scored) == len(sample_candidates)


def test_predictor_when_trained(sample_candidates):
    model_path = "artifacts/models/lgbm_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained — run: python train.py --model lgbm")
    from src.inference.predictor import CSAOPredictor
    pred = CSAOPredictor.get_instance()
    ranked_df, scores, latency_ms = pred.predict(sample_candidates, top_n=8)
    assert len(ranked_df) == 8
    assert len(scores) == 8
    assert latency_ms < 500
