"""Tests for preprocessor — csao_ml_final dataset."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.preprocessor import engineer_features, load_and_split, build_pipeline, CATEGORICAL_COLS, NUMERIC_COLS


@pytest.fixture
def sample_df():
    n = 100
    np.random.seed(42)
    df = pd.DataFrame({
        "label": np.random.randint(0, 2, n),
        "split": np.random.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
        "user_segment": np.random.randint(0, 6, n),
        "city": np.random.randint(0, 10, n),
        "cart_size_at_show": np.random.randint(1, 5, n),
        "cart_value_at_show": np.random.uniform(100, 500, n),
        "discount_on_order": np.random.uniform(0, 0.2, n),
        "has_main_in_cart": np.random.randint(0, 2, n),
        "has_side_in_cart": np.random.randint(0, 2, n),
        "has_drink_in_cart": np.random.randint(0, 2, n),
        "has_dessert_in_cart": np.random.randint(0, 2, n),
        "item_category": np.random.randint(0, 7, n),
        "item_price": np.random.uniform(50, 200, n),
        "item_popularity": np.random.uniform(0.2, 0.9, n),
        "is_complement_category": np.random.randint(0, 2, n),
        "is_bestseller": np.random.randint(0, 2, n),
        "is_ghost_kitchen": np.random.randint(0, 2, n),
        "meal_slot": np.random.randint(0, 4, n),
        "hour": np.random.randint(0, 24, n),
        "is_weekend": np.random.randint(0, 2, n),
        "is_festival_order": np.random.randint(0, 2, n),
        "is_rainy_weather": np.random.randint(0, 2, n),
        "rank_shown": np.random.randint(1, 10, n),
        "city_tier": np.random.randint(1, 3, n),
        "user_preference_drift": np.random.uniform(-0.1, 0.1, n),
        "meal_complete_score": np.random.randint(0, 5, n),
        "missing_drink": np.random.randint(0, 2, n),
        "missing_dessert": np.random.randint(0, 2, n),
    })
    return df


def test_engineer_features(sample_df):
    out = engineer_features(sample_df)
    assert "hour_sin" in out.columns
    assert "cart_completion_urgency" in out.columns
    assert "price_vs_cart_ratio" in out.columns
    assert "meal_gap_score" in out.columns
    assert "value_per_item" in out.columns


def test_load_and_split():
    path = "csao_ml_final.csv"
    if not os.path.exists(path):
        pytest.skip("csao_ml_final.csv not found")
    train_df, test_df = load_and_split(path)
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert (train_df["split"] < 2).all()
    assert (test_df["split"] == 2).all()


def test_build_pipeline(sample_df):
    train_df = sample_df[sample_df["split"] < 2].copy()
    test_df = sample_df[sample_df["split"] == 2].copy()
    if len(train_df) < 10 or len(test_df) < 5:
        train_df = sample_df.iloc[:80].copy()
        test_df = sample_df.iloc[80:].copy()
        train_df["split"] = 0
        test_df["split"] = 2
    X_train, X_test, y_train, y_test, feat_cols = build_pipeline(train_df, test_df)
    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert len(feat_cols) > 0
