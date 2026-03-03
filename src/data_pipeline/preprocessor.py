"""
Feature engineering pipeline for csao_ml_final dataset.
- Uses provided 'split' column for train/test (split < 2 = train, split == 2 = test)
- Engineer derived features from available columns only
"""
import pandas as pd
import numpy as np
import joblib
import os

try:
    from src.config import DATA_PATH, TARGET, ENCODER_DIR
except ImportError:
    from config import DATA_PATH, TARGET, ENCODER_DIR

# Columns in csao_ml_final.csv
CATEGORICAL_COLS = ["user_segment", "city", "item_category", "meal_slot", "city_tier"]
NUMERIC_COLS = [
    "cart_size_at_show", "cart_value_at_show", "discount_on_order",
    "has_main_in_cart", "has_drink_in_cart", "has_dessert_in_cart", "has_side_in_cart",
    "item_price", "item_popularity", "is_complement_category", "is_bestseller",
    "is_ghost_kitchen", "hour", "is_weekend", "is_festival_order", "is_rainy_weather",
    "rank_shown", "user_preference_drift", "meal_complete_score",
    "missing_drink", "missing_dessert"
]


def load_and_split(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)
    # Use provided split: 0,1 = train, 2 = test
    train_df = df[df["split"] < 2].copy()
    test_df = df[df["split"] == 2].copy()
    return train_df, test_df


def engineer_features(df):
    df = df.copy()
    # Cyclical time encoding
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Cart completion urgency
    meal = df.get("meal_complete_score", pd.Series(0.5, index=df.index))
    df["cart_completion_urgency"] = 1 - np.clip(meal / 4.0, 0, 1)  # scale 0-4 to 0-1
    # Price ratio vs cart value
    cart_val = df.get("cart_value_at_show", pd.Series(200, index=df.index))
    item_price = df.get("item_price", pd.Series(100, index=df.index))
    df["price_vs_cart_ratio"] = item_price / (cart_val + 1)
    # Meal gap: items missing from a complete meal
    has_main = df.get("has_main_in_cart", pd.Series(0, index=df.index))
    has_side = df.get("has_side_in_cart", pd.Series(0, index=df.index))
    has_drink = df.get("has_drink_in_cart", pd.Series(0, index=df.index))
    has_dessert = df.get("has_dessert_in_cart", pd.Series(0, index=df.index))
    df["meal_gap_score"] = (1 - has_main) + (1 - has_side) + (1 - has_drink) + (1 - has_dessert)
    # Cart value per item
    cart_size = df.get("cart_size_at_show", pd.Series(1, index=df.index))
    df["value_per_item"] = cart_val / (cart_size + 1)
    return df


EXTRA_NUMERIC = [
    "hour_sin", "hour_cos", "cart_completion_urgency",
    "price_vs_cart_ratio", "meal_gap_score", "value_per_item"
]


def build_pipeline(train_df, test_df):
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    all_cat = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    all_num = [c for c in NUMERIC_COLS + EXTRA_NUMERIC if c in train_df.columns]

    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = train_df[all_cat].fillna("missing").astype(str)
    test_cat = test_df[all_cat].fillna("missing").astype(str)
    train_df[all_cat] = enc.fit_transform(train_cat)
    test_df[all_cat] = enc.transform(test_cat)

    scaler = StandardScaler()
    train_num = train_df[all_num].fillna(0)
    test_num = test_df[all_num].fillna(0)
    train_df[all_num] = scaler.fit_transform(train_num)
    test_df[all_num] = scaler.transform(test_num)

    feature_cols = all_cat + all_num
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    os.makedirs(ENCODER_DIR, exist_ok=True)
    joblib.dump(enc, os.path.join(ENCODER_DIR, "ordinal_encoder.pkl"))
    joblib.dump(scaler, os.path.join(ENCODER_DIR, "standard_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(ENCODER_DIR, "feature_cols.pkl"))

    return X_train, X_test, y_train, y_test, feature_cols
