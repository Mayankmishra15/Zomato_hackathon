# All hyperparameters, paths, and constants
import os

DATA_PATH = "csao_ml_final.csv"
TRAIN_PATH = "data/processed/train.parquet"
TEST_PATH = "data/processed/test.parquet"
MODEL_DIR = "artifacts/models/"
ENCODER_DIR = "artifacts/encoders/"
SHAP_DIR = "artifacts/shap/"

TARGET = "label"
TOP_K = [5, 8, 10]

# Temporal split: last 20% of rows as test (simulate real deployment)
TEST_SIZE = 0.2

# XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 1.5,   # handle class imbalance
    "random_state": 42,
}

# LightGBM params
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}

# Latency budget (ms)
LATENCY_BUDGET_MS = 200

# Meal slot mapping
MEAL_SLOTS = {
    0: "Early Morning",
    1: "Breakfast",
    2: "Brunch",
    3: "Lunch",
    4: "Evening Snack",
    5: "Dinner",
    6: "Late Night"
}

# Category mapping
ITEM_CATEGORIES = {
    0: "Beverages",
    1: "Desserts",
    2: "Sides",
    3: "Main Course",
    4: "Starters",
    5: "Breads",
    6: "Combos"
}
