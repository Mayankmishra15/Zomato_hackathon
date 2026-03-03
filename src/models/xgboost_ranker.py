"""
XGBoost binary classifier used as a pointwise ranker.
Scores each candidate item with P(acceptance). Items ranked by score descending.
"""
import os
import xgboost as xgb
import joblib

try:
    from src.config import XGB_PARAMS, MODEL_DIR
except ImportError:
    from config import XGB_PARAMS, MODEL_DIR


def train(X_train, y_train, X_val=None, y_val=None):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    return model


def load():
    return joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
