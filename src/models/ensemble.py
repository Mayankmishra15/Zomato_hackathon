"""
Weighted ensemble of XGBoost and LightGBM rankers.
"""
import numpy as np
import joblib
import os

try:
    from src.config import MODEL_DIR
except ImportError:
    from config import MODEL_DIR


def load_ensemble(xgb_weight=0.4, lgb_weight=0.6):
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    return xgb_model, lgb_model, xgb_weight, lgb_weight


def predict_proba_ensemble(X, xgb_weight=0.4, lgb_weight=0.6):
    xgb_model, lgb_model, _, _ = load_ensemble(xgb_weight, lgb_weight)
    xgb_scores = xgb_model.predict_proba(X)[:, 1]
    lgb_scores = lgb_model.predict_proba(X)[:, 1]
    return xgb_weight * xgb_scores + lgb_weight * lgb_scores
