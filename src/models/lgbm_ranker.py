"""
LightGBM classifier used as a pointwise ranker. Faster than XGBoost for inference.
This is the PRIMARY production model due to lower latency.
"""
import os
import lightgbm as lgb
import joblib

try:
    from src.config import LGBM_PARAMS, MODEL_DIR
except ImportError:
    from config import LGBM_PARAMS, MODEL_DIR


def train(X_train, y_train, X_val=None, y_val=None):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)] if (X_val is not None and y_val is not None) else []
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)] if (X_val is not None and y_val is not None) else None,
        callbacks=callbacks
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    return model


def load():
    return joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
