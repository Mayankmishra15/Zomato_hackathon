"""
Real-time inference engine targeting <200ms latency.

Design:
- Model loaded once at startup (singleton pattern)
- Feature transforms pre-fitted and cached
- Batch scoring even for single requests (vectorized)
- Returns top-N ranked items with scores + explanations
"""
import time
import os
import numpy as np
import pandas as pd
import joblib

try:
    from src.config import MODEL_DIR, ENCODER_DIR
except ImportError:
    from config import MODEL_DIR, ENCODER_DIR


class CSAOPredictor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
        enc_path = os.path.join(ENCODER_DIR, "ordinal_encoder.pkl")
        scaler_path = os.path.join(ENCODER_DIR, "standard_scaler.pkl")
        feat_path = os.path.join(ENCODER_DIR, "feature_cols.pkl")
        if not all(os.path.exists(p) for p in [model_path, enc_path, scaler_path, feat_path]):
            raise FileNotFoundError(
                "Model artifacts not found. Run: python train.py --model lgbm"
            )
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(enc_path)
        self.scaler = joblib.load(scaler_path)
        self.feat_cols = joblib.load(feat_path)

    def predict(self, candidates_df: pd.DataFrame, top_n: int = 8):
        """
        candidates_df: DataFrame where each row = one candidate add-on item
                       with all required feature columns.
        Returns: (ranked_df, scores, latency_ms)
        """
        t0 = time.time()
        from src.data_pipeline.preprocessor import (
            engineer_features,
            CATEGORICAL_COLS,
            NUMERIC_COLS,
            EXTRA_NUMERIC,
        )
        df = engineer_features(candidates_df.copy())

        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns and c in self.feat_cols]
        num_cols = [c for c in NUMERIC_COLS + EXTRA_NUMERIC if c in df.columns and c in self.feat_cols]

        if cat_cols:
            df[cat_cols] = self.encoder.transform(df[cat_cols].fillna("missing").astype(str))
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols].fillna(0))

        # Ensure feature order matches
        avail_feats = [c for c in self.feat_cols if c in df.columns]
        X = df[avail_feats].reindex(columns=self.feat_cols, fill_value=0)

        scores = self.model.predict_proba(X)[:, 1]

        ranked_idx = np.argsort(scores)[::-1][:top_n]
        ranked_df = candidates_df.iloc[ranked_idx].copy()
        ranked_df["acceptance_probability"] = scores[ranked_idx]

        latency_ms = (time.time() - t0) * 1000
        return ranked_df, scores[ranked_idx], latency_ms
