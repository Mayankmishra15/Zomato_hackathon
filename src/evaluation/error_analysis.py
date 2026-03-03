"""
Segment-level error analysis for model performance breakdown.
"""
import pandas as pd
import numpy as np


def segment_metrics(df, y_true, y_pred, segment_col):
    """Compute metrics per segment."""
    results = []
    for seg in df[segment_col].dropna().unique():
        mask = df[segment_col] == seg
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 10:
            continue
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:
            auc = 0.0
        results.append({
            "segment": seg,
            "n": mask.sum(),
            "auc": round(auc, 4),
            "mean_pred": round(float(np.mean(yp)), 4),
        })
    return pd.DataFrame(results)
