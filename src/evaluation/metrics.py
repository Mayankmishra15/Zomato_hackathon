"""
All offline evaluation metrics.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auc_score(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def precision_at_k(y_true, y_score, k):
    """Precision@K averaged across queries (here: each row is a candidate)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_k_idx]))


def recall_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0
    hits = np.sum(y_true[top_k_idx])
    return float(hits / total_relevant)


def dcg_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    gains = y_true[top_k_idx]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ideal_order = np.argsort(y_true)[::-1]
    ideal_gains = y_true[ideal_order][:k]
    ideal_dcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))
    if ideal_dcg == 0:
        return 0.0
    return float(dcg_at_k(y_true, y_score, k) / ideal_dcg)


def full_report(y_true, y_score, ks=None):
    if ks is None:
        try:
            from src.config import TOP_K
        except ImportError:
            from config import TOP_K
        ks = TOP_K
    report = {
        "AUC": round(auc_score(y_true, y_score), 4),
        "AP": round(average_precision_score(y_true, y_score), 4),
    }
    for k in ks:
        report[f"Precision@{k}"] = round(precision_at_k(y_true, y_score, k), 4)
        report[f"Recall@{k}"] = round(recall_at_k(y_true, y_score, k), 4)
        report[f"NDCG@{k}"] = round(ndcg_at_k(y_true, y_score, k), 4)
    return report
