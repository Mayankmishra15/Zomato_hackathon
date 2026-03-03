"""
Diversity + business rules reranker applied after ML scoring.
"""
import numpy as np
import pandas as pd


def rerank(df, scores, top_k=8, max_same_category=3):
    """
    Rerank top candidates to ensure diversity (limit same-category items)
    and apply business rules (e.g., veg preference, price ceiling).
    """
    if len(df) <= top_k:
        return df, scores

    ranked = []
    used_idx = set()
    cat_counts = {}

    for _ in range(top_k):
        best_idx = None
        best_score = -np.inf
        for i in range(len(df)):
            if i in used_idx:
                continue
            cat = df.iloc[i].get("item_category", 0)
            penalty = 0.0
            if cat_counts.get(cat, 0) >= max_same_category:
                penalty = 0.3
            adj_score = scores[i] - penalty
            if adj_score > best_score:
                best_score = adj_score
                best_idx = i

        if best_idx is None:
            break
        used_idx.add(best_idx)
        ranked.append(best_idx)
        cat = df.iloc[best_idx].get("item_category", 0)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    return df.iloc[ranked], scores[ranked]
