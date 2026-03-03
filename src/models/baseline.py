"""
Popularity-based baseline: ranks by item_popularity.
For csao_ml_final dataset.
"""
import pandas as pd


def score(df):
    """Return probability-like scores for ranking."""
    pop = df.get("item_popularity", pd.Series(0.5, index=df.index))
    return pop


def predict_top_k(df, k=8):
    scores = score(df)
    top_idx = scores.nlargest(k).index
    return df.loc[top_idx], scores.loc[top_idx]
