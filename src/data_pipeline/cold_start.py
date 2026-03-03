"""
Cold-start fallback for csao_ml_final: uses city_tier + meal_slot + item_popularity.
"""
import pandas as pd

CITY_MEAL_DEFAULTS = {
    (1, 1): [3, 2, 0], (1, 2): [3, 2, 0], (1, 3): [3, 2, 0],
    (2, 1): [3, 2, 0], (2, 2): [3, 2, 0], (2, 3): [3, 2, 0],
}


def cold_start_score(candidates_df, city_tier=1, meal_slot=2):
    df = candidates_df.copy()
    pop = df.get("item_popularity", pd.Series(0.5, index=df.index))
    df["cold_start_score"] = pop
    bs = df.get("is_bestseller", pd.Series(0, index=df.index))
    df["cold_start_score"] = df["cold_start_score"] + bs * 0.15
    preferred_cats = CITY_MEAL_DEFAULTS.get((city_tier, meal_slot), [3, 2, 0])
    item_cat = df.get("item_category", pd.Series(3, index=df.index))

    def cat_boost(c):
        try:
            return 0.2 if c in preferred_cats else 0
        except (TypeError, ValueError):
            return 0

    df["cold_start_score"] = df["cold_start_score"] + item_cat.apply(cat_boost)
    return df.sort_values("cold_start_score", ascending=False)
