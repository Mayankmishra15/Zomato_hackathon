"""Pydantic schemas for csao_ml_final dataset."""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class CandidateItem(BaseModel):
    user_segment: int = 5
    city: int = 9
    cart_size_at_show: int = 3
    cart_value_at_show: float = 345.0
    discount_on_order: float = 0.1
    has_main_in_cart: int = 1
    has_drink_in_cart: int = 1
    has_dessert_in_cart: int = 0
    has_side_in_cart: int = 1
    item_category: int = 2
    item_price: float = 187.0
    item_popularity: float = 0.245
    is_complement_category: int = 1
    is_bestseller: int = 0
    is_ghost_kitchen: int = 0
    meal_slot: int = 2
    hour: int = 0
    is_weekend: int = 0
    is_festival_order: int = 1
    is_rainy_weather: int = 0
    rank_shown: int = 1
    city_tier: int = 2
    user_preference_drift: float = 0.0
    meal_complete_score: float = 3.0
    missing_drink: int = 0
    missing_dessert: int = 0

    class Config:
        extra = "allow"


class RecommendRequest(BaseModel):
    candidates: List[CandidateItem]
    top_n: int = 8
    user_id: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    model_version: str
