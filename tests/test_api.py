"""Tests for FastAPI."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Avoid loading predictor at import (may fail if no model)
import api.main as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "model" in data


def test_recommend(client):
    candidates = [
        {
            "item_category": 3,
            "item_price": 120,
            "is_veg": 1,
            "is_bestseller": 1,
            "item_popularity": 0.7,
            "cart_value_at_show": 300,
            "cart_size_at_show": 2,
            "meal_slot": 3,
            "city_tier": 1,
            "user_segment": 1,
        }
        for _ in range(5)
    ]
    r = client.post("/recommend", json={"candidates": candidates, "top_n": 3})
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert "latency_ms" in data
    assert "model_version" in data
    assert len(data["recommendations"]) <= 3
