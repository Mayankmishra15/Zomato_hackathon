"""
CLI evaluation script.
Usage: python evaluate.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline.preprocessor import load_and_split, build_pipeline
from src.models.lgbm_ranker import load
from src.evaluation.metrics import full_report
from src.evaluation.business_metrics import project_business_impact
from src.config import DATA_PATH, MODEL_DIR


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(os.path.join(MODEL_DIR, "lgbm_model.pkl")):
        print("❌ Model not found. Run: python train.py --model lgbm")
        sys.exit(1)

    train_df, test_df = load_and_split()
    X_train, X_test, y_train, y_test, _ = build_pipeline(train_df, test_df)

    model = load()
    scores = model.predict_proba(X_test)[:, 1]
    report = full_report(y_test, scores)
    print(json.dumps(report, indent=2))
    impact = project_business_impact(report.get("Precision@8", 0.3))
    print("\nBusiness Impact:", json.dumps(impact, indent=2))


if __name__ == "__main__":
    main()
