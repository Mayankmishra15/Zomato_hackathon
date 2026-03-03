"""
CLI training script.
Usage: python train.py --model all
"""
import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline.preprocessor import load_and_split, build_pipeline
from src.evaluation.metrics import full_report
from src.evaluation.business_metrics import project_business_impact
from src.config import DATA_PATH, SHAP_DIR, MODEL_DIR


def main(args):
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Ensure csao_ml_final.csv is in project root.")
        sys.exit(1)

    print("Loading and preprocessing data...")
    train_df, test_df = load_and_split()
    X_train, X_test, y_train, y_test, feat_cols = build_pipeline(train_df, test_df)

    results = {}

    if args.model in ["xgb", "all"]:
        from src.models import xgboost_ranker
        print("Training XGBoost...")
        xgb_model = xgboost_ranker.train(X_train, y_train, X_test, y_test)
        xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
        results["XGBoost"] = full_report(y_test, xgb_scores)
        print("XGBoost:", results["XGBoost"])

    if args.model in ["lgbm", "all"]:
        from src.models import lgbm_ranker
        print("Training LightGBM...")
        lgbm_model = lgbm_ranker.train(X_train, y_train, X_test, y_test)
        lgbm_scores = lgbm_model.predict_proba(X_test)[:, 1]
        results["LightGBM"] = full_report(y_test, lgbm_scores)
        print("LightGBM:", results["LightGBM"])

        if args.shap:
            print("Computing SHAP values...")
            try:
                import shap
                import joblib
                explainer = shap.TreeExplainer(lgbm_model)
                X_sample = X_test[:min(500, len(X_test))]
                shap_values = explainer.shap_values(X_sample)
                os.makedirs(SHAP_DIR, exist_ok=True)
                joblib.dump(
                    {"shap_values": shap_values, "X_test": X_sample, "feature_names": feat_cols},
                    os.path.join(SHAP_DIR, "shap_values.pkl")
                )
                print("SHAP values saved")
            except Exception as e:
                print(f"SHAP computation skipped: {e}")

    if args.model in ["baseline", "all"]:
        from src.models import baseline
        test_df_reset = test_df.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        base_scores = baseline.score(test_df_reset)
        results["Baseline"] = full_report(y_test_reset, base_scores)
        print("Baseline:", results["Baseline"])

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to artifacts/eval_results.json")

    if "LightGBM" in results:
        p8 = results["LightGBM"].get("Precision@8", 0.3)
        impact = project_business_impact(p8)
        print("\nBusiness Impact Projection:")
        for k, v in impact.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["xgb", "lgbm", "baseline", "all"])
    parser.add_argument("--shap", action="store_true", help="Compute SHAP values after LGBM training")
    args = parser.parse_args()
    main(args)
