"""
services/model_trainer.py
Trains RandomForest, XGBoost, LightGBM models for each betting market.
Saves trained models to /models directory.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[ModelTrainer] XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[ModelTrainer] LightGBM not installed")


MODELS_DIR = os.getenv("MODELS_DIR", "models")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

MARKETS = [
    "target_over15",
    "target_over25",
    "target_over35",
    "target_under25",
    "target_btts",
    "target_over95c",
    "target_over35k",
]


class ModelTrainer:
    """Trains and saves ML models for all betting markets."""

    def __init__(self):
        self.results = {}

    def _get_models(self) -> Dict:
        """Return dict of models to train."""
        models = {
            "logistic_regression": CalibratedClassifierCV(
                LogisticRegression(C=0.1, max_iter=1000),
                method="isotonic", cv=3
            ),
            "random_forest": CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200, max_depth=8,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
                ),
                method="isotonic", cv=3
            ),
        }
        if HAS_XGB:
            models["xgboost"] = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, use_label_encoder=False,
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                ),
                method="isotonic", cv=3
            )
        if HAS_LGB:
            models["lightgbm"] = CalibratedClassifierCV(
                lgb.LGBMClassifier(
                    n_estimators=300, num_leaves=63,
                    learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, n_jobs=-1,
                ),
                method="isotonic", cv=3
            )
        return models

    def train_market(self, X: np.ndarray, y: np.ndarray,
                     feature_cols: List[str], market: str) -> Dict:
        """Train all models for one market, return best model results."""
        if len(X) < 100:
            print(f"[ModelTrainer] Not enough data for {market}: {len(X)} rows")
            return {}

        # Time-series split (no future leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2:
            print(f"[ModelTrainer] Only one class in {market}, skipping")
            return {}

        models = self._get_models()
        results = {}
        best_model = None
        best_score = 999

        for name, model in models.items():
            try:
                print(f"[ModelTrainer] Training {name} for {market}...")
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)

                acc     = accuracy_score(y_test, preds)
                ll      = log_loss(y_test, probs)
                auc     = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
                brier   = brier_score_loss(y_test, probs)

                results[name] = {
                    "accuracy": round(acc, 4),
                    "log_loss": round(ll, 4),
                    "roc_auc":  round(auc, 4),
                    "brier":    round(brier, 4),
                    "n_train":  len(X_train),
                    "n_test":   len(X_test),
                }

                print(f"  {name}: acc={acc:.3f} ll={ll:.3f} auc={auc:.3f}")

                if ll < best_score:
                    best_score = ll
                    best_model = (name, model)

            except Exception as e:
                print(f"[ModelTrainer] {name} failed for {market}: {e}")

        # Save best model
        if best_model:
            model_name, model_obj = best_model
            path = os.path.join(MODELS_DIR, f"{market}_{model_name}.pkl")
            with open(path, "wb") as f:
                pickle.dump({
                    "model": model_obj,
                    "feature_cols": feature_cols,
                    "market": market,
                    "best_model_name": model_name,
                    "trained_at": datetime.utcnow().isoformat(),
                    "metrics": results[model_name],
                }, f)
            print(f"[ModelTrainer] Saved best model: {path}")
            results["best"] = model_name

        return results

    def train_all(self, df: pd.DataFrame, feature_engineer) -> Dict:
        """Train models for all markets."""
        all_results = {}

        for market in MARKETS:
            if market not in df.columns:
                print(f"[ModelTrainer] Missing target: {market}")
                continue

            X, y, feature_cols = feature_engineer.prepare_for_training(df, market)
            if len(X) < 100:
                print(f"[ModelTrainer] Skipping {market}: only {len(X)} rows")
                continue

            print(f"\n[ModelTrainer] === Training market: {market} ===")
            results = self.train_market(X, y, feature_cols, market)
            all_results[market] = results

        # Save summary
        summary_path = os.path.join(MODELS_DIR, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "trained_at": datetime.utcnow().isoformat(),
                "markets": all_results,
                "n_rows": len(df),
            }, f, indent=2)

        print(f"\n[ModelTrainer] Training complete. Results saved to {summary_path}")
        self.results = all_results
        return all_results

    def get_summary(self) -> Dict:
        path = os.path.join(MODELS_DIR, "training_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}


if __name__ == "__main__":
    import sqlite3
    from services.dataset_builder import DatasetBuilder
    from services.feature_engineering import FeatureEngineer

    builder = DatasetBuilder()
    df_raw  = builder.get_dataframe()

    if len(df_raw) < 100:
        print("Not enough data. Run dataset_builder.py first.")
    else:
        engineer = FeatureEngineer()
        df_feat  = engineer.transform(df_raw)
        trainer  = ModelTrainer()
        results  = trainer.train_all(df_feat, engineer)
        print(json.dumps(results, indent=2))
