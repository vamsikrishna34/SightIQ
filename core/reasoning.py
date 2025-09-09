from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from utils import MODELS_DIR, logger

MODEL_PATH = MODELS_DIR / "xgb_model.json"

ESSENTIAL_FEATURES = [
    "obj_count", "avg_conf", "mean_box_area_pct", "max_box_area_pct", "min_box_area_pct",
    "min_center_dist", "mean_center_x", "mean_center_y", "density_per_mpx"
]

@dataclass
class ReasoningOutput:
    y_pred: float
    y_label: Optional[int]
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    feature_vector: np.ndarray

class Reasoner:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[xgb.Booster] = None
        if model_path.exists():
            self._load_model()
        else:
            logger.warning("XGBoost model not found. Using heuristic fallback until you train or provide one.")

    def _load_model(self):
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        logger.info(f"Loaded XGBoost model from {self.model_path}")

    def save_model(self):
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save_model(str(self.model_path))
        logger.info(f"Saved XGBoost model to {self.model_path}")

    def _to_feature_vector(self, features: dict) -> Tuple[np.ndarray, List[str]]:
        names = list(ESSENTIAL_FEATURES)
        class_feats = sorted(k for k in features.keys() if k.startswith("class_") and k.endswith("_count"))
        names += class_feats
        vec = np.array([features.get(k, 0.0) for k in names], dtype=float)[None, :]
        return vec, names

    def predict(self, features: dict) -> ReasoningOutput:
        vec, names = self._to_feature_vector(features)

        if self.model is None:
            score = 0.0
            score += 0.4 * min(1.0, float(features.get("obj_count", 0)) / 10.0)
            score += 0.4 * float(features.get("avg_conf", 0.0))
            score += 0.2 * min(1.0, float(features.get("density_per_mpx", 0.0)) / 50.0)
            shap_vals = np.zeros(vec.shape[1], dtype=float)
            return ReasoningOutput(float(score), None, shap_vals, 0.0, names, vec[0])

        dm = xgb.DMatrix(vec, feature_names=names)
        y_pred = float(self.model.predict(dm)[0])

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(pd.DataFrame(vec, columns=names))[0]
        expected_value = explainer.expected_value
        try:
            expected_value = float(expected_value)
        except Exception:
            expected_value = float(np.mean(shap_values))
        return ReasoningOutput(y_pred, None, shap_values, expected_value, names, vec[0])

    def train(self, df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, seed: int = 42):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        y = df[label_col].values
        X_cols = [c for c in df.columns if c != label_col]
        X = df[X_cols].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_cols)
        dtest  = xgb.DMatrix(X_test, label=y_test, feature_names=X_cols)

        params = dict(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            nthread=4
        )
        model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtrain, "train"), (dtest, "valid")], early_stopping_rounds=30, verbose_eval=False)
        self.model = model
        self.save_model()

        try:
            yhat = model.predict(dtest)
            auc = roc_auc_score(y_test, yhat)
            logger.info(f"Trained XGBoost | AUC={auc:.3f} | features={len(X_cols)}")
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")