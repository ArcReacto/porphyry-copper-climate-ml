"""Frozen XGBoost classifier for matched rebuttal experiments."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PARAMETERS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 20260622,
    "n_jobs": 4,
}


def make_xgb_model(y_train) -> Pipeline:
    labels = np.asarray(y_train, dtype=int)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if not positive or not negative:
        raise ValueError("XGBoost requires both classes in every training fold")
    model = XGBClassifier(**PARAMETERS, scale_pos_weight=negative / positive)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
