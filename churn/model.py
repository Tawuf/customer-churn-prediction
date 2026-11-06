"""Model training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline(kind: str = "logreg", seed: int = 0) -> Pipeline:
    """Preprocessing + classifier in one scikit-learn pipeline."""
    if kind == "logreg":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
            ]
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif kind == "gbm":
        pre = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
            ]
        )
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=seed,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("clf", clf)])


@dataclass
class Evaluation:
    name: str
    cv_auc_mean: float
    cv_auc_std: float
    test_auc: float
    test_pr_auc: float
    test_brier: float
    model: Pipeline
    test_proba: np.ndarray


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def evaluate(kind: str, X_train, X_test, y_train, y_test, seed: int = 0) -> Evaluation:
    """5-fold CV on the training set, then a single fit scored on the hold-out set."""
    pipe = build_pipeline(kind, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return Evaluation(
        name=kind,
        cv_auc_mean=float(scores.mean()),
        cv_auc_std=float(scores.std()),
        test_auc=float(roc_auc_score(y_test, proba)),
        test_pr_auc=float(average_precision_score(y_test, proba)),
        test_brier=float(brier_score_loss(y_test, proba)),
        model=pipe,
        test_proba=proba,
    )


def feature_importance(model: Pipeline, X_test, y_test, seed: int = 0) -> pd.Series:
    """Permutation importance (drop in ROC-AUC when a feature is shuffled)."""
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
