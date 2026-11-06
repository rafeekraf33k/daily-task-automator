"""Machine-learning signal filter.

A gradient-boosting classifier predicts whether price will be higher
`horizon` bars from now. It is trained with a chronological split (no
shuffling) so the reported accuracy is an honest out-of-sample estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .indicators import FEATURE_COLUMNS, add_indicators


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype(float)
    label[future.isna()] = np.nan
    return label


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    baseline: float  # accuracy of always predicting the majority class

    def __str__(self) -> str:
        return (
            f"train={self.n_train} test={self.n_test} "
            f"accuracy={self.accuracy:.3f} (baseline {self.baseline:.3f}) auc={self.auc:.3f}"
        )


class SignalModel:
    def __init__(self, clf=None):
        self.clf = clf

    @property
    def ready(self) -> bool:
        return self.clf is not None

    def train(self, bars: pd.DataFrame, strategy_cfg, test_frac: float = 0.25) -> TrainReport:
        feats = add_indicators(bars, strategy_cfg)
        feats["label"] = make_labels(feats, strategy_cfg.ml_horizon)
        data = feats.dropna(subset=FEATURE_COLUMNS + ["label"])
        data = data.iloc[strategy_cfg.trend_ema:]  # drop indicator warm-up
        if len(data) < 300:
            raise ValueError(f"Not enough data to train ({len(data)} rows, need >= 300)")

        split = int(len(data) * (1 - test_frac))
        # Leave a gap of `horizon` bars so test labels don't leak into training
        train = data.iloc[: split - strategy_cfg.ml_horizon]
        test = data.iloc[split:]

        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            l2_regularization=1.0, random_state=42,
        )
        clf.fit(train[FEATURE_COLUMNS], train["label"])
        proba = clf.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        y = test["label"].to_numpy()
        acc = accuracy_score(y, proba > 0.5)
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
        baseline = max(y.mean(), 1 - y.mean())

        # Refit on everything for live use
        clf.fit(data[FEATURE_COLUMNS], data["label"])
        self.clf = clf
        return TrainReport(len(train), len(test), acc, auc, baseline)

    def prob_up(self, features_row: pd.DataFrame) -> float:
        if not self.ready:
            raise RuntimeError("Model is not trained/loaded")
        return float(self.clf.predict_proba(features_row[FEATURE_COLUMNS])[:, 1][0])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path: str | Path) -> "SignalModel":
        return cls(joblib.load(path))
