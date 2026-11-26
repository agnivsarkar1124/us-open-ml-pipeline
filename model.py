"""
Match-win model: an equal-weight ensemble of logistic regression and a
monotonic-constrained HistGradientBoosting classifier over the leakage-free
features in features.py.

Training uses symmetric augmentation (every match is also added with the
players swapped and the label flipped), and prediction averages p(A beats B)
with 1 - p(B beats A), so the output never depends on argument order.

This exact pipeline is what evaluate.py backtests and what main.py serves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import MAIN_TOUR_LEVELS, FeatureBuilder

BO5_INTERACTIONS = ["elo_bo5", "eloh_bo5"]
MODEL_FEATURES = FeatureBuilder.FEATURES + ["bo5"] + BO5_INTERACTIONS
TRAIN_FROM_YEAR = 2014  # 2010–2013 only warm up the ratings

GBM_PARAMS = dict(
    max_iter=300, learning_rate=0.03, min_samples_leaf=100,
    l2_regularization=1.0, max_depth=4, random_state=42,
)


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bo5"] = (df["best_of"] == 5).astype(float)
    df["elo_bo5"] = df["elo_diff"] * df["bo5"]
    df["eloh_bo5"] = df["elo_hard_diff"] * df["bo5"]
    return df


def training_rows(rows: pd.DataFrame, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Main-tour matches (Slams, Masters, ATP 250/500, Finals) from TRAIN_FROM_YEAR, optionally before a date."""
    mask = (rows["year"] >= TRAIN_FROM_YEAR) & rows["tourney_level"].isin(MAIN_TOUR_LEVELS)
    if before is not None:
        mask &= rows["date"] < before
    return rows[mask]


def _flip(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c != "bo5":
            X[c] = -X[c]
    return X


class MatchModel:
    def __init__(self):
        mono = [FeatureBuilder.MONOTONIC.get(f, 1 if f in BO5_INTERACTIONS else 0)
                for f in MODEL_FEATURES]
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self.gbm = HistGradientBoostingClassifier(monotonic_cst=mono, **GBM_PARAMS)

    def fit(self, rows: pd.DataFrame) -> "MatchModel":
        X = rows[MODEL_FEATURES]
        y = rows["target"].to_numpy()
        X_aug = pd.concat([X, _flip(X)], ignore_index=True)
        y_aug = np.r_[y, 1 - y]
        self.lr.fit(X_aug.fillna(0), y_aug)   # LR needs no NaNs; 0 = "no difference"
        self.gbm.fit(X_aug, y_aug)            # GBM handles NaN natively
        return self

    def _p(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 * (self.lr.predict_proba(X.fillna(0))[:, 1] + self.gbm.predict_proba(X)[:, 1])

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """P(player A wins), symmetric in A/B."""
        X = rows[MODEL_FEATURES]
        return 0.5 * (self._p(X) + 1 - self._p(_flip(X)))


def elo_expected(elo_diff) -> np.ndarray:
    return 1.0 / (1.0 + 10 ** (-np.asarray(elo_diff, dtype=float) / 400.0))
