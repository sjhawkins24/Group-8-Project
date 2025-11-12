"""Utilities for training and loading the hybrid AP ranking model.

This module centralises all data preparation, model training, and persistence
logic so other components (CLI tools, Streamlit UI, batch jobs) can reuse the
same pipelines without duplicating feature engineering code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path("artifacts")
REGRESSOR_PATH = ARTIFACT_DIR / "rank_change_regressor.pkl"
CLASSIFIER_PATH = ARTIFACT_DIR / "rank_direction_classifier.pkl"
FEATURE_METADATA_PATH = ARTIFACT_DIR / "feature_metadata.json"

DATA_MERGED_DEFAULT = Path("03 - Cleaned Data Space") / "mergedTrainingData.csv"
ELO_SEASON_DEFAULT = Path("04 - Elo Space") / "elo_ratings_by_season.csv"

FEATURE_COLUMNS: List[str] = [
    "prev_ap_rank",
    "team_elo",
    "opponent_elo",
    "elo_diff",
    "elo_advantage",
    "is_win",
    "margin",
    "home_bool",
    "opp_ranked",
    "win_streak",
    "avg_margin_3games",
]


@dataclass
class HybridDataset:
    """Container for the processed dataset used by the hybrid model."""

    features: pd.DataFrame
    reg_target: pd.Series
    cls_target: pd.Series
    raw: pd.DataFrame


def _normalise_boolean(value: object) -> int:
    """Normalise TRUE/FALSE style values to integers."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().upper()
    return int(text in {"TRUE", "T", "YES", "1", "Y"})


def _load_sources(
    merged_path: Path = DATA_MERGED_DEFAULT,
    elo_path: Path = ELO_SEASON_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.read_csv(merged_path)
    elo = pd.read_csv(elo_path)
    return merged, elo


def build_dataset(
    merged_path: Path = DATA_MERGED_DEFAULT,
    elo_path: Path = ELO_SEASON_DEFAULT,
) -> HybridDataset:
    """Prepare the modelling dataset with all engineered features."""

    merged_df, elo_ratings = _load_sources(merged_path, elo_path)

    # Merge Elo ratings for teams and opponents (per season).
    merged = merged_df.merge(
        elo_ratings.rename(columns={"Elo": "team_elo"}),
        on=["Team", "season"],
        how="left",
    ).merge(
        elo_ratings.rename(columns={"Team": "opponent", "Elo": "opponent_elo"}),
        on=["opponent", "season"],
        how="left",
    )

    # Core feature engineering (mirrors the established workflow).
    merged = merged.dropna(subset=["AP_rank", "team_elo"])
    merged["is_win"] = (merged["win_loss"].astype(str).str.upper() == "W").astype(int)
    merged["margin"] = merged["points_scored"].astype(float) - merged["points_allowed"].astype(float)
    merged["home_bool"] = merged["home_game"].apply(_normalise_boolean)
    merged = merged.sort_values(["season", "week", "Team"]).reset_index(drop=True)

    merged["prev_ap_rank"] = merged.groupby("Team")["AP_rank"].shift(1)
    merged["rank_change"] = merged["prev_ap_rank"] - merged["AP_rank"]

    merged["elo_diff"] = merged["team_elo"] - merged["opponent_elo"].fillna(merged["team_elo"].mean())
    merged["elo_advantage"] = (merged["elo_diff"] > 0).astype(int)
    merged["opp_ranked"] = (merged["opponent_rank"].fillna(0) > 0).astype(int)

    merged["win_streak"] = merged.groupby("Team")["is_win"].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    merged["avg_margin_3games"] = merged.groupby("Team")["margin"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    usable = merged[merged["prev_ap_rank"].notna()].copy()

    features = usable[FEATURE_COLUMNS].copy()
    features = features.fillna(features.mean(numeric_only=True))

    reg_target = usable["rank_change"].astype(float)
    usable["dir"] = pd.cut(reg_target, bins=[-99, -2, 2, 99], labels=["down", "flat", "up"])
    cls_target = usable["dir"].dropna()
    valid_mask = cls_target.index

    features = features.loc[valid_mask]
    reg_target = reg_target.loc[valid_mask]

    return HybridDataset(features=features, reg_target=reg_target, cls_target=cls_target, raw=usable.loc[valid_mask])


def train_models(dataset: HybridDataset) -> Dict[str, Pipeline]:
    """Train the regression and classification pipelines."""

    num_cols = dataset.features.columns.tolist()
    transformer = ColumnTransformer([("num", StandardScaler(), num_cols)])

    reg_model = Pipeline([("pre", transformer), ("lin", LinearRegression())])
    cls_model = Pipeline([
        ("pre", transformer),
        ("logit", LogisticRegression(max_iter=300, random_state=42))
    ])

    reg_model.fit(dataset.features, dataset.reg_target)
    cls_model.fit(dataset.features, dataset.cls_target)

    return {"regressor": reg_model, "classifier": cls_model}


def evaluate_time_series(dataset: HybridDataset) -> Dict[str, Tuple[float, float]]:
    """Run 5-fold time-series CV to benchmark model performance."""
    tscv = TimeSeriesSplit(n_splits=5)
    reg_errors: List[float] = []
    clf_scores: List[float] = []

    num_cols = dataset.features.columns.tolist()
    transformer = ColumnTransformer([("num", StandardScaler(), num_cols)])

    reg_model = Pipeline([("pre", transformer), ("lin", LinearRegression())])
    cls_model = Pipeline([
        ("pre", transformer),
        ("logit", LogisticRegression(max_iter=300, random_state=42))
    ])

    for train_idx, test_idx in tscv.split(dataset.features):
        X_train, X_test = dataset.features.iloc[train_idx], dataset.features.iloc[test_idx]
        y_reg_train, y_reg_test = dataset.reg_target.iloc[train_idx], dataset.reg_target.iloc[test_idx]
        y_cls_train, y_cls_test = dataset.cls_target.iloc[train_idx], dataset.cls_target.iloc[test_idx]

        reg_model.fit(X_train, y_reg_train)
        cls_model.fit(X_train, y_cls_train)

        reg_pred = reg_model.predict(X_test)
        clf_pred = cls_model.predict(X_test)

        reg_errors.append(np.mean(np.abs(y_reg_test - reg_pred)))
        clf_scores.append(np.mean(clf_pred == y_cls_test))

    return {
        "reg_mae": (float(np.mean(reg_errors)), float(np.std(reg_errors))),
        "clf_acc": (float(np.mean(clf_scores)), float(np.std(clf_scores))),
    }


def save_models(models: Dict[str, Pipeline]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(models["regressor"], REGRESSOR_PATH)
    joblib.dump(models["classifier"], CLASSIFIER_PATH)
    with open("regression_model.pkl", 'wb') as file:
            pickle.dump(models["regressor"], file)
    meta = {
        "features": FEATURE_COLUMNS,
    }
    import json

    FEATURE_METADATA_PATH.write_text(json.dumps(meta, indent=2))


def load_models() -> Dict[str, Pipeline]:
    import joblib

    if not (REGRESSOR_PATH.exists() and CLASSIFIER_PATH.exists()):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python train_hybrid_model.py` first to create them."
        )

    reg_model: Pipeline = joblib.load(REGRESSOR_PATH)
    cls_model: Pipeline = joblib.load(CLASSIFIER_PATH)

    return {"regressor": reg_model, "classifier": cls_model}


def ensure_models(
    merged_path: Path = DATA_MERGED_DEFAULT,
    elo_path: Path = ELO_SEASON_DEFAULT,
) -> Dict[str, Pipeline]:
    """Load existing models, training fresh ones if artefacts are missing."""
    try:
        return load_models()
    except FileNotFoundError:
        dataset = build_dataset(merged_path, elo_path)
        models = train_models(dataset)
        save_models(models)
        return models
