"""High-level interface for generating ranking predictions from model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.elo_hybrid import (
    DATA_MERGED_DEFAULT,
    ELO_SEASON_DEFAULT,
    FEATURE_COLUMNS,
    ensure_models,
)


@dataclass
class PredictionResult:
    team: str
    season: int
    week: int
    opponent: str
    current_rank: float
    predicted_rank_change: float
    predicted_new_rank: int
    predicted_direction: str
    direction_probabilities: Dict[str, float]
    feature_payload: Dict[str, float]


class RankPredictor:
    """Prediction helper that wraps feature engineering and model inference."""

    def __init__(
        self,
        merged_path: Path = DATA_MERGED_DEFAULT,
        elo_path: Path = ELO_SEASON_DEFAULT,
    ) -> None:
        self.merged_path = merged_path
        self.elo_path = elo_path

        self.merged_df = pd.read_csv(self.merged_path)
        self.elo_df = pd.read_csv(self.elo_path)

        self.models = ensure_models(self.merged_path, self.elo_path)

    def _compute_elo(self, team: str, season: int) -> Optional[float]:
        mask = (self.elo_df["Team"] == team) & (self.elo_df["season"] == season)
        if mask.any():
            return float(self.elo_df.loc[mask, "Elo"].iloc[0])
        return None

    def _build_feature_vector(
        self,
        team: str,
        opponent: str,
        season: int,
        week: int,
        points_scored: float,
        points_allowed: float,
        home_game: bool,
        won_game: Optional[bool] = None,
        current_rank: Optional[float] = None,
        opponent_rank: Optional[float] = None,
    ) -> Dict[str, float]:
        df_team = self.merged_df[
            (self.merged_df["Team"] == team)
            & (self.merged_df["season"] == season)
            & (self.merged_df["week"] <= week)
        ].copy()

        if df_team.empty:
            raise ValueError(f"No historical data for {team} in season {season} up to week {week}")

        df_team = df_team.sort_values("week")

        prev_rank = (
            df_team.loc[df_team["week"] == week, "AP_rank"].iloc[0]
            if current_rank is None and (df_team["week"] == week).any()
            else current_rank
        )
        if prev_rank is None:
            historical_ranks = df_team["AP_rank"].dropna()
            if historical_ranks.empty:
                # Provide a neutral fallback when the team has never been ranked so we keep the pipeline running.
                prev_rank = 26.0
            else:
                prev_rank = float(historical_ranks.iloc[-1])

        is_win = won_game
        if is_win is None:
            if (df_team["week"] == week).any():
                is_win = str(df_team.loc[df_team["week"] == week, "win_loss"].iloc[0]).upper() == "W"
            else:
                is_win = points_scored > points_allowed

        margin = points_scored - points_allowed
        home_bool = int(bool(home_game))

        team_elo = self._compute_elo(team, season)
        opponent_elo = self._compute_elo(opponent, season)
        if team_elo is None:
            raise ValueError(f"Missing Elo rating for {team} in season {season}")
        if opponent_elo is None:
            raise ValueError(f"Missing Elo rating for {opponent} in season {season}")

        if opponent_rank is None:
            opp_rows = self.merged_df[
                (self.merged_df["Team"] == opponent)
                & (self.merged_df["season"] == season)
                & (self.merged_df["week"] == week)
            ]
            if not opp_rows.empty and opp_rows["AP_rank"].notna().any():
                opponent_rank = float(opp_rows["AP_rank"].iloc[0])
            else:
                opponent_rank = np.nan

        opp_ranked_flag = 0 if pd.isna(opponent_rank) else float(opponent_rank > 0)
        unranked_margin = margin if opp_ranked_flag == 0 else 0.0
        unranked_blowout = float((opp_ranked_flag == 0) and (margin >= 21))

        win_history = df_team.copy()
        win_history.loc[win_history["week"] == week, "points_scored"] = points_scored
        win_history.loc[win_history["week"] == week, "points_allowed"] = points_allowed
        win_history.loc[win_history["week"] == week, "win_loss"] = "W" if is_win else "L"

        win_history["is_win"] = (win_history["win_loss"].astype(str).str.upper() == "W").astype(int)
        win_history["margin"] = win_history["points_scored"].astype(float) - win_history["points_allowed"].astype(float)

        # Rolling windows using up to the current week (inclusive).
        rolling_is_win = win_history["is_win"].rolling(window=3, min_periods=1).sum()
        rolling_margin = win_history["margin"].rolling(window=3, min_periods=1).mean()

        win_streak = float(rolling_is_win.iloc[-1])
        avg_margin = float(rolling_margin.iloc[-1])

        feature_row = {
            "prev_ap_rank": float(prev_rank),
            "team_elo": float(team_elo),
            "opponent_elo": float(opponent_elo),
            "elo_diff": float(team_elo - opponent_elo),
            "elo_advantage": float(team_elo > opponent_elo),
            "is_win": float(is_win),
            "margin": float(margin),
            "home_bool": float(home_bool),
            "opp_ranked": float(opp_ranked_flag),
            "win_streak": float(win_streak),
            "avg_margin_3games": float(avg_margin),
            "unranked_margin": float(unranked_margin),
            "unranked_blowout": float(unranked_blowout),
        }

        return feature_row

    def predict(
        self,
        team: str,
        opponent: str,
        season: int,
        week: int,
        points_scored: float,
        points_allowed: float,
        home_game: bool,
        current_rank: Optional[float] = None,
        opponent_rank: Optional[float] = None,
    ) -> PredictionResult:
        """Generate predictions for a specific matchup using the trained models."""

        feature_row = self._build_feature_vector(
            team=team,
            opponent=opponent,
            season=season,
            week=week,
            points_scored=points_scored,
            points_allowed=points_allowed,
            home_game=home_game,
            won_game=None,
            current_rank=current_rank,
            opponent_rank=opponent_rank,
        )

        reg_pipeline = self.models["regressor"]
        clf_pipeline = self.models["classifier"]

        # Preserve column order the pipelines expect.
        X = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
        rank_change_pred = float(reg_pipeline.predict(X)[0])
        direction_pred = str(clf_pipeline.predict(X)[0])
        direction_proba = clf_pipeline.predict_proba(X)[0]
        proba_map = {
            label: float(prob)
            for label, prob in zip(clf_pipeline.classes_, direction_proba)
        }

        classes = list(clf_pipeline.classes_)

        def _bias_probabilities(primary: str, secondary: str | None = None) -> Dict[str, float]:
            weights = {label: 0.05 for label in classes}
            if primary in weights:
                weights[primary] = 0.8
            if secondary and secondary in weights:
                weights[secondary] = 0.15
            total = sum(weights.values()) or 1.0
            return {label: weights[label] / total for label in classes}

        is_ranked_team = feature_row["prev_ap_rank"] <= 25
        opp_unranked = feature_row["opp_ranked"] == 0
        won_game_flag = feature_row["is_win"] >= 0.5
        blowout_win = feature_row["margin"] >= 21

        if is_ranked_team and opp_unranked and won_game_flag:
            if blowout_win:
                rank_change_pred = max(rank_change_pred, 2.0)
                predicted_direction = "up"
                proba_map = _bias_probabilities("up", "flat")
            else:
                rank_change_pred = max(rank_change_pred, 0.0)
                predicted_direction = "up" if rank_change_pred > 0.5 else "flat"
                target_primary = "up" if predicted_direction == "up" else "flat"
                target_secondary = "flat" if target_primary == "up" else "up"
                proba_map = _bias_probabilities(target_primary, target_secondary)

        prev_rank = feature_row["prev_ap_rank"] if current_rank is None else current_rank
        new_rank = int(np.clip(round(prev_rank - rank_change_pred), 1, 25))

        return PredictionResult(
            team=team,
            season=int(season),
            week=int(week),
            opponent=opponent,
            current_rank=float(prev_rank),
            predicted_rank_change=round(rank_change_pred, 3),
            predicted_new_rank=new_rank,
            predicted_direction=direction_pred,
            direction_probabilities=proba_map,
            feature_payload=feature_row,
        )

    def predict_from_dataset(self, team: str, season: int, week: int) -> PredictionResult:
        """Convenience wrapper that uses actual results from the merged dataset."""
        row = self.merged_df[
            (self.merged_df["Team"] == team)
            & (self.merged_df["season"] == season)
            & (self.merged_df["week"] == week)
        ]
        if row.empty:
            raise ValueError(f"No record for {team} in {season} week {week}")

        row = row.iloc[0]

        return self.predict(
            team=team,
            opponent=str(row["opponent"]),
            season=int(season),
            week=int(week),
            points_scored=float(row["points_scored"]),
            points_allowed=float(row["points_allowed"]),
            home_game=bool(row["home_game"])
            if isinstance(row["home_game"], (bool, np.bool_))
            else str(row["home_game"]).strip().upper() in {"TRUE", "T", "YES", "1"},
            current_rank=float(row["prev_ap_rank"]) if pd.notna(row["prev_ap_rank"]) else None,
            opponent_rank=float(row["opponent_rank"]) if pd.notna(row["opponent_rank"]) else None,
        )
