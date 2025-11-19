
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from services.ranking_predictor import RankPredictor


@st.cache_resource
def load_predictor() -> RankPredictor:
    return RankPredictor()


def latest_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[-1])


def safe_rank(value: float | int | None) -> str | int:
    if value is None or pd.isna(value):
        return "Unranked"
    return int(round(float(value)))


def load_latest_rank_map(predictor: RankPredictor, ranking_path: Path) -> dict[str, int]:
    if not ranking_path.exists():
        return {}

    df = pd.read_csv(ranking_path)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    if df.empty:
        return {}

    latest_season = df["season"].max()
    df = df[df["season"] == latest_season]
    df["ap_rank"] = pd.to_numeric(df["AP/CFP"], errors="coerce")
    df = df.dropna(subset=["ap_rank"])
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    latest_by_code = (
        df.assign(Team_id=df["Team_id"].apply(lambda value: str(int(value)) if pd.notna(value) else None))
        .dropna(subset=["Team_id"])
        .groupby("Team_id")["ap_rank"]
        .last()
    )

    def _normalise_code(value: object) -> str | None:
        if pd.isna(value):
            return None
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value).strip()

    team_code_map = (
        predictor.merged_df[["Team", "code"]]
        .dropna()
        .drop_duplicates()
        .assign(code=lambda frame: frame["code"].map(_normalise_code))
        .dropna(subset=["code"])
        .set_index("code")["Team"]
        .to_dict()
    )

    rank_map: dict[str, int] = {}
    for code, rank in latest_by_code.items():
        team = team_code_map.get(code)
        if team and pd.notna(rank):
            rank_map[team] = int(round(float(rank)))
    return rank_map


predictor = load_predictor()
data = predictor.merged_df.copy()
latest_rank_map = load_latest_rank_map(
    predictor, Path("00 - Raw Data Space") / "CollegeFootballRankings2025.csv"
)

season_numeric = pd.to_numeric(data["season"], errors="coerce")
current_season = int(season_numeric.dropna().max())
season_df = data[season_numeric == current_season].copy()

teams = sorted(season_df["Team"].unique())

st.title("College Football Ranking Predictor")
st.image("StreamlitPic.jpg", width=1000)

st.header("Home Team")
home_team = st.selectbox("Select the home team:", teams)

home_history = season_df[season_df["Team"] == home_team].copy()
home_history = home_history.assign(
    _week=pd.to_numeric(home_history["week"], errors="coerce")
).sort_values("_week")

if home_history.empty:
    st.error(f"No data available for {home_team} in {current_season}.")
    st.stop()

home_rank = (
    home_history["AP_rank"].dropna().iloc[-1]
    if home_history["AP_rank"].notna().any()
    else np.nan
)
if home_team in latest_rank_map:
    home_rank = float(latest_rank_map[home_team])
st.write(
    f"Home team current rank: {'unranked' if pd.isna(home_rank) else int(home_rank)}"
)

last_week = latest_numeric(home_history["week"]) or 0
next_week = int(last_week) + 1

st.header("Away Team")
away_choices = [t for t in teams if t != home_team]
away_team = st.selectbox("Select the away team:", away_choices)

away_history = season_df[season_df["Team"] == away_team].copy()
away_history = away_history.assign(
    _week=pd.to_numeric(away_history["week"], errors="coerce")
).sort_values("_week")

away_rank = (
    away_history["AP_rank"].dropna().iloc[-1]
    if away_history["AP_rank"].notna().any()
    else np.nan
)
if away_team in latest_rank_map:
    away_rank = float(latest_rank_map[away_team])
st.write(
    f"Away team current rank: {'unranked' if pd.isna(away_rank) else int(away_rank)}"
)

st.header("Points Scored and Allowed")
home_points_default = latest_numeric(home_history["points_scored"]) or 28.0
away_points_default = latest_numeric(away_history["points_scored"]) or 21.0
points_scored = st.number_input("Home team points scored:", min_value=0, step=1, value=int(home_points_default))
points_allowed = st.number_input("Away team points scored:", min_value=0, step=1, value=int(away_points_default))

point_differential = points_scored - points_allowed
game_outcome = "Won" if point_differential > 0 else ("Tied" if point_differential == 0 else "Lost")
st.subheader("Game Outcome")
st.write(f"{home_team} {game_outcome.lower()} by {point_differential} points.")

summary_rows = pd.DataFrame(
    {
        "Option": [
            "Home Team",
            "Home Rank",
            "Away Team",
            "Away Rank",
            "Home Points",
            "Away Points",
            "Game Outcome",
        ],
        "Selection": [
            home_team,
            safe_rank(home_rank),
            away_team,
            safe_rank(away_rank),
            points_scored,
            points_allowed,
            game_outcome,
        ],
    }
)

st.header("Selected Options")
st.table(summary_rows)

st.header("Results")

if st.button("Predict Ranking Impact", type="primary"):
    try:
        result = predictor.predict(
            team=home_team,
            opponent=away_team,
            season=current_season,
            week=next_week,
            points_scored=float(points_scored),
            points_allowed=float(points_allowed),
            home_game=True,
            current_rank=None if pd.isna(home_rank) else float(home_rank),
            opponent_rank=None if pd.isna(away_rank) else float(away_rank),
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        game_result = "beat" if point_differential > 0 else ("tie" if point_differential == 0 else "lose to")
        movement = result.predicted_direction
        if result.predicted_rank_change > 0:
            movement_text = "climb up the rankings"
        elif result.predicted_rank_change < 0:
            movement_text = "slide down the rankings"
        else:
            movement_text = "hold steady"

        st.write(
            f"If {home_team} {game_result} {away_team} by {abs(point_differential)} points, "
            f"the model projects they will {movement_text} to #{result.predicted_new_rank} "
            f"(change {result.predicted_rank_change:+.2f})."
        )

        probs = pd.DataFrame(
            [
                {"Direction": label.title(), "Probability": f"{prob*100:.1f}%"}
                for label, prob in result.direction_probabilities.items()
            ]
        )
        st.subheader("Direction Confidence")
        st.table(probs)
        st.caption(
            "These probabilities indicate how confident the classifier is that the home team's ranking will move in each direction after the game."
        )
        with st.expander("Model feature snapshot"):
            st.json(result.feature_payload)
