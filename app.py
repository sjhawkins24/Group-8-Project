from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.ranking_predictor import RankPredictor

try:
    from services.chat_insights import ChatInsightGenerator
except RuntimeError:
    ChatInsightGenerator = None  # type: ignore


@st.cache_resource
def load_predictor() -> RankPredictor:
    return RankPredictor()


def latest_value(df: pd.DataFrame, team: str, season: int, column: str) -> float | None:
    season_rows = df[(df["Team"] == team) & (df["season"] == season) & df[column].notna()]
    if not season_rows.empty:
        return float(season_rows.iloc[-1][column])
    fallback = df[(df["Team"] == team) & df[column].notna()]
    if not fallback.empty:
        return float(fallback.iloc[-1][column])
    return None


predictor = load_predictor()
data = predictor.merged_df

st.title("College Football Ranking Predictor")
image_path = st.session_state.get("_image_path", "StreamlitPic.jpg")
if image_path:
    st.image(image_path, width=1000)

st.header("Team Selection")
teams = sorted(data["Team"].unique())
team = st.selectbox("Select your team", teams)

seasons = sorted(int(s) for s in data["season"].dropna().unique())
default_season = seasons[-1]
season = st.selectbox("Season", seasons, index=seasons.index(default_season))

team_fpi = latest_value(data, team, season, "FPI")
if team_fpi is None:
    team_fpi = latest_value(data, team, default_season, "FPI")
if team_fpi is not None:
    st.write(f"Current FPI rating: {team_fpi:.1f}")

season_weeks = data.loc[data["season"] == season, "week"].dropna()
default_week = int(season_weeks.max()) if not season_weeks.empty else 1
week = st.number_input("Week (use upcoming week for future games)", min_value=1, max_value=25, value=default_week, step=1)

st.subheader("Opponent Selection")
opponent_choices = [t for t in teams if t != team]
opponent = st.selectbox("Choose the opponent", opponent_choices)

st.subheader("Game Setup")
current_rank_default = latest_value(data, team, season, "AP_rank")
current_rank = st.number_input(
    "Current AP Rank",
    min_value=1,
    max_value=40,
    value=int(current_rank_default) if current_rank_default else 10,
    step=1,
)
st.write(f"The {team} enter at #{current_rank} in the AP Poll.")
opponent_rank_default = latest_value(data, opponent, season, "AP_rank")
opponent_rank = st.number_input(
    "Opponent AP Rank (use best guess if unranked)",
    min_value=1,
    max_value=50,
    value=int(opponent_rank_default) if opponent_rank_default else 26,
    step=1,
)
opponent_fpi = latest_value(data, opponent, season, "FPI")
if opponent_fpi is not None:
    st.write(f"Opponent FPI rating: {opponent_fpi:.1f}")
st.write(f"{opponent} enter at #{opponent_rank} in the AP Poll.")
home_pick = st.selectbox("Is this a home game?", ["Home", "Away", "Neutral"], index=0)
home_game = home_pick == "Home"

st.subheader("Score Projection")
team_last_margin = latest_value(data, team, season, "points_scored")
opp_last_margin = latest_value(data, team, season, "points_allowed")
points_scored = st.number_input("Projected points scored", min_value=0.0, step=1.0, value=float(team_last_margin) if team_last_margin else 28.0)
points_allowed = st.number_input("Projected points allowed", min_value=0.0, step=1.0, value=float(opp_last_margin) if opp_last_margin else 21.0)

point_diff = points_scored - points_allowed
game_outcome = "beat" if point_diff > 0 else ("tie" if point_diff == 0 else "lose to")

st.subheader("Your Scenario")
pred_table = pd.DataFrame(
    {
        "Option": ["Season", "Week", "Team", "Opponent", "Projected Score", "Venue"],
        "Selection": [season, int(week), team, opponent, f"{int(points_scored)}-{int(points_allowed)}", home_pick],
    }
)
st.table(pred_table)

run_clicked = st.button("Predict Ranking Impact", type="primary")

if run_clicked:
    try:
        st.session_state["_latest_prediction"] = asdict(
            predictor.predict(
                team=team,
                opponent=opponent,
                season=season,
                week=int(week),
                points_scored=points_scored,
                points_allowed=points_allowed,
                home_game=home_game,
                current_rank=float(current_rank),
                opponent_rank=float(opponent_rank),
            )
        )
    except Exception as exc:
        st.session_state.pop("_latest_prediction", None)
        st.error(f"Prediction failed: {exc}")

prediction_payload = st.session_state.get("_latest_prediction")

if prediction_payload:
    from services.ranking_predictor import PredictionResult

    result = PredictionResult(**prediction_payload)

    st.header("Results")
    move = {
        "up": "move up",
        "down": "move down",
        "flat": "stay level",
    }[result.predicted_direction]
    verb = game_outcome
    margin_value = abs(int(round(point_diff)))
    margin_phrase = "" if margin_value == 0 else f" by {margin_value} points"
    st.write(
        f"If the {team} {verb} {opponent}{margin_phrase}, "
        f"the model projects they will {move} to #{result.predicted_new_rank} (change {result.predicted_rank_change:+.2f})."
    )

    probs = pd.DataFrame([
        {"Direction": direction.title(), "Probability": f"{prob*100:.1f}%"}
        for direction, prob in result.direction_probabilities.items()
    ])
    st.subheader("Direction Confidence")
    st.table(probs)

    use_chat = st.toggle("Explain this with AI", key="ai_toggle")
    if use_chat:
        try:
            @st.cache_resource
            def load_chat():
                if ChatInsightGenerator is None:
                    raise RuntimeError("Chat assistant unavailable; install openai and set OPENROUTER_API_KEY.")
                return ChatInsightGenerator()

            chat = load_chat()
            summary = chat.explain_prediction(prediction_payload)
        except Exception as chat_error:
            st.warning(f"AI summary unavailable: {chat_error}")
        else:
            st.subheader("AI Take")
            st.write(summary)