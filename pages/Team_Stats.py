import streamlit as st
import pandas as pd

st.title("Team Stats Analysis")

df = pd.read_csv("mergedTrainingData.csv")

# Select team
selected_team = st.selectbox("Select a team:", sorted(df["Team"].unique()))

# Filter to that team
team_df = df[df["Team"] == selected_team].copy()

if team_df.empty:
    st.warning(f"No data found for {selected_team}.")
    st.stop()

# --- Process win/loss and numeric aggregations over all weeks ---
team_df["win_binary"] = team_df["win_loss"].apply(lambda x: 1 if x == "W" else 0)

agg_df = (
    team_df.groupby("season", as_index=False)
    .agg({
        "win_binary": "sum",             # total wins for season
        "pass": "mean",                  # average per week
        "rush": "mean",
        "rec": "mean",
        "points_allowed": "mean",
        "points_scored": "mean",
        "home_game": "sum",              # number of home games
    })
)

# --- Pull the AP rank only for week 18 (per season) ---
ap_rank_df = (
    team_df[team_df["week"] == 18][["season", "AP_rank"]]
    .rename(columns={"AP_rank": "Week 18 AP Rank"})
)

# Merge both frames
merged_df = pd.merge(agg_df, ap_rank_df, on="season", how="left")

# Rename columns for readability
merged_df = merged_df.rename(columns={
    "win_binary": "Total Wins",
    "pass": "Avg Passing Yds",
    "rush": "Avg Rushing Yds",
    "rec": "Avg Receiving Yds",
    "points_allowed": "Avg Points Allowed",
    "points_scored": "Avg Points Scored",
    "home_game": "Home Games"
})

# Handle NaN AP ranks → 'Unranked'
merged_df["Week 18 AP Rank"] = merged_df["Week 18 AP Rank"].fillna("Unranked")

# Display final result
st.subheader(f"📊 Seasonal Summary for {selected_team} (2021–2024)")
st.dataframe(merged_df, use_container_width=True)
