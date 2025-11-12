import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Team Stats Analysis")

# Setting an image
st.image('TeamStatImg.webp', width = 1000)

df = pd.read_csv("mergedTrainingData.csv")

### TABLE CREATION
# --- Select team ---
selected_team = st.selectbox("Select a team:", sorted(df["Team"].unique()))

# --- Filter dataset for selected team ---
team_df = df[df["Team"] == selected_team].copy()

if team_df.empty:
    st.warning(f"No data found for {selected_team}.")
    st.stop()

# --- Convert win/loss to numeric (1 = W, 0 = L) ---
team_df["win_binary"] = team_df["win_loss"].apply(lambda x: 1 if x == "W" else 0)

# --- Aggregate season-level stats (Home Games removed) ---
agg_df = (
    team_df.groupby("season", as_index=False)
    .agg({
        "win_binary": "sum",             # total wins
        "pass": "mean",                  # average per week
        "rush": "mean",
        "rec": "mean",
        "points_allowed": "mean",
        "points_scored": "mean",
        "week": "count"                  # number of games (weeks)
    })
)

# --- Rename columns for clarity ---
agg_df = agg_df.rename(columns={
    "win_binary": "Total Wins",
    "week": "Games Played",
    "pass": "Avg Passing Yds",
    "rush": "Avg Rushing Yds",
    "rec": "Avg Receiving Yds",
    "points_allowed": "Avg Points Allowed",
    "points_scored": "Avg Points Scored"
})

# --- Compute Total Losses ---
agg_df["Total Losses"] = agg_df["Games Played"] - agg_df["Total Wins"]

# --- Round averages to 2 decimals ---
cols_to_round = [
    "Avg Passing Yds", "Avg Rushing Yds", "Avg Receiving Yds",
    "Avg Points Allowed", "Avg Points Scored"
]
agg_df[cols_to_round] = agg_df[cols_to_round].round(2)

# --- Pull the AP Rank for Week 14 (typical last regular-season week) ---
ap_rank_df = (
    team_df[team_df["week"] == 14][["season", "AP_rank"]]
    .rename(columns={"AP_rank": "Week 14 AP Rank"})
)

# --- Merge aggregated stats with Week 14 AP Rank ---
merged_df = pd.merge(agg_df, ap_rank_df, on="season", how="left")

# --- Handle missing AP rank values ---
merged_df["Week 14 AP Rank"] = merged_df["Week 14 AP Rank"].fillna("Unranked")

# --- Display the final summary table ---
st.subheader(f"📊 Seasonal Summary for {selected_team} (2021–2024)")
st.dataframe(
    merged_df[
        [
            "season", "Games Played", "Total Wins", "Total Losses",
            "Avg Passing Yds", "Avg Rushing Yds", "Avg Receiving Yds",
            "Avg Points Scored", "Avg Points Allowed", "Week 14 AP Rank"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

### WEEKLY DATA
st.markdown("---")
st.subheader(f"📅 Weekly Game Details by Season for {selected_team}")

# Seasons to display
seasons_to_show = [2021, 2022, 2023, 2024]

# --- Build lookup dictionary for opponent rankings ---
# Key: (team, season, week), Value: AP_rank
ap_rank_lookup = (
    df[["Team", "season", "week", "AP_rank"]]
    .dropna(subset=["AP_rank"])  # keep only rows where AP_rank exists
    .set_index(["Team", "season", "week"])["AP_rank"]
    .to_dict()
)

# --- Columns for weekly display (includes Opponent Rank) ---
display_cols = [
    "week", "opponent", "win_loss",
    "pass", "rush", "rec",
    "points_allowed", "points_scored", "AP_rank"
]

for season in seasons_to_show:
    # Filter for this team & season
    season_df = team_df[team_df["season"] == season][display_cols].copy()

    if season_df.empty:
        st.info(f"No data available for {selected_team} in {season}.")
        continue

    # --- Lookup opponent rank for each row ---
    opponent_ranks = []
    for _, row in season_df.iterrows():
        opp_rank = ap_rank_lookup.get((row["opponent"], season, row["week"]), "Unranked")
        opponent_ranks.append(opp_rank)
    season_df["Opponent Rank"] = opponent_ranks

    # Round numeric stats
    cols_to_round = ["pass", "rush", "rec", "points_allowed", "points_scored"]
    for c in cols_to_round:
        season_df[c] = pd.to_numeric(season_df[c], errors="coerce").round(2)

    # Sort by week ascending
    season_df = season_df.sort_values("week")

    # Rename for display
    season_df = season_df.rename(columns={
        "week": "Week",
        "opponent": "Opponent",
        "win_loss": "Win/Loss",
        "pass": "Pass Yds",
        "rush": "Rush Yds",
        "rec": "Receiving Yds",
        "points_allowed": "Points Allowed",
        "points_scored": "Points Scored",
        "AP_rank": "AP Rank"
    })

    # Fill missing AP ranks with "Unranked"
    season_df["AP Rank"] = season_df["AP Rank"].fillna("Unranked")

    # Reorder columns to place Opponent Rank after Opponent
    season_df = season_df[
        [
            "Week", "Opponent", "Opponent Rank", "Win/Loss",
            "Pass Yds", "Rush Yds", "Receiving Yds",
            "Points Allowed", "Points Scored", "AP Rank"
        ]
    ]

    # Reset index and show table
    season_df.reset_index(drop=True, inplace=True)
    with st.expander(f"📆 Season {season} — Weekly Breakdown"):
        st.dataframe(season_df, use_container_width=True, hide_index=True)

### BUILDING VISUALIZATIONS
st.markdown("---")
st.subheader(f"📈 Year‑over‑Year Trends for {selected_team} (2021 – 2024)")

# Restrict to relevant seasons
viz_df = merged_df[(merged_df["season"] >= 2021) & (merged_df["season"] <= 2024)]

if viz_df.empty:
    st.warning("No data available for seasons 2021–2024.")
else:
    # Ensure all 4 years appear even if data missing
    viz_df["season"] = viz_df["season"].astype(int)
    viz_df = viz_df.set_index("season").reindex([2021, 2022, 2023, 2024]).reset_index()

    # --- Numeric columns to visualize (Home Games removed) ---
    numeric_cols = [
        "Total Wins", "Total Losses",
        "Avg Passing Yds", "Avg Rushing Yds",
        "Avg Receiving Yds", "Avg Points Scored",
        "Avg Points Allowed"
    ]

    for col in numeric_cols:
        col_data = viz_df[col].dropna()
        if col_data.empty:
            continue

        # Calculate safe y-axis range
        y_min = 0
        y_max = col_data.max() if col_data.max() > 0 else 1
        y_max = y_max * 1.15  # add 15% top padding

        # Build line chart with text labels
        fig = px.line(
            viz_df,
            x="season",
            y=col,
            markers=True,
            text=col,  # show values as text
            title=f"{col} Trend (2021–2024)",
            labels={"season": "Season", col: col},
            color_discrete_sequence=["#1f77b4"]
        )

        # Format text display
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="top right",
            line=dict(width=3),
            marker=dict(size=8)
        )

        # Axis settings
        fig.update_yaxes(range=[y_min, y_max], autorange=False)
        fig.update_xaxes(tickvals=[2021, 2022, 2023, 2024])
        fig.update_layout(
            margin=dict(l=40, r=20, t=60, b=40),
            yaxis_showgrid=True,
            xaxis_showgrid=True,
            yaxis_gridcolor="rgba(128,128,128,0.3)",
            xaxis_gridcolor="rgba(128,128,128,0.3)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Week 18 AP Rank Visualization (with labels) ---
    if "Week 18 AP Rank" in viz_df:
        ap_rank_viz = viz_df.copy()
        ap_rank_viz["Week 18 AP Rank"] = pd.to_numeric(
            ap_rank_viz["Week 18 AP Rank"], errors="coerce"
        )

        if ap_rank_viz["Week 18 AP Rank"].notna().any():
            ap_data = ap_rank_viz["Week 18 AP Rank"].dropna()
            y_min = 0
            y_max = ap_data.max() * 1.15

            fig_ap = px.line(
                ap_rank_viz,
                x="season",
                y="Week 18 AP Rank",
                markers=True,
                text="Week 18 AP Rank",
                title="Week 18 AP Rank Trend (2021–2024, Lower = Better)",
                labels={"season": "Season", "Week 18 AP Rank": "AP Rank"},
                color_discrete_sequence=["#d62728"]
            )

            fig_ap.update_traces(
                texttemplate="%{text}",
                textposition="top right",
                line=dict(width=3),
                marker=dict(size=8)
            )

            # Reverse axis (rank 1 = top) with headroom
            fig_ap.update_yaxes(range=[y_max, y_min], autorange=False)
            fig_ap.update_xaxes(tickvals=[2021, 2022, 2023, 2024])
            fig_ap.update_layout(
                margin=dict(l=40, r=20, t=60, b=40),
                yaxis_showgrid=True,
                xaxis_showgrid=True,
                yaxis_gridcolor="rgba(128,128,128,0.3)",
                xaxis_gridcolor="rgba(128,128,128,0.3)"
            )

            st.plotly_chart(fig_ap, use_container_width=True)
        else:
            st.info("No AP Rank data available for this team.")