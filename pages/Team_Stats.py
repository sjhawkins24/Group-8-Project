import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Team Stats Analysis")

df = pd.read_csv("mergedTrainingData.csv")

# Select team
selected_team = st.selectbox("Select a team:", sorted(df["Team"].unique()))

# Filter to that team
team_df = df[df["Team"] == selected_team].copy()

if team_df.empty:
    st.warning(f"No data found for {selected_team}.")
    st.stop()

# --- Convert win/loss to numeric (1 = W, 0 = L) ---
team_df["win_binary"] = team_df["win_loss"].apply(lambda x: 1 if x == "W" else 0)

# --- Aggregate season data ---
agg_df = (
    team_df.groupby("season", as_index=False)
    .agg({
        "win_binary": "sum",             # total wins
        "pass": "mean",                  # average per week
        "rush": "mean",
        "rec": "mean",
        "points_allowed": "mean",
        "points_scored": "mean",
        "home_game": "sum",              # number of home games
        "week": "count"                  # number of games (weeks)
    })
)

# Rename columns for clarity
agg_df = agg_df.rename(columns={
    "win_binary": "Total Wins",
    "week": "Games Played",
    "pass": "Avg Passing Yds",
    "rush": "Avg Rushing Yds",
    "rec": "Avg Receiving Yds",
    "points_allowed": "Avg Points Allowed",
    "points_scored": "Avg Points Scored",
    "home_game": "Home Games"
})

# --- Compute Total Losses ---
agg_df["Total Losses"] = agg_df["Games Played"] - agg_df["Total Wins"]

# --- Round average numeric columns to 2 decimals ---
cols_to_round = [
    "Avg Passing Yds", "Avg Rushing Yds", "Avg Receiving Yds",
    "Avg Points Allowed", "Avg Points Scored"
]
agg_df[cols_to_round] = agg_df[cols_to_round].round(2)

# --- Pull the AP rank only for week 18 (per season) ---
ap_rank_df = (
    team_df[team_df["week"] == 18][["season", "AP_rank"]]
    .rename(columns={"AP_rank": "Week 18 AP Rank"})
)

# Merge both frames
merged_df = pd.merge(agg_df, ap_rank_df, on="season", how="left")

# Handle NaN AP ranks → 'Unranked'
merged_df["Week 18 AP Rank"] = merged_df["Week 18 AP Rank"].fillna("Unranked")

# --- Display final result ---
st.subheader(f"📊 Seasonal Summary for {selected_team} (2021–2024)")
st.dataframe(
    merged_df[
        [
            "season", "Games Played", "Total Wins", "Total Losses",
            "Home Games", "Avg Passing Yds", "Avg Rushing Yds",
            "Avg Receiving Yds", "Avg Points Scored", "Avg Points Allowed",
            "Week 18 AP Rank"
        ]
    ],
    use_container_width=True
)

### BUILDING VISUALIZATIONS
st.markdown("---")
st.subheader(f"📈 Year‑over‑Year Trends for {selected_team} (2021 – 2024)")

st.markdown("---")
st.subheader(f"📈 Year‑over‑Year Trends for {selected_team} (2021 – 2024)")

# Restrict to relevant seasons
viz_df = merged_df[(merged_df["season"] >= 2021) & (merged_df["season"] <= 2024)]

if viz_df.empty:
    st.warning("No data available for seasons 2021–2024.")
else:
    # Make sure all four seasons are shown even if missing
    viz_df["season"] = viz_df["season"].astype(int)
    viz_df = viz_df.set_index("season").reindex([2021, 2022, 2023, 2024]).reset_index()

    # --- Numeric columns to visualize (Home Games removed) ---
    numeric_cols = [
        "Total Wins", "Total Losses",
        "Avg Passing Yds", "Avg Rushing Yds",
        "Avg Receiving Yds", "Avg Points Scored",
        "Avg Points Allowed"
    ]

    # --- Build individual line charts ---
    for col in numeric_cols:
        col_data = viz_df[col].dropna()
        if col_data.empty:
            continue

        # Dynamic Y‑axis with 15 % headroom
        y_min = 0
        y_max = col_data.max()
        if y_max == 0:
            y_max = 1  # prevent zero scale
        y_max = y_max * 1.15  # add headroom

        fig = px.line(
            viz_df,
            x="season",
            y=col,
            markers=True,
            title=f"{col} Trend (2021–2024)",
            labels={"season": "Season", col: col},
            color_discrete_sequence=["#1f77b4"]
        )

        # Styling and axis configuration
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
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

    # --- Special chart for Week 18 AP Rank (lower = better) ---
    if "Week 18 AP Rank" in viz_df:
        ap_rank_viz = viz_df.copy()
        ap_rank_viz["Week 18 AP Rank"] = pd.to_numeric(
            ap_rank_viz["Week 18 AP Rank"], errors="coerce"
        )

        if ap_rank_viz["Week 18 AP Rank"].notna().any():
            ap_data = ap_rank_viz["Week 18 AP Rank"].dropna()
            y_min = 0
            y_max = ap_data.max()
            y_max = y_max * 1.15  # add 15% padding

            fig_ap = px.line(
                ap_rank_viz,
                x="season",
                y="Week 18 AP Rank",
                markers=True,
                title="Week 18 AP Rank Trend (2021–2024, Lower = Better)",
                labels={"season": "Season", "Week 18 AP Rank": "AP Rank"},
                color_discrete_sequence=["#d62728"]
            )

            fig_ap.update_traces(line=dict(width=3), marker=dict(size=8))
            
            # Reverse Y‑axis (Rank 1 at top) but keep top padding
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
            st.info("No AP Rank data available for this team.")
