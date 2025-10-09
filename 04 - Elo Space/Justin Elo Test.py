# Justin_Elo.py
import os
from pathlib import Path
import pandas as pd
import numpy as np

# --- 1) READ THE RAW CSV FROM GITHUB ---
RAW_URL = "https://raw.githubusercontent.com/sjhawkins24/Group-8-Project/main/CollegeFootballRankings24to20.csv"
df = pd.read_csv(RAW_URL)

# Remove rows where either side is a Bye week entry
df = df[~df['Team'].eq('Bye') & ~df['opponent'].eq('Bye')]

# --- 2) CLEAN / PREP ---
# Build a real date using the season (year) + the text date in your file
# Example strings look like "Sat, Sep 26" -> append year to disambiguate
df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')
df['date_str'] = df['date'].astype(str).str.strip() + " " + df['season'].astype(str)
df['date_parsed'] = pd.to_datetime(df['date_str'], errors='coerce', infer_datetime_format=True)

# Core features
df['points_scored'] = pd.to_numeric(df['points_scored'], errors='coerce')
df['points_allowed'] = pd.to_numeric(df['points_allowed'], errors='coerce')
df['margin'] = df['points_scored'] - df['points_allowed']

# Robust booleans (your file uses TRUE/FALSE strings)
def as_bool(x):
    s = str(x).strip().upper()
    return s in {"TRUE", "T", "YES", "1"}

df['is_win'] = (df['win_loss'].astype(str).str.upper() == 'W').astype(int)
df['is_ot'] = df['OT'].apply(as_bool)
df['home_bool'] = df['home_game'].apply(as_bool)

seasons = sorted([int(s) for s in df['season'].dropna().unique()])

# --- 3) ELO PARAMS ---
BASE_K = 22.0
HFA = 55.0  # home-field advantage in Elo points

def expected_score(ra, rb):
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))

def mov_multiplier(margin, dr):
    # FiveThirtyEight-style MOV multiplier
    dr = max(1.0, abs(dr))
    margin = 1.0 if pd.isna(margin) or margin == 0 else float(margin)
    return np.log(1.0 + abs(margin)) * (2.2 / (0.001 * dr + 2.2))

ratings_by_season = []
for season in seasons:
    sdf = df[df['season'] == season].copy().sort_values('date_parsed')
    R = {}  # team -> rating
    seen = set()

    for _, row in sdf.iterrows():
        team = row.get('Team')
        opp  = row.get('opponent')
        if not isinstance(team, str) or not isinstance(opp, str):
            continue

        # De-dup key: (season, date, pair of teams)
        gid = (season, row['date_parsed'], tuple(sorted([team, opp])))
        if gid in seen:
            continue
        seen.add(gid)

        # Try to fetch both perspectives (team and opponent)
        mask = (
            (sdf['date_parsed'] == row['date_parsed']) &
            (sdf['Team'].isin([team, opp])) &
            (sdf['opponent'].isin([team, opp]))
        )
        game_rows = sdf[mask]

        if len(game_rows['Team'].unique()) == 2:
            a, b = sorted(game_rows['Team'].unique().tolist())
            ra_row = game_rows[game_rows['Team'] == a].iloc[0]
            rb_row = game_rows[game_rows['Team'] == b].iloc[0]

            margin_a = float(ra_row['points_scored'] - ra_row['points_allowed'])
            sa = 1.0 if str(ra_row['win_loss']).upper() == 'W' else 0.0
            sb = 1.0 if str(rb_row['win_loss']).upper() == 'W' else 0.0
            ot = as_bool(ra_row['OT']) or as_bool(rb_row['OT'])

            h_a = HFA if as_bool(ra_row['home_game']) else 0.0
            h_b = HFA if as_bool(rb_row['home_game']) else 0.0
            # If both show home or both show away/neutral, treat as neutral
            if (h_a > 0 and h_b > 0) or (h_a == 0 and h_b == 0):
                h_a = h_b = 0.0
        else:
            # Single-sided fallback
            a, b = team, opp
            ra_row = row
            margin_a = float(row['points_scored'] - row['points_allowed'])
            sa = 1.0 if str(row['win_loss']).upper() == 'W' else 0.0
            sb = 1.0 - sa
            ot = as_bool(row['OT'])
            h_a = HFA if as_bool(row['home_game']) else 0.0
            h_b = 0.0  # assume away/neutral

        # Init ratings
        R.setdefault(a, 1500.0)
        R.setdefault(b, 1500.0)

        # Expected result with home adjustment
        ra_eff = R[a] + h_a
        rb_eff = R[b] + h_b
        dr = ra_eff - rb_eff

        ea = expected_score(ra_eff, rb_eff)
        eb = 1.0 - ea

        K = BASE_K * (1.35 if ot else 1.0)
        m = mov_multiplier(margin_a, dr)

        # Update ratings
        R[a] = R[a] + K * m * (sa - ea)
        R[b] = R[b] + K * m * (sb - eb)

    # Save season standings
    if R:
        ratings = (
            pd.DataFrame({'season': season, 'Team': list(R.keys()), 'Elo': list(R.values())})
            .sort_values('Elo', ascending=False)
            .reset_index(drop=True)
        )
        ratings_by_season.append(ratings)

# --- 4) WRITE THE CSV INTO YOUR PROJECT FOLDER ---
# Save next to this script
try:
    script_dir = Path(__file__).parent  # works for .py files
except NameError:
    script_dir = Path(os.getcwd())      # fallback for notebooks

out_path = script_dir / "elo_ratings_by_season.csv"
if ratings_by_season:
    all_ratings = pd.concat(ratings_by_season, ignore_index=True)
    # Double-check no lingering Bye rows
    all_ratings = all_ratings[all_ratings['Team'] != 'Bye']
    all_ratings.to_csv(out_path, index=False)
    print(f"Saved Elo ratings → {out_path}")

    # Show latest season’s Top 25 in console and save to its own CSV
    latest_season = all_ratings['season'].max()
    latest_df = all_ratings[all_ratings['season'] == latest_season]
    top_25 = latest_df.nlargest(25, 'Elo').reset_index(drop=True)
    print(f"\nTop 25 Elo — Season {latest_season}")
    print(top_25)

    top25_path = script_dir / f"elo_top25_{latest_season}.csv"
    top_25.to_csv(top25_path, index=False)
    print(f"Top 25 for {latest_season} saved → {top25_path}")
else:
    print("No ratings were generated (ratings_by_season is empty). Check input data.")
