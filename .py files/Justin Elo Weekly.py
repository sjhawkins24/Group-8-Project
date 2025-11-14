# Justin_Elo_Weekly.py
# Generates WEEKLY Elo ratings (after each game) instead of just end-of-season
import os
from pathlib import Path
import pandas as pd
import numpy as np

# --- 1) READ THE RAW CSV FROM GITHUB ---
RAW_URL = "https://raw.githubusercontent.com/sjhawkins24/Group-8-Project/main/mergedTrainingData.csv"
df = pd.read_csv(RAW_URL)

# Remove rows where either side is a Bye week entry
df = df[~df['Team'].eq('Bye') & ~df['opponent'].eq('Bye')]

# --- 2) CLEAN / PREP ---
df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')
df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int64')
df['date_str'] = df['date'].astype(str).str.strip() + " " + df['season'].astype(str)
df['date_parsed'] = pd.to_datetime(df['date_str'], errors='coerce', infer_datetime_format=True)

df['points_scored'] = pd.to_numeric(df['points_scored'], errors='coerce')
df['points_allowed'] = pd.to_numeric(df['points_allowed'], errors='coerce')
df['margin'] = df['points_scored'] - df['points_allowed']

def as_bool(x):
    s = str(x).strip().upper()
    return s in {"TRUE", "T", "YES", "1"}

df['is_win'] = (df['win_loss'].astype(str).str.upper() == 'W').astype(int)
df['is_ot'] = df['OT'].apply(as_bool)
df['home_bool'] = df['home_game'].apply(as_bool)

seasons = sorted([int(s) for s in df['season'].dropna().unique()])

# --- 3) ELO PARAMS ---
BASE_K = 22.0
HFA = 55.0

def expected_score(ra, rb):
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))

def mov_multiplier(margin, dr):
    dr = max(1.0, abs(dr))
    margin = 1.0 if pd.isna(margin) or margin == 0 else float(margin)
    return np.log(1.0 + abs(margin)) * (2.2 / (0.001 * dr + 2.2))

# --- 4) GENERATE WEEKLY ELO RATINGS ---
weekly_ratings = []

for season in seasons:
    sdf = df[df['season'] == season].copy().sort_values(['week', 'date_parsed'])
    R = {}  # team -> current rating
    seen = set()
    
    print(f"Processing season {season}...")

    for _, row in sdf.iterrows():
        team = row.get('Team')
        opp  = row.get('opponent')
        week = row.get('week')
        
        if not isinstance(team, str) or not isinstance(opp, str):
            continue
        if pd.isna(week):
            continue

        # De-dup key
        gid = (season, week, row['date_parsed'], tuple(sorted([team, opp])))
        if gid in seen:
            continue
        seen.add(gid)

        # Try to fetch both perspectives
        mask = (
            (sdf['week'] == week) &
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
            if (h_a > 0 and h_b > 0) or (h_a == 0 and h_b == 0):
                h_a = h_b = 0.0
        else:
            a, b = team, opp
            ra_row = row
            margin_a = float(row['points_scored'] - row['points_allowed'])
            sa = 1.0 if str(row['win_loss']).upper() == 'W' else 0.0
            sb = 1.0 - sa
            ot = as_bool(row['OT'])
            h_a = HFA if as_bool(row['home_game']) else 0.0
            h_b = 0.0

        # Init ratings (carry over from previous season or start at 1500)
        R.setdefault(a, 1500.0)
        R.setdefault(b, 1500.0)

        # SAVE PRE-GAME RATINGS (ratings going INTO this week)
        weekly_ratings.append({
            'season': season,
            'week': week,
            'Team': a,
            'Elo_pregame': R[a]
        })
        weekly_ratings.append({
            'season': season,
            'week': week,
            'Team': b,
            'Elo_pregame': R[b]
        })

        # Calculate expected and update
        ra_eff = R[a] + h_a
        rb_eff = R[b] + h_b
        dr = ra_eff - rb_eff

        ea = expected_score(ra_eff, rb_eff)
        eb = 1.0 - ea

        K = BASE_K * (1.35 if ot else 1.0)
        m = mov_multiplier(margin_a, dr)

        R[a] = R[a] + K * m * (sa - ea)
        R[b] = R[b] + K * m * (sb - eb)

# --- 5) CONSOLIDATE AND SAVE ---
try:
    script_dir = Path(__file__).parent
except NameError:
    script_dir = Path(os.getcwd())

if weekly_ratings:
    weekly_df = pd.DataFrame(weekly_ratings)
    
    # Remove duplicates (keep first occurrence of each team-week)
    weekly_df = weekly_df.drop_duplicates(subset=['season', 'week', 'Team'], keep='first')
    
    # Sort for readability
    weekly_df = weekly_df.sort_values(['season', 'Team', 'week']).reset_index(drop=True)
    
    # Save
    out_path = script_dir / "elo_ratings_weekly.csv"
    weekly_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved weekly Elo ratings → {out_path}")
    print(f"   Total records: {len(weekly_df)}")
    print(f"   Seasons: {weekly_df['season'].min()} - {weekly_df['season'].max()}")
    print(f"   Weeks per season: {weekly_df.groupby('season')['week'].max().to_dict()}")
    
    # Show sample
    print("\nSample of weekly ratings:")
    print(weekly_df.head(20))
    
    # Show a specific team's progression through a season
    sample_team = weekly_df[weekly_df['season'] == weekly_df['season'].max()]['Team'].iloc[0]
    sample_season = weekly_df['season'].max()
    team_progression = weekly_df[(weekly_df['Team'] == sample_team) & (weekly_df['season'] == sample_season)]
    print(f"\n{sample_team}'s Elo progression in {sample_season}:")
    print(team_progression)
    
else:
    print("No weekly ratings generated. Check input data.")
