"""
COMPARISON: Season-End Elo vs Weekly Elo
Tests whether using weekly Elo ratings improves model accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

print("="*70)
print("COMPARISON: SEASON-END ELO vs WEEKLY ELO")
print("="*70)

# Load data
merged_df = pd.read_csv("mergedTrainingData.csv")
elo_season_end = pd.read_csv("elo_ratings_by_season.csv")
elo_weekly = pd.read_csv("elo_ratings_weekly.csv")

print("\n1. Data loaded:")
print(f"   Merged games: {merged_df.shape}")
print(f"   Season-end Elo: {elo_season_end.shape}")
print(f"   Weekly Elo: {elo_weekly.shape}")

# Helper function to build features
def build_features(df, elo_data, is_weekly=False):
    """Build feature set with either season-end or weekly Elo"""
    
    # Merge team Elo
    if is_weekly:
        # For weekly, merge on season, week, and team
        df = df.merge(
            elo_data.rename(columns={'Elo_pregame': 'team_elo'}),
            on=['Team', 'season', 'week'],
            how='left'
        )
        # Merge opponent Elo
        df = df.merge(
            elo_data.rename(columns={'Team': 'opponent', 'Elo_pregame': 'opponent_elo'}),
            left_on=['opponent', 'season', 'week'],
            right_on=['opponent', 'season', 'week'],
            how='left'
        )
    else:
        # For season-end, merge only on team and season
        df = df.merge(
            elo_data.rename(columns={'Elo': 'team_elo'}),
            on=['Team', 'season'],
            how='left'
        )
        df = df.merge(
            elo_data.rename(columns={'Team': 'opponent', 'Elo': 'opponent_elo'}),
            left_on=['opponent', 'season'],
            right_on=['opponent', 'season'],
            how='left'
        )
    
    return df

# Build dataset with season-end Elo
print("\n2. Building dataset with SEASON-END Elo...")
df_season = merged_df.copy()
df_season = build_features(df_season, elo_season_end, is_weekly=False)

# Build dataset with weekly Elo
print("   Building dataset with WEEKLY Elo...")
df_weekly = merged_df.copy()
df_weekly = build_features(df_weekly, elo_weekly, is_weekly=True)

# Feature engineering (same for both)
def engineer_features(df):
    df = df.dropna(subset=['AP_rank', 'team_elo'])
    df['is_win'] = (df['win_loss'] == 'W').astype(int)
    df['margin'] = df['points_scored'] - df['points_allowed']
    df['home_bool'] = df['home_game'].astype(int)
    df = df.sort_values(['season', 'week', 'Team']).reset_index(drop=True)
    
    df['prev_ap_rank'] = df.groupby('Team')['AP_rank'].shift(1)
    df['rank_change'] = df['prev_ap_rank'] - df['AP_rank']
    
    df['elo_diff'] = df['team_elo'] - df['opponent_elo'].fillna(df['team_elo'].mean())
    df['elo_advantage'] = (df['elo_diff'] > 0).astype(int)
    df['opp_ranked'] = (df['opponent_rank'] > 0).astype(int)
    
    df['win_streak'] = df.groupby('Team')['is_win'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    df['avg_margin_3games'] = df.groupby('Team')['margin'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    df = df[df['prev_ap_rank'].notna()].copy()
    return df

df_season = engineer_features(df_season)
df_weekly = engineer_features(df_weekly)

print(f"   Season-end dataset: {df_season.shape[0]} samples")
print(f"   Weekly dataset: {df_weekly.shape[0]} samples")

# Define features and targets
features = [
    'prev_ap_rank', 'team_elo', 'opponent_elo', 'elo_diff', 'elo_advantage',
    'is_win', 'margin', 'home_bool', 'opp_ranked', 'win_streak', 'avg_margin_3games'
]

def prepare_data(df):
    X = df[features].copy()
    X = X.fillna(X.mean())
    
    y_reg = df['rank_change'].astype(float)
    df['dir'] = pd.cut(df['rank_change'], bins=[-99, -2, 2, 99], labels=['down', 'flat', 'up'])
    y_clf = df['dir']
    
    mask = y_clf.notna()
    return X[mask], y_reg[mask], y_clf[mask], df[mask]

X_season, y_reg_season, y_clf_season, _ = prepare_data(df_season)
X_weekly, y_reg_weekly, y_clf_weekly, _ = prepare_data(df_weekly)

# Train and evaluate both models
def evaluate_model(X, y_reg, y_clf, model_name):
    print(f"\n   Evaluating {model_name}...")
    
    num_cols = X.columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder='drop')
    
    reg_model = Pipeline([("pre", pre), ("lin", LinearRegression())])
    clf_model = Pipeline([("pre", pre), ("logit", LogisticRegression(max_iter=300, random_state=42))])
    
    tscv = TimeSeriesSplit(n_splits=5)
    reg_mae_list = []
    clf_acc_list = []
    
    for train, test in tscv.split(X):
        reg_model.fit(X.iloc[train], y_reg.iloc[train])
        clf_model.fit(X.iloc[train], y_clf.iloc[train])
        
        reg_mae_list.append(mean_absolute_error(y_reg.iloc[test], reg_model.predict(X.iloc[test])))
        clf_acc_list.append(accuracy_score(y_clf.iloc[test], clf_model.predict(X.iloc[test])))
    
    return np.mean(reg_mae_list), np.std(reg_mae_list), np.mean(clf_acc_list), np.std(clf_acc_list)

print("\n3. Running cross-validation (5 folds)...")

# Evaluate season-end Elo model
reg_mae_season, reg_std_season, clf_acc_season, clf_std_season = evaluate_model(
    X_season, y_reg_season, y_clf_season, "Season-End Elo"
)

# Evaluate weekly Elo model
reg_mae_weekly, reg_std_weekly, clf_acc_weekly, clf_std_weekly = evaluate_model(
    X_weekly, y_reg_weekly, y_clf_weekly, "Weekly Elo"
)

# Print results
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print("\nREGRESSION (Rank Change Magnitude - Lower is Better):")
print(f"  Season-End Elo MAE: {reg_mae_season:.4f} (+/- {reg_std_season:.4f})")
print(f"  Weekly Elo MAE:     {reg_mae_weekly:.4f} (+/- {reg_std_weekly:.4f})")
improvement_reg = ((reg_mae_season - reg_mae_weekly) / reg_mae_season) * 100
print(f"  --> Improvement:    {improvement_reg:+.2f}%")

print("\nCLASSIFICATION (Direction Accuracy - Higher is Better):")
print(f"  Season-End Elo Acc: {clf_acc_season:.4f} (+/- {clf_std_season:.4f})")
print(f"  Weekly Elo Acc:     {clf_acc_weekly:.4f} (+/- {clf_std_weekly:.4f})")
improvement_clf = ((clf_acc_weekly - clf_acc_season) / clf_acc_season) * 100
print(f"  --> Improvement:    {improvement_clf:+.2f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if improvement_reg > 5 or improvement_clf > 3:
    print("✅ WEEKLY ELO IS SIGNIFICANTLY BETTER!")
    print(f"   - Rank change predictions improved by {improvement_reg:.1f}%")
    print(f"   - Direction accuracy improved by {improvement_clf:.1f}%")
    print("\n   RECOMMENDATION: Use weekly Elo ratings for better predictions!")
elif improvement_reg > 0 or improvement_clf > 0:
    print("✅ Weekly Elo shows modest improvement")
    print(f"   - Rank change: {improvement_reg:+.1f}%")
    print(f"   - Direction: {improvement_clf:+.1f}%")
    print("\n   RECOMMENDATION: Weekly Elo is worth using.")
else:
    print("⚠️  Weekly Elo shows no significant improvement")
    print("   Season-end Elo may be sufficient for this task.")

print("="*70)
