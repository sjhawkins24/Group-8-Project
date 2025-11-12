import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

print("="*60)
print("ENHANCED HYBRID MODEL WITH ELO RATINGS")
print("="*60)

# Read the data
print("\n1. Loading data...")
merged_df = pd.read_csv("03 - Cleaned Data Space/mergedTrainingData.csv")
weekly_ranks = pd.read_csv("00 - Raw Data Space/WeekByWeekRankings2020to2025.csv")
elo_ratings = pd.read_csv("04 - Elo Space/elo_ratings_by_season.csv")

print(f"   - Merged data: {merged_df.shape}")
print(f"   - Weekly ranks: {weekly_ranks.shape}")
print(f"   - Elo ratings: {elo_ratings.shape}")

# Merge Elo ratings with the game data
print("\n2. Merging Elo ratings with game data...")
merged_df = merged_df.merge(
    elo_ratings.rename(columns={'Team': 'Team', 'season': 'season', 'Elo': 'team_elo'}),
    on=['Team', 'season'],
    how='left'
)

# Also merge opponent Elo ratings
merged_df = merged_df.merge(
    elo_ratings.rename(columns={'Team': 'opponent', 'season': 'season', 'Elo': 'opponent_elo'}),
    left_on=['opponent', 'season'],
    right_on=['opponent', 'season'],
    how='left'
)

# Clean and prepare the data
print("\n3. Feature engineering...")
merged_df = merged_df.dropna(subset=['AP_rank', 'team_elo'])
merged_df['is_win'] = (merged_df['win_loss'] == 'W').astype(int)
merged_df['margin'] = merged_df['points_scored'] - merged_df['points_allowed']
merged_df['home_bool'] = merged_df['home_game'].astype(int)

# Sort by season and week to ensure proper time ordering
merged_df = merged_df.sort_values(['season', 'week', 'Team']).reset_index(drop=True)

# Create target variables: rank change week-over-week
merged_df['prev_ap_rank'] = merged_df.groupby('Team')['AP_rank'].shift(1)
merged_df['rank_change'] = merged_df['prev_ap_rank'] - merged_df['AP_rank']  # Positive = moved up

# Create additional Elo-based features
merged_df['elo_diff'] = merged_df['team_elo'] - merged_df['opponent_elo'].fillna(merged_df['team_elo'].mean())
merged_df['elo_advantage'] = (merged_df['elo_diff'] > 0).astype(int)
merged_df['opp_ranked'] = (merged_df['opponent_rank'] > 0).astype(int)

# Win streak and recent performance features
merged_df['win_streak'] = merged_df.groupby('Team')['is_win'].transform(
    lambda x: x.rolling(window=3, min_periods=1).sum()
)
merged_df['avg_margin_3games'] = merged_df.groupby('Team')['margin'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Only keep rows where we have previous rank (for prediction)
df = merged_df[merged_df['prev_ap_rank'].notna()].copy()

print(f"   - Dataset size after filtering: {df.shape}")

# Select features for the model
features = [
    'prev_ap_rank',      # Previous AP rank
    'team_elo',          # Team's Elo rating
    'opponent_elo',      # Opponent's Elo rating
    'elo_diff',          # Elo differential
    'elo_advantage',     # Binary: Elo advantage
    'is_win',            # Did they win?
    'margin',            # Point differential
    'home_bool',         # Home game?
    'opp_ranked',        # Opponent ranked?
    'win_streak',        # Recent wins
    'avg_margin_3games'  # Average margin last 3 games
]

X = df[features].copy()
X = X.fillna(X.mean())  # Fill any remaining NaNs

# Target variables
y_reg = df['rank_change'].astype(float)

# Create direction labels based on rank change
# Positive = moved up in ranking (lower number), Negative = moved down (higher number)
df['dir'] = pd.cut(df['rank_change'], bins=[-99, -2, 2, 99], labels=['down', 'flat', 'up'])
y_clf = df['dir']

# Filter out any remaining NaNs
mask = y_clf.notna()
X = X[mask]
y_reg = y_reg[mask]
y_clf = y_clf[mask]
df_filtered = df[mask]

print(f"\n4. Final dataset statistics:")
print(f"   - Total samples: {X.shape[0]}")
print(f"   - Number of features: {X.shape[1]}")
print(f"   - Features used: {features}")
print(f"\n   - Target distribution:")
print(f"     {y_clf.value_counts().to_dict()}")
print(f"\n   - Rank change stats:")
print(f"     Mean: {y_reg.mean():.2f}")
print(f"     Std: {y_reg.std():.2f}")
print(f"     Min: {y_reg.min():.2f}, Max: {y_reg.max():.2f}")

# Create pipelines for both regression and classification
num_cols = X.columns.tolist()
pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder='drop')

# Use both linear models and ensemble methods
print("\n5. Training models...")
print("   Models: Linear Regression, Random Forest Regression")
print("   Models: Logistic Regression, Random Forest Classification")

lin_reg = Pipeline([("pre", pre), ("lin", LinearRegression())])
rf_reg = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))])

log_clf = Pipeline([("pre", pre), ("logit", LogisticRegression(max_iter=300, random_state=42))])
rf_clf = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))])

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
lin_reg_mae, rf_reg_mae = [], []
log_clf_acc, rf_clf_acc = [], []

print("\n6. Running time series cross-validation (5 folds)...")
for i, (train, test) in enumerate(tscv.split(X)):
    print(f"   Fold {i+1}/5...", end=" ")
    
    # Train regression models
    lin_reg.fit(X.iloc[train], y_reg.iloc[train])
    rf_reg.fit(X.iloc[train], y_reg.iloc[train])
    
    # Train classification models
    log_clf.fit(X.iloc[train], y_clf.iloc[train])
    rf_clf.fit(X.iloc[train], y_clf.iloc[train])
    
    # Evaluate regression models
    lin_reg_mae.append(mean_absolute_error(y_reg.iloc[test], lin_reg.predict(X.iloc[test])))
    rf_reg_mae.append(mean_absolute_error(y_reg.iloc[test], rf_reg.predict(X.iloc[test])))
    
    # Evaluate classification models
    log_clf_acc.append(accuracy_score(y_clf.iloc[test], log_clf.predict(X.iloc[test])))
    rf_clf_acc.append(accuracy_score(y_clf.iloc[test], rf_clf.predict(X.iloc[test])))
    
    print("Done")

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print("\nREGRESSION (Predicting Rank Change Magnitude):")
print(f"  Linear Regression MAE:       {np.mean(lin_reg_mae):.4f} (+/- {np.std(lin_reg_mae):.4f})")
print(f"  Random Forest Regression MAE: {np.mean(rf_reg_mae):.4f} (+/- {np.std(rf_reg_mae):.4f})")

print("\nCLASSIFICATION (Predicting Direction: Up/Flat/Down):")
print(f"  Logistic Regression Accuracy:      {np.mean(log_clf_acc):.4f} (+/- {np.std(log_clf_acc):.4f})")
print(f"  Random Forest Classification Acc:  {np.mean(rf_clf_acc):.4f} (+/- {np.std(rf_clf_acc):.4f})")

# Feature importance from Random Forest
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Random Forest Regression)")
print("="*60)
rf_reg.fit(X, y_reg)
importances = rf_reg.named_steps['rf'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

print("\n" + "="*60)
print("Model training complete!")
print("="*60)
