
"""
Visualization script for the Hybrid Ranking Model
Requires: matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("Loading and preparing data...")

# Load data
merged_df = pd.read_csv("mergedTrainingData.csv")
elo_ratings = pd.read_csv("elo_ratings_by_season.csv")

# Merge Elo ratings
merged_df = merged_df.merge(
    elo_ratings.rename(columns={'Team': 'Team', 'season': 'season', 'Elo': 'team_elo'}),
    on=['Team', 'season'],
    how='left'
)
merged_df = merged_df.merge(
    elo_ratings.rename(columns={'Team': 'opponent', 'season': 'season', 'Elo': 'opponent_elo'}),
    left_on=['opponent', 'season'],
    right_on=['opponent', 'season'],
    how='left'
)

# Feature engineering
merged_df = merged_df.dropna(subset=['AP_rank', 'team_elo'])
merged_df['is_win'] = (merged_df['win_loss'] == 'W').astype(int)
merged_df['margin'] = merged_df['points_scored'] - merged_df['points_allowed']
merged_df['home_bool'] = merged_df['home_game'].astype(int)
merged_df = merged_df.sort_values(['season', 'week', 'Team']).reset_index(drop=True)

merged_df['prev_ap_rank'] = merged_df.groupby('Team')['AP_rank'].shift(1)
merged_df['rank_change'] = merged_df['prev_ap_rank'] - merged_df['AP_rank']

merged_df['elo_diff'] = merged_df['team_elo'] - merged_df['opponent_elo'].fillna(merged_df['team_elo'].mean())
merged_df['elo_advantage'] = (merged_df['elo_diff'] > 0).astype(int)
merged_df['opp_ranked'] = (merged_df['opponent_rank'] > 0).astype(int)

merged_df['win_streak'] = merged_df.groupby('Team')['is_win'].transform(
    lambda x: x.rolling(window=3, min_periods=1).sum()
)
merged_df['avg_margin_3games'] = merged_df.groupby('Team')['margin'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df = merged_df[merged_df['prev_ap_rank'].notna()].copy()

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Distribution of Rank Changes
ax1 = fig.add_subplot(gs[0, :2])
df['rank_change'].hist(bins=30, ax=ax1, edgecolor='black', alpha=0.7)
ax1.axvline(df['rank_change'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["rank_change"].mean():.2f}')
ax1.set_xlabel('Rank Change (Positive = Moved Up)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of AP Rank Changes', fontsize=14, fontweight='bold')
ax1.legend()

# 2. Rank Change by Win/Loss
ax2 = fig.add_subplot(gs[0, 2])
win_data = df[df['is_win'] == 1]['rank_change']
loss_data = df[df['is_win'] == 0]['rank_change']
ax2.boxplot([win_data, loss_data], labels=['Win', 'Loss'])
ax2.set_ylabel('Rank Change', fontsize=12)
ax2.set_title('Rank Change:\nWin vs Loss', fontsize=12, fontweight='bold')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

# 3. Elo Rating vs Rank Change
ax3 = fig.add_subplot(gs[1, 0])
scatter = ax3.scatter(df['team_elo'], df['rank_change'], 
                      c=df['is_win'], cmap='RdYlGn', alpha=0.5, s=30)
ax3.set_xlabel('Team Elo Rating', fontsize=12)
ax3.set_ylabel('Rank Change', fontsize=12)
ax3.set_title('Elo Rating vs Rank Change', fontsize=12, fontweight='bold')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.colorbar(scatter, ax=ax3, label='Win (1) / Loss (0)')

# 4. Point Margin vs Rank Change
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(df['margin'], df['rank_change'], alpha=0.4, s=30)
z = np.polyfit(df['margin'], df['rank_change'], 1)
p = np.poly1d(z)
ax4.plot(df['margin'].sort_values(), p(df['margin'].sort_values()), 
         "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
ax4.set_xlabel('Point Margin', fontsize=12)
ax4.set_ylabel('Rank Change', fontsize=12)
ax4.set_title('Point Margin vs Rank Change', fontsize=12, fontweight='bold')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax4.legend()

# 5. Win Streak Impact
ax5 = fig.add_subplot(gs[1, 2])
win_streak_avg = df.groupby('win_streak')['rank_change'].mean()
ax5.bar(win_streak_avg.index, win_streak_avg.values, color='steelblue', edgecolor='black')
ax5.set_xlabel('Win Streak (last 3 games)', fontsize=12)
ax5.set_ylabel('Avg Rank Change', fontsize=12)
ax5.set_title('Win Streak Impact', fontsize=12, fontweight='bold')
ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)

# 6. Model Performance by Season
ax6 = fig.add_subplot(gs[2, :2])

features = [
    'prev_ap_rank', 'team_elo', 'opponent_elo', 'elo_diff', 'elo_advantage',
    'is_win', 'margin', 'home_bool', 'opp_ranked', 'win_streak', 'avg_margin_3games'
]

X = df[features].copy()
X = X.fillna(X.mean())
y = df['rank_change'].astype(float)

# Performance by season
seasons = sorted(df['season'].unique())
season_mae = []

for season in seasons:
    mask = df['season'] == season
    if mask.sum() > 10:  # Need enough samples
        X_season = X[mask]
        y_season = y[mask]
        
        # Simple train/test split
        split_idx = int(len(X_season) * 0.7)
        X_train, X_test = X_season.iloc[:split_idx], X_season.iloc[split_idx:]
        y_train, y_test = y_season.iloc[:split_idx], y_season.iloc[split_idx:]
        
        if len(X_test) > 0:
            pre = ColumnTransformer([("num", StandardScaler(), features)], remainder='drop')
            model = Pipeline([("pre", pre), ("lin", LinearRegression())])
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            season_mae.append(mae)
        else:
            season_mae.append(np.nan)
    else:
        season_mae.append(np.nan)

ax6.plot(seasons, season_mae, marker='o', linewidth=2, markersize=8, color='darkgreen')
ax6.set_xlabel('Season', fontsize=12)
ax6.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
ax6.set_title('Model Performance by Season', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Correlation Heatmap
ax7 = fig.add_subplot(gs[2, 2])
corr_features = ['rank_change', 'team_elo', 'elo_diff', 'margin', 'win_streak', 'prev_ap_rank']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax7, cbar_kws={'shrink': 0.8})
ax7.set_title('Feature Correlations', fontsize=12, fontweight='bold')

plt.suptitle('College Football AP Ranking Prediction Model - Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.savefig('model_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("\nDashboard saved as 'model_analysis_dashboard.png'")

plt.show()

print("\n" + "="*60)
print("Key Insights from Visualizations:")
print("="*60)
print(f"1. Average rank change: {df['rank_change'].mean():.2f} positions")
print(f"2. Winners move up avg: {win_data.mean():.2f} positions")
print(f"3. Losers move down avg: {loss_data.mean():.2f} positions")
print(f"4. Correlation (Elo vs Rank Change): {df['team_elo'].corr(df['rank_change']):.3f}")
print(f"5. Correlation (Margin vs Rank Change): {df['margin'].corr(df['rank_change']):.3f}")
print(f"6. Teams with 3-game win streak gain avg: {win_streak_avg.get(3, 0):.2f} positions")
print("="*60)
