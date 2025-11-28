"""
Prediction Script for College Football AP Ranking Changes
Uses the trained hybrid model with Elo ratings
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle

def train_and_save_models():
    """Train the models and save them for future predictions"""
    print("Training models on full dataset...")
    
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
    
    # Features
    features = [
        'prev_ap_rank', 'team_elo', 'opponent_elo', 'elo_diff', 'elo_advantage',
        'is_win', 'margin', 'home_bool', 'opp_ranked', 'win_streak', 'avg_margin_3games'
    ]
    
    X = df[features].copy()
    X = X.fillna(X.mean())
    
    y_reg = df['rank_change'].astype(float)
    df['dir'] = pd.cut(df['rank_change'], bins=[-99, -2, 2, 99], labels=['down', 'flat', 'up'])
    y_clf = df['dir']
    
    mask = y_clf.notna()
    X = X[mask]
    y_reg = y_reg[mask]
    y_clf = y_clf[mask]
    
    # Train models
    num_cols = X.columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder='drop')
    
    reg_model = Pipeline([("pre", pre), ("lin", LinearRegression())])
    clf_model = Pipeline([("pre", pre), ("logit", LogisticRegression(max_iter=300, random_state=42))])
    
    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)
    
    # Save models
    with open('rank_change_regressor.pkl', 'wb') as f:
        pickle.dump(reg_model, f)
    with open('rank_direction_classifier.pkl', 'wb') as f:
        pickle.dump(clf_model, f)
    
    print("Models saved successfully!")
    return reg_model, clf_model, features


def predict_rank_change(team_name, current_rank, team_elo, opponent_elo, 
                        won_game, point_margin, is_home, opponent_ranked,
                        recent_win_count, recent_avg_margin):
    """
    Predict how a team's AP ranking will change after a game
    
    Parameters:
    -----------
    team_name : str
        Name of the team
    current_rank : int
        Current AP ranking (1-25)
    team_elo : float
        Team's Elo rating
    opponent_elo : float
        Opponent's Elo rating
    won_game : bool
        Did the team win?
    point_margin : int
        Point differential (positive if won, negative if lost)
    is_home : bool
        Was it a home game?
    opponent_ranked : bool
        Was opponent ranked?
    recent_win_count : int
        Number of wins in last 3 games (0-3)
    recent_avg_margin : float
        Average point margin in last 3 games
    
    Returns:
    --------
    dict : Predictions including magnitude and direction
    """
    try:
        # Load models
        with open('rank_change_regressor.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        with open('rank_direction_classifier.pkl', 'rb') as f:
            clf_model = pickle.load(f)
    except FileNotFoundError:
        print("Models not found. Training new models...")
        reg_model, clf_model, _ = train_and_save_models()
    
    # Prepare features
    features_dict = {
        'prev_ap_rank': current_rank,
        'team_elo': team_elo,
        'opponent_elo': opponent_elo,
        'elo_diff': team_elo - opponent_elo,
        'elo_advantage': 1 if team_elo > opponent_elo else 0,
        'is_win': 1 if won_game else 0,
        'margin': point_margin,
        'home_bool': 1 if is_home else 0,
        'opp_ranked': 1 if opponent_ranked else 0,
        'win_streak': recent_win_count,
        'avg_margin_3games': recent_avg_margin
    }
    
    X_pred = pd.DataFrame([features_dict])
    
    # Make predictions
    rank_change_pred = reg_model.predict(X_pred)[0]
    direction_pred = clf_model.predict(X_pred)[0]
    direction_proba = clf_model.predict_proba(X_pred)[0]
    
    # Calculate new predicted rank
    new_rank_pred = current_rank - rank_change_pred  # Subtract because lower number = better rank
    new_rank_pred = max(1, min(25, round(new_rank_pred)))  # Clamp to 1-25
    
    return {
        'team': team_name,
        'current_rank': current_rank,
        'predicted_rank_change': round(rank_change_pred, 2),
        'predicted_new_rank': new_rank_pred,
        'predicted_direction': direction_pred,
        'direction_probabilities': {
            'down': round(direction_proba[0], 3),
            'flat': round(direction_proba[1], 3),
            'up': round(direction_proba[2], 3)
        }
    }


def print_prediction(result):
    """Pretty print prediction results"""
    print("\n" + "="*60)
    print(f"RANKING PREDICTION FOR {result['team'].upper()}")
    print("="*60)
    print(f"\nCurrent Rank: #{result['current_rank']}")
    print(f"Predicted Rank Change: {result['predicted_rank_change']:+.2f} positions")
    print(f"Predicted New Rank: #{result['predicted_new_rank']}")
    print(f"\nDirection: {result['predicted_direction'].upper()}")
    print(f"Confidence Breakdown:")
    print(f"  Up:   {result['direction_probabilities']['up']:.1%}")
    print(f"  Flat: {result['direction_probabilities']['flat']:.1%}")
    print(f"  Down: {result['direction_probabilities']['down']:.1%}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Train models if they don't exist
    import os
    if not os.path.exists('rank_change_regressor.pkl'):
        print("Training models for the first time...")
        train_and_save_models()
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example 1: Top team wins convincingly at home against ranked opponent
    result1 = predict_rank_change(
        team_name="Ohio State",
        current_rank=5,
        team_elo=1750,
        opponent_elo=1680,
        won_game=True,
        point_margin=21,
        is_home=True,
        opponent_ranked=True,
        recent_win_count=3,
        recent_avg_margin=18.5
    )
    print_prediction(result1)
    
    # Example 2: Mid-ranked team loses close road game
    result2 = predict_rank_change(
        team_name="Tennessee",
        current_rank=12,
        team_elo=1620,
        opponent_elo=1700,
        won_game=False,
        point_margin=-3,
        is_home=False,
        opponent_ranked=True,
        recent_win_count=2,
        recent_avg_margin=10.3
    )
    print_prediction(result2)
    
    # Example 3: Lower-ranked team upsets higher-ranked opponent
    result3 = predict_rank_change(
        team_name="Indiana",
        current_rank=20,
        team_elo=1580,
        opponent_elo=1720,
        won_game=True,
        point_margin=7,
        is_home=True,
        opponent_ranked=True,
        recent_win_count=2,
        recent_avg_margin=5.7
    )
    print_prediction(result3)
