"""Example script to make a prediction and examine the feature payload."""
from services.ranking_predictor import RankPredictor

# Load the predictor
predictor = RankPredictor()

# Make a prediction for a specific game
result = predictor.predict(
    team="Ohio State",
    opponent="Michigan",
    season=2024,
    week=13,
    points_scored=42,
    points_allowed=27,
    home_game=True,
    current_rank=2,
    opponent_rank=3,
)

# Print the prediction results
print("=" * 60)
print(f"PREDICTION FOR {result.team} vs {result.opponent}")
print("=" * 60)
print(f"\nCurrent Rank: #{result.current_rank}")
print(f"Predicted Rank Change: {result.predicted_rank_change:+.2f}")
print(f"Predicted New Rank: #{result.predicted_new_rank}")
print(f"Direction: {result.predicted_direction.upper()}")

# Print direction probabilities
print(f"\nDirection Probabilities:")
for direction, prob in result.direction_probabilities.items():
    print(f"  {direction:>6s}: {prob:.1%}")

# Print the feature payload (what drove the prediction)
print(f"\nFeature Payload (what the model saw):")
print("-" * 60)
for feature, value in result.feature_payload.items():
    print(f"  {feature:20s}: {value:>10.2f}")

print("\n" + "=" * 60)
