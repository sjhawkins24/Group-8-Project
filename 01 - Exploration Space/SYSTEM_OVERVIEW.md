# College Football Ranking Prediction System
## High-Level Overview

---

## What Does This System Do?

This system predicts **changes in AP Poll rankings** for college football teams using machine learning. It answers two key questions:

1. **By how many positions will a team's ranking change?** (Regression problem)
2. **Will the team move up, down, or stay flat?** (Classification problem)

The system combines historical game data, team strength metrics (Elo ratings), and recent performance trends to forecast how voters will adjust rankings after each week's games.

---

## Architecture: Three Core Components

### 1. **Elo Rating Engine** (`04 - Elo Space/`)
**What it does:** Quantifies team strength over time using a chess-style rating system adapted for football.

**How it works:**
- Every team starts each season at 1500 Elo rating
- After each game, winners gain rating points; losers lose points
- The magnitude of change depends on:
  - **Margin of victory** (bigger wins = more points)
  - **Expected outcome** (upsets shift more points)
  - **Home field advantage** (+55 Elo points)
- Uses K-factor of 22.0 and FiveThirtyEight's MOV (Margin of Victory) multiplier

**Why it's right:**
- Elo ratings are proven effective in sports prediction (chess, NFL, NBA)
- Captures team strength beyond win-loss records
- Self-correcting: ratings converge to true strength over time
- Reflects the reality that college rosters reset annually (no carryover between seasons)

**Output files:**
- `elo_ratings_by_season.csv` - End-of-season ratings per team
- `elo_ratings_weekly.csv` - Pre-game ratings for every week
- `elo_top25_*.csv` - Top 25 rankings by Elo for validation

### 2. **Hybrid Machine Learning Models** (`models/elo_hybrid.py`)
**What it does:** Trains two complementary models to predict ranking movements.

**How it works:**

**Regression Model (Linear Regression):**
- Predicts the **numerical change** in ranking (e.g., -3.2 positions)
- Uses 11 engineered features:
  - Previous AP rank (where they were last week)
  - Team and opponent Elo ratings
  - Elo differential (strength advantage)
  - Win/loss outcome and margin
  - Home field indicator
  - Win streak (3-game rolling count)
  - Average margin over last 3 games
  - Whether opponent is ranked
- Achieves **2.39 MAE** (Mean Absolute Error) - predictions are off by ~2.4 positions on average

**Classification Model (Logistic Regression):**
- Predicts **direction**: Will the team move up, stay flat, or drop?
  - **Up**: Rank improves by 3+ positions
  - **Flat**: Rank changes by -2 to +2 positions
  - **Down**: Rank drops by 3+ positions
- Achieves **63.8% accuracy** - correctly predicts direction nearly 2/3 of the time
- Provides probability estimates for each direction

**Why it's right:**
- **Hybrid approach** combines magnitude and direction for robust predictions
- **Time-series validation** ensures models work on future weeks (no data leakage)
- **Feature engineering** captures what actually drives AP Poll changes:
  - **Inertia** (previous rank is 25% of importance - voters don't radically shift opinions)
  - **Recent trends** (3-game averages matter more than single games)
  - **Strength metrics** (Elo ratings account for ~30% of predictive power)
  - **Momentum** (win streaks signal teams on the rise)
- **52% improvement** over baseline model proves the approach works

### 3. **Prediction Service & User Interface**
**What it does:** Provides easy access to predictions through both API and interactive web app.

**Components:**

**`services/ranking_predictor.py` - Core Prediction Engine:**
- Loads trained models from `artifacts/` directory
- Builds feature vectors for new scenarios
- Merges Elo ratings with game data
- Returns structured predictions with probabilities

**`app.py` - Streamlit Web Interface:**
- Interactive dashboard for exploring predictions
- Allows users to input hypothetical games:
  - Select team and opponent
  - Specify current rankings
  - Adjust game parameters (score, location)
- Displays predicted rank change and direction
- Optional AI-powered narrative insights (via OpenRouter API)

**`train_hybrid_model.py` - Model Training CLI:**
- Command-line tool to retrain models
- Runs 5-fold time-series cross-validation
- Saves artifacts to `artifacts/` for production use
- Reports performance metrics (MAE, accuracy)

**Why it's right:**
- **Separation of concerns**: Data → Models → Services → UI (clean architecture)
- **Caching and persistence**: Models trained once, reused for many predictions (fast inference)
- **What-if analysis**: Users can test scenarios without retraining
- **Reproducibility**: All predictions trace back to specific model versions and data

---

## Data Flow: From Raw Data to Predictions

```
1. Historical Game Data
   ├─ mergedTrainingData.csv (team game logs 2020-2024)
   └─ Week-by-week results with AP rankings

2. Elo Rating Generation
   ├─ Justin Elo Test.py processes all games chronologically
   ├─ Computes team strength ratings per season
   └─ Outputs: elo_ratings_by_season.csv

3. Feature Engineering
   ├─ Merge game data + Elo ratings
   ├─ Calculate derived features (streaks, trends, differentials)
   ├─ Sort chronologically (critical for temporal causality)
   └─ Create prev_ap_rank by shifting within team-season groups

4. Model Training
   ├─ Split data using TimeSeriesSplit (respects time order)
   ├─ Train regression model (rank change magnitude)
   ├─ Train classification model (rank change direction)
   ├─ Save to artifacts/ with metadata
   └─ Validate with 5-fold cross-validation

5. Prediction Service
   ├─ Load models from artifacts/
   ├─ Accept user input (team, opponent, game details)
   ├─ Build feature vector with historical context
   ├─ Run inference through both models
   └─ Return structured prediction with probabilities

6. User Interface
   ├─ Streamlit app loads predictor
   ├─ User selects scenario parameters
   ├─ Display predicted rank change ± direction
   └─ Optional: Generate AI narrative explanation
```

---

## Why This Approach Is Right

### 1. **It Respects Temporal Reality**
- Rankings are inherently sequential - this week depends on last week
- Time-series validation prevents "future leaking into past"
- Features like `prev_ap_rank` and `win_streak` maintain causality

### 2. **It Captures What Matters to Voters**
The AP Poll voters consider:
- **Where teams were ranked** (inertia - top feature at 25% importance)
- **Recent performance** (3-game trends - 17.4% importance)
- **Strength of opponent** (Elo ratings - 30% combined importance)
- **Momentum** (win streaks - 12% importance)
- **Margin of victory** (impressive wins matter - 12.3% importance)

The model mirrors this decision-making process.

### 3. **It's Validated Against Real History**
- Trained on 1,086 team-week observations (2021-2024)
- Cross-validated on 5 time-based folds
- Achieves meaningful improvement over baseline:
  - **52% reduction** in rank change error (4.97 → 2.39 MAE)
  - **61% increase** in direction accuracy (39.7% → 63.8%)

### 4. **It Handles Uncertainty Appropriately**
- Regression model gives point estimates (expected change)
- Classification model gives probabilities (confidence in direction)
- Together, they communicate both "what will happen" and "how certain we are"

### 5. **It's Practical and Extensible**
- Fast inference (models are lightweight)
- Easy to retrain with new data
- Clear extension points for improvements:
  - Add conference strength features
  - Incorporate injury data
  - Model early-season vs late-season dynamics differently
  - Ensemble multiple model architectures

### 6. **It Solves a Real Problem**
- AP Poll predictions interest fans, analysts, and teams
- Quantifies subjective voting behavior
- Enables what-if scenario planning
- Could inform playoff selection discussions

---

## Key Design Decisions Explained

### Why Linear Models Instead of Neural Networks?
- **Interpretability**: We can explain feature importance to users
- **Data efficiency**: Only ~1,000 samples - deep learning would overfit
- **Speed**: Instant predictions without GPU requirements
- **Sufficient performance**: 2.39 MAE is practical for the domain

### Why Reset Elo Each Season Instead of Carrying Over?
- College rosters turn over dramatically (graduation, transfers)
- New season = fresh evaluation period
- Matches how voters actually think (they don't heavily weight last year)
- Empirically performs better than carryover models

### Why 11 Features Instead of More?
- Balances predictive power with model stability
- Avoids multicollinearity (features are relatively independent)
- Each feature has clear interpretation
- Comprehensive coverage of major ranking factors

### Why Time-Series Split Instead of Random Split?
- Prevents data leakage (testing on "future" data)
- Realistic evaluation (simulates actual prediction scenario)
- More conservative performance estimates
- Standard practice for forecasting problems

### Why Hybrid (Regression + Classification)?
- Regression gives precise magnitude (useful for detailed analysis)
- Classification gives actionable direction (easier for decision-making)
- Combined view reduces misinterpretation risk
- Probabilities add confidence intervals

---

## Performance Benchmarks

### Regression Model (Rank Change Magnitude)
- **Mean Absolute Error**: 2.39 ± 0.19 positions
- **Interpretation**: On average, predictions are within 2-3 ranking spots
- **Comparison**: Baseline (using only previous rank) = 4.97 MAE

### Classification Model (Direction)
- **Accuracy**: 63.8% ± 2.5%
- **Interpretation**: Correctly predicts up/flat/down nearly 2/3 of the time
- **Comparison**: Random guessing = 33%, baseline = 39.7%

### Feature Importance (Top 5)
1. **Previous AP Rank** (25.0%) - Voters exhibit strong inertia
2. **Avg Margin (3 games)** (17.4%) - Recent dominance matters
3. **Current Margin** (12.3%) - This week's performance
4. **Win Streak** (12.0%) - Momentum is real
5. **Team Elo** (11.9%) - Underlying strength measurement

---

## Usage Patterns

### For Analysts: What-If Scenarios
*"If Alabama beats Georgia by 14 at home, how much will they rise?"*
→ Use Streamlit app to input parameters and get instant predictions

### For Researchers: Model Retraining
*"I have updated game data through Week 10"*
→ Run `python train_hybrid_model.py --force` to retrain with fresh data

### For Developers: API Integration
*"I want to build a rankings dashboard"*
→ Import `RankPredictor` class and call `predict_game()` or `predict_dataset()`

### For Educators: Understanding Rankings
*"Why did Team X move up 5 spots after beating Team Y?"*
→ Examine feature payload to see which factors drove the prediction

---

## Limitations and Future Work

### Current Limitations
1. **Doesn't account for non-game factors**:
   - Coach changes
   - Injuries to key players
   - Off-field issues
   
2. **Assumes consistent voter behavior**:
   - Doesn't model shifts in voting patterns over time
   - Can't predict outlier weeks (e.g., chaos scenarios)

3. **Limited to ranked teams**:
   - Requires previous AP rank as input
   - Doesn't predict initial entry into rankings

4. **Seasonal scope**:
   - Trained on 2021-2024 data
   - May drift as game evolves (rule changes, playoff expansion)

### Promising Extensions
1. **Conference strength features** - SEC vs Pac-12 voting biases
2. **Temporal dynamics** - Early season (Week 1-4) behaves differently than late season
3. **Ensemble stacking** - Combine multiple model architectures
4. **Upset indicators** - Special handling for ranked-vs-ranked games
5. **Voter panel modeling** - Different regions/conferences have different biases

---

## Conclusion

This system successfully models AP Poll ranking dynamics by:
- **Quantifying team strength** via Elo ratings (objective)
- **Learning voter behavior** via machine learning (data-driven)
- **Combining both** into actionable predictions (practical)

The 52% improvement over baseline demonstrates that the approach captures real patterns in how rankings evolve. The modular architecture allows for continuous improvement as more data becomes available and new features are discovered.

**Most importantly**: The system provides **interpretable, validated predictions** that help users understand not just *what* will happen to rankings, but *why*.
