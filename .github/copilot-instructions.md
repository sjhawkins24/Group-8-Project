# Copilot Instructions

## Project Snapshot
- College football ranking predictor using ML to predict AP Poll rank changes. Architecture centers on `models/elo_hybrid.py` for feature engineering and model persistence, with Streamlit UI (`app.py`) and prediction services under `services/`.
- **Core production path**: `train_hybrid_model.py` → `artifacts/*.pkl` → `services/ranking_predictor.py` → `app.py`
- Data sources: `03 - Cleaned Data Space/mergedTrainingData.csv` (team game logs) + `04 - Elo Space/elo_ratings_by_season.csv` (computed team strengths)
- Elo pipelines (`04 - Elo Space/Justin Elo Test.py`, `Justin Elo Weekly.py`) generate strength ratings from game outcomes
- Legacy experiments (`HybridModelTest.py`, `PredictRankings.py`, `apputil.py`) exist but avoid—use production services instead
- Notebooks in `01 - Exploration Space/` are research only—keep production logic in `.py` scripts

## Data Sources & Prep
- **Critical**: `mergedTrainingData.csv` and `CleanedTrainingData.csv` store booleans as strings "TRUE"/"FALSE"—use `as_bool()` pattern from `Justin Elo Test.py:26` for normalization: `s.strip().upper() in {"TRUE", "T", "YES", "1"}`
- **Time ordering**: Always process data chronologically with `sort_values(['season','week','Team'])` before feature engineering—target derivation depends on correct temporal sequence
- **Elo reset behavior**: Each season starts all teams at 1500 rating (no carryover)—reflects college football roster turnover reality
- **Path handling**: Windows + spaced directories need quotes: `python "04 - Elo Space/Justin Elo Weekly.py"`
- **Dual perspectives**: CSVs contain both team/opponent views of same game—use `seen` guards or explicit deduplication to avoid double-counting

## Modeling Workflow
- **Feature schema**: `models/elo_hybrid.py:FEATURE_COLUMNS` defines the canonical 11-feature set—extend here to propagate changes across both regression/classification pipelines
- **Target engineering**: `prev_ap_rank` comes from shifting `ap_rank` within team-season groups—maintains temporal causality for ranking predictions
- **Validation**: Time-series split (`TimeSeriesSplit(n_splits=5)`) respects chronological order; `evaluate_time_series()` reports MAE for regression + accuracy for classification
- **Artifacts**: Run `python train_hybrid_model.py --force` to retrain and save models to `artifacts/` (auto-created on first load if missing)
- **Performance**: Current hybrid achieves ~2.4 MAE rank change, 63.8% direction accuracy (52% improvement over baseline—see `MODEL_IMPROVEMENTS.md`)
- **Pipeline updates**: When changing features, update both regressors and classifiers + regenerate metadata + document in `MODEL_IMPROVEMENTS.md`

## Elo Rating Pipelines
- **Season-end Elo**: `04 - Elo Space/Justin Elo Test.py` reads master CSV from GitHub, removes `Bye` weeks, computes ratings with MOV multiplier (FiveThirtyEight style), outputs `elo_ratings_by_season.csv` + seasonal top-25 rankings
- **Weekly Elo**: `04 - Elo Space/Justin Elo Weekly.py` records pre-game team ratings per week in `elo_ratings_weekly.csv`—merge on `['Team','season','week']` for method comparisons
- **Deduplication**: Both scripts handle dual team perspectives using `seen` sets—mimic this pattern when processing game-level data to avoid double-counting matchups
- **Algorithm**: K=22.0, HFA=55.0 points, MOV multiplier based on log(1+margin) with Elo difference scaling—reuses established rating formulas

## Prediction APIs
- **Primary interface**: `services/ranking_predictor.RankPredictor` handles all inference—loads artifacts via `ensure_models()`, builds feature vectors with Elo lookups, supports both dataset and what-if predictions
- **Return format**: `PredictionResult` dataclass with team, opponent, current rank, predicted change, new rank, direction, probabilities, and feature payload
- **Direction mapping**: `pd.cut` bins `[-99,-2,2,99]` → `['down','flat','up']` labels—all downstream consumers must align with this schema
- **Model sync**: Keep `artifacts/rank_change_regressor.pkl` and `artifacts/rank_direction_classifier.pkl` consistent with code; delete before retraining if schema drifts

## Streamlit Front End
- **Architecture**: `app.py` consumes `RankPredictor` for model inference; optional `services/chat_insights.ChatInsightGenerator` provides OpenRouter-powered narrative summaries
- **Environment**: Requires `OPENROUTER_API_KEY` for chat features (optional: `OPENROUTER_MODEL`, `OPENROUTER_REFERER`, `OPENROUTER_TITLE`)
- **Data flow**: Season/week/team selectors → merged dataset → user input adjustments → what-if feature builder → prediction display
- **Performance**: Use `st.cache_resource`/`st.cache_data` for expensive operations; follow existing caching patterns when adding new data loaders

## Conventions & Gotchas
- **Data duplication**: Many CSVs track both team perspectives per game—always drop or reconcile duplicate rows before aggregations
- **Boolean handling**: Handle `home_game` as ints via `.astype(int)` only after cleaning `TRUE`/`FALSE` strings; otherwise regressors see mixed dtypes
- **Path issues**: Windows paths plus spaced directories require quoting in scripts and tests; prefer `Path` objects for new code
- **Legacy code**: `apputil.py` retains unresolved merge markers—avoid depending on it until cleaned or expect inconsistent behavior
- **Temporal causality**: Previous rank features must respect chronological order—always sort by season/week before shifting values

## Typical Commands (PowerShell)
- Train & save hybrid model: `python train_hybrid_model.py --force`
- Generate season Elo: `python "04 - Elo Space/Justin Elo Test.py"`
- Generate weekly Elo: `python "04 - Elo Space/Justin Elo Weekly.py"`
- Compare Elo methods: `python CompareEloMethods.py`
- Launch Streamlit UI: `streamlit run app.py`

## When Updating
- **Data refresh**: Regenerate Elo CSVs before retraining if source data changes; downstream merges assume fresh values
- **Model artifacts**: Remove and rebuild artifacts under `artifacts/` whenever modifying feature engineering or model hyperparameters
- **Documentation**: Document material modeling shifts in `MODEL_IMPROVEMENTS.md` so analysts can track feature importance deltas
- **Dependencies**: Add any new third-party integrations (e.g., chat providers) to `services/` and extend `requirements.txt` accordingly; keep environment variables documented in `README`/deployment notes
