# Copilot Instructions

## Project Snapshot
- College football ranking predictor that now standardises on `models/elo_hybrid.py` for feature engineering and model persistence, with Streamlit UI (`app.py`) and helper services under `services/`.
- Core data lives in `03 - Cleaned Data Space/mergedTrainingData.csv` plus season Elo tables in `04 - Elo Space/elo_ratings_by_season.csv`. Elo pipelines (`Justin Elo Test.py`, `Justin Elo Weekly.py`) still refresh those inputs.
- Legacy experiments (`HybridModelTest.py`, `PredictRankings.py`) remain for reference but the production path goes through `train_hybrid_model.py` and `services/ranking_predictor.py`.
- Notebooks inside `01 - Exploration Space/` are exploratory only—keep production logic in `.py` scripts.

## Data Sources & Prep
- `mergedTrainingData.csv` and `CleanedTrainingData.csv` contain boolean flags as "TRUE"/"FALSE"; reuse helper `as_bool` pattern from `Justin Elo Test.py` when normalizing.
- Elo scripts start every season from the same baseline rating (1500) for each team—do not carry ratings across seasons.
- Travel paths include spaces; when scripting in PowerShell use quotes: `python "04 - Elo Space/Justin Elo Weekly.py"`.

## Modeling Workflow
- `models/elo_hybrid.py` exports constants for the 11-feature set (`prev_ap_rank`, `team_elo`, `opponent_elo`, `elo_diff`, `elo_advantage`, `is_win`, `margin`, `home_bool`, `opp_ranked`, `win_streak`, `avg_margin_3games`). Extend features there and regenerate metadata.
- Keep the time-aware ordering (`sort_values(['season','week','Team'])`) and target derivation (`prev_ap_rank` shift) whenever training or augmenting models.
- Cross-validation uses `TimeSeriesSplit(n_splits=5)`; `evaluate_time_series()` already reports MAE/accuracy—update this when changing the pipeline.
- Run `python train_hybrid_model.py [--force]` to (re)train and save artifacts inside `artifacts/`. The helper autocreates models if they are missing when predictors load.
- When adding features, propagate them to both regression and classification pipelines and update documentation (`MODEL_IMPROVEMENTS.md`).

## Elo Rating Pipelines
- Season-end Elo: `04 - Elo Space/Justin Elo Test.py` reads master CSV from GitHub, removes `Bye`, computes ratings with MOV multiplier, and writes `elo_ratings_by_season.csv` plus a per-season top 25.
- Weekly Elo: `04 - Elo Space/Justin Elo Weekly.py` records pre-game ratings per team-week in `elo_ratings_weekly.csv`; merge on `['Team','season','week']` when comparing methods (`CompareEloMethods.py`).
- Both scripts de-duplicate dual team rows; mimic their `seen` guard when ingesting mirrored team/opponent records.

## Prediction APIs
- Prefer `services/ranking_predictor.RankPredictor` for inference. It loads artifacts via `ensure_models()`, auto-builds feature vectors (including Elo lookups), and exposes both dataset-backed and what-if predictions.
- Direction labels rely on `pd.cut` bins `[-99,-2,2,99]` mapping to `['down','flat','up']`; align any downstream consumers (e.g., UI) with these labels and probabilities.
- Keep `artifacts/rank_change_regressor.pkl` and `artifacts/rank_direction_classifier.pkl` in sync with code changes; delete them before retraining if schema drifts.

## Streamlit Front End
- `app.py` now consumes `RankPredictor` for model output and optionally calls `services/chat_insights.ChatInsightGenerator` via OpenRouter. Provide `OPENROUTER_API_KEY` (plus optional `OPENROUTER_MODEL`, `OPENROUTER_REFERER`, `OPENROUTER_TITLE`) before enabling the toggle.
- Season/week/team selectors pull from the merged dataset; user-adjusted points feed the what-if feature builder. Elo resets are handled automatically by the training scripts.
- Cache expensive loaders with `st.cache_resource`/`st.cache_data` to keep UI snappy; follow the existing pattern when adding new data hooks.

## Conventions & Gotchas
- Many CSVs track both team perspectives per game; always drop or reconcile duplicate rows before aggregations.
- Handle `home_game` as ints via `.astype(int)` only after cleaning `TRUE`/`FALSE` strings; otherwise regressors see mixed dtypes.
- Windows paths plus spaced directories require quoting in scripts and tests; prefer `Path` objects for new code.
- `apputil.py` retains unresolved merge markers—avoid depending on it until cleaned or expect inconsistent behavior.

## Typical Commands (PowerShell)
- Train & save hybrid model: `python train_hybrid_model.py --force`
- Generate season Elo: `python "04 - Elo Space/Justin Elo Test.py"`
- Generate weekly Elo: `python "04 - Elo Space/Justin Elo Weekly.py"`
- Compare Elo methods: `python CompareEloMethods.py`
- Launch Streamlit UI: `streamlit run app.py`

## When Updating
- Regenerate Elo CSVs before retraining if source data changes; downstream merges assume fresh values.
- Remove and rebuild artifacts under `artifacts/` whenever modifying feature engineering or model hyperparameters.
- Document material modeling shifts in `MODEL_IMPROVEMENTS.md` so analysts can track feature importance deltas.
- Add any new third-party integrations (e.g., chat providers) to `services/` and extend `requirements.txt` accordingly; keep environment variables documented in `README`/deployment notes.
