# NHL Player Performance Predictor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nhl-acquired-predictions-jxeklsgca5en282fvawzyg.streamlit.app/)

A machine learning application that predicts how an NHL player will perform after joining a new team. Given a player's historical statistics, the model forecasts their expected production in a new context, helping answer one of hockey's most persistent questions: how will this player translate?

---

## What It Does

When an NHL team acquires a player, past performance in a different system is a poor direct predictor of future output. This application accounts for contextual factors like linemate quality, power play usage, and per-60 rates to produce adjusted performance forecasts for acquired players.

Users can select a player and a destination team and receive a predicted statistical output across key offensive and defensive metrics.

---

## Tech Stack

- **Language:** Python
- **ML Models:** CatBoost, LightGBM, scikit-learn ridge (separate blends for forwards and defensemen)
- **Feature Engineering:** Per-60 rate normalization, power play context, linemate quality, shooting danger metrics
- **App Framework:** Streamlit
- **Deployment:** Streamlit Cloud (live link above)
- **Containerization:** Docker

---

## Project Structure

```
app.py                          # Streamlit entry point (page setup + tab dispatch)
refresh_and_retrain.py          # Rebuild all data files, then clear the model caches
nhl_predictor/
  config.py                     # Paths, teams, model settings, feature lists
  data_io.py                    # CSV loading, ages
  features.py                   # Shared feature-engineering helpers
  training.py                   # Residual-model CV training loop, ModelBundle cache
  offense.py                    # Forward model: features, training, team-fit predictions
  defense.py                    # Defenseman model: features, training, predictions
  grading.py                    # Percentile grades, D-man archetypes
  pairing.py                    # Defensive pairing / cascade insertion
  contract.py                   # Age curves, contract projections, CBA limits
  nhl_api.py                    # All NHL API calls + name/headshot and shift caches
  charts.py                     # Plotly / matplotlib figures
  theme.py                      # Dark theme + team-coloured background
  assets/                       # Injected CSS / JS (tour, tab persistence, background)
  ui/                           # One module per tab + shared components
pipeline/
  fetch_player_ages.py          # Player ages from the NHL API (runs daily via GitHub Actions)
  season_dataset.py             # MoneyPuck game-level → season_dataset.csv
  power_play.py                 # → pp_features.csv
  defensive_dataset.py          # → defensive_dataset.csv
  linemates.py                  # MoneyPuck lines → linemate_features.csv
  data_sources.py               # Reads every export in raw_data/
  nhl_api_download.py           # NHL API alternative: raw play-by-play + shifts → raw_data/nhl_api/
  nhl_api_parse.py              #   → shots / line-change segments / player-game tables
  nhl_api_xg.py                 #   → own expected-goals model (cross-fitted by season)
  nhl_api_datasets.py           #   → the same four CSVs as the MoneyPuck scripts
  nhl_api_rapm.py               #   → regularised adjusted plus-minus (research; not used by the app)
  legacy/                       # One-off scripts from the original raw-data workflow
tests/                          # Unit tests for pure logic (python -m pytest tests)
```

Trained models are cached as `trained_models_forwards_v7.joblib` and
`defensive_models_v3.joblib`; if they are missing the app trains on first launch.

---

## Data Pipeline

1. **Collection:** Every regular-season game's play-by-play and shift charts from the public NHL API (2010-11 on), with our own expected-goals model
2. **Cleaning:** Duplicate removal, missing value handling, team name normalization
3. **Feature Engineering:** Per-60 rate construction, power play usage, linemate context, shooting danger zones, player age curves
4. **Modeling:** Separate model blends for forwards (CatBoost + ridge) and defensemen (LightGBM + CatBoost), each predicting how far a player will land above or below a recent-form baseline
5. **Serving:** Streamlit interface surfaces predictions with feature importance context

---

## Running Locally

Requires Python 3.12 (3.11+ works). Run every command from the repo root.

**With Python:**
```bash
git clone https://github.com/zach340/NHL-acquired-predictions.git
cd NHL-acquired-predictions
git lfs pull                       # the CSV data files are stored in Git LFS
pip install -r requirements.txt
python -m streamlit run app.py     # opens http://localhost:8501
```

The first launch trains both models (~15 min) and caches them as
`trained_models_forwards_v7.joblib` / `defensive_models_v3.joblib`; later
launches load them in seconds. Delete those files (or use the retrain buttons
on the **Models** tab) to retrain.

**With Docker:**
```bash
docker build -t nhl-predictor .
docker run -p 7860:7860 nhl-predictor   # opens http://localhost:7860
```

**Tests:**
```bash
pip install pytest
python -m pytest tests
```

**Refreshing the data** straight from the NHL API (no manual downloads; the
first run fetches every game since 2010-11 into `raw_data/nhl_api/` and takes
~4-5 hours, later runs only fetch new games):
```bash
python refresh_and_retrain.py      # rebuild every CSV, then clear the model caches
python pipeline/fetch_player_ages.py   # ages only (also runs daily via GitHub Actions)
```

The original MoneyPuck pipeline still works (needs the raw exports in
`raw_data/game_level/` and `raw_data/line_level/` — see `pipeline/data_sources.py`):
```bash
python refresh_and_retrain.py --source moneypuck
```

Seasons are labelled by their start year throughout (2024 = the 2024-25 season).

---

## Key Features

- Separate models for forwards and defensemen
- Per-60 rate normalization to account for ice time differences
- Power play context features to adjust for usage differences between teams
- Linemate quality scoring to isolate individual player contribution
- Feature importance visualization to explain each prediction
- Live deployed application accessible without local setup
