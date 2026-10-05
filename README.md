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
  validation.py                 # Holdout comparison shared by the Validation tab and weekly snapshot
  models.py                     # Model blends (CatBoost / LightGBM / ridge)
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
  nhl_api_download.py           # Raw play-by-play + shift charts → raw_data/nhl_api/
  nhl_api_parse.py              #   → shots / line-change segments / player-game tables
  nhl_api_xg.py                 #   → expected-goals model (cross-fitted by season)
  nhl_api_datasets.py           #   → season_dataset / defensive_dataset / pp_features / linemate_features CSVs
scripts/
  validation_snapshot.py        # Weekly: saved models vs this season's NHL API stats → validation_history.csv
  yearly_retrain.py / .bat      # Yearly: refresh data, retrain, log old vs new metrics (no commit)
  daily_refresh_ages.bat        # Local copy of the daily ages refresh
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

Seasons are labelled by their start year throughout (2024 = the 2024-25 season).

### Scheduled jobs

**Weekly validation snapshot** — `.github/workflows/weekly-validation-snapshot.yml`
runs every Monday: it loads the saved models (no retraining), compares them with
the current season's NHL API stats exactly as the Validation tab does, and
commits one row to `validation_history.csv`. It skips quietly until players have
10+ games. The Validation tab charts this history. Run it by hand from the
repo's **Actions** tab (*Run workflow*) or locally with
`python scripts/validation_snapshot.py`.

**Yearly retrain** — `scripts/yearly_retrain.bat` (Windows Task Scheduler, July):
refreshes the data (only new or changed seasons are downloaded, parsed and
xG-scored), retrains both models, saves the caches, and appends old vs new
holdout and CV metrics to `yearly_retrain.log`. Nothing is committed: check the
log's comparison tables (anything >5% worse is marked `<-- CHECK`), then commit
the CSVs and `.joblib` files yourself. Register it once from PowerShell:
```powershell
$repo = "C:\Users\Zachc\OneDrive\Documents\GitHub\NHL-acquired-predictions"
schtasks /Create /TN "NHL Predictor yearly retrain" /SC MONTHLY /M JUL /D 15 /ST 03:00 `
  /TR "`"$repo\scripts\yearly_retrain.bat`""
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Set-ScheduledTask -TaskName "NHL Predictor yearly retrain" -Settings $s
```

---

## Key Features

- Separate models for forwards and defensemen
- Per-60 rate normalization to account for ice time differences
- Power play context features to adjust for usage differences between teams
- Linemate quality scoring to isolate individual player contribution
- Feature importance visualization to explain each prediction
- Live deployed application accessible without local setup
