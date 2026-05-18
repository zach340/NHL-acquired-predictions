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
- **ML Models:** scikit-learn, LightGBM (separate models for forwards and defensemen)
- **Feature Engineering:** Per-60 rate normalization, power play context, linemate quality, shooting danger metrics
- **App Framework:** Streamlit
- **Deployment:** Streamlit Cloud (live link above)
- **Containerization:** Docker

---

## Project Structure

```
app.py                        # Streamlit application entry point
model_utils.py                # Model loading and prediction logic
process_hockey_data.py        # Core data processing pipeline
cleaning_and_shrinking.py     # Data validation and cleaning
combining_by_season.py        # Season-level aggregation
addingx60.py                  # Per-60 rate feature construction
adding_Power_play.py          # Power play context features
lines.py                      # Linemate quality features
shooting_danger.py            # Shooting danger zone features
defensive.py                  # Defensive metric processing
fetch_player_ages.py          # Player age data pipeline
trained_models_forwards_v5.joblib   # Serialized forward models
defensive_models.joblib             # Serialized defensive models
dockerfile                    # Container configuration
requirements.txt              # Python dependencies
```

---

## Data Pipeline

1. **Collection:** Historical NHL player statistics aggregated by season
2. **Cleaning:** Duplicate removal, missing value handling, team name normalization
3. **Feature Engineering:** Per-60 rate construction, power play usage, linemate context, shooting danger zones, player age curves
4. **Modeling:** Separate LightGBM models trained for forwards and defensemen across multiple statistical targets
5. **Serving:** Streamlit interface surfaces predictions with feature importance context

---

## Running Locally

**With Python:**
```bash
git clone https://github.com/zach340/NHL-acquired-predictions.git
cd NHL-acquired-predictions
pip install -r requirements.txt
streamlit run app.py
```

**With Docker:**
```bash
docker build -t nhl-predictor .
docker run -p 8501:8501 nhl-predictor
```

Then open `http://localhost:8501` in your browser.

---

## Key Features

- Separate models for forwards and defensemen
- Per-60 rate normalization to account for ice time differences
- Power play context features to adjust for usage differences between teams
- Linemate quality scoring to isolate individual player contribution
- Feature importance visualization to explain each prediction
- Live deployed application accessible without local setup
