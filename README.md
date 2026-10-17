# Calibrated IPL edition: start here

This edition adds a calibrated chase forecast with Bayesian team/venue context.
The fitted models and results are included. No new dependencies are needed.
Open this extracted project folder in VS Code, then run Streamlit using your
existing Python environment. See docs/UPDATED_MODEL_GUIDE.md and
 docs/UPDATED_RESULTS.md first; the baseline documentation below is historical.

To reproduce THIS edition, run:

```bash
python improve_model.py --input data/raw/ipl_deliveries.csv
python -m streamlit run app.py
```

Running run_pipeline.py instead switches back to the original baseline mode.

## Updated IPL edition: start here

The real-data model and reports are already included. In your active environment:

```bash
python -m streamlit run app.py
```

If you replace an existing copy, stop Streamlit first (Ctrl+C), keep your existing
.venv-1 folder, and copy this package's files into the project folder. Restart
the app. The yellow synthetic warning should disappear. No dependency changes
are required. To avoid mixing old files, you can instead extract to a separate
folder and run it using the absolute path to your old .venv-1/bin/python.

# Live Win-Probability Model for Cricket

A connected three-module Python student project: Markov state transitions,
Bayesian match-level inference, Monte Carlo simulation and a Streamlit demo.
This updated edition includes real converted IPL data and a fitted baseline.
The real-data holdout results are in reports/. This remains an educational
baseline, not a production predictor. The app uses manual entry, not an external live-score API.

## Quick start (Mac/Linux)

Open Terminal in this extracted folder (the one containing app.py):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --input data/raw/ipl_deliveries.csv
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Stop with Ctrl+C.
On Windows activate with `.venv\Scripts\activate` instead of `source ...`.
Python 3.11 or 3.12 is recommended. Internet is required to install dependencies,
not to run the included real-data pipeline. No keys, accounts or paid APIs are needed.

## What to try

Second innings: 30 balls remaining, 4 wickets lost, 40 runs required.
Calculate to compare exact Markov probability with Monte Carlo.
Try 0 runs required (100%), or 0 balls remaining with runs required (0%).
First innings: enter runs ALREADY scored. Simulation finishes that innings,
then simulates the opponent's chase, reporting the batting-first win estimate.

## Files

| File | Purpose |
|---|---|
| cricket/preprocessing.py | CSV validation, pre-delivery states, season split |
| cricket/bayes.py | Beta posterior, toss test, team/venue estimates |
| cricket/synthetic.py | Artificial demonstration matches |
| cricket/state.py | Shared immutable state, event and update |
| cricket/markov.py | Transition counts, smoothing and exact chase recursion |
| cricket/simulation.py | Chase and first-innings Monte Carlo |
| cricket/evaluation.py | Holdout predictions, scores, calibration |
| run_pipeline.py | Connect modules and produce model/results |
| app.py | Manual-entry Streamlit application |
| tests/ | Automated correctness and app smoke tests |
| docs/TEAM_SPLIT.md | Responsibilities and integration contract |
| docs/DATA_FORMAT.md | Real dataset mapping and filtering requirements |
| docs/METHODS.md | Equations, interpretation and limitations |

The processed CSVs, model and reports can be regenerated. Only load model.pkl
created by your own pipeline; never unpickle untrusted downloaded models.

## Test

```bash
python -m unittest discover -s tests -v
```

## Convert a new Cricsheet ZIP

The original IPL ZIP is included in data/raw/. The converter applies conservative
filters and logs a reason for every exclusion. Run:

```bash
python convert_cricsheet.py --zip data/raw/ipl_male_json.zip
python run_pipeline.py --input data/raw/ipl_deliveries.csv
python -m streamlit run app.py
```

The converter supports Cricsheet nested JSON. Other CSV/Kaggle schemas require
explicit mapping. To switch back to artificial data: python run_pipeline.py --demo.
That command OVERWRITES the real-data model/results; rerun the real input command
to restore real-data mode.

## Output and interpretation

reports/metrics.json: Brier score, log loss and split metadata.
reports/predictions.csv: held-out sampled match states and predicted probabilities.
reports/calibration.csv and .png: calibration summary and chart.
reports/win_curve.png: probability curve for one held-out chase.
reports/toss_analysis.json: one observation per match, exact binomial test and
Beta posterior; chi-square is included only when a 2x2 table exists.
reports/team_venue_priors.json: Bayesian win-rate estimates from training games.

The probability curve samples every 24 delivery events for a quick demo. To
produce a prediction after every ball, call predict_matches(..., stride=1).
States are BEFORE each selected delivery; the next state is the after-ball
position. Delivery index includes illegal deliveries.

Bayesian team/venue estimates are a separate analysis, not currently fused
into the Markov delivery probabilities. Do not describe the model as already
adjusting for specific teams/players. First innings uses simulation, not exact
backward recursion. See docs/REAL_DATA_RESULTS.md for measured results and limitations.

See docs/TEAM_SPLIT.md for separate ownership; this structure supports parallel
work, but each person must still understand and verify their module.
