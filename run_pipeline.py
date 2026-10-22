"""Complete End-to-End Data, Model, and Evaluation Pipeline.
Usage:
  python run_pipeline.py --input data/raw/ipl_deliveries.csv
  python run_pipeline.py --demo
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cricket.preprocessing import prepare, split_seasons
from cricket.synthetic import generate
from cricket.markov import TransitionModel, WinModel
from cricket.state import State
from cricket.adjustment import BayesianContext, ChaseCalibrator
from cricket.bayes import toss_analysis, venue_chase_posteriors, team_venue_priors
from cricket.evaluation import predict_matches, metrics, calibration, calibration_summary
from cricket.simulation import compare_analytical_vs_mc

def sample_chase_predictions(df, win_model, context=None, stride=24):
    """Compute chase state predictions with context and terminal flags."""
    rows = []
    for mid, g in df[df.innings == 2].groupby('match_id', sort=False):
        chasing_team = g.iloc[0].batting_team
        bowling_team = g.iloc[0].bowling_team
        venue = g.iloc[0].venue
        winner = g.iloc[0].winner
        ctx_val = 0.0
        if context is not None:
            ctx_val = context.feature(chasing_team, bowling_team, venue)

        for i, r in enumerate(g.itertuples()):
            if i % stride != 0:
                continue
            s = State(int(r.balls_left), int(r.wickets_lost), int(r.runs_required))
            p_raw = win_model.probability(s)
            rows.append({
                'match_id': mid,
                'ball_index': i + 1,
                'balls_left': int(r.balls_left),
                'wickets_lost': int(r.wickets_lost),
                'runs_required': int(r.runs_required),
                'probability': p_raw,
                'actual': int(chasing_team == winner),
                'context': ctx_val,
                'chasing_team': chasing_team,
                'bowling_team': bowling_team,
                'venue': venue,
                'terminal': s.terminal
            })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Cricket Win Probability Pipeline")
    parser.add_argument('--demo', action='store_true', help='Generate and run on synthetic demonstration data')
    parser.add_argument('--input', type=Path, default='data/raw/ipl_deliveries.csv', help='Path to canonical real IPL delivery CSV')
    args = parser.parse_args()

    for p in ['data/raw', 'data/processed', 'reports']:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Data Ingestion and Preprocessing")
    print("=" * 60)
    if args.demo:
        print("Generating synthetic demonstration data...")
        raw = generate(matches=100)
        raw.to_csv('data/raw/synthetic_deliveries.csv', index=False)
        is_demo = True
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file {args.input} not found. Run convert_cricsheet.py first.")
        print(f"Loading Cricsheet IPL data from {args.input}...")
        raw = pd.read_csv(args.input, low_memory=False)
        is_demo = False

    clean = prepare(raw)
    print(f"Total valid deliveries: {len(clean):,}")
    print(f"Total unique matches: {clean.match_id.nunique()}")
    print(f"Seasons covered: {sorted(clean.season.unique())}")

    print("\n" + "=" * 60)
    print("STEP 2: Chronological Train / Test Split")
    print("=" * 60)
    # Split: 2008-2023 for training, 2024 for testing
    train, test = split_seasons(clean, test_season=2024)
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)

    train_seasons = sorted(int(x) for x in train.season.unique())
    test_seasons = sorted(int(x) for x in test.season.unique())
    print(f"Training set: {train_seasons} ({train.match_id.nunique()} matches, {len(train):,} deliveries)")
    print(f"Holdout test set: {test_seasons} ({test.match_id.nunique()} matches, {len(test):,} deliveries)")

    print("\n" + "=" * 60)
    print("STEP 3: Core Probability Engine - Fitting Markov Transition Model")
    print("=" * 60)
    # MLE with Laplace smoothing alpha=0.5 over 16 (phase x wicket) bands
    model = TransitionModel(alpha=0.5).fit(train)
    win_model = WinModel(model)
    with open('data/processed/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Markov model fitted and cached to data/processed/model.pkl")

    print("\n" + "=" * 60)
    print("STEP 4: Bayesian Team & Venue Context + Calibration")
    print("=" * 60)
    context = BayesianContext().fit(train)
    print(f"Fitted Bayesian priors for {len(context.teams)} teams and {len(context.venues)} venues.")

    # Fit calibrator on training chase states
    print("Sampling training chase states for calibration...")
    train_cal_data = sample_chase_predictions(train, win_model, context, stride=24)
    calibrator = ChaseCalibrator().fit(train_cal_data)
    print("Fitted Calibrator parameters:", calibrator.as_dict())

    adjustment_data = {'context': context, 'calibrator': calibrator}
    with open('data/processed/adjustment.pkl', 'wb') as f:
        pickle.dump(adjustment_data, f)
    print("Adjustment models saved to data/processed/adjustment.pkl")

    print("\n" + "=" * 60)
    print("STEP 5: Holdout Evaluation on Test Season")
    print("=" * 60)
    test_preds = sample_chase_predictions(test, win_model, context, stride=12)
    test_preds['adjusted_probability'] = [
        calibrator.adjust(r.probability, r.context, terminal=None if pd.isna(r.terminal) else r.terminal)
        for r in test_preds.itertuples()
    ]
    test_preds.to_csv('reports/predictions.csv', index=False)
    test_preds.to_csv('reports/adjusted_predictions.csv', index=False)

    raw_scores = metrics(test_preds.actual, test_preds.probability)
    adj_scores = metrics(test_preds.actual, test_preds.adjusted_probability)
    base_scores = metrics(test_preds.actual, [0.5] * len(test_preds))

    raw_cal_summary = calibration_summary(test_preds.actual, test_preds.probability)
    adj_cal_summary = calibration_summary(test_preds.actual, test_preds.adjusted_probability)

    eval_report = {
        'synthetic_demo': is_demo,
        'train_seasons': train_seasons,
        'test_seasons': test_seasons,
        'train_matches': int(train.match_id.nunique()),
        'test_matches': int(test.match_id.nunique()),
        'test_predictions_count': len(test_preds),
        'raw_markov': {
            **raw_scores,
            'expected_calibration_error': raw_cal_summary['expected_calibration_error'],
            'max_calibration_error': raw_cal_summary['max_calibration_error']
        },
        'calibrated_hybrid': {
            **adj_scores,
            'expected_calibration_error': adj_cal_summary['expected_calibration_error'],
            'max_calibration_error': adj_cal_summary['max_calibration_error']
        },
        'constant_50_percent_baseline': base_scores,
        'calibrator_parameters': calibrator.as_dict()
    }

    Path('reports/metrics.json').write_text(json.dumps(eval_report, indent=2))
    Path('reports/adjusted_metrics.json').write_text(json.dumps(eval_report, indent=2))

    # Decile calibration CSVs
    raw_cal_table = calibration(test_preds.actual, test_preds.probability)
    raw_cal_table.to_csv('reports/calibration.csv', index=False)
    adj_cal_table = calibration(test_preds.actual, test_preds.adjusted_probability)
    adj_cal_table.to_csv('reports/adjusted_calibration.csv', index=False)

    print(f"Holdout Brier Score: Raw Markov = {raw_scores['brier_score']:.4f} | Calibrated Hybrid = {adj_scores['brier_score']:.4f} | Baseline = {base_scores['brier_score']:.4f}")
    print(f"Holdout Log Loss:    Raw Markov = {raw_scores['log_loss']:.4f} | Calibrated Hybrid = {adj_scores['log_loss']:.4f} | Baseline = {base_scores['log_loss']:.4f}")
    print(f"Holdout Accuracy:    Raw Markov = {raw_scores['accuracy']:.1%} | Calibrated Hybrid = {adj_scores['accuracy']:.1%}")
    print(f"Expected Calib Error (ECE): Raw = {raw_cal_summary['expected_calibration_error']:.4f} | Calibrated = {adj_cal_summary['expected_calibration_error']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 6: Statistical Analytics (Toss & Venue Bayesian Inference)")
    print("=" * 60)
    # Toss analysis on test set and full dataset
    test_toss = toss_analysis(test)
    full_toss = toss_analysis(clean)
    combined_toss = {
        'test_season': test_toss,
        'full_dataset': full_toss
    }
    Path('reports/toss_analysis.json').write_text(json.dumps(combined_toss, indent=2))
    print(f"Toss Win Advantage (Test Season): Z={test_toss.get('z_statistic', 0):.4f}, p={test_toss.get('z_test_p_value', 1):.4f}, Exact binom p={test_toss['p_value']:.4f}")
    print(f"Toss Win Advantage (All Seasons):  Z={full_toss.get('z_statistic', 0):.4f}, p={full_toss.get('z_test_p_value', 1):.4f}, Exact binom p={full_toss['p_value']:.4f}")

    # Venue Bayesian Chasing
    venue_stats = venue_chase_posteriors(clean, prior_a=10.0, prior_b=10.0)
    Path('reports/venue_chase_stats.json').write_text(json.dumps(venue_stats, indent=2))
    Path('reports/team_venue_priors.json').write_text(json.dumps(team_venue_priors(train), indent=2))
    Path('reports/adjusted_team_venue_priors.json').write_text(json.dumps(context.summary(), indent=2))
    print(f"Computed Bayesian chasing statistics for {len(venue_stats)} venues.")

    print("\n" + "=" * 60)
    print("STEP 7: Analytical Markov DP vs 10,000-Trial Monte Carlo Benchmark")
    print("=" * 60)
    mc_bench = compare_analytical_vs_mc(win_model, n=10000, seed=42)
    Path('reports/mc_vs_markov.json').write_text(json.dumps(mc_bench, indent=2))
    print(f"Monte Carlo vs Analytical DP Consistency: {mc_bench['consistent_cases']}/{mc_bench['total_cases']} states within 95% CI.")

    print("\n" + "=" * 60)
    print("STEP 8: Generating High-Resolution Diagnostic Plots")
    print("=" * 60)
    # Calibration Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration (y = x)')
    ax.plot(raw_cal_table.mean_prediction, raw_cal_table.observed_win_rate, 'o--', color='#FF5722', label='Raw Markov')
    ax.plot(adj_cal_table.mean_prediction, adj_cal_table.observed_win_rate, 's-', color='#00E676', lw=2, label='Calibrated Hybrid')
    ax.set(xlabel='Mean Predicted Probability', ylabel='Observed Win Proportion',
           title=f'Decile Reliability Diagram (Holdout {test_seasons})', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('reports/calibration.png')
    fig.savefig('reports/adjusted_calibration.png')
    plt.close(fig)

    # Win Curve Plot for first held-out match
    first_mid = test_preds.match_id.iloc[0]
    match_sample = test_preds[test_preds.match_id == first_mid]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(match_sample.ball_index, match_sample.probability, 'o--', color='#FF9800', label='Raw Markov')
    ax.plot(match_sample.ball_index, match_sample.adjusted_probability, 's-', color='#2196F3', lw=2, label='Calibrated Hybrid')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set(xlabel='Delivery Index', ylabel='Chasing Team Win Probability', ylim=(-0.05, 1.05),
           title=f'Holdout Match Win-Probability Trajectory (Match #{first_mid})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('reports/win_curve.png')
    fig.savefig('reports/adjusted_win_curve.png')
    plt.close(fig)

    # Metadata
    metadata = {
        'synthetic_demo': is_demo,
        'adjusted_model': True,
        'train_seasons': train_seasons,
        'test_seasons': test_seasons,
        'updated_at': str(pd.Timestamp.now())
    }
    Path('data/processed/metadata.json').write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All models and reports saved.")
    print("Launch UI with: streamlit run app.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
