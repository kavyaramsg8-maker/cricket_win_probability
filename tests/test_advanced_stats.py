"""Unit tests for advanced statistics: Bayesian venue posteriors, toss Z/Chi2 tests, and simulation."""
import unittest
import pandas as pd
import numpy as np
from cricket.state import State
from cricket.markov import TransitionModel, WinModel
from cricket.bayes import toss_analysis, venue_chase_posteriors, posterior
from cricket.simulation import simulate_chase, simulate_first_innings, compare_analytical_vs_mc
from cricket.historical import analyze_chase_trajectory

class AdvancedStatsTests(unittest.TestCase):
    def setUp(self):
        self.model = TransitionModel()
        self.win = WinModel(self.model)

    def test_venue_chase_posteriors(self):
        # Sample match deliveries
        df = pd.DataFrame([
            # Match 1: Chaser wins at Eden Gardens
            dict(match_id='m1', innings=2, batting_team='KKR', bowling_team='RCB', winner='KKR', venue='Eden Gardens'),
            # Match 2: Defender wins at Eden Gardens
            dict(match_id='m2', innings=2, batting_team='CSK', bowling_team='KKR', winner='KKR', venue='Eden Gardens'),
            # Match 3: Chaser wins at Wankhede
            dict(match_id='m3', innings=2, batting_team='MI', bowling_team='DC', winner='MI', venue='Wankhede')
        ])
        v_stats = venue_chase_posteriors(df, prior_a=2.0, prior_b=2.0)
        self.assertEqual(len(v_stats), 2)
        eden = next(v for v in v_stats if v['venue'] == 'Eden Gardens')
        self.assertEqual(eden['matches'], 2)
        self.assertEqual(eden['chase_wins'], 1)
        # Posterior with Beta(2,2): (2+1)/(2+2+2) = 3/6 = 0.5
        self.assertAlmostEqual(eden['posterior_mean'], 0.5)

    def test_toss_z_and_chi2(self):
        df = pd.DataFrame([
            dict(match_id='1', toss_winner='A', winner='A', toss_decision='field'),
            dict(match_id='2', toss_winner='A', winner='A', toss_decision='field'),
            dict(match_id='3', toss_winner='A', winner='B', toss_decision='bat'),
            dict(match_id='4', toss_winner='B', winner='B', toss_decision='field'),
        ])
        res = toss_analysis(df)
        self.assertIn('z_statistic', res)
        self.assertIn('z_test_p_value', res)
        self.assertIn('toss_decision_chi_square', res)
        self.assertTrue(res['matches'] == 4)

    def test_fast_vectorized_10000_trials(self):
        s = State(24, 3, 25)
        res = simulate_chase(self.model, s, n=10000, seed=123)
        self.assertEqual(res['simulations'], 10000)
        self.assertTrue(0.0 <= res['win_probability'] <= 1.0)
        self.assertTrue(res['conf_interval_95'][0] <= res['win_probability'] <= res['conf_interval_95'][1])

    def test_analytical_vs_mc_agreement(self):
        comp = compare_analytical_vs_mc(self.win, n=5000, seed=42)
        self.assertEqual(comp['total_cases'], 8)
        # At least 7 out of 8 should fall within 95% CI
        self.assertGreaterEqual(comp['consistent_cases'], 7)

    def test_chase_trajectory_calculation(self):
        # Mock chase deliveries
        chase_df = pd.DataFrame([
            dict(match_id='m1', innings=2, balls_left=12, wickets_lost=4, runs_required=15,
                 total_runs=4, wicket=0, legal=1, batting_team='A', bowling_team='B', venue='V',
                 target=150, winner='A'),
            dict(match_id='m1', innings=2, balls_left=11, wickets_lost=4, runs_required=11,
                 total_runs=6, wicket=0, legal=1, batting_team='A', bowling_team='B', venue='V',
                 target=150, winner='A'),
            dict(match_id='m1', innings=2, balls_left=10, wickets_lost=4, runs_required=5,
                 total_runs=6, wicket=0, legal=1, batting_team='A', bowling_team='B', venue='V',
                 target=150, winner='A'),
        ])
        traj = analyze_chase_trajectory(chase_df, self.win)
        self.assertEqual(len(traj), 3)
        self.assertTrue('wpa_raw' in traj.columns)
        self.assertTrue('leverage_index' in traj.columns)
        self.assertGreater(traj.iloc[0]['next_raw_probability'], traj.iloc[0]['raw_probability'])

if __name__ == '__main__':
    unittest.main()
