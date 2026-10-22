"""Unit tests for Delivery Leverage Index and outcome probability swings."""
import unittest
from cricket.state import State
from cricket.markov import TransitionModel, WinModel

class LeverageTests(unittest.TestCase):
    def setUp(self):
        self.model = TransitionModel()
        self.win = WinModel(self.model)

    def test_boundary_swings_monotonic(self):
        # State: 12 balls, 4 wickets, 18 runs
        s = State(12, 4, 18)
        swings = self.win.outcome_swings(s)
        self.assertTrue(len(swings) > 0)

        # 6 should strictly improve win probability
        six_swing = next(item['swing'] for item in swings if item['runs'] == 6 and item['wicket'] == 0)
        self.assertGreater(six_swing, 0.0)

        # Wicket with 0 runs should strictly decrease win probability
        wicket_swing = next(item['swing'] for item in swings if item['wicket'] == 1 and item['runs'] == 0)
        self.assertLess(wicket_swing, 0.0)

    def test_terminal_state_leverage_is_zero(self):
        # Target reached
        s_win = State(10, 2, 0)
        lev_win = self.win.leverage_index(s_win)
        self.assertEqual(lev_win['leverage_index'], 0.0)
        self.assertEqual(lev_win['rms_swing'], 0.0)

        # Balls exhausted
        s_loss = State(0, 2, 5)
        lev_loss = self.win.leverage_index(s_loss)
        self.assertEqual(lev_loss['leverage_index'], 0.0)

    def test_death_overs_higher_leverage_than_comfortable_state(self):
        # High leverage: final over, 10 runs needed, 6 down
        tight = State(6, 6, 10)
        lev_tight = self.win.leverage_index(tight)

        # Low leverage: comfortable chase, 60 balls, 10 runs, 2 down
        easy = State(60, 2, 10)
        lev_easy = self.win.leverage_index(easy)

        self.assertGreater(lev_tight['leverage_index'], lev_easy['leverage_index'])

    def test_lookup_table_shape_and_bounds(self):
        balls = [6, 12, 18]
        runs = [10, 20, 30]
        grid_data = self.win.generate_lookup_table(balls, runs, wickets_lost=3)
        self.assertEqual(grid_data['grid'].shape, (3, 3))
        self.assertTrue((grid_data['grid'] >= 0.0).all() and (grid_data['grid'] <= 1.0).all())

if __name__ == '__main__':
    unittest.main()
