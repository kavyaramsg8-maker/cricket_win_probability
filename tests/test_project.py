import unittest
import pandas as pd
from cricket.state import State, Outcome, advance
from cricket.markov import TransitionModel, WinModel
from cricket.simulation import simulate_chase
from cricket.preprocessing import prepare
from cricket.evaluation import metrics
from cricket.bayes import toss_analysis

class ProjectTests(unittest.TestCase):
    def test_wide_does_not_consume_ball(self):
        self.assertEqual(advance(State(3,2,5), Outcome(1,0,0)), State(3,2,4))
    def test_target_on_last_ball_is_win(self):
        self.assertEqual(WinModel(TransitionModel()).probability(State(0,2,0)),1)
    def test_exhaustion_is_loss(self):
        self.assertEqual(WinModel(TransitionModel()).probability(State(0,2,1)),0)
    def test_exact_one_ball_probability(self):
        m=TransitionModel(); m.global_counts={Outcome(0,0,1):1,Outcome(4,0,1):1};m.alpha=0
        self.assertAlmostEqual(WinModel(m).probability(State(1,0,4)),.5)
    def test_simulation_matches_exact(self):
        m=TransitionModel();m.global_counts={Outcome(0,0,1):1,Outcome(4,0,1):1};m.alpha=0
        result=simulate_chase(m,State(1,0,4),10000,7)
        self.assertLess(abs(result['win_probability']-.5),.025)
    def test_state_before_delivery(self):
        raw=pd.DataFrame([dict(match_id='a',season=2020,innings=1,total_runs=4,wicket=0,legal=1),dict(match_id='a',season=2020,innings=2,total_runs=1,wicket=0,legal=0),dict(match_id='a',season=2020,innings=2,total_runs=4,wicket=0,legal=1)])
        out=prepare(raw);chase=out[out.innings==2]
        self.assertEqual(list(chase.runs_required),[5,4]);self.assertEqual(list(chase.balls_left),[120,120])
    def test_perfect_metrics(self):
        self.assertEqual(metrics([0,1],[0,1])['brier_score'],0)
    def test_toss_counts_one_per_match(self):
        df=pd.DataFrame([dict(match_id='x',toss_winner='A',winner='A')]*4)
        self.assertEqual(toss_analysis(df)['matches'],1)
    def test_invalid_outcome(self):
        with self.assertRaises(ValueError): Outcome(0,0,0)

if __name__=='__main__': unittest.main()
