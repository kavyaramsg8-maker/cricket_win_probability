import unittest
import numpy as np
from cricket.synthetic import generate
from cricket.preprocessing import prepare,split_seasons
from cricket.markov import TransitionModel,WinModel
from cricket.state import State

class FittedModelTests(unittest.TestCase):
    def test_refit_refreshes_distribution(self):
        df=prepare(generate(matches=6));model=TransitionModel().fit(df)
        s=State(30,4,20)
        self.assertAlmostEqual(sum(p for _,p in model.distribution(s)),1)
        # A training set with dots only must have dot as its most likely outcome.
        dots=df.copy();dots['total_runs']=0;dots['wicket']=0;dots['legal']=1
        model.fit(dots)
        best=max(model.distribution(s),key=lambda x:x[1])[0]
        self.assertEqual((best.runs,best.wicket,best.legal),(0,0,1))
    def test_all_training_outcomes_supported(self):
        df=prepare(generate(matches=6));model=TransitionModel().fit(df)
        events={o for o,p in model.distribution(State(30,4,20))}
        self.assertTrue(all(any((o.runs,o.wicket,o.legal)==(int(r.total_runs),int(r.wicket),int(r.legal)) for o in events) for r in df.itertuples()))
