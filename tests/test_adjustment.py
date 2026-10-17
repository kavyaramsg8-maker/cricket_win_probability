import unittest
import pandas as pd
import numpy as np
from cricket.adjustment import BayesianContext,ChaseCalibrator

class AdjustmentTests(unittest.TestCase):
    def test_unknown_context_is_neutral(self):
        context=BayesianContext().fit(pd.DataFrame([dict(match_id='1',batting_team='A',bowling_team='B',winner='A',venue='V')]))
        self.assertEqual(context.feature('unknown','other','new'),0)
    def test_priors_favour_historically_stronger_team(self):
        df=pd.DataFrame([dict(match_id=str(i),batting_team='A',bowling_team='B',winner='A',venue='V') for i in range(20)])
        c=BayesianContext().fit(df)
        self.assertGreater(c.feature('A','B','V'),0)
        self.assertAlmostEqual(c.feature('A','B','V'),-c.feature('B','A','V'))
    def test_terminal_values_remain_exact(self):
        c=ChaseCalibrator();c.parameters=np.array([.2,.5,.3])
        self.assertEqual(c.adjust(0,1,terminal=0),0)
        self.assertEqual(c.adjust(1,-1,terminal=1),1)
    def test_monotone_and_context_changes_probability(self):
        c=ChaseCalibrator();c.parameters=np.array([0,.5,.3])
        self.assertLess(c.adjust(.2,0),c.adjust(.8,0))
        self.assertGreater(c.adjust(.5,1),c.adjust(.5,-1))
    def test_fit_reduces_overconfidence(self):
        df=pd.DataFrame({'match_id':list(range(100)),'probability':[.99]*50+[.01]*50,'context':[0.]*100,'actual':[1]*35+[0]*15+[1]*15+[0]*35})
        c=ChaseCalibrator().fit(df)
        self.assertLess(c.adjust(.99,0),.9)
        self.assertGreater(c.adjust(.01,0),.1)
        self.assertAlmostEqual(c.parameters[1],c.parameters[2])
