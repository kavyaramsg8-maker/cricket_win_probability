"""Bayesian context features plus monotone logistic calibration for CHASES.
Fit context and transitions before the calibration season; fit the two
calibration coefficients using only that calibration season. Preserve terminal
states explicitly. This is Bayesian feature estimation plus frequentist
calibration, not a full Bayesian posterior over match-state probabilities.
"""
from collections import defaultdict
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize

ALIASES={'Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings',
         'Royal Challengers Bangalore':'Royal Challengers Bengaluru',
         'Rising Pune Supergiant':'Rising Pune Supergiants'}
def team_name(team): return ALIASES.get(str(team),str(team))

class BayesianContext:
    def __init__(self):
        self.global_counts={};self.venue_counts={};self.teams=[];self.venues=[]
    def fit(self,df):
        global_counts=defaultdict(lambda:[0,0]);venue_counts=defaultdict(lambda:[0,0])
        matches=df.drop_duplicates('match_id')
        for r in matches.itertuples():
            venue=str(r.venue);winner=team_name(r.winner)
            for team in [team_name(r.batting_team),team_name(r.bowling_team)]:
                win=int(team==winner)
                global_counts[team][0]+=win;global_counts[team][1]+=1
                venue_counts[(team,venue)][0]+=win;venue_counts[(team,venue)][1]+=1
        self.global_counts=dict(global_counts);self.venue_counts=dict(venue_counts)
        self.teams=sorted(self.global_counts);self.venues=sorted({v for _,v in self.venue_counts})
        return self
    def rate(self,team,venue=None):
        team=team_name(team);w,n=self.global_counts.get(team,(0,0))
        # Beta(10,10) smooths team rates, then 20 prior equivalent matches
        # shrink sparse venue rates toward that team's global posterior mean.
        global_mean=(w+10)/(n+20)
        vw,vn=self.venue_counts.get((team,str(venue)),(0,0))
        return (vw+20*global_mean)/(vn+20)
    def feature(self,chasing_team=None,bowling_team=None,venue=None):
        if not chasing_team or not bowling_team: return 0.
        a=self.rate(chasing_team,venue);b=self.rate(bowling_team,venue)
        return float(.5*(logit(a)-logit(b)))
    def summary(self):
        return [{'team':t,'venue':v,'posterior_mean':self.rate(t,v),'venue_matches':n,'venue_wins':w}
                for (t,v),(w,n) in sorted(self.venue_counts.items())]

class ChaseCalibrator:
    def __init__(self): self.parameters=np.array([0.,1.,0.])
    def fit(self,df):
        p=np.clip(df.probability.to_numpy(float),1e-5,1-1e-5)
        # Add the smoothed Bayesian context to the raw Markov log odds before
        # a shared positive slope/intercept calibration. This is a hybrid odds
        # heuristic, not an exact Bayes theorem update (evidence overlaps).
        x=np.column_stack([np.ones(len(df)),logit(p)+df.context.to_numpy(float)])
        y=df.actual.to_numpy(float)
        counts=df.groupby('match_id').match_id.transform('size').to_numpy()
        weights=1/counts;weights/=weights.sum()
        def objective(theta):
            z=x@theta
            return float(np.sum(weights*(np.logaddexp(0,z)-y*z))+.01*np.sum(theta**2))
        result=minimize(objective,[0.,1.],method='L-BFGS-B',bounds=[(-5,5),(.01,3)])
        if not result.success: raise RuntimeError('Calibration optimization failed: '+str(result.message))
        a,b=result.x
        self.parameters=np.array([a,b,b]);return self
    def adjust(self,p,context=0.,terminal=None):
        if terminal is not None: return float(terminal)
        a,b,c=self.parameters
        return float(expit(a+b*logit(np.clip(p,1e-5,1-1e-5))+c*context))
    def as_dict(self):
        return dict(zip(['intercept','markov_log_odds_slope','bayesian_context_weight'],map(float,self.parameters)))
