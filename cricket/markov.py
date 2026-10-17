"""Person 2: sparse Markov transition distributions and memoized recursion.
State bands pool overs and wickets. Runs-required remains an exact DP state;
transition estimation pools across required scores to avoid sparse data.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from .state import State, Outcome, advance

BASE=[Outcome(r,0,1) for r in (0,1,2,3,4,6)]+[Outcome(0,1,1),Outcome(1,0,0)]

class TransitionModel:
    def __init__(self,alpha=0.5):
        if alpha<0: raise ValueError('alpha must be non-negative')
        self.alpha=alpha
        self.counts=defaultdict(Counter)
        # Explicit synthetic fallback, never silently used for IPL training.
        self.global_counts=Counter(dict(zip(BASE,[32,30,8,1,14,7,5,3])))
        self.source='synthetic fallback'
    @staticmethod
    def key(balls,wickets): return (min(3,(120-int(balls))//30),min(3,int(wickets)//3))
    def fit(self,df):
        if df.empty: raise ValueError('Empty training data')
        self.counts=defaultdict(Counter);self.global_counts=Counter()
        for r in df.itertuples():
            o=Outcome(int(r.total_runs),int(r.wicket),int(r.legal))
            self.counts[self.key(r.balls_left,r.wickets_lost)][o]+=1
            self.global_counts[o]+=1
        self.source='fitted training deliveries'
        return self
    def distribution(self,state):
        counts=self.counts.get(self.key(state.balls_left,state.wickets_lost),self.global_counts)
        outcomes=sorted(set(BASE)|set(self.global_counts),key=lambda o:(o.legal,o.wicket,o.runs)) if self.alpha else sorted(counts,key=lambda o:(o.legal,o.wicket,o.runs))
        denominator=sum(counts.values())+self.alpha*len(outcomes)
        if denominator<=0: raise ValueError('No transition counts')
        return [(o,(counts.get(o,0)+self.alpha)/denominator) for o in outcomes if counts.get(o,0)+self.alpha>0]

class WinModel:
    def __init__(self,transitions):
        self.transitions=transitions
        self._solve=lru_cache(maxsize=500000)(self._uncached)
    def _uncached(self,b,w,r):
        s=State(b,w,r)
        if s.terminal is not None: return s.terminal
        # Every event decreases balls left or runs required: no self-loop.
        return sum(p*self._solve(ns.balls_left,ns.wickets_lost,ns.runs_required)
                   for o,p in self.transitions.distribution(s) for ns in [advance(s,o)])
    def probability(self,state):
        return self._solve(state.balls_left,state.wickets_lost,state.runs_required)
    def clear_cache(self): self._solve.cache_clear()
