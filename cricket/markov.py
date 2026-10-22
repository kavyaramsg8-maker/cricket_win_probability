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
    def __init__(self,transitions,precompute=True,max_balls=120,max_wickets=10,max_runs=300):
        import sys
        sys.setrecursionlimit(25000)
        self.transitions=transitions
        self.max_balls=max_balls
        self.max_wickets=max_wickets
        self.max_runs=max_runs
        self._table=None
        if precompute:
            self._build_table()
        self._solve=lru_cache(maxsize=500000)(self._uncached)

    def _build_table(self):
        """Construct exact backward induction dynamic programming lookup table."""
        import numpy as np
        band_dist = {}
        for b_idx in range(4):
            for w_idx in range(4):
                s = State(120 - b_idx * 30, w_idx * 3, 1)
                band_dist[(b_idx, w_idx)] = self.transitions.distribution(s)

        V = np.zeros((self.max_balls + 1, self.max_wickets + 1, self.max_runs + 1), dtype=np.float64)
        # Terminal win: runs_required <= 0
        V[:, :, 0] = 1.0

        for b in range(1, self.max_balls + 1):
            b_band = min(3, (120 - b) // 30)
            for w in range(self.max_wickets):
                w_band = min(3, w // 3)
                dist = band_dist[(b_band, w_band)]
                legal_terms = [(p, o.runs, o.wicket) for o, p in dist if o.legal]
                illegal_terms = [(p, o.runs, o.wicket) for o, p in dist if not o.legal]

                for r in range(1, self.max_runs + 1):
                    val = 0.0
                    for p, r_run, w_w in legal_terms:
                        val += p * V[b - 1, min(self.max_wickets, w + w_w), max(0, r - r_run)]
                    for p, r_run, w_w in illegal_terms:
                        val += p * V[b, min(self.max_wickets, w + w_w), max(0, r - r_run)]
                    V[b, w, r] = val

        self._table = V

    def _uncached(self,b,w,r):
        s=State(b,w,r)
        if s.terminal is not None: return s.terminal
        # Every event decreases balls left or runs required: no self-loop.
        return sum(p*self._solve(ns.balls_left,ns.wickets_lost,ns.runs_required)
                   for o,p in self.transitions.distribution(s) for ns in [advance(s,o)])

    def probability(self,state):
        if state.terminal is not None: return float(state.terminal)
        b, w, r = state.balls_left, state.wickets_lost, state.runs_required
        if self._table is not None and 0 <= b <= self.max_balls and 0 <= w <= self.max_wickets and 0 <= r <= self.max_runs:
            return float(self._table[b, w, r])
        return self._solve(b, w, r)

    def clear_cache(self):
        self._solve.cache_clear()
        if self._table is not None:
            self._build_table()

    def outcome_swings(self,state):
        """Compute win probability swing for each possible delivery outcome from state."""
        p0 = self.probability(state)
        dist = self.transitions.distribution(state)
        results = []
        for o, p in dist:
            ns = advance(state, o)
            p_next = self.probability(ns)
            results.append({
                'outcome': o,
                'runs': o.runs,
                'wicket': o.wicket,
                'legal': o.legal,
                'prob': float(p),
                'next_prob': float(p_next),
                'swing': float(p_next - p0),
                'abs_swing': float(abs(p_next - p0))
            })
        return results

    def leverage_index(self,state,baseline_swing=0.065):
        """Compute state Leverage Index (LI), expected swing, and potential spread."""
        if state.terminal is not None:
            return {'leverage_index': 0.0, 'rms_swing': 0.0, 'expected_swing': 0.0, 'max_swing': 0.0, 'swings': []}
        swings = self.outcome_swings(state)
        if not swings:
            return {'leverage_index': 0.0, 'rms_swing': 0.0, 'expected_swing': 0.0, 'max_swing': 0.0, 'swings': []}
        rms = float((sum(s['prob'] * (s['swing'] ** 2) for s in swings)) ** 0.5)
        exp_swing = float(sum(s['prob'] * s['abs_swing'] for s in swings))
        p_nexts = [s['next_prob'] for s in swings]
        max_swing = float(max(p_nexts) - min(p_nexts))
        li = float(rms / baseline_swing) if baseline_swing > 0 else 1.0
        return {
            'leverage_index': li,
            'rms_swing': rms,
            'expected_swing': exp_swing,
            'max_swing': max_swing,
            'swings': swings
        }

    def generate_lookup_table(self,balls_values=None,runs_values=None,wickets_lost=4):
        """Generate a 2D grid of win probabilities across (balls_left x runs_required)."""
        import numpy as np
        if balls_values is None: balls_values = list(range(0, 121, 6))
        if runs_values is None: runs_values = list(range(0, 201, 10))
        grid = np.zeros((len(runs_values), len(balls_values)), dtype=float)
        for i, r in enumerate(runs_values):
            for j, b in enumerate(balls_values):
                grid[i, j] = self.probability(State(int(b), int(wickets_lost), int(r)))
        return {'grid': grid, 'balls': balls_values, 'runs': runs_values, 'wickets': wickets_lost}
