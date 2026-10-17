"""Person 3: independent sampling using Person 2's shared transition interface."""
import numpy as np
from .state import State, advance

def sample(model,state,rng):
    dist=model.distribution(state)
    return dist[int(rng.choice(len(dist),p=[p for _,p in dist]))][0]

def chase_once(model,state,rng):
    while state.terminal is None: state=advance(state,sample(model,state,rng))
    return state.terminal

def simulate_chase(model,state,n=2000,seed=42):
    if n<1: raise ValueError('n must be positive')
    rng=np.random.default_rng(seed)
    p=float(np.mean([chase_once(model,state,rng) for _ in range(n)]))
    return {'win_probability':p,'standard_error':float(np.sqrt(p*(1-p)/n)),'simulations':n}

def simulate_first_innings(model,balls_left,wickets_lost,runs_scored,n=1000,seed=42):
    """Complete first innings, then simulate opponent chase; probability for batting-first team.
    Same pooled model for both teams. No claimed team-strength correction.
    """
    if n<1 or runs_scored<0: raise ValueError('Invalid simulation inputs')
    State(balls_left,wickets_lost,1)
    rng=np.random.default_rng(seed);wins=[];totals=[]
    for _ in range(n):
        b,w,total=balls_left,wickets_lost,runs_scored
        while b>0 and w<10:
            o=sample(model,State(b,w,1),rng);b-=o.legal;w+=o.wicket;total+=o.runs
        totals.append(total)
        wins.append(1-chase_once(model,State(120,0,total+1),rng))
    p=float(np.mean(wins))
    return {'win_probability':p,'standard_error':float(np.sqrt(p*(1-p)/n)),
            'mean_final_score':float(np.mean(totals)),'simulations':n}
