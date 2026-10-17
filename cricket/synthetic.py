"""Artificial demonstration data only; no real IPL matches or player records."""
import numpy as np
import pandas as pd
from .markov import TransitionModel
from .state import State
from .simulation import sample

def generate(matches=24,seed=2026):
    rng=np.random.default_rng(seed);model=TransitionModel();rows=[]
    teams=['Demo A','Demo B','Demo C','Demo D']
    for i in range(matches):
        t1,t2=rng.choice(teams,2,replace=False);toss=str(rng.choice([t1,t2]))
        records=[];target=None
        for innings in [1,2]:
            b,w,total,delivery=120,0,0,0
            while b>0 and w<10 and (innings==1 or total<target):
                o=sample(model,State(b,w,1),rng);delivery+=1
                records.append(dict(match_id=f'SYNTH-{i:03}',season=2023+i%3,venue=f'Demo Ground {i%2+1}',innings=innings,
                    delivery_index=delivery,batting_team=t1 if innings==1 else t2,bowling_team=t2 if innings==1 else t1,
                    total_runs=o.runs,wicket=o.wicket,legal=o.legal,toss_winner=toss,toss_decision='bat' if toss==t1 else 'field'))
                b-=o.legal;w+=o.wicket;total+=o.runs
            if innings==1: target=total+1
        # Exclude tied regulation scores; binary model has no Super Over model.
        if total==target-1: continue
        winner=t2 if total>=target else t1
        for r in records: r['winner']=winner;rows.append(r)
    return pd.DataFrame(rows)
