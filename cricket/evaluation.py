"""Person 3: holdout evaluation; all rows from a match stay in one split."""
import numpy as np
import pandas as pd

def metrics(actual,predicted):
    y=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float)
    if len(y)==0 or y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected nonempty binary labels and probabilities in [0,1]')
    clipped=np.clip(p,1e-12,1-1e-12)
    return {'brier_score':float(np.mean((p-y)**2)),
            'log_loss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'predictions':len(y)}

def calibration(actual,predicted,bins=10):
    metrics(actual,predicted)
    df=pd.DataFrame({'actual':actual,'predicted':predicted})
    df['bin']=pd.cut(df.predicted,np.linspace(0,1,bins+1),include_lowest=True)
    return df.groupby('bin',observed=True).agg(mean_prediction=('predicted','mean'),observed_win_rate=('actual','mean'),count=('actual','size')).reset_index()

def predict_matches(df,win_model,stride=12):
    from .state import State
    rows=[]
    for mid,g in df[df.innings==2].groupby('match_id',sort=False):
        for r in g.iloc[::stride].itertuples():
            s=State(int(r.balls_left),int(r.wickets_lost),int(r.runs_required))
            rows.append({'match_id':mid,'ball_index':len(g[g.index<=r.Index]),
                         'probability':win_model.probability(s),'actual':int(r.batting_team==r.winner)})
    if not rows: raise ValueError('No second innings predictions')
    return pd.DataFrame(rows)
