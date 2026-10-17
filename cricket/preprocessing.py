"""Person 1: canonical delivery CSV -> states before delivery.
Input order within a match/innings must already be chronological.
Only standard 20-over, completed matches; no DLS/ties/Super Overs.
"""
import pandas as pd

REQUIRED=['match_id','season','innings','total_runs','wicket','legal']

def prepare(raw):
    missing=set(REQUIRED)-set(raw.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df=raw.copy()
    if df[REQUIRED].isna().any().any(): raise ValueError('Required fields contain missing values')
    for c in ['season','innings','total_runs','wicket','legal']:
        x=pd.to_numeric(df[c],errors='raise')
        if (x%1!=0).any(): raise ValueError(f'{c} must contain integers')
        df[c]=x.astype(int)
    if not df.innings.isin([1,2]).all(): raise ValueError('Only innings 1 and 2 supported')
    if (df.total_runs<0).any() or not df.wicket.isin([0,1]).all() or not df.legal.isin([0,1]).all():
        raise ValueError('Invalid delivery values')
    if ((df.legal==0)&(df.total_runs==0)).any(): raise ValueError('Illegal deliveries need extras')
    keys=['match_id','innings']
    df['runs_scored']=df.groupby(keys,sort=False).total_runs.cumsum()-df.total_runs
    df['wickets_lost']=df.groupby(keys,sort=False).wicket.cumsum()-df.wicket
    df['balls_left']=120-(df.groupby(keys,sort=False).legal.cumsum()-df.legal)
    if (df.groupby(keys).legal.sum()>120).any() or (df.groupby(keys).wicket.sum()>10).any():
        raise ValueError('Innings exceeds 120 legal balls or 10 wickets')
    totals=df[df.innings==1].groupby('match_id').total_runs.sum()
    df['target']=df.match_id.map(totals+1)
    if df.target.isna().any(): raise ValueError('Missing first innings for a match')
    df['runs_required']=df.target-df.runs_scored
    df.loc[df.innings==1,'runs_required']=0
    return df

def split_seasons(df):
    seasons=sorted(df.season.unique())
    if len(seasons)<2: raise ValueError('At least two seasons required for chronological holdout')
    cutoff=seasons[-1]
    return df[df.season<cutoff].copy(),df[df.season==cutoff].copy()
