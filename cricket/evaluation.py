"""Person 3: holdout evaluation; all rows from a match stay in one split."""
import numpy as np
import pandas as pd

def metrics(actual,predicted):
    y=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float)
    if len(y)==0 or y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected nonempty binary labels and probabilities in [0,1]')
    clipped=np.clip(p,1e-12,1-1e-12)
    brier = float(np.mean((p-y)**2))
    brier_skill = float(1.0 - (brier / 0.25))
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    return {'brier_score': brier,
            'brier_skill_score': brier_skill,
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'accuracy': acc,
            'predictions': len(y)}

def calibration(actual,predicted,bins=10):
    metrics(actual,predicted)
    df=pd.DataFrame({'actual':actual,'predicted':predicted})
    df['bin']=pd.cut(df.predicted,np.linspace(0,1,bins+1),include_lowest=True)
    cal=df.groupby('bin',observed=True).agg(
        mean_prediction=('predicted','mean'),
        observed_win_rate=('actual','mean'),
        count=('actual','size')
    ).reset_index()
    cal['mean_prediction'] = cal['mean_prediction'].fillna(0.0)
    cal['observed_win_rate'] = cal['observed_win_rate'].fillna(0.0)
    # Bin standard error: sqrt(p*(1-p)/n)
    cal['std_error'] = np.where(cal['count'] > 0,
                                np.sqrt(cal['observed_win_rate'] * (1.0 - cal['observed_win_rate']) / np.maximum(1, cal['count'])),
                                0.0)
    cal['diff'] = np.abs(cal['observed_win_rate'] - cal['mean_prediction'])
    return cal

def calibration_summary(actual,predicted,bins=10):
    """Compute ECE (Expected Calibration Error) and MCE (Max Calibration Error)."""
    cal = calibration(actual, predicted, bins=bins)
    total_n = len(actual)
    ece = float(np.sum((cal['count'] / total_n) * cal['diff'])) if total_n > 0 else 0.0
    mce = float(cal['diff'].max()) if not cal.empty else 0.0
    return {
        'expected_calibration_error': ece,
        'max_calibration_error': mce,
        'bins': bins,
        'table': cal.to_dict(orient='records')
    }

def predict_matches(df,win_model,stride=12):
    from .state import State
    rows=[]
    for mid,g in df[df.innings==2].groupby('match_id',sort=False):
        for r in g.iloc[::stride].itertuples():
            s=State(int(r.balls_left),int(r.wickets_lost),int(r.runs_required))
            rows.append({'match_id':mid,'ball_index':len(g[g.index<=r.Index]),
                         'probability':win_model.probability(s),'actual':int(r.batting_team==r.winner),
                         'chasing_team': r.batting_team, 'bowling_team': r.bowling_team,
                         'venue': r.venue, 'balls_left': int(r.balls_left),
                         'wickets_lost': int(r.wickets_lost), 'runs_required': int(r.runs_required)})
    if not rows: raise ValueError('No second innings predictions')
    return pd.DataFrame(rows)
