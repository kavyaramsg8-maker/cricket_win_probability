"""Fixed chronological recipe. Do not tune coefficients to the final test scores.
python improve_model.py --input data/raw/ipl_deliveries.csv
"""
import argparse,json,pickle
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cricket.preprocessing import prepare
from cricket.markov import TransitionModel,WinModel
from cricket.state import State
from cricket.adjustment import BayesianContext,ChaseCalibrator
from cricket.evaluation import metrics,calibration

def predictions(df,model,context,stride=24):
    win=WinModel(model);rows=[]
    for mid,g in df[df.innings==2].groupby('match_id',sort=False):
        for i,r in enumerate(g.itertuples()):
            if i%stride: continue
            state=State(int(r.balls_left),int(r.wickets_lost),int(r.runs_required))
            rows.append(dict(match_id=mid,ball_index=i+1,probability=win.probability(state),
                actual=int(r.batting_team==r.winner),context=context.feature(r.batting_team,r.bowling_team,r.venue),
                chasing_team=r.batting_team,bowling_team=r.bowling_team,venue=r.venue,
                terminal=state.terminal))
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='data/raw/ipl_deliveries.csv')
    args=parser.parse_args();df=prepare(pd.read_csv(args.input,low_memory=False))
    years=sorted(df.season.unique())
    if len(years)<3: raise ValueError('Need at least three seasons')
    calibration_year,test_year=years[-2:]
    train=df[df.season<calibration_year];cal=df[df.season==calibration_year];test=df[df.season==test_year]
    train.to_csv('data/processed/train.csv',index=False)
    cal.to_csv('data/processed/calibration.csv',index=False)
    test.to_csv('data/processed/test.csv',index=False)
    print('Fitting transitions/context through',int(calibration_year)-1,flush=True)
    model=TransitionModel().fit(train);context=BayesianContext().fit(train)
    print('Computing calibration-season states',int(calibration_year),flush=True)
    cal_predictions=predictions(cal,model,context)
    calibrator=ChaseCalibrator().fit(cal_predictions)
    print('Calibration parameters',calibrator.as_dict(),flush=True)
    print('Computing final test states',int(test_year),flush=True)
    test_predictions=predictions(test,model,context)
    test_predictions['adjusted_probability']=[calibrator.adjust(r.probability,r.context,
        terminal=None if pd.isna(r.terminal) else r.terminal) for r in test_predictions.itertuples()]
    scores={'synthetic_demo':False,'model':'Bayesian context plus calibrated Markov chase',
        'transition_training_years':[int(x) for x in sorted(train.season.unique())],
        'calibration_year':int(calibration_year),'test_year':int(test_year),
        'train_matches':int(train.match_id.nunique()),'calibration_matches':int(cal.match_id.nunique()),
        'test_matches':int(test.match_id.nunique()),'sample_stride_deliveries':24,
        'raw_markov':metrics(test_predictions.actual,test_predictions.probability),
        'adjusted_model':metrics(test_predictions.actual,test_predictions.adjusted_probability),
        'constant_50_percent_baseline':metrics(test_predictions.actual,[.5]*len(test_predictions)),
        'parameters':calibrator.as_dict(),
        'disclosure':'2026 was previously inspected. This is a reused test set, not new independent confirmation.'}
    for name,obj in [('model',model),('adjustment',{'context':context,'calibrator':calibrator})]:
        with open(f'data/processed/{name}.pkl','wb') as f: pickle.dump(obj,f)
    Path('data/processed/metadata.json').write_text(json.dumps({'synthetic_demo':False,'adjusted_model':True}))
    test_predictions.to_csv('reports/adjusted_predictions.csv',index=False)
    cal_predictions.to_csv('reports/calibration_training_predictions.csv',index=False)
    Path('reports/adjusted_metrics.json').write_text(json.dumps(scores,indent=2))
    Path('reports/adjusted_team_venue_priors.json').write_text(json.dumps(context.summary(),indent=2))
    table=calibration(test_predictions.actual,test_predictions.adjusted_probability)
    table.to_csv('reports/adjusted_calibration.csv',index=False)
    fig,ax=plt.subplots();ax.plot([0,1],[0,1],'--',color='gray')
    ax.plot(table.mean_prediction,table.observed_win_rate,'o-',label='Adjusted')
    ax.set(xlabel='Predicted probability',ylabel='Observed win rate',title='Reused 2026 holdout calibration');ax.legend();fig.tight_layout();fig.savefig('reports/adjusted_calibration.png');plt.close(fig)
    first=test_predictions[test_predictions.match_id==test_predictions.match_id.iloc[0]]
    fig,ax=plt.subplots();ax.plot(first.ball_index,first.probability,'o-',label='Raw Markov');ax.plot(first.ball_index,first.adjusted_probability,'o-',label='Adjusted')
    ax.set(xlabel='Delivery index',ylabel='Chasing team win probability',ylim=(0,1),title=str(first.match_id.iloc[0]));ax.legend();fig.tight_layout();fig.savefig('reports/adjusted_win_curve.png');plt.close(fig)
    print(json.dumps(scores,indent=2),flush=True)

if __name__=='__main__': main()
