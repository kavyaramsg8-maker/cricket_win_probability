"""Run from project root: python run_pipeline.py --demo"""
import argparse,json,pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from cricket.preprocessing import prepare,split_seasons
from cricket.synthetic import generate
from cricket.markov import TransitionModel,WinModel
from cricket.bayes import toss_analysis,team_venue_priors
from cricket.evaluation import predict_matches,metrics,calibration

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--demo',action='store_true',help='Generate artificial demonstration data')
    parser.add_argument('--input',type=Path,help='Canonical real-data CSV; see docs/DATA_FORMAT.md')
    args=parser.parse_args()
    if args.demo==bool(args.input): parser.error('Choose exactly one of --demo or --input')
    for p in ['data/raw','data/processed','reports']: Path(p).mkdir(parents=True,exist_ok=True)
    raw=generate() if args.demo else pd.read_csv(args.input,low_memory=False)
    if args.demo: raw.to_csv('data/raw/synthetic_deliveries.csv',index=False)
    clean=prepare(raw);train,test=split_seasons(clean)
    train.to_csv('data/processed/train.csv',index=False);test.to_csv('data/processed/test.csv',index=False)
    model=TransitionModel().fit(train)
    with open('data/processed/model.pkl','wb') as f: pickle.dump(model,f)
    predictions=predict_matches(test,WinModel(model),stride=24)
    predictions.to_csv('reports/predictions.csv',index=False)
    scores=metrics(predictions.actual,predictions.probability)
    scores['constant_50_percent_baseline']=metrics(predictions.actual,[0.5]*len(predictions))
    scores.update(synthetic_demo=bool(args.demo),train_seasons=[int(x) for x in sorted(train.season.unique())],test_seasons=[int(x) for x in sorted(test.season.unique())],test_matches=int(test.match_id.nunique()))
    Path('reports/metrics.json').write_text(json.dumps(scores,indent=2))
    Path('reports/toss_analysis.json').write_text(json.dumps(toss_analysis(test),indent=2))
    Path('reports/team_venue_priors.json').write_text(json.dumps(team_venue_priors(train),indent=2))
    cal=calibration(predictions.actual,predictions.probability);cal.to_csv('reports/calibration.csv',index=False)
    fig,ax=plt.subplots();ax.plot([0,1],[0,1],'--',color='gray');ax.plot(cal.mean_prediction,cal.observed_win_rate,'o-');ax.set(xlabel='Predicted probability',ylabel='Observed win rate',title='Synthetic calibration demo' if args.demo else 'Holdout calibration');fig.tight_layout();fig.savefig('reports/calibration.png');plt.close(fig)
    first=predictions[predictions.match_id==predictions.match_id.iloc[0]]
    fig,ax=plt.subplots();ax.plot(first.ball_index,first.probability,'o-');ax.set(xlabel='Delivery index',ylabel='Chasing team win probability',ylim=(0,1),title=str(first.match_id.iloc[0]));fig.tight_layout();fig.savefig('reports/win_curve.png');plt.close(fig)
    Path('data/processed/metadata.json').write_text(json.dumps({'synthetic_demo':bool(args.demo)}))
    print(json.dumps(scores,indent=2));print('Artifacts written to reports/. Launch: streamlit run app.py')

if __name__=='__main__': main()
