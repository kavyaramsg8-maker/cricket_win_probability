"""Person 3: manual match-state demo, not a live score feed."""
import json,pickle
from pathlib import Path
import streamlit as st
from cricket.state import State
from cricket.markov import WinModel
from cricket.simulation import simulate_chase,simulate_first_innings
st.set_page_config(page_title='Cricket Win Probability',layout='centered')
st.title('Cricket Win Probability')
st.caption('Markov chains, Monte Carlo simulation and Bayesian match analysis')
model_path=Path('data/processed/model.pkl')
if not model_path.exists():
    st.info('Run python run_pipeline.py --demo in the project folder first.');st.stop()
meta=json.loads(Path('data/processed/metadata.json').read_text())
if meta['synthetic_demo']: st.warning('SYNTHETIC DEMO: artificial matches. These are not validated IPL predictions.')
st.caption('Standard 20-over matches only. No rain adjustments, Super Overs or live API. A regulation tie counts as failure to reach the target.')
@st.cache_resource
def load_model(stamp):
    # Local trusted file produced by run_pipeline.py. Never load uploaded pickle files.
    with model_path.open('rb') as f: model=pickle.load(f)
    return model,WinModel(model)
model,win=load_model(model_path.stat().st_mtime_ns)
adjustment_path=Path('data/processed/adjustment.pkl')
adjustment=None
if meta.get('adjusted_model') and adjustment_path.exists():
    with adjustment_path.open('rb') as f: adjustment=pickle.load(f)
innings=st.radio('Innings',[ 'Second innings chase','First innings'],horizontal=True)
with st.form('match_state'):
    chasing_team=bowling_team=venue=None
    if adjustment and innings.startswith('Second'):
        context=adjustment['context']
        teams=['Unspecified']+context.teams
        chasing_team=st.selectbox('Chasing team',teams,index=0)
        bowling_team=st.selectbox('Bowling team',teams,index=0)
        venue=st.selectbox('Venue',['Unspecified']+context.venues,index=0)
        st.caption('Unspecified teams use neutral context. Unseen venues fall back to overall team rates.')
    balls=st.number_input('Legal balls remaining',0,120,30)
    wickets=st.number_input('Wickets lost',0,10,4)
    runs=st.number_input('Runs required to win' if innings.startswith('Second') else 'Runs scored so far',0,400,40 if innings.startswith('Second') else 120)
    n=st.select_slider('Simulations',options=[500,1000,2000,5000],value=1000)
    submit=st.form_submit_button('Calculate probability')
if submit:
    with st.spinner('Calculating…'):
        if innings.startswith('Second'):
            if chasing_team and chasing_team!='Unspecified' and chasing_team==bowling_team:
                st.error('Choose two different teams.');st.stop()
            state=State(int(balls),int(wickets),int(runs));p=win.probability(state)
            if adjustment:
                ctx=adjustment['context'].feature(None if chasing_team=='Unspecified' else chasing_team,
                    None if bowling_team=='Unspecified' else bowling_team,venue)
                adjusted=adjustment['calibrator'].adjust(p,ctx,terminal=state.terminal)
                st.metric('Chasing team: calibrated probability with Bayesian context',f'{100*adjusted:.1f}%')
                st.caption('The adjustment uses historical team/venue estimates and coefficients fitted on a separate earlier season.')
            result=simulate_chase(model,state,n)
            st.metric('Chasing team: raw Markov win probability',f'{100*p:.1f}%')
            st.metric('Chasing team: raw Monte Carlo estimate',f'{100*result["win_probability"]:.1f}%')
        else:
            result=simulate_first_innings(model,int(balls),int(wickets),int(runs),n)
            st.caption('First-innings simulation uses the pooled model. The chase calibration is not applied here.')
            st.metric('Batting-first team: Monte Carlo win estimate',f'{100*result["win_probability"]:.1f}%')
            st.write(f'Average simulated final score: {result["mean_final_score"]:.1f}')
        st.write(f'Monte Carlo standard error: {100*result["standard_error"]:.2f} percentage points')
        st.caption('Sampling error for the raw Monte Carlo estimate only. Raw Markov and Monte Carlo should agree; the calibrated probability is a separate adjusted forecast.')
st.subheader('Holdout results')
report_name='adjusted_metrics' if adjustment else 'metrics'
if Path(f'reports/{report_name}.json').exists(): st.json(json.loads(Path(f'reports/{report_name}.json').read_text()))
for name in (['adjusted_win_curve','adjusted_calibration'] if adjustment else ['win_curve','calibration']):
    if Path(f'reports/{name}.png').exists(): st.image(f'reports/{name}.png')
with st.expander('Toss study and Bayesian team/venue estimates'):
    st.caption('Toss analysis is separate. In adjusted chase mode, Bayesian team/venue features enter the final calibrated probability; raw delivery probabilities remain pooled.')
    for name in ['toss_analysis','adjusted_team_venue_priors' if adjustment else 'team_venue_priors']:
        if Path(f'reports/{name}.json').exists(): st.json(json.loads(Path(f'reports/{name}.json').read_text()))
