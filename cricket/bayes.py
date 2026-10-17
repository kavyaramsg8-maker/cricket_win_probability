"""Person 1: match-level Beta-Binomial inference, separate from delivery model."""
from scipy.stats import beta, binomtest, chi2_contingency
import pandas as pd

def posterior(wins,n,a=1,b=1):
    if not 0<=wins<=n or a<=0 or b<=0: raise ValueError('Invalid Beta-Binomial inputs')
    aa,bb=a+wins,b+n-wins
    return {'matches':int(n),'wins':int(wins),'posterior_mean':float(aa/(aa+bb)),
            'credible_interval_95':[float(x) for x in beta.ppf([.025,.975],aa,bb)]}

def match_table(df):
    cols=['match_id','toss_winner','winner']
    if not set(cols)<=set(df.columns): raise ValueError('Match/toss columns missing')
    out=df.drop_duplicates('match_id').copy()
    return out.dropna(subset=['winner','toss_winner'])

def toss_analysis(df):
    m=match_table(df);n=len(m)
    if n==0: raise ValueError('No completed matches')
    success=(m.toss_winner==m.winner);k=int(success.sum())
    result=posterior(k,n);test=binomtest(k,n,.5)
    result.update(p_value=float(test.pvalue),confidence_interval_95=[float(x) for x in test.proportion_ci()])
    if 'toss_decision' in m:
        table=pd.crosstab(m.toss_decision,success)
        if table.shape==(2,2):
            _,p,_,expected=chi2_contingency(table)
            result['toss_decision_chi_square']={'p_value':float(p),'minimum_expected_count':float(expected.min())}
    return result

def team_venue_priors(df):
    m=match_table(df)
    needed={'batting_team','bowling_team','venue'}
    if not needed<=set(m.columns): return []
    rows=[]
    for venue,group in m.groupby('venue'):
        for team in sorted(set(group.batting_team)|set(group.bowling_team)):
            games=group[(group.batting_team==team)|(group.bowling_team==team)]
            rows.append({'team':team,'venue':venue,**posterior(int((games.winner==team).sum()),len(games))})
    return rows
