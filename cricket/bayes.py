"""Person 1: match-level Beta-Binomial inference, separate from delivery model."""
import numpy as np
from scipy.stats import beta, binomtest, chi2_contingency, norm
import pandas as pd

def posterior(wins,n,a=1,b=1):
    if not 0<=wins<=n or a<=0 or b<=0: raise ValueError('Invalid Beta-Binomial inputs')
    aa,bb=a+wins,b+n-wins
    return {'matches':int(n),'wins':int(wins),'posterior_mean':float(aa/(aa+bb)),
            'credible_interval_95':[float(x) for x in beta.ppf([.025,.975],aa,bb)],
            'posterior_alpha':float(aa),'posterior_beta':float(bb),
            'posterior_std':float(np.sqrt((aa*bb)/(((aa+bb)**2)*(aa+bb+1))))}

def match_table(df):
    cols=['match_id','toss_winner','winner']
    if not set(cols)<=set(df.columns): raise ValueError('Match/toss columns missing')
    out=df.drop_duplicates('match_id').copy()
    return out.dropna(subset=['winner','toss_winner'])

def toss_analysis(df):
    m=match_table(df);n=len(m)
    if n==0: raise ValueError('No completed matches')
    success=(m.toss_winner==m.winner);k=int(success.sum())
    result=posterior(k,n)
    test=binomtest(k,n,.5)

    # Z-test for proportions
    p_hat = k / n
    z_stat = float((2 * k - n) / np.sqrt(n)) if n > 0 else 0.0
    z_pvalue = float(2 * (1 - norm.cdf(abs(z_stat))))

    result.update({
        'p_value': float(test.pvalue),
        'confidence_interval_95': [float(x) for x in test.proportion_ci()],
        'z_statistic': z_stat,
        'z_test_p_value': z_pvalue,
        'observed_proportion': float(p_hat),
        'null_hypothesis': 'P(Toss winner wins match) == 0.5',
        'is_statistically_significant': bool(test.pvalue < 0.05)
    })

    if 'toss_decision' in m:
        table=pd.crosstab(m.toss_decision,success)
        if table.shape==(2,2):
            chi2_stat,p,dof,expected=chi2_contingency(table)
            # Odds ratio
            try:
                # table: index=['bat', 'field'], cols=[False, True]
                o_bat_win = float(table.loc['bat', True]) if True in table.columns and 'bat' in table.index else 0.0
                o_bat_loss = float(table.loc['bat', False]) if False in table.columns and 'bat' in table.index else 1.0
                o_fld_win = float(table.loc['field', True]) if True in table.columns and 'field' in table.index else 0.0
                o_fld_loss = float(table.loc['field', False]) if False in table.columns and 'field' in table.index else 1.0
                odds_bat = o_bat_win / max(1.0, o_bat_loss)
                odds_field = o_fld_win / max(1.0, o_fld_loss)
                odds_ratio = float(odds_field / max(1e-6, odds_bat))
            except Exception:
                odds_ratio = 1.0

            result['toss_decision_chi_square']={
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p),
                'degrees_of_freedom': int(dof),
                'minimum_expected_count': float(expected.min()),
                'cochran_condition_met': bool(expected.min() >= 5.0),
                'odds_ratio_field_vs_bat': odds_ratio,
                'is_statistically_significant': bool(p < 0.05),
                'contingency_table': {
                    'bat': {'win': int(table.loc['bat', True]) if True in table.columns and 'bat' in table.index else 0,
                            'loss': int(table.loc['bat', False]) if False in table.columns and 'bat' in table.index else 0},
                    'field': {'win': int(table.loc['field', True]) if True in table.columns and 'field' in table.index else 0,
                              'loss': int(table.loc['field', False]) if False in table.columns and 'field' in table.index else 0}
                }
            }
    return result

def venue_chase_posteriors(df, prior_a=1.0, prior_b=1.0):
    """Bayesian venue chasing analysis via Beta-Binomial updating with 95% credible intervals."""
    # Innings 2 batting team is the chasing team
    chases = df[df.innings == 2].drop_duplicates('match_id')
    if chases.empty or 'venue' not in chases.columns or 'winner' not in chases.columns:
        return []
    records = []
    for venue, group in chases.groupby('venue'):
        n = len(group)
        wins = int((group.batting_team == group.winner).sum())
        post = posterior(wins, n, a=prior_a, b=prior_b)
        records.append({
            'venue': str(venue),
            'matches': n,
            'chase_wins': wins,
            'empirical_win_rate': float(wins / n) if n > 0 else 0.5,
            'posterior_mean': post['posterior_mean'],
            'credible_interval_95': post['credible_interval_95'],
            'ci_lower': post['credible_interval_95'][0],
            'ci_upper': post['credible_interval_95'][1],
            'posterior_std': post['posterior_std']
        })
    return sorted(records, key=lambda x: x['matches'], reverse=True)

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
