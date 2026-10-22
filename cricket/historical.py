"""Historical match query and ball-by-ball win-probability trajectory analyzer."""
import pandas as pd
import numpy as np
from .state import State, Outcome, advance

def get_match_list(df):
    """Return deduplicated list of matches with comprehensive metadata."""
    matches = []
    for mid, group in df.groupby('match_id', sort=False):
        row0 = group.iloc[0]
        inn1 = group[group.innings == 1]
        inn2 = group[group.innings == 2]
        if inn1.empty or inn2.empty:
            continue
        bat1 = inn1.iloc[0].batting_team
        bat2 = inn2.iloc[0].batting_team
        score1 = int(inn1.total_runs.sum())
        score2 = int(inn2.total_runs.sum())
        w1 = int(inn1.wicket.sum())
        w2 = int(inn2.wicket.sum())
        overs1 = f"{inn1.legal.sum() // 6}.{inn1.legal.sum() % 6}"
        overs2 = f"{inn2.legal.sum() // 6}.{inn2.legal.sum() % 6}"
        winner = row0.winner
        toss_winner = row0.toss_winner
        toss_decision = row0.toss_decision
        venue = row0.venue
        season = int(row0.season)
        date = str(row0.date)

        # Label for UI selector
        label = f"[{season}] {bat1} ({score1}/{w1}) vs {bat2} ({score2}/{w2}) - {winner} won"
        matches.append({
            'match_id': str(mid),
            'season': season,
            'date': date,
            'venue': str(venue),
            'team1': str(bat1),
            'team2': str(bat2),
            'score1': score1,
            'wickets1': w1,
            'overs1': overs1,
            'target': score1 + 1,
            'score2': score2,
            'wickets2': w2,
            'overs2': overs2,
            'winner': str(winner),
            'toss_winner': str(toss_winner),
            'toss_decision': str(toss_decision),
            'label': label
        })
    return pd.DataFrame(matches).sort_values(['season', 'date'], ascending=[False, False])

def analyze_chase_trajectory(match_deliveries, win_model, calibrator=None, context=None):
    """Compute ball-by-ball win probabilities, WPA swings, and leverage for a chase."""
    chase = match_deliveries[match_deliveries.innings == 2].copy()
    if chase.empty:
        return pd.DataFrame()

    rows = []
    batting_team = chase.iloc[0].batting_team
    bowling_team = chase.iloc[0].bowling_team
    venue = chase.iloc[0].venue
    target = int(chase.iloc[0].target)
    winner = chase.iloc[0].winner

    ctx_val = 0.0
    if context is not None:
        ctx_val = context.feature(batting_team, bowling_team, venue)

    cumulative_runs = 0
    cumulative_wickets = 0
    legal_balls_bowled = 0

    for idx, r in enumerate(chase.itertuples()):
        b = int(r.balls_left)
        w = int(r.wickets_lost)
        req = int(r.runs_required)
        s_before = State(b, w, req)

        p_raw = win_model.probability(s_before)
        p_cal = p_raw
        if calibrator is not None:
            p_cal = calibrator.adjust(p_raw, ctx_val, terminal=s_before.terminal)

        o = Outcome(int(r.total_runs), int(r.wicket), int(r.legal))
        s_after = advance(s_before, o)
        p_after_raw = win_model.probability(s_after)
        p_after_cal = p_after_raw
        if calibrator is not None:
            p_after_cal = calibrator.adjust(p_after_raw, ctx_val, terminal=s_after.terminal)

        wpa_raw = p_after_raw - p_raw
        wpa_cal = p_after_cal - p_cal

        lev = win_model.leverage_index(s_before)

        cumulative_runs += o.runs
        cumulative_wickets += o.wicket
        legal_balls_bowled += o.legal
        over_num = (legal_balls_bowled - 1) // 6 if legal_balls_bowled > 0 else 0
        ball_in_over = (legal_balls_bowled - 1) % 6 + 1 if legal_balls_bowled > 0 else 0
        over_label = f"{over_num}.{ball_in_over}"

        # Event description
        desc_parts = []
        if o.wicket > 0:
            desc_parts.append("WICKET")
        if o.legal == 0:
            desc_parts.append("EXTRA (Illegal)")
        if o.runs == 4:
            desc_parts.append("FOUR")
        elif o.runs == 6:
            desc_parts.append("SIX")
        elif o.runs == 0 and o.wicket == 0:
            desc_parts.append("Dot ball")
        else:
            desc_parts.append(f"{o.runs} run{'s' if o.runs > 1 else ''}")

        event_str = " | ".join(desc_parts)

        rows.append({
            'delivery_index': idx + 1,
            'over_label': over_label,
            'legal_balls_bowled': legal_balls_bowled,
            'balls_left': b,
            'wickets_lost': w,
            'runs_required': req,
            'runs_scored_on_ball': o.runs,
            'wicket': o.wicket,
            'legal': o.legal,
            'cumulative_runs': cumulative_runs,
            'cumulative_wickets': cumulative_wickets,
            'raw_probability': p_raw,
            'calibrated_probability': p_cal,
            'next_raw_probability': p_after_raw,
            'next_calibrated_probability': p_after_cal,
            'wpa_raw': wpa_raw,
            'wpa_calibrated': wpa_cal,
            'leverage_index': lev['leverage_index'],
            'max_swing': lev['max_swing'],
            'event': event_str,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'target': target,
            'winner': winner
        })

    traj_df = pd.DataFrame(rows)
    return traj_df
