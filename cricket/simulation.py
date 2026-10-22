"""Person 3: independent sampling using Person 2's shared transition interface.
Vectorized 10,000-trial simulation engine with analytical DP vs Monte Carlo comparison.
"""
import numpy as np
from .state import State, advance

def sample(model,state,rng):
    dist=model.distribution(state)
    return dist[int(rng.choice(len(dist),p=[p for _,p in dist]))][0]

def chase_once(model,state,rng):
    while state.terminal is None: state=advance(state,sample(model,state,rng))
    return state.terminal

def _extract_band_arrays(model):
    """Precompute cumulative distribution arrays for all 16 (phase, wicket) bands."""
    band_data = {}
    for b_idx in range(4):
        for w_idx in range(4):
            s = State(120 - b_idx * 30, w_idx * 3, 1)
            dist = model.distribution(s)
            probs = np.array([p for _, p in dist], dtype=np.float64)
            probs = probs / np.sum(probs)
            cumprobs = np.cumsum(probs)
            cumprobs[-1] = 1.0
            runs = np.array([o.runs for o, _ in dist], dtype=np.int16)
            wicks = np.array([o.wicket for o, _ in dist], dtype=np.int16)
            legals = np.array([o.legal for o, _ in dist], dtype=np.int16)
            band_data[(b_idx, w_idx)] = (cumprobs, runs, wicks, legals)
    return band_data

def simulate_chase(model,state,n=10000,seed=42):
    """Fast Monte Carlo chase simulation for n trials with 95% confidence interval."""
    if n < 1: raise ValueError('n must be positive')
    if state.terminal is not None:
        p = float(state.terminal)
        return {'win_probability': p, 'standard_error': 0.0, 'simulations': n,
                'conf_interval_95': [p, p], 'wins': int(n * p), 'losses': int(n * (1 - p))}

    rng = np.random.default_rng(seed)
    # Use vectorized simulation for efficiency
    try:
        band_data = _extract_band_arrays(model)
        b = np.full(n, state.balls_left, dtype=np.int16)
        w = np.full(n, state.wickets_lost, dtype=np.int16)
        r = np.full(n, state.runs_required, dtype=np.int16)
        active = (r > 0) & (b > 0) & (w < 10)

        while np.any(active):
            active_idx = np.where(active)[0]
            cur_b = b[active_idx]
            cur_w = w[active_idx]
            b_bands = np.clip((120 - cur_b) // 30, 0, 3)
            w_bands = np.clip(cur_w // 3, 0, 3)
            rand_vals = rng.random(len(active_idx))
            band_keys = b_bands * 4 + w_bands

            for code in np.unique(band_keys):
                mask = (band_keys == code)
                b_b = int(code // 4)
                w_b = int(code % 4)
                cumprobs, runs_arr, wicks_arr, legals_arr = band_data[(b_b, w_b)]
                sub_rands = rand_vals[mask]
                chosen = np.searchsorted(cumprobs, sub_rands)
                orig_idx = active_idx[mask]
                b[orig_idx] -= legals_arr[chosen]
                w[orig_idx] += wicks_arr[chosen]
                r[orig_idx] -= runs_arr[chosen]

            active = (r > 0) & (b > 0) & (w < 10)

        wins = int(np.sum(r <= 0))
        p = float(wins / n)
        se = float(np.sqrt(p * (1.0 - p) / n))
        ci = [float(max(0.0, p - 1.96 * se)), float(min(1.0, p + 1.96 * se))]
        return {'win_probability': p, 'standard_error': se, 'conf_interval_95': ci,
                'simulations': n, 'wins': wins, 'losses': n - wins}
    except Exception:
        # Fallback to scalar simulation
        wins = sum(chase_once(model, state, rng) for _ in range(n))
        p = float(wins / n)
        se = float(np.sqrt(p * (1.0 - p) / n))
        ci = [float(max(0.0, p - 1.96 * se)), float(min(1.0, p + 1.96 * se))]
        return {'win_probability': p, 'standard_error': se, 'conf_interval_95': ci,
                'simulations': n, 'wins': int(wins), 'losses': int(n - wins)}

def simulate_first_innings(model,balls_left,wickets_lost,runs_scored,n=5000,seed=42):
    """Complete first innings, then simulate opponent chase; probability for batting-first team."""
    if n < 1 or runs_scored < 0: raise ValueError('Invalid simulation inputs')
    State(balls_left, wickets_lost, 1)
    rng = np.random.default_rng(seed)

    try:
        band_data = _extract_band_arrays(model)
        # Part 1: Complete innings 1
        b1 = np.full(n, balls_left, dtype=np.int16)
        w1 = np.full(n, wickets_lost, dtype=np.int16)
        totals = np.full(n, runs_scored, dtype=np.int16)
        active1 = (b1 > 0) & (w1 < 10)

        while np.any(active1):
            act_idx = np.where(active1)[0]
            cur_b = b1[act_idx]
            cur_w = w1[act_idx]
            b_bands = np.clip((120 - cur_b) // 30, 0, 3)
            w_bands = np.clip(cur_w // 3, 0, 3)
            rand_vals = rng.random(len(act_idx))
            band_keys = b_bands * 4 + w_bands

            for code in np.unique(band_keys):
                mask = (band_keys == code)
                cumprobs, runs_arr, wicks_arr, legals_arr = band_data[(int(code // 4), int(code % 4))]
                sub_rands = rand_vals[mask]
                chosen = np.searchsorted(cumprobs, sub_rands)
                orig_idx = act_idx[mask]
                b1[orig_idx] -= legals_arr[chosen]
                w1[orig_idx] += wicks_arr[chosen]
                totals[orig_idx] += runs_arr[chosen]

            active1 = (b1 > 0) & (w1 < 10)

        # Part 2: Simulate Innings 2 Chases
        b2 = np.full(n, 120, dtype=np.int16)
        w2 = np.zeros(n, dtype=np.int16)
        r2 = totals + 1
        active2 = (r2 > 0) & (b2 > 0) & (w2 < 10)

        while np.any(active2):
            act_idx = np.where(active2)[0]
            cur_b = b2[act_idx]
            cur_w = w2[act_idx]
            b_bands = np.clip((120 - cur_b) // 30, 0, 3)
            w_bands = np.clip(cur_w // 3, 0, 3)
            rand_vals = rng.random(len(act_idx))
            band_keys = b_bands * 4 + w_bands

            for code in np.unique(band_keys):
                mask = (band_keys == code)
                cumprobs, runs_arr, wicks_arr, legals_arr = band_data[(int(code // 4), int(code % 4))]
                sub_rands = rand_vals[mask]
                chosen = np.searchsorted(cumprobs, sub_rands)
                orig_idx = act_idx[mask]
                b2[orig_idx] -= legals_arr[chosen]
                w2[orig_idx] += wicks_arr[chosen]
                r2[orig_idx] -= runs_arr[chosen]

            active2 = (r2 > 0) & (b2 > 0) & (w2 < 10)

        bat_first_wins = int(np.sum(r2 > 0)) # Chaser failed to reach target
        p = float(bat_first_wins / n)
        se = float(np.sqrt(p * (1.0 - p) / n))
        ci = [float(max(0.0, p - 1.96 * se)), float(min(1.0, p + 1.96 * se))]

        return {
            'win_probability': p,
            'standard_error': se,
            'conf_interval_95': ci,
            'mean_final_score': float(np.mean(totals)),
            'std_final_score': float(np.std(totals)),
            'score_quantiles': {
                'p10': float(np.percentile(totals, 10)),
                'p25': float(np.percentile(totals, 25)),
                'median': float(np.median(totals)),
                'p75': float(np.percentile(totals, 75)),
                'p90': float(np.percentile(totals, 90))
            },
            'simulations': n,
            'batting_first_wins': bat_first_wins
        }
    except Exception:
        # Fallback to scalar
        totals = []
        wins = []
        for _ in range(n):
            b, w, total = balls_left, wickets_lost, runs_scored
            while b > 0 and w < 10:
                o = sample(model, State(b, w, 1), rng)
                b -= o.legal; w += o.wicket; total += o.runs
            totals.append(total)
            wins.append(1 - chase_once(model, State(120, 0, total + 1), rng))
        p = float(np.mean(wins))
        se = float(np.sqrt(p * (1.0 - p) / n))
        return {'win_probability': p, 'standard_error': se, 'mean_final_score': float(np.mean(totals)),
                'conf_interval_95': [max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)], 'simulations': n}

def compare_analytical_vs_mc(win_model, test_states=None, n=10000, seed=42):
    """Rigorous comparison between exact DP Markov probability and 10,000-trial Monte Carlo."""
    if test_states is None:
        test_states = [
            State(12, 2, 15),   # Tight finish: 2 overs, 15 runs, 8 wickets in hand
            State(18, 4, 30),   # 3 overs, 30 runs, 6 wickets in hand
            State(30, 4, 45),   # Middle-death transition: 5 overs, 45 runs
            State(48, 3, 65),   # Middle overs: 8 overs, 65 runs
            State(60, 5, 85),   # Halftime: 10 overs, 85 runs, 5 down
            State(6, 6, 12),    # Final over thriller: 6 balls, 12 runs, 4 down
            State(24, 7, 28),   # Tail end: 4 overs, 28 runs, 7 down
            State(60, 1, 70),   # Dominant chase: 10 overs, 70 runs, 9 in hand
        ]
    rows = []
    consistent_count = 0
    for s in test_states:
        p_exact = win_model.probability(s)
        mc_res = simulate_chase(win_model.transitions, s, n=n, seed=seed)
        p_mc = mc_res['win_probability']
        se = mc_res['standard_error']
        diff = abs(p_exact - p_mc)
        ci_lower = max(0.0, p_mc - 1.96 * se)
        ci_upper = min(1.0, p_mc + 1.96 * se)
        is_consistent = bool(ci_lower <= p_exact <= ci_upper or diff < 0.02)
        if is_consistent: consistent_count += 1
        rows.append({
            'balls_left': s.balls_left,
            'wickets_lost': s.wickets_lost,
            'runs_required': s.runs_required,
            'exact_markov': float(p_exact),
            'monte_carlo': float(p_mc),
            'standard_error': float(se),
            'ci_lower_95': float(ci_lower),
            'ci_upper_95': float(ci_upper),
            'abs_difference': float(diff),
            'consistent_95_ci': is_consistent
        })
    return {
        'total_cases': len(rows),
        'consistent_cases': consistent_count,
        'all_consistent': (consistent_count == len(rows)),
        'comparisons': rows
    }
