"""Live Win-Probability Engine for T20 Cricket.
Comprehensive sports-broadcast analytics dashboard implementing Markov chains,
Bayesian inference, Monte Carlo simulation, and calibration.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import beta as beta_dist, norm

# Ensure deep recursion limit for any uncached states
sys.setrecursionlimit(25000)

from cricket.state import State, Outcome, advance
from cricket.markov import TransitionModel, WinModel
from cricket.simulation import simulate_chase, simulate_first_innings, compare_analytical_vs_mc
from cricket.bayes import toss_analysis, venue_chase_posteriors, team_venue_priors
from cricket.adjustment import BayesianContext, ChaseCalibrator, team_name
from cricket.evaluation import metrics, calibration, calibration_summary
from cricket.historical import get_match_list, analyze_chase_trajectory

# ==========================================
# PAGE CONFIGURATION & CUSTOM BROADCAST CSS
# ==========================================
st.set_page_config(
    page_title="T20 Cricket Win-Probability Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Modern Sports Broadcast Dark Theme */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .broadcast-title {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Glowing broadcast card */
.broadcast-card {
    background: linear-gradient(135deg, rgba(13, 27, 42, 0.9) 0%, rgba(27, 38, 59, 0.85) 100%);
    border: 1px solid rgba(0, 230, 118, 0.25);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
}

.scoreboard-banner {
    background: linear-gradient(90deg, #0A1128 0%, #001F54 50%, #034078 100%);
    border-left: 6px solid #00E676;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 230, 118, 0.15);
}

.metric-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pill-live { background: rgba(255, 51, 102, 0.2); color: #FF3366; border: 1px solid #FF3366; }
.pill-chase { background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
.pill-info { background: rgba(58, 134, 255, 0.2); color: #3A86FF; border: 1px solid #3A86FF; }
.pill-gold { background: rgba(255, 209, 102, 0.2); color: #FFD166; border: 1px solid #FFD166; }

/* Pulse animation */
@keyframes pulse {
    0% { transform: scale(0.97); opacity: 0.8; }
    50% { transform: scale(1.03); opacity: 1; }
    100% { transform: scale(0.97); opacity: 0.8; }
}
.pulse-badge {
    animation: pulse 2s infinite;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# RESOURCE CACHING & ARTIFACT LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    model_path = Path('data/processed/model.pkl')
    adj_path = Path('data/processed/adjustment.pkl')
    meta_path = Path('data/processed/metadata.json')

    if not model_path.exists():
        return None, None, None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    win_model = WinModel(model)

    adjustment = None
    if adj_path.exists():
        with open(adj_path, 'rb') as f:
            adjustment = pickle.load(f)

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return model, win_model, adjustment, meta

@st.cache_data(show_spinner=False)
def load_datasets():
    train_path = Path('data/processed/train.csv')
    test_path = Path('data/processed/test.csv')
    raw_path = Path('data/raw/ipl_deliveries.csv')

    train_df = pd.read_csv(train_path, low_memory=False) if train_path.exists() else None
    test_df = pd.read_csv(test_path, low_memory=False) if test_path.exists() else None
    raw_df = pd.read_csv(raw_path, low_memory=False) if raw_path.exists() else None

    # Fallback to test_df or train_df if raw_df not found
    full_df = raw_df if raw_df is not None else (test_df if train_df is None else pd.concat([train_df, test_df], ignore_index=True))
    matches_list = get_match_list(full_df) if full_df is not None else pd.DataFrame()
    return train_df, test_df, full_df, matches_list

@st.cache_data(show_spinner=False)
def load_reports():
    reports = {}
    report_files = [
        'metrics.json', 'adjusted_metrics.json', 'toss_analysis.json',
        'venue_chase_stats.json', 'team_venue_priors.json', 'mc_vs_markov.json',
        'calibration.csv', 'adjusted_calibration.csv'
    ]
    for rf in report_files:
        p = Path(f'reports/{rf}')
        if p.exists():
            if rf.endswith('.json'):
                try: reports[rf.replace('.json', '')] = json.loads(p.read_text())
                except: pass
            elif rf.endswith('.csv'):
                try: reports[rf.replace('.csv', '')] = pd.read_csv(p)
                except: pass
    return reports

model, win_model, adjustment, meta = load_models()
train_df, test_df, full_df, matches_list = load_datasets()
reports = load_reports()

if model is None or win_model is None:
    st.error("⚠️ Model files not found! Please run `python run_pipeline.py` from the project root first.")
    st.stop()

# Context and teams setup
context = adjustment['context'] if adjustment else None
calibrator = adjustment['calibrator'] if adjustment else None
team_options = ['Unspecified'] + (context.teams if context else [])
venue_options = ['Unspecified'] + (context.venues if context else [])

# ==========================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 16px 0;">
    <h2 style="color: #00E676; margin-bottom: 2px;">🏏 CRIC-PROB 2.0</h2>
    <p style="color: #90A4AE; font-size: 0.85rem; margin: 0;">T20 Win-Probability Engine</p>
</div>
""", unsafe_allow_html=True)

nav_section = st.sidebar.selectbox(
    "Select Analytics Module",
    [
        "🎯 Win Probability Predictor",
        "🏏 Live Match Dashboard",
        "⚡ Ball-by-Ball Match Simulator",
        "📈 Probability Worm Chart",
        "⚖️ Delivery Leverage Analysis",
        "🎲 Monte Carlo Simulation",
        "🏟️ Bayesian Venue Analysis",
        "🪙 Toss Hypothesis Testing",
        "📊 Model Validation and Calibration",
        "📜 Historical Match Analysis"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Telemetry")
st.sidebar.markdown(f"**Model Architecture:** Absorbing Markov Chain")
st.sidebar.markdown(f"**Smoothing:** Laplace (α = 0.5)")
st.sidebar.markdown(f"**Context:** Bayesian Beta-Binomial")
st.sidebar.markdown(f"**Calibration:** Monotone Platt Scaling")
if meta:
    st.sidebar.markdown(f"**Training Seasons:** {min(meta.get('train_seasons', [2008]))}–{max(meta.get('train_seasons', [2023]))}")
    st.sidebar.markdown(f"**Holdout Test Season:** {meta.get('test_seasons', [2024])[0]}")

# ==========================================
# HELPER FUNCTIONS: PLOTLY CHARTS
# ==========================================
def create_probability_gauge(prob, title="Win Probability", team_name="Chasing Team", raw_prob=None):
    """Create a vibrant broadcast radial probability gauge."""
    fig = go.Figure()
    pct = prob * 100
    
    # Gradient color logic
    if pct >= 70:
        bar_color = "#00E676"
    elif pct >= 50:
        bar_color = "#00F5D4"
    elif pct >= 30:
        bar_color = "#FFD166"
    else:
        bar_color = "#FF3366"

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 44, 'family': 'Rajdhani', 'color': '#FFFFFF'}},
        title={'text': f"<b>{team_name}</b><br><span style='font-size:0.85em;color:#90A4AE'>{title}</span>",
               'font': {'size': 18, 'family': 'Rajdhani', 'color': '#E0E1DD'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#415A77", 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(13, 27, 42, 0.6)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.15)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(255, 51, 102, 0.15)"},
                {'range': [30, 50], 'color': "rgba(255, 209, 102, 0.15)"},
                {'range': [50, 70], 'color': "rgba(0, 245, 212, 0.15)"},
                {'range': [70, 100], 'color': "rgba(0, 230, 118, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.75,
                'value': pct
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#FFFFFF"}
    )
    return fig

def create_worm_chart(traj_df, chasing_team="Chaser", bowling_team="Defenders"):
    """Create an interactive Plotly worm chart with critical wickets and boundaries."""
    fig = go.Figure()
    if traj_df.empty:
        return fig

    # 50% baseline
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)",
                  annotation_text="50-50 Balance", annotation_position="bottom right",
                  annotation_font=dict(color="#90A4AE", size=10))

    # Calibrated probability line
    if 'calibrated_probability' in traj_df.columns:
        fig.add_trace(go.Scatter(
            x=traj_df['delivery_index'],
            y=traj_df['calibrated_probability'] * 100,
            mode='lines',
            name=f'{chasing_team} (Calibrated)',
            line=dict(color='#00E676', width=3.5, shape='spline'),
            hovertemplate="Ball %{x} (%{customdata[0]})<br>Score: %{customdata[1]}/%{customdata[2]}<br>Req: %{customdata[3]} in %{customdata[4]} balls<br>Win Prob: <b>%{y:.1f}%</b><br>Event: %{customdata[5]}<extra></extra>",
            customdata=np.column_stack([
                traj_df['over_label'], traj_df['cumulative_runs'], traj_df['cumulative_wickets'],
                traj_df['runs_required'], traj_df['balls_left'], traj_df['event']
            ])
        ))

    # Raw Markov line
    if 'raw_probability' in traj_df.columns:
        fig.add_trace(go.Scatter(
            x=traj_df['delivery_index'],
            y=traj_df['raw_probability'] * 100,
            mode='lines',
            name=f'{chasing_team} (Raw Markov)',
            line=dict(color='#FF9800', width=1.5, dash='dot'),
            hoverinfo='skip'
        ))

    # Markers for Wickets
    wickets = traj_df[traj_df['wicket'] > 0]
    if not wickets.empty:
        fig.add_trace(go.Scatter(
            x=wickets['delivery_index'],
            y=(wickets['calibrated_probability'] if 'calibrated_probability' in wickets else wickets['raw_probability']) * 100,
            mode='markers',
            name='Wicket Lost',
            marker=dict(symbol='x', size=11, color='#FF3366', line=dict(width=2, color='#FFFFFF')),
            hovertemplate="💥 WICKET! Ball %{x}<br>Score: %{customdata[0]}/%{customdata[1]}<extra></extra>",
            customdata=np.column_stack([wickets['cumulative_runs'], wickets['cumulative_wickets']])
        ))

    # Markers for Sixes
    sixes = traj_df[traj_df['runs_scored_on_ball'] == 6]
    if not sixes.empty:
        fig.add_trace(go.Scatter(
            x=sixes['delivery_index'],
            y=(sixes['calibrated_probability'] if 'calibrated_probability' in sixes else sixes['raw_probability']) * 100,
            mode='markers',
            name='Six (6)',
            marker=dict(symbol='star', size=11, color='#FFD166', line=dict(width=1, color='#000')),
            hovertemplate="🚀 MAXIMUM SIX! Ball %{x}<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>Win-Probability Trajectory (Worm Chart)</b>",
        xaxis_title="Deliveries Bowled in Chase",
        yaxis_title="Chasing Team Win Probability (%)",
        yaxis=dict(range=[-2, 102], ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 27, 42, 0.4)",
        font=dict(color="#FFFFFF", family='Inter'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Initialize simulator in session state if missing
if 'sim_balls' not in st.session_state:
    st.session_state.sim_balls = 30
    st.session_state.sim_wickets = 4
    st.session_state.sim_runs = 40
    st.session_state.sim_target = 180
    st.session_state.sim_chaser = 'Kolkata Knight Riders'
    st.session_state.sim_bowler = 'Royal Challengers Bengaluru'
    st.session_state.sim_venue = 'Eden Gardens'
    st.session_state.sim_history = []


# =========================================================================
# MODULE 1: LIVE MATCH DASHBOARD
# =========================================================================
if nav_section == "🏏 Live Match Dashboard":
    st.markdown("""
    <div class="scoreboard-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="metric-pill pill-live pulse-badge">● LIVE MATCH BROADCAST</span>
                <span class="metric-pill pill-chase">CHASE PHASE</span>
                <span class="metric-pill pill-info">IPL MATCH CENTER</span>
            </div>
            <div style="color: #90A4AE; font-size: 0.9rem;">
                Target: <b style="color: #FFD166; font-size: 1.1rem;">{}</b>
            </div>
        </div>
    </div>
    """.format(st.session_state.sim_target), unsafe_allow_html=True)

    # Current match situation
    b = st.session_state.sim_balls
    w = st.session_state.sim_wickets
    r = st.session_state.sim_runs
    target = st.session_state.sim_target
    chaser = st.session_state.sim_chaser
    bowler = st.session_state.sim_bowler
    venue = st.session_state.sim_venue

    score_so_far = max(0, target - r)
    overs_bowled = (120 - b) // 6 + ((120 - b) % 6) / 10
    crr = (score_so_far / ((120 - b) / 6)) if b < 120 else 0.0
    rrr = (r / (b / 6)) if b > 0 else 0.0

    cur_state = State(b, w, r)
    p_raw = win_model.probability(cur_state)
    ctx_val = context.feature(chaser, bowler, venue) if context else 0.0
    p_cal = calibrator.adjust(p_raw, ctx_val, terminal=cur_state.terminal) if calibrator else p_raw

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="broadcast-card" style="text-align: center;">
            <div style="color: #90A4AE; font-size: 0.85rem;">CHASING SCORE</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #00E676;">{score_so_far}/{w}</div>
            <div style="color: #E0E1DD; font-size: 0.9rem;">{overs_bowled:.1f} / 20.0 Overs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="broadcast-card" style="text-align: center;">
            <div style="color: #90A4AE; font-size: 0.85rem;">RUNS REQUIRED</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #FFD166;">{r}</div>
            <div style="color: #E0E1DD; font-size: 0.9rem;">From {b} Balls Left</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="broadcast-card" style="text-align: center;">
            <div style="color: #90A4AE; font-size: 0.85rem;">REQUIRED RUN RATE</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {'#FF3366' if rrr > 12 else '#3A86FF'};">{rrr:.2f}</div>
            <div style="color: #E0E1DD; font-size: 0.9rem;">Current RR: {crr:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="broadcast-card" style="text-align: center;">
            <div style="color: #90A4AE; font-size: 0.85rem;">WICKETS IN HAND</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #00F5D4;">{10 - w}</div>
            <div style="color: #E0E1DD; font-size: 0.9rem;">{w} Dismissals</div>
        </div>
        """, unsafe_allow_html=True)

    # Gauges & Momentum
    g_col1, g_col2 = st.columns([1.2, 1])
    with g_col1:
        st.plotly_chart(
            create_probability_gauge(p_cal, title="Calibrated Win Probability", team_name=chaser, raw_prob=p_raw),
            use_container_width=True
        )
    with g_col2:
        lev = win_model.leverage_index(cur_state)
        li_val = lev['leverage_index']
        li_badge = "EXTREME HIGH LEVERAGE" if li_val > 2.0 else ("HIGH LEVERAGE" if li_val > 1.3 else ("MODERATE" if li_val > 0.8 else "LOW LEVERAGE"))
        li_color = "#FF3366" if li_val > 2.0 else ("#FFD166" if li_val > 1.3 else "#00E676")

        st.markdown(f"""
        <div class="broadcast-card" style="height: 260px; display: flex; flex-direction: column; justify-content: center;">
            <div style="color: #90A4AE; font-size: 0.85rem;">DELIVERY LEVERAGE INDEX (LI)</div>
            <div style="font-size: 2.6rem; font-weight: 700; color: {li_color};">{li_val:.2f}x</div>
            <div style="margin-bottom: 12px;"><span class="metric-pill" style="background: rgba(255,255,255,0.1); color: {li_color}; border: 1px solid {li_color};">{li_badge}</span></div>
            <div style="color: #B0BEC5; font-size: 0.85rem;">
                Max swing potential on next ball: <b style="color: #FFFFFF;">±{lev['max_swing']*100:.1f}%</b><br>
                Expected swing magnitude: <b style="color: #FFFFFF;">{lev['expected_swing']*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Live Match Controls quick summary
    st.markdown("### 🎮 Quick Match State Adjuster")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        new_chaser = st.selectbox("Chasing Team", team_options[1:], index=team_options[1:].index(chaser) if chaser in team_options[1:] else 0)
    with qc2:
        new_bowler = st.selectbox("Bowling Team", team_options[1:], index=team_options[1:].index(bowler) if bowler in team_options[1:] else 1)
    with qc3:
        new_venue = st.selectbox("Venue", venue_options[1:], index=venue_options[1:].index(venue) if venue in venue_options[1:] else 0)
    with qc4:
        new_target = st.number_input("Target Score", 50, 300, target)

    if new_chaser != chaser or new_bowler != bowler or new_venue != venue or new_target != target:
        st.session_state.sim_chaser = new_chaser
        st.session_state.sim_bowler = new_bowler
        st.session_state.sim_venue = new_venue
        st.session_state.sim_target = new_target
        st.rerun()


# =========================================================================
# MODULE 2: BALL-BY-BALL MATCH SIMULATOR
# =========================================================================
elif nav_section == "⚡ Ball-by-Ball Match Simulator":
    st.markdown("## ⚡ Interactive Ball-by-Ball Match Simulator")
    st.markdown("Simulate individual deliveries or full overs and watch the win-probability needle respond instantly.")

    b = st.session_state.sim_balls
    w = st.session_state.sim_wickets
    r = st.session_state.sim_runs
    target = st.session_state.sim_target
    chaser = st.session_state.sim_chaser
    bowler = st.session_state.sim_bowler
    venue = st.session_state.sim_venue
    ctx_val = context.feature(chaser, bowler, venue) if context else 0.0

    cur_state = State(b, w, r)
    p_cur = calibrator.adjust(win_model.probability(cur_state), ctx_val, terminal=cur_state.terminal) if calibrator else win_model.probability(cur_state)

    # Status Banner
    st.markdown(f"""
    <div class="scoreboard-banner">
        <h3 style="margin: 0; color: #FFFFFF;">{chaser} vs {bowler}</h3>
        <p style="margin: 4px 0 0 0; color: #90A4AE;">
            Target: <b>{target}</b> | Need <b>{r} runs</b> from <b>{b} legal balls</b> | <b>{10 - w} wickets</b> in hand | Ground: <b>{venue}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Outcome action buttons
    st.markdown("#### 🎯 Play Next Delivery")
    btn_cols = st.columns(9)

    outcomes_to_test = [
        ("⚪ Dot (0)", 0, 0, 1),
        ("1 Single", 1, 0, 1),
        ("2 Double", 2, 0, 1),
        ("3 Three", 3, 0, 1),
        ("🔥 FOUR (4)", 4, 0, 1),
        ("🚀 SIX (6)", 6, 0, 1),
        ("💥 WICKET", 0, 1, 1),
        ("🟡 Wide (+1)", 1, 0, 0),
        ("⚠️ No-Ball (+1)", 1, 0, 0)
    ]

    delivery_played = None
    for idx, (label, runs, wicket, legal) in enumerate(outcomes_to_test):
        with btn_cols[idx]:
            if st.button(label, key=f"btn_play_{idx}", disabled=(cur_state.terminal is not None), use_container_width=True):
                delivery_played = Outcome(runs, wicket, legal)

    # Auto Simulation buttons
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns([1, 1, 1, 1])
    with sim_c1:
        if st.button("🎲 Auto-Simulate Ball", use_container_width=True, disabled=(cur_state.terminal is not None)):
            dist = win_model.transitions.distribution(cur_state)
            probs = [p for _, p in dist]
            chosen_outcome = dist[np.random.choice(len(dist), p=probs)][0]
            delivery_played = chosen_outcome

    with sim_c2:
        if st.button("⚡ Simulate Full Over", use_container_width=True, disabled=(cur_state.terminal is not None)):
            for _ in range(6):
                if cur_state.terminal is not None: break
                dist = win_model.transitions.distribution(cur_state)
                chosen_o = dist[np.random.choice(len(dist), p=[p for _, p in dist])][0]
                ns = advance(cur_state, chosen_o)
                p_next = calibrator.adjust(win_model.probability(ns), ctx_val, terminal=ns.terminal) if calibrator else win_model.probability(ns)
                st.session_state.sim_history.append({
                    'ball_num': len(st.session_state.sim_history) + 1,
                    'runs': chosen_o.runs, 'wicket': chosen_o.wicket, 'legal': chosen_o.legal,
                    'balls_left': ns.balls_left, 'wickets_lost': ns.wickets_lost, 'runs_required': ns.runs_required,
                    'prob_before': p_cur, 'prob_after': p_next, 'wpa': p_next - p_cur
                })
                cur_state = ns
                p_cur = p_next
            st.session_state.sim_balls = cur_state.balls_left
            st.session_state.sim_wickets = cur_state.wickets_lost
            st.session_state.sim_runs = cur_state.runs_required
            st.rerun()

    with sim_c3:
        if st.button("↩️ Undo Delivery", use_container_width=True, disabled=(len(st.session_state.sim_history) == 0)):
            last = st.session_state.sim_history.pop()
            # restore prior state
            st.session_state.sim_balls = last['balls_left'] + last['legal']
            st.session_state.sim_wickets = max(0, last['wickets_lost'] - last['wicket'])
            st.session_state.sim_runs = last['runs_required'] + last['runs']
            st.rerun()

    with sim_c4:
        if st.button("🔄 Reset Match", use_container_width=True):
            st.session_state.sim_balls = 30
            st.session_state.sim_wickets = 4
            st.session_state.sim_runs = 40
            st.session_state.sim_history = []
            st.rerun()

    # Process single delivery
    if delivery_played is not None:
        next_s = advance(cur_state, delivery_played)
        p_next = calibrator.adjust(win_model.probability(next_s), ctx_val, terminal=next_s.terminal) if calibrator else win_model.probability(next_s)
        st.session_state.sim_history.append({
            'ball_num': len(st.session_state.sim_history) + 1,
            'runs': delivery_played.runs, 'wicket': delivery_played.wicket, 'legal': delivery_played.legal,
            'balls_left': next_s.balls_left, 'wickets_lost': next_s.wickets_lost, 'runs_required': next_s.runs_required,
            'prob_before': p_cur, 'prob_after': p_next, 'wpa': p_next - p_cur
        })
        st.session_state.sim_balls = next_s.balls_left
        st.session_state.sim_wickets = next_s.wickets_lost
        st.session_state.sim_runs = next_s.runs_required
        st.rerun()

    # Display Current Probability Gauge and Status
    if cur_state.terminal == 1.0:
        st.success(f"🏆 VICTORY! {chaser} reached the target and won the match!")
    elif cur_state.terminal == 0.0:
        st.error(f"❌ DEFEAT! {bowler} defended the total successfully!")

    pg_col, worm_col = st.columns([1, 2])
    with pg_col:
        st.plotly_chart(create_probability_gauge(p_cur, title="Live Win Probability", team_name=chaser), use_container_width=True)
    with worm_col:
        # Build history dataframe for live worm chart
        if st.session_state.sim_history:
            hist_df = pd.DataFrame(st.session_state.sim_history)
            hist_df['delivery_index'] = hist_df['ball_num']
            hist_df['calibrated_probability'] = hist_df['prob_after']
            hist_df['cumulative_runs'] = target - hist_df['runs_required']
            hist_df['cumulative_wickets'] = hist_df['wickets_lost']
            hist_df['runs_scored_on_ball'] = hist_df['runs']
            hist_df['over_label'] = [(120 - b)//6 for b in hist_df['balls_left']]
            hist_df['event'] = [f"{r} runs {'(W)' if w else ''}" for r, w in zip(hist_df['runs'], hist_df['wicket'])]
            st.plotly_chart(create_worm_chart(hist_df, chaser, bowler), use_container_width=True)
        else:
            st.info("Delivery worm chart will generate as you play deliveries.")

    # Ball log table
    if st.session_state.sim_history:
        with st.expander("📜 Live Delivery Log & WPA Swings", expanded=True):
            log_table = pd.DataFrame(st.session_state.sim_history)[::-1]
            log_table['Win Prob'] = (log_table['prob_after'] * 100).map("{:.1f}%".format)
            log_table['WPA Swing'] = (log_table['wpa'] * 100).map("{:+.1f}%".format)
            log_table['Balls Left'] = log_table['balls_left']
            log_table['Wickets'] = log_table['wickets_lost']
            log_table['Req Runs'] = log_table['runs_required']
            st.dataframe(log_table[['ball_num', 'runs', 'wicket', 'legal', 'Balls Left', 'Wickets', 'Req Runs', 'Win Prob', 'WPA Swing']], use_container_width=True)


# =========================================================================
# MODULE 3: WIN PROBABILITY PREDICTOR
# =========================================================================
elif nav_section == "🎯 Win Probability Predictor":
    st.markdown("## 🎯 Custom Match State Predictor")
    st.markdown("Enter any exact match situation to evaluate the Exact Dynamic Programming Markov probability, Calibrated Hybrid probability, and Monte Carlo simulation.")

    innings = st.radio('Innings', ['Second innings chase', 'First innings'], horizontal=True)

    with st.form('match_state'):
        chasing_team = bowling_team = venue = None
        if adjustment and innings.startswith('Second'):
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                chasing_team = st.selectbox('Chasing team', team_options, index=team_options.index('Kolkata Knight Riders') if 'Kolkata Knight Riders' in team_options else 0)
            with col_t2:
                bowling_team = st.selectbox('Bowling team', team_options, index=team_options.index('Royal Challengers Bengaluru') if 'Royal Challengers Bengaluru' in team_options else 0)
            with col_t3:
                venue = st.selectbox('Venue', venue_options, index=venue_options.index('Eden Gardens') if 'Eden Gardens' in venue_options else 0)
            st.caption('Unspecified teams use neutral context. Known teams activate historical Bayesian strength updates.')

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            balls = st.number_input('Legal balls remaining', 0, 120, 30)
        with fc2:
            wickets = st.number_input('Wickets lost', 0, 10, 4)
        with fc3:
            runs = st.number_input('Runs required to win' if innings.startswith('Second') else 'Runs scored so far', 0, 400, 40 if innings.startswith('Second') else 120)

        n = st.select_slider('Simulations', options=[500, 1000, 2000, 5000, 10000], value=1000)
        submit = st.form_submit_button('Calculate probability', use_container_width=True)

    if submit:
        with st.spinner('Computing exact absorbing Markov probability & Monte Carlo trials…'):
            if innings.startswith('Second'):
                if chasing_team and chasing_team != 'Unspecified' and chasing_team == bowling_team:
                    st.error('Please choose two different teams.'); st.stop()

                state = State(int(balls), int(wickets), int(runs))
                p = win_model.probability(state)

                if adjustment:
                    ctx = adjustment['context'].feature(
                        None if chasing_team == 'Unspecified' else chasing_team,
                        None if bowling_team == 'Unspecified' else bowling_team,
                        venue
                    )
                    adjusted = adjustment['calibrator'].adjust(p, ctx, terminal=state.terminal)
                    st.metric('Chasing team: calibrated probability with Bayesian context', f'{100*adjusted:.1f}%')
                    st.caption('The adjustment uses historical team/venue estimates and monotone logistic calibration.')

                result = simulate_chase(model, state, n)
                st.metric('Chasing team: raw Markov win probability', f'{100*p:.1f}%')
                st.metric('Chasing team: raw Monte Carlo estimate', f'{100*result["win_probability"]:.1f}%')

                # 2D Sensitivity Heatmap
                st.markdown("### 🗺️ 2D Sensitivity Matrix: Balls Remaining vs Runs Required")
                balls_grid = list(range(6, 61, 6))
                runs_grid = list(range(10, 101, 10))
                lookup_data = win_model.generate_lookup_table(balls_values=balls_grid, runs_values=runs_grid, wickets_lost=int(wickets))
                heatmap_fig = px.imshow(
                    lookup_data['grid'] * 100,
                    x=lookup_data['balls'],
                    y=lookup_data['runs'],
                    labels=dict(x="Balls Remaining", y="Runs Required", color="Win Prob (%)"),
                    color_continuous_scale="Viridis",
                    title=f"Win Probability Heatmap ({wickets} Wickets Lost)"
                )
                heatmap_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
                st.plotly_chart(heatmap_fig, use_container_width=True)

            else:
                result = simulate_first_innings(model, int(balls), int(wickets), int(runs), n)
                st.caption('First-innings simulation uses the pooled transition model to project final score and opponent chase.')
                st.metric('Batting-first team: Monte Carlo win estimate', f'{100*result["win_probability"]:.1f}%')
                st.write(f'Average simulated final score: {result["mean_final_score"]:.1f}')

            st.write(f'Monte Carlo standard error: {100*result["standard_error"]:.2f} percentage points')
            st.caption('Monte Carlo standard error measures simulation noise. In nonterminal states, raw Markov and Monte Carlo align closely.')


# =========================================================================
# MODULE 4: PROBABILITY WORM CHART
# =========================================================================
elif nav_section == "📈 Probability Worm Chart":
    st.markdown("## 📈 Real-Match Win-Probability Worm Chart")
    st.markdown("Visualize full ball-by-ball chase win trajectories from real held-out IPL matches.")

    if not matches_list.empty:
        selected_match_label = st.selectbox("Select Historical Match to Inspect", matches_list['label'].tolist(), index=0)
        match_info = matches_list[matches_list['label'] == selected_match_label].iloc[0]
        match_mid = match_info['match_id']

        match_deliveries = full_df[full_df['match_id'] == match_mid]
        traj_df = analyze_chase_trajectory(match_deliveries, win_model, calibrator, context)

        if not traj_df.empty:
            st.plotly_chart(create_worm_chart(traj_df, match_info['team2'], match_info['team1']), use_container_width=True)

            # Cumulative score worm
            st.markdown("### 📊 Runs & Score Trajectory")
            score_fig = go.Figure()
            score_fig.add_trace(go.Scatter(
                x=traj_df['delivery_index'], y=traj_df['cumulative_runs'],
                name=f"{match_info['team2']} Chase Score",
                line=dict(color='#00F5D4', width=3)
            ))
            score_fig.add_hline(y=match_info['target'], line_dash="dash", line_color="#FFD166", annotation_text="Target Line")
            score_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13, 27, 42, 0.4)",
                font=dict(color="#FFFFFF"), xaxis_title="Deliveries Bowled", yaxis_title="Score"
            )
            st.plotly_chart(score_fig, use_container_width=True)
    else:
        st.info("Historical match data not found in workspace.")


# =========================================================================
# MODULE 5: DELIVERY LEVERAGE ANALYSIS
# =========================================================================
elif nav_section == "⚖️ Delivery Leverage Analysis":
    st.markdown("## ⚖️ Delivery Leverage Index (LI) & Probability Swings")
    st.markdown("Quantify how critical the next delivery is to the match outcome across all possible events.")

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        lev_balls = st.slider("Balls Remaining", 1, 120, 24)
    with lc2:
        lev_wickets = st.slider("Wickets Lost", 0, 9, 4)
    with lc3:
        lev_runs = st.slider("Runs Required", 1, 100, 32)

    s = State(lev_balls, lev_wickets, lev_runs)
    lev_data = win_model.leverage_index(s)
    swings = lev_data['swings']

    st.markdown(f"""
    <div class="scoreboard-banner">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="metric-pill pill-gold">LEVERAGE INDEX: {lev_data['leverage_index']:.2f}x</span>
                <span class="metric-pill pill-info">MAX SWING SPREAD: ±{lev_data['max_swing']*100:.1f}%</span>
            </div>
            <div style="color: #90A4AE;">
                Current Win Prob: <b>{win_model.probability(s)*100:.1f}%</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if swings:
        labels = []
        deltas = []
        colors = []
        for item in swings:
            o = item['outcome']
            name = f"{o.runs}R" if o.wicket == 0 else f"WICKET (+{o.runs}R)"
            if not o.legal: name += " [EXTRA]"
            labels.append(name)
            deltas.append(item['swing'] * 100)
            colors.append('#00E676' if item['swing'] >= 0 else '#FF3366')

        fig = go.Figure(go.Bar(
            x=labels, y=deltas,
            marker_color=colors,
            text=[f"{d:+.1f}%" for d in deltas],
            textposition='auto'
        ))
        fig.update_layout(
            title="<b>Win Probability Added (WPA / Swing) by Outcome on Next Ball</b>",
            xaxis_title="Potential Delivery Outcome",
            yaxis_title="Probability Swing (Percentage Points)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13, 27, 42, 0.4)",
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# MODULE 6: MONTE CARLO SIMULATION
# =========================================================================
elif nav_section == "🎲 Monte Carlo Simulation":
    st.markdown("## 🎲 10,000-Trial Monte Carlo Simulation Laboratory")
    st.markdown("Test stochastic convergence, simulate chase trajectories, and examine empirical score distributions.")

    mc_mode = st.radio("Simulation Objective", ["Chase Target Simulation", "First Innings Setting Simulation"], horizontal=True)

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("#### Match Configuration")
        mc_trials = st.slider("Number of Independent Trials (N)", 1000, 20000, 10000, step=1000)
        mc_seed = st.number_input("Random Generator Seed", 0, 999999, 42)
        mc_balls = st.slider("Balls Left", 0, 120, 36)
        mc_wickets = st.slider("Wickets Down", 0, 9, 3)
        mc_runs = st.number_input("Runs (Required / Scored)", 0, 250, 48)
        run_mc = st.button("🚀 Execute Monte Carlo Engine", use_container_width=True)

    with col_m2:
        if run_mc:
            with st.spinner(f"Simulating {mc_trials:,} stochastic matches…"):
                if mc_mode.startswith("Chase"):
                    s = State(mc_balls, mc_wickets, mc_runs)
                    res = simulate_chase(model, s, n=mc_trials, seed=mc_seed)
                    p_exact = win_model.probability(s)

                    st.markdown(f"""
                    <div class="broadcast-card">
                        <div style="font-size: 1.1rem; color: #90A4AE;">SIMULATION RESULTS ({mc_trials:,} TRIALS)</div>
                        <div style="font-size: 2.4rem; font-weight: 700; color: #00E676;">{res['win_probability']*100:.2f}% Win Probability</div>
                        <div style="color: #E0E1DD;">
                            Standard Error: <b>±{res['standard_error']*100:.2f}%</b> | 
                            95% CI: <b>[{res['conf_interval_95'][0]*100:.2f}%, {res['conf_interval_95'][1]*100:.2f}%]</b><br>
                            Analytical Exact DP Probability: <b style="color: #00F5D4;">{p_exact*100:.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    res = simulate_first_innings(model, mc_balls, mc_wickets, mc_runs, n=mc_trials, seed=mc_seed)
                    st.markdown(f"""
                    <div class="broadcast-card">
                        <div style="font-size: 1.1rem; color: #90A4AE;">FIRST INNINGS PROJECTION ({mc_trials:,} TRIALS)</div>
                        <div style="font-size: 2.4rem; font-weight: 700; color: #3A86FF;">{res['win_probability']*100:.2f}% Win Probability</div>
                        <div style="color: #E0E1DD;">
                            Expected Final Total: <b>{res['mean_final_score']:.1f} runs</b> (Std Dev: ±{res['std_final_score']:.1f})<br>
                            Median Score: <b>{res['score_quantiles']['median']:.0f}</b> | 
                            80% Range: <b>{res['score_quantiles']['p10']:.0f} – {res['score_quantiles']['p90']:.0f}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Benchmark consistency table
    st.markdown("### 🔬 Analytical Backward Induction DP vs Monte Carlo Formal Verification")
    if 'mc_vs_markov' in reports:
        mc_comp = reports['mc_vs_markov']
        df_bench = pd.DataFrame(mc_comp['comparisons'])
        df_bench['Exact DP'] = (df_bench['exact_markov'] * 100).map("{:.2f}%".format)
        df_bench['Monte Carlo (10k)'] = (df_bench['monte_carlo'] * 100).map("{:.2f}%".format)
        df_bench['SE'] = (df_bench['standard_error'] * 100).map("{:.2f}%".format)
        df_bench['95% CI'] = df_bench.apply(lambda r: f"[{r['ci_lower_95']*100:.1f}%, {r['ci_upper_95']*100:.1f}%]", axis=1)
        df_bench['Diff'] = (df_bench['abs_difference'] * 100).map("{:.2f}%".format)
        df_bench['Verified within CI'] = df_bench['consistent_95_ci'].map(lambda x: "✅ PASS" if x else "⚠️")
        st.dataframe(df_bench[['balls_left', 'wickets_lost', 'runs_required', 'Exact DP', 'Monte Carlo (10k)', 'SE', '95% CI', 'Diff', 'Verified within CI']], use_container_width=True)


# =========================================================================
# MODULE 7: BAYESIAN VENUE ANALYSIS
# =========================================================================
elif nav_section == "🏟️ Bayesian Venue Analysis":
    st.markdown("## 🏟️ Bayesian Venue Chasing Probabilities")
    st.markdown("Empirical Bayesian Beta-Binomial updating to model venue chasing bias and ground dynamics.")

    if 'venue_chase_stats' in reports:
        venues_data = reports['venue_chase_stats']
        v_df = pd.DataFrame(venues_data)

        vc1, vc2 = st.columns([1, 1.5])
        with vc1:
            sel_venue = st.selectbox("Select Ground to Examine", v_df['venue'].tolist(), index=0)
            v_match = v_df[v_df['venue'] == sel_venue].iloc[0]

            n = int(v_match['matches'])
            k = int(v_match['chase_wins'])
            post_mean = float(v_match['posterior_mean'])
            ci = v_match['credible_interval_95']

            st.markdown(f"""
            <div class="broadcast-card">
                <h3 style="color: #00E676; margin: 0;">{sel_venue}</h3>
                <div style="font-size: 2.2rem; font-weight: 700; color: #FFFFFF; margin: 10px 0;">{post_mean*100:.1f}%</div>
                <div style="color: #90A4AE; font-size: 0.9rem;">
                    Matches Analyzed: <b>{n}</b> | Chases Won: <b>{k}</b> ({k/n*100:.1f}%)<br>
                    95% Bayesian Credible Interval: <b style="color: #FFD166;">[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]</b><br>
                    Prior: <b>Beta(10, 10)</b> (Shrinkage toward league mean)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with vc2:
            # Plot Beta Posterior distribution curve
            x = np.linspace(0.2, 0.8, 200)
            prior_y = beta_dist.pdf(x, 10, 10)
            post_y = beta_dist.pdf(x, 10 + k, 10 + n - k)

            beta_fig = go.Figure()
            beta_fig.add_trace(go.Scatter(x=x, y=prior_y, mode='lines', name='Prior Beta(10,10)', line=dict(dash='dash', color='#90A4AE')))
            beta_fig.add_trace(go.Scatter(x=x, y=post_y, mode='lines', name='Posterior Distribution', line=dict(color='#00E676', width=3), fill='tozeroy', fillcolor='rgba(0,230,118,0.15)'))
            beta_fig.add_vline(x=post_mean, line_color="#FFD166", line_dash="dot", annotation_text="Posterior Mean")
            beta_fig.update_layout(
                title=f"<b>Beta Posterior Density for Chase Win Probability</b>",
                xaxis_title="Chase Win Probability θ",
                yaxis_title="Probability Density",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13, 27, 42, 0.4)",
                font=dict(color="#FFFFFF")
            )
            st.plotly_chart(beta_fig, use_container_width=True)

        # Forest plot of all venues with >= 15 matches
        st.markdown("### 🌲 League-Wide Venue Chase Bias (Venues with ≥ 15 Matches)")
        top_v = v_df[v_df['matches'] >= 15].sort_values('posterior_mean', ascending=True)
        forest_fig = go.Figure()
        forest_fig.add_trace(go.Scatter(
            x=top_v['posterior_mean'] * 100,
            y=top_v['venue'],
            mode='markers',
            error_x=dict(
                type='data',
                symmetric=False,
                array=(top_v['ci_upper'] - top_v['posterior_mean']) * 100,
                arrayminus=(top_v['posterior_mean'] - top_v['ci_lower']) * 100,
                color='#00E676'
            ),
            marker=dict(color='#00F5D4', size=10),
            name="95% Credible Interval"
        ))
        forest_fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50% Neutral")
        forest_fig.update_layout(
            xaxis_title="Chasing Win Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13, 27, 42, 0.4)",
            font=dict(color="#FFFFFF"), height=550
        )
        st.plotly_chart(forest_fig, use_container_width=True)


# =========================================================================
# MODULE 8: TOSS HYPOTHESIS TESTING
# =========================================================================
elif nav_section == "🪙 Toss Hypothesis Testing":
    st.markdown("## 🪙 Toss Advantage Statistical Hypothesis Testing")
    st.markdown("Rigorous hypothesis tests (Proportion Z-Test, Exact Binomial Test, and Chi-Square Independence Test) to determine if winning the coin toss provides an unfair advantage.")

    toss_scope = st.radio("Dataset Scope", ["2024 Holdout Season (70 matches)", "All Historical IPL Matches (1,034 matches)"], horizontal=True)

    if 'toss_analysis' in reports:
        t_data = reports['toss_analysis']['test_season'] if toss_scope.startswith("2024") else reports['toss_analysis']['full_dataset']

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(f"""
            <div class="broadcast-card" style="text-align: center;">
                <div style="color: #90A4AE; font-size: 0.85rem;">TOSS WINNER WIN RATE</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #FFD166;">{t_data['observed_proportion']*100:.1f}%</div>
                <div style="color: #E0E1DD;">{t_data['wins']} Wins / {t_data['matches']} Matches</div>
            </div>
            """, unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="broadcast-card" style="text-align: center;">
                <div style="color: #90A4AE; font-size: 0.85rem;">PROPORTION Z-TEST</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #3A86FF;">Z = {t_data['z_statistic']:.3f}</div>
                <div style="color: #E0E1DD;">p-value: <b>{t_data['z_test_p_value']:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""
            <div class="broadcast-card" style="text-align: center;">
                <div style="color: #90A4AE; font-size: 0.85rem;">STATISTICAL SIGNIFICANCE</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {'#FF3366' if t_data['is_statistically_significant'] else '#00E676'};">
                    {'REJECT H0' if t_data['is_statistically_significant'] else 'FAIL TO REJECT H0'}
                </div>
                <div style="color: #E0E1DD;">Exact Binomial p = {t_data['p_value']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Chi-Square Analysis
        if 'toss_decision_chi_square' in t_data:
            chi_data = t_data['toss_decision_chi_square']
            st.markdown("### 📋 Chi-Square Test of Independence: Toss Decision (Bat vs Field)")
            st.markdown(f"""
            Contingency test evaluates whether choosing to bat first vs field first is statistically independent of winning the match.
            - **Chi-Square Statistic (χ²):** `{chi_data['chi2_statistic']:.4f}` (df = 1)
            - **p-value:** `{chi_data['p_value']:.4f}` ({'Significant at α=0.05' if chi_data['is_statistically_significant'] else 'Not Significant at α=0.05'})
            - **Odds Ratio (Fielding vs Batting):** `{chi_data['odds_ratio_field_vs_bat']:.2f}x`
            - **Cochran Condition Met:** `{'Yes (Min expected cell >= 5)' if chi_data['cochran_condition_met'] else 'No'}`
            """)


# =========================================================================
# MODULE 9: MODEL VALIDATION AND CALIBRATION
# =========================================================================
elif nav_section == "📊 Model Validation and Calibration":
    st.markdown("## 📊 Model Validation & Decile Calibration")
    st.markdown("Formal empirical evaluation of probability forecasts against ground-truth holdout results.")

    if 'adjusted_metrics' in reports:
        m_rep = reports['adjusted_metrics']
        raw = m_rep['raw_markov']
        adj = m_rep['calibrated_hybrid']
        base = m_rep['constant_50_percent_baseline']

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown(f"""
            <div class="broadcast-card">
                <div style="color: #90A4AE; font-size: 0.85rem;">BRIER SCORE (LOWER IS BETTER)</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #00E676;">{adj['brier_score']:.4f}</div>
                <div style="color: #E0E1DD;">Raw Markov: {raw['brier_score']:.4f}<br>Baseline: {base['brier_score']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with vc2:
            st.markdown(f"""
            <div class="broadcast-card">
                <div style="color: #90A4AE; font-size: 0.85rem;">LOG LOSS / CROSS-ENTROPY</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #00F5D4;">{adj['log_loss']:.4f}</div>
                <div style="color: #E0E1DD;">Raw Markov: {raw['log_loss']:.4f}<br>Baseline: {base['log_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with vc3:
            st.markdown(f"""
            <div class="broadcast-card">
                <div style="color: #90A4AE; font-size: 0.85rem;">EXPECTED CALIBRATION ERROR (ECE)</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #FFD166;">{adj['expected_calibration_error']:.4f}</div>
                <div style="color: #E0E1DD;">Accuracy: {adj['accuracy']*100:.1f}%<br>Brier Skill Score: +{adj['brier_skill_score']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Reliability diagram
    if 'adjusted_calibration' in reports:
        cal_df = reports['adjusted_calibration']
        raw_cal = reports.get('calibration', cal_df)

        rel_fig = go.Figure()
        rel_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#90A4AE'), name='Perfect Calibration (y = x)'))
        rel_fig.add_trace(go.Scatter(
            x=raw_cal['mean_prediction'], y=raw_cal['observed_win_rate'],
            mode='lines+markers', name='Raw Markov Model', line=dict(color='#FF5722', dash='dot')
        ))
        rel_fig.add_trace(go.Scatter(
            x=cal_df['mean_prediction'], y=cal_df['observed_win_rate'],
            mode='lines+markers', name='Calibrated Hybrid Model', line=dict(color='#00E676', width=3)
        ))
        rel_fig.update_layout(
            title="<b>Decile Reliability Diagram (Holdout Season Evaluation)</b>",
            xaxis_title="Mean Predicted Win Probability",
            yaxis_title="Empirical Observed Win Proportion",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13, 27, 42, 0.4)",
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(rel_fig, use_container_width=True)

        with st.expander("📋 View Decile Reliability Table"):
            st.dataframe(cal_df, use_container_width=True)


# =========================================================================
# MODULE 10: HISTORICAL MATCH ANALYSIS
# =========================================================================
elif nav_section == "📜 Historical Match Analysis":
    st.markdown("## 📜 Historical Match Database & Turning Points Analysis")
    st.markdown("Review any match from the 2008–2024 IPL dataset, replay ball-by-ball win probabilities, and inspect top momentum swings.")

    if not matches_list.empty:
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            seasons_available = sorted(matches_list['season'].unique(), reverse=True)
            sel_s = st.selectbox("Filter Season", seasons_available, index=0)
            sub_matches = matches_list[matches_list['season'] == sel_s]
            sel_match_label = st.selectbox("Select Match", sub_matches['label'].tolist(), index=0)
            m_row = sub_matches[sub_matches['label'] == sel_match_label].iloc[0]

        with col_s2:
            st.markdown(f"""
            <div class="broadcast-card">
                <h3 style="color: #00E676; margin: 0;">{m_row['team1']} vs {m_row['team2']}</h3>
                <div style="color: #E0E1DD; margin-top: 6px;">
                    Date: <b>{m_row['date']}</b> | Venue: <b>{m_row['venue']}</b><br>
                    Toss: <b>{m_row['toss_winner']}</b> elected to <b>{m_row['toss_decision']}</b><br>
                    Result: <b style="color: #FFD166;">{m_row['winner']} WON</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Analyze chase trajectory
        match_balls = full_df[full_df['match_id'] == m_row['match_id']]
        traj = analyze_chase_trajectory(match_balls, win_model, calibrator, context)

        if not traj.empty:
            st.plotly_chart(create_worm_chart(traj, m_row['team2'], m_row['team1']), use_container_width=True)

            # Top 5 Pivotal Moments
            st.markdown("### ⚡ Top 5 Match Turning Points (Highest Win Probability Added)")
            top_swings = traj.reindex(traj['wpa_calibrated'].abs().sort_values(ascending=False).index).head(5)
            for _, sw in top_swings.iterrows():
                sign = "+" if sw['wpa_calibrated'] > 0 else ""
                swing_col = "#00E676" if sw['wpa_calibrated'] > 0 else "#FF3366"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-left: 4px solid {swing_col}; padding: 10px 16px; margin-bottom: 8px; border-radius: 6px;">
                    <b>Over {sw['over_label']}</b> — <b>{sw['event']}</b> | 
                    Score: <b>{sw['cumulative_runs']}/{sw['cumulative_wickets']}</b> (Req: {sw['runs_required']} off {sw['balls_left']} balls)<br>
                    Win Prob: {sw['calibrated_probability']*100:.1f}% → {sw['next_calibrated_probability']*100:.1f}% 
                    (<b style="color: {swing_col};">{sign}{sw['wpa_calibrated']*100:.1f}% WPA</b>)
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No historical matches found in dataset.")
