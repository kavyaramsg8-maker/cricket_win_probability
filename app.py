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
# PAGE CONFIGURATION & INSTITUTIONAL CSS
# ==========================================
st.set_page_config(
    page_title="T20 Cricket Win-Probability Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #121417;
    --bg-card: #1A1E24;
    --bg-card-hover: #222730;
    --border-color: rgba(226, 232, 240, 0.08);
    --border-accent: rgba(5, 150, 105, 0.35);
    
    /* Elegant Editorial Palette: Forest Green, Editorial Dusty Pink & Slate Gray */
    --emerald-deep: #059669;
    --emerald-light: #10B981;
    --emerald-soft: #34D399;
    --pink-dusty: #F43F5E;
    --pink-rose: #FB7185;
    --pink-blush: #FDA4AF;
    --pink-glow: rgba(244, 63, 94, 0.12);
    
    --slate-50: #F8FAFC;
    --slate-200: #E2E8F0;
    --slate-400: #94A3B8;
    --slate-600: #475569;
    --slate-800: #1E2228;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--slate-200);
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.02em;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace;
}

/* Subtle Geometric Pattern Canvas */
.stApp {
    background-color: var(--bg-main);
    background-image: 
        radial-gradient(circle at 12% 18%, rgba(5, 150, 105, 0.04) 0%, transparent 28%),
        radial-gradient(circle at 88% 82%, rgba(244, 63, 94, 0.04) 0%, transparent 26%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.01) 25%, transparent 25%),
        linear-gradient(225deg, rgba(255, 255, 255, 0.01) 25%, transparent 25%);
    background-size: 100% 100%, 100% 100%, 60px 60px, 60px 60px;
}

/* Brand Title Styling */
.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.18rem;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #10B981 0%, #34D399 45%, #FDA4AF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* Sidebar styling with geometric edge */
[data-testid="stSidebar"] {
    background-color: #16191E;
    border-right: 1px solid rgba(226, 232, 240, 0.07);
}
[data-testid="stSidebar"] * {
    color: var(--slate-200);
}

/* Organic Asymmetric Geometric Cards */
.cric-card {
    background: linear-gradient(145deg, #1A1E24 0%, #171B20 100%);
    border: 1px solid var(--border-color);
    border-top: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 20px 6px 20px 6px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.cric-card::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, transparent 50%, rgba(244, 63, 94, 0.09) 50%);
    pointer-events: none;
}

.cric-card:hover {
    border-color: rgba(52, 211, 153, 0.35);
    transform: translateY(-1px);
}

/* Scoreboard Header: Modern Chamfered Architectural Silhouette */
.scoreboard-banner {
    background: linear-gradient(135deg, #181E24 0%, #151A20 60%, #1C1A22 100%);
    border: 1px solid rgba(226, 232, 240, 0.09);
    border-left: 5px solid #10B981;
    border-right: 5px solid #F43F5E;
    border-radius: 16px 4px 16px 4px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
    position: relative;
}

/* Metric Display Unit with Asymmetric Contour */
.metric-box {
    background: linear-gradient(140deg, #1A1E24 0%, #181C21 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px 4px 16px 4px;
    padding: 16px 20px;
    text-align: center;
    height: 100%;
    position: relative;
}

.metric-box-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--slate-400);
    margin-bottom: 4px;
}

.metric-box-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    margin-bottom: 4px;
}

.metric-box-sub {
    font-size: 0.8rem;
    color: #94A3B8;
}

/* Geometric Hex/Capsule Badges */
.cric-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 12px 2px 12px 2px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 6px;
}

.badge-live { background: rgba(244, 63, 94, 0.14); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.35); }
.badge-chase { background: rgba(5, 150, 105, 0.15); color: #6EE7B7; border: 1px solid rgba(5, 150, 105, 0.4); }
.badge-neutral { background: rgba(148, 163, 184, 0.1); color: #E2E8F0; border: 1px solid rgba(148, 163, 184, 0.22); }
.badge-gold { background: rgba(251, 113, 133, 0.12); color: #FDA4AF; border: 1px solid rgba(251, 113, 133, 0.3); }

/* Clean UI Expander */
div[data-testid="stExpander"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px 3px 12px 3px;
}

/* Polished Streamlit Buttons */
.stButton > button {
    border-radius: 12px 3px 12px 3px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border: 1px solid rgba(5, 150, 105, 0.4) !important;
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #FFFFFF !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    border-color: #34D399 !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Unified Editorial Dark Theme (Green, Pink, Slate)
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26, 30, 36, 0.5)",
    font=dict(color="#E2E8F0", family='Plus Jakarta Sans, -apple-system, sans-serif'),
    xaxis=dict(gridcolor="rgba(226, 232, 240, 0.06)", zerolinecolor="rgba(226, 232, 240, 0.12)"),
    yaxis=dict(gridcolor="rgba(226, 232, 240, 0.06)", zerolinecolor="rgba(226, 232, 240, 0.12)"),
    hoverlabel=dict(bgcolor="#222730", font_color="#F8FAFC", font_family="Plus Jakarta Sans", bordercolor="rgba(226, 232, 240, 0.2)")
)

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

    # Crucial: Use processed train + test datasets as full_df because they contain
    # precomputed 'target', 'balls_left', 'runs_required', 'wickets_lost', 'runs_scored'.
    if train_df is not None and test_df is not None:
        full_df = pd.concat([train_df, test_df], ignore_index=True)
    elif test_df is not None:
        full_df = test_df.copy()
    elif train_df is not None:
        full_df = train_df.copy()
    elif raw_path.exists():
        raw_df = pd.read_csv(raw_path, low_memory=False)
        from cricket.preprocessing import prepare
        try:
            full_df = prepare(raw_df)
        except Exception:
            full_df = raw_df
    else:
        full_df = None

    if full_df is not None:
        full_df['match_id'] = full_df['match_id'].astype(str)
        matches_list = get_match_list(full_df)
    else:
        matches_list = pd.DataFrame()

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
# SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# ==========================================
st.sidebar.markdown("""
<div style="padding: 12px 0 16px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.15); margin-bottom: 14px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.6rem;">🏏</span>
        <div>
            <div class="brand-title">CRIC-PROB ANALYTICS</div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">T20 Absorbing Markov Win Engine</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_section = st.sidebar.selectbox(
    "Analytics Navigation",
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
st.sidebar.markdown("<div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.06em; margin-bottom: 8px;'>Engine Telemetry</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Model Core:** Absorbing Markov Chain")
st.sidebar.markdown(f"**Smoothing:** Laplace (α = 0.5)")
st.sidebar.markdown(f"**Context Layer:** Bayesian Beta-Binomial")
st.sidebar.markdown(f"**Calibration:** Monotone Platt Scaling")
if meta:
    st.sidebar.markdown(f"**Training Epoch:** {min(meta.get('train_seasons', [2008]))}–{max(meta.get('train_seasons', [2023]))}")
    st.sidebar.markdown(f"**Holdout Season:** {meta.get('test_seasons', [2024])[0]}")

# ==========================================
# HELPER FUNCTIONS: PLOTLY CHARTS
# ==========================================
def create_probability_gauge(prob, title="Win Probability", team_name="Chasing Team", raw_prob=None):
    """Create an elegant institutional radial probability indicator."""
    fig = go.Figure()
    pct = prob * 100
    
    if pct >= 60:
        bar_color = "#10B981"  # Forest/Emerald
    elif pct >= 45:
        bar_color = "#34D399"  # Soft mint green
    elif pct >= 30:
        bar_color = "#FDA4AF"  # Soft dusty pink
    else:
        bar_color = "#F43F5E"  # Rich pink-rose

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 42, 'family': 'Space Grotesk, sans-serif', 'color': '#F8FAFC'}},
        title={'text': f"<b>{team_name}</b><br><span style='font-size:0.8rem;color:#94A3B8'>{title}</span>",
               'font': {'size': 16, 'family': 'Plus Jakarta Sans, sans-serif', 'color': '#E2E8F0'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(30, 34, 40, 0.7)",
            'borderwidth': 1,
            'bordercolor': "rgba(226, 232, 240, 0.12)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(244, 63, 94, 0.12)"},
                {'range': [30, 50], 'color': "rgba(253, 164, 175, 0.08)"},
                {'range': [50, 70], 'color': "rgba(52, 211, 153, 0.08)"},
                {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.12)"}
            ],
            'threshold': {
                'line': {'color': "#F8FAFC", 'width': 3},
                'thickness': 0.75,
                'value': pct
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#F8FAFC", 'family': 'Plus Jakarta Sans, sans-serif'}
    )
    return fig

def create_worm_chart(traj_df, chasing_team="Chaser", bowling_team="Defenders"):
    """Create an editorial Plotly worm chart with critical wickets and boundaries."""
    fig = go.Figure()
    if traj_df.empty:
        return fig

    # 50% equilibrium baseline
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)",
                  annotation_text="50% Equilibrium", annotation_position="bottom right",
                  annotation_font=dict(color="#94A3B8", size=10))

    # Calibrated probability trajectory (Green)
    if 'calibrated_probability' in traj_df.columns:
        fig.add_trace(go.Scatter(
            x=traj_df['delivery_index'],
            y=traj_df['calibrated_probability'] * 100,
            mode='lines',
            name=f'{chasing_team} (Calibrated Hybrid)',
            line=dict(color='#10B981', width=3, shape='spline'),
            hovertemplate="Ball %{x} (Over %{customdata[0]})<br>Score: %{customdata[1]}/%{customdata[2]}<br>Req: %{customdata[3]} from %{customdata[4]} balls<br>Win Prob: <b>%{y:.1f}%</b><br>Event: %{customdata[5]}<extra></extra>",
            customdata=np.column_stack([
                traj_df['over_label'], traj_df['cumulative_runs'], traj_df['cumulative_wickets'],
                traj_df['runs_required'], traj_df['balls_left'], traj_df['event']
            ])
        ))

    # Raw Markov baseline (Pink blush)
    if 'raw_probability' in traj_df.columns:
        fig.add_trace(go.Scatter(
            x=traj_df['delivery_index'],
            y=traj_df['raw_probability'] * 100,
            mode='lines',
            name=f'{chasing_team} (Raw Markov)',
            line=dict(color='#FDA4AF', width=1.8, dash='dot'),
            hoverinfo='skip'
        ))

    # Markers for Wickets (Pink Rose)
    wickets = traj_df[traj_df['wicket'] > 0]
    if not wickets.empty:
        w_prob = wickets['calibrated_probability'] if 'calibrated_probability' in wickets else wickets['raw_probability']
        fig.add_trace(go.Scatter(
            x=wickets['delivery_index'],
            y=w_prob * 100,
            mode='markers',
            name='Wicket Fallen',
            marker=dict(symbol='x', size=9, color='#F43F5E', line=dict(width=2, color='#FFFFFF')),
            hovertemplate="💥 WICKET! Ball %{x}<br>Score: %{customdata[0]}/%{customdata[1]}<extra></extra>",
            customdata=np.column_stack([wickets['cumulative_runs'], wickets['cumulative_wickets']])
        ))

    # Markers for Sixes (Bright Mint Emerald)
    sixes = traj_df[traj_df['runs_scored_on_ball'] == 6]
    if not sixes.empty:
        s_prob = sixes['calibrated_probability'] if 'calibrated_probability' in sixes else sixes['raw_probability']
        fig.add_trace(go.Scatter(
            x=sixes['delivery_index'],
            y=s_prob * 100,
            mode='markers',
            name='Six (6)',
            marker=dict(symbol='diamond', size=8, color='#34D399', line=dict(width=1, color='#121417')),
            hovertemplate="🚀 SIX! Ball %{x}<extra></extra>"
        ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=f"<b>Chase Win Probability Trajectory</b>",
        xaxis_title="Deliveries Bowled in Chase",
        yaxis_title="Win Probability (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(range=[-2, 102], ticksuffix="%")
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

    st.markdown(f"""
    <div class="scoreboard-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span class="cric-badge badge-live">● LIVE MATCH</span>
                    <span class="cric-badge badge-chase">2ND INNINGS CHASE</span>
                    <span class="cric-badge badge-neutral">{venue}</span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC;">
                    {chaser} <span style="color: #64748B; font-weight: 400; font-size: 1rem;">vs</span> {bowler}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Target Total</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #FBBF24;">{target} <span style="font-size: 0.9rem; font-weight: 500; color: #94A3B8;">runs</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-label">Current Score</div>
            <div class="metric-box-value" style="color: #34D399;">{score_so_far}/{w}</div>
            <div class="metric-box-sub">{overs_bowled:.1f} / 20.0 Overs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-label">Runs Required</div>
            <div class="metric-box-value" style="color: #FDA4AF;">{r}</div>
            <div class="metric-box-sub">From {b} Balls Remaining</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        rrr_col = "#F43F5E" if rrr > 12 else ("#FB7185" if rrr > 9 else "#34D399")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-label">Required Run Rate</div>
            <div class="metric-box-value" style="color: {rrr_col};">{rrr:.2f}</div>
            <div class="metric-box-sub">Current RR: {crr:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-label">Wickets in Hand</div>
            <div class="metric-box-value" style="color: #E2E8F0;">{10 - w}</div>
            <div class="metric-box-sub">{w} Dismissals</div>
        </div>
        """, unsafe_allow_html=True)

    # Gauges & Momentum
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns([1.2, 1])
    with g_col1:
        st.plotly_chart(
            create_probability_gauge(p_cal, title="Calibrated Win Probability", team_name=chaser, raw_prob=p_raw),
            use_container_width=True
        )
    with g_col2:
        lev = win_model.leverage_index(cur_state)
        li_val = lev['leverage_index']
        li_badge = "CRITICAL LEVERAGE" if li_val > 2.0 else ("HIGH LEVERAGE" if li_val > 1.3 else ("MODERATE" if li_val > 0.8 else "LOW LEVERAGE"))
        li_color = "#F43F5E" if li_val > 2.0 else ("#FB7185" if li_val > 1.3 else "#10B981")

        st.markdown(f"""
        <div class="cric-card" style="height: 250px; display: flex; flex-direction: column; justify-content: center;">
            <div class="metric-box-label">Next Ball Leverage Index (LI)</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; color: {li_color}; margin: 4px 0;">{li_val:.2f}x</div>
            <div style="margin-bottom: 12px;"><span class="cric-badge" style="background: rgba(244,63,94,0.1); color: {li_color}; border: 1px solid {li_color};">{li_badge}</span></div>
            <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">
                Max swing potential on next ball: <b style="color: #F8FAFC;">±{lev['max_swing']*100:.1f}%</b><br>
                Expected swing magnitude: <b style="color: #F8FAFC;">{lev['expected_swing']*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Match state configuration
    with st.expander("⚙️ Adjust Match Situation Parameters", expanded=False):
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            new_chaser = st.selectbox("Chasing Team", team_options[1:], index=team_options[1:].index(chaser) if chaser in team_options[1:] else 0)
        with qc2:
            new_bowler = st.selectbox("Bowling Team", team_options[1:], index=team_options[1:].index(bowler) if bowler in team_options[1:] else 1)
        with qc3:
            new_venue = st.selectbox("Ground Venue", venue_options[1:], index=venue_options[1:].index(venue) if venue in venue_options[1:] else 0)
        with qc4:
            new_target = st.number_input("Target Total", 50, 300, target)

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
    st.caption("Simulate individual deliveries or full overs to observe instant win probability adjustments and momentum swings.")

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
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">{chaser} vs {bowler}</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                    Target: <b style="color: #FBBF24;">{target}</b> | Need <b>{r} runs</b> from <b>{b} legal balls</b> | <b>{10 - w} wickets</b> in hand | Venue: <b>{venue}</b>
                </div>
            </div>
            <div>
                <span class="cric-badge badge-neutral">Live Simulation</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Outcome action buttons
    st.markdown("#### 🎯 Play Next Delivery Outcome")
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

    # Simulation control row
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        if st.button("🎲 Auto-Simulate Ball", use_container_width=True, disabled=(cur_state.terminal is not None)):
            dist = win_model.transitions.distribution(cur_state)
            probs = [p for _, p in dist]
            chosen_outcome = dist[np.random.choice(len(dist), p=probs)][0]
            delivery_played = chosen_outcome

    with sim_c2:
        if st.button("⚡ Simulate Full Over (6 Balls)", use_container_width=True, disabled=(cur_state.terminal is not None)):
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
        if st.button("↩️ Undo Last Delivery", use_container_width=True, disabled=(len(st.session_state.sim_history) == 0)):
            last = st.session_state.sim_history.pop()
            st.session_state.sim_balls = last['balls_left'] + last['legal']
            st.session_state.sim_wickets = max(0, last['wickets_lost'] - last['wicket'])
            st.session_state.sim_runs = last['runs_required'] + last['runs']
            st.rerun()

    with sim_c4:
        if st.button("🔄 Reset Match State", use_container_width=True):
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

    # Display Current Status Banner if terminal
    if cur_state.terminal == 1.0:
        st.success(f"🏆 VICTORY: {chaser} achieved the target of {target} successfully!")
    elif cur_state.terminal == 0.0:
        st.error(f"❌ DEFEAT: {bowler} defended the total successfully!")

    pg_col, worm_col = st.columns([1, 2])
    with pg_col:
        st.plotly_chart(create_probability_gauge(p_cur, title="Live Win Probability", team_name=chaser), use_container_width=True)
    with worm_col:
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
            st.info("Delivery worm chart trajectory updates automatically as you play deliveries.")

    # Ball log table
    if st.session_state.sim_history:
        with st.expander("📜 Delivery Log & WPA Volatility", expanded=True):
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
    st.markdown("## 🎯 Match Situation Win-Probability Predictor")
    st.caption("Calculate Dynamic Programming Absorbing Markov win probability, Calibrated Bayesian Context Hybrid, and Monte Carlo confidence intervals.")

    innings = st.radio('Innings Selection', ['Second innings chase', 'First innings'], horizontal=True)

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

        n = st.select_slider('Monte Carlo Iterations', options=[500, 1000, 2000, 5000, 10000], value=1000)
        submit = st.form_submit_button('Calculate Probability', use_container_width=True)

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
                st.markdown("### 🗺️ Sensitivity Matrix: Balls Remaining vs Runs Required")
                balls_grid = list(range(6, 61, 6))
                runs_grid = list(range(10, 101, 10))
                lookup_data = win_model.generate_lookup_table(balls_values=balls_grid, runs_values=runs_grid, wickets_lost=int(wickets))
                heatmap_fig = px.imshow(
                    lookup_data['grid'] * 100,
                    x=lookup_data['balls'],
                    y=lookup_data['runs'],
                    labels=dict(x="Balls Remaining", y="Runs Required", color="Win Prob (%)"),
                    color_continuous_scale=[[0, "#F43F5E"], [0.4, "#1E2228"], [0.6, "#059669"], [1, "#34D399"]],
                    title=f"Win Probability Heatmap ({wickets} Wickets Lost)"
                )
                heatmap_fig.update_layout(**PLOTLY_THEME)
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
    st.caption("Inspect ball-by-ball chase probability trajectories, run-rate worms, and match turning points from historical matches.")

    if not matches_list.empty:
        # Filter controls for quick browsing
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            all_seasons = ['All Seasons'] + sorted(matches_list['season'].unique().tolist(), reverse=True)
            sel_season = st.selectbox("Filter Season", all_seasons, index=0)

        filtered_matches = matches_list if sel_season == 'All Seasons' else matches_list[matches_list['season'] == sel_season]

        with sc2:
            selected_match_label = st.selectbox("Select Match to Inspect", filtered_matches['label'].tolist(), index=0)

        match_info = filtered_matches[filtered_matches['label'] == selected_match_label].iloc[0]
        match_mid = str(match_info['match_id'])

        # Match Overview Header Card
        st.markdown(f"""
        <div class="cric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="cric-badge badge-neutral">Season {match_info['season']}</span>
                    <span class="cric-badge badge-gold">{match_info['venue']}</span>
                    <span class="cric-badge badge-live">{match_info['date']}</span>
                    <h3 style="margin: 8px 0 4px 0; color: #F8FAFC;">{match_info['team1']} vs {match_info['team2']}</h3>
                    <div style="color: #94A3B8; font-size: 0.85rem;">
                        Toss: <b>{match_info['toss_winner']}</b> elected to <b>{match_info['toss_decision']}</b>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Match Winner</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #10B981;">{match_info['winner']}</div>
                    <div style="font-size: 0.85rem; color: #E2E8F0;">
                        1st Inn: {match_info['score1']}/{match_info['wickets1']} ({match_info['overs1']} ov) | Target: <b>{match_info['target']}</b>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        match_deliveries = full_df[full_df['match_id'].astype(str) == match_mid]
        traj_df = analyze_chase_trajectory(match_deliveries, win_model, calibrator, context)

        if not traj_df.empty:
            tab_worm, tab_scoring, tab_moments = st.tabs([
                "📊 Win Probability Trajectory",
                "📈 Score & Run Rate Progression",
                "⚡ Critical Match Turning Points"
            ])

            with tab_worm:
                st.plotly_chart(create_worm_chart(traj_df, match_info['team2'], match_info['team1']), use_container_width=True)

            with tab_scoring:
                # Runs progression worm
                score_fig = go.Figure()
                score_fig.add_trace(go.Scatter(
                    x=traj_df['delivery_index'], y=traj_df['cumulative_runs'],
                    name=f"{match_info['team2']} Chase Runs",
                    line=dict(color='#10B981', width=3)
                ))
                score_fig.add_hline(
                    y=match_info['target'], line_dash="dash", line_color="#FDA4AF",
                    annotation_text=f"Target: {match_info['target']} runs", annotation_position="top left",
                    annotation_font=dict(color="#FDA4AF")
                )
                score_fig.update_layout(
                    **PLOTLY_THEME,
                    title="<b>Cumulative Chase Score Progression</b>",
                    xaxis_title="Deliveries Bowled in Chase",
                    yaxis_title="Runs Scored"
                )
                st.plotly_chart(score_fig, use_container_width=True)

            with tab_moments:
                st.markdown("#### ⚡ Top 5 Match Volatility Swings (WPA)")
                top_swings = traj_df.reindex(traj_df['wpa_calibrated'].abs().sort_values(ascending=False).index).head(5)
                for _, sw in top_swings.iterrows():
                    sign = "+" if sw['wpa_calibrated'] > 0 else ""
                    swing_col = "#10B981" if sw['wpa_calibrated'] > 0 else "#F43F5E"
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); border-left: 4px solid {swing_col}; padding: 10px 16px; margin-bottom: 8px; border-radius: 6px;">
                        <b>Over {sw['over_label']}</b> — <b>{sw['event']}</b> | 
                        Score: <b>{sw['cumulative_runs']}/{sw['cumulative_wickets']}</b> (Required: {sw['runs_required']} off {sw['balls_left']} balls)<br>
                        Win Prob: {sw['calibrated_probability']*100:.1f}% → {sw['next_calibrated_probability']*100:.1f}% 
                        (<b style="color: {swing_col};">{sign}{sw['wpa_calibrated']*100:.1f}% WPA</b>)
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("📋 Full Ball-by-Ball Chase Ledger"):
                    view_df = traj_df[['over_label', 'event', 'cumulative_runs', 'cumulative_wickets', 'runs_required', 'balls_left', 'calibrated_probability', 'wpa_calibrated']].copy()
                    view_df['calibrated_probability'] = (view_df['calibrated_probability'] * 100).map("{:.1f}%".format)
                    view_df['wpa_calibrated'] = (view_df['wpa_calibrated'] * 100).map("{:+.1f}%".format)
                    view_df.columns = ['Over', 'Event', 'Runs', 'Wickets', 'Req Runs', 'Balls Left', 'Win Prob', 'WPA']
                    st.dataframe(view_df, use_container_width=True)

        else:
            st.warning("⚠️ Chase trajectory data could not be computed for this fixture. Verify innings 2 records are present.")
    else:
        st.info("Historical match data not found in workspace.")

# =========================================================================
# MODULE 5: DELIVERY LEVERAGE ANALYSIS
# =========================================================================
elif nav_section == "⚖️ Delivery Leverage Analysis":
    st.markdown("## ⚖️ Delivery Leverage Index (LI) & Win Probability Swings")
    st.caption("Evaluate how critical an impending delivery is to match outcome across all possible scoring outcomes.")

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
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span class="cric-badge badge-gold">LEVERAGE INDEX: {lev_data['leverage_index']:.2f}x</span>
                <span class="cric-badge badge-neutral">SWING POTENTIAL: ±{lev_data['max_swing']*100:.1f}%</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.9rem;">
                Current Win Probability: <b style="color: #10B981; font-size: 1.15rem;">{win_model.probability(s)*100:.1f}%</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if swings:
        labels, deltas, colors = [], [], []
        for item in swings:
            o = item['outcome']
            name = f"{o.runs}R" if o.wicket == 0 else f"WICKET (+{o.runs}R)"
            if not o.legal: name += " [EXTRA]"
            labels.append(name)
            deltas.append(item['swing'] * 100)
            colors.append('#10B981' if item['swing'] >= 0 else '#F43F5E')

        fig = go.Figure(go.Bar(
            x=labels, y=deltas,
            marker_color=colors,
            marker_line=dict(color='rgba(255,255,255,0.08)', width=1),
            text=[f"{d:+.1f}%" for d in deltas],
            textposition='auto'
        ))
        fig.update_layout(
            **PLOTLY_THEME,
            title="<b>Win Probability Added (WPA) by Outcome on Next Ball</b>",
            xaxis_title="Potential Delivery Outcome",
            yaxis_title="Probability Swing (Percentage Points)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Integrated Feature: Match Phase Leverage Volatility Curve
    st.markdown("### 📊 Leverage Volatility Across Chase Phases")
    st.caption("How Delivery Leverage Index scales with balls remaining at standard T20 chase run-rates.")

    sample_balls = list(range(6, 121, 6))
    sample_li = [win_model.leverage_index(State(sb, min(9, int(10 * (120 - sb) / 120)), max(1, int(160 * sb / 120))))['leverage_index'] for sb in sample_balls]
    
    phase_fig = go.Figure()
    phase_fig.add_trace(go.Scatter(
        x=[120 - sb for sb in sample_balls],
        y=sample_li,
        mode='lines+markers',
        line=dict(color='#34D399', width=2.5),
        marker=dict(size=7, color='#FDA4AF', line=dict(color='#1E2228', width=1.5)),
        name="Mean Leverage (LI)"
    ))
    phase_fig.add_vrect(x0=0, x1=36, fillcolor="rgba(148, 163, 184, 0.08)", layer="below", line_width=0, annotation_text="Powerplay (1-6)", annotation_position="top left", annotation_font=dict(color="#94A3B8"))
    phase_fig.add_vrect(x0=36, x1=90, fillcolor="rgba(5, 150, 105, 0.08)", layer="below", line_width=0, annotation_text="Middle (7-15)", annotation_position="top left", annotation_font=dict(color="#6EE7B7"))
    phase_fig.add_vrect(x0=90, x1=120, fillcolor="rgba(244, 63, 94, 0.08)", layer="below", line_width=0, annotation_text="Death Overs (16-20)", annotation_position="top left", annotation_font=dict(color="#FDA4AF"))
    phase_fig.update_layout(
        **PLOTLY_THEME,
        title="<b>Leverage Volatility Across Match Phases</b>",
        xaxis_title="Deliveries Bowled in Chase",
        yaxis_title="Leverage Index (LI)"
    )
    st.plotly_chart(phase_fig, use_container_width=True)

# =========================================================================
# MODULE 6: MONTE CARLO SIMULATION
# =========================================================================
elif nav_section == "🎲 Monte Carlo Simulation":
    st.markdown("## 🎲 Stochastic Monte Carlo Simulation Engine")
    st.caption("Test stochastic convergence against analytical Markov dynamic programming, empirical score distributions, and confidence bounds.")

    mc_mode = st.radio("Simulation Objective", ["Chase Target Simulation", "First Innings Setting Simulation"], horizontal=True)

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("#### Match Configuration")
        mc_trials = st.slider("Number of Independent Trials (N)", 1000, 20000, 10000, step=1000)
        mc_seed = st.number_input("Random Generator Seed", 0, 999999, 42)
        mc_balls = st.slider("Balls Remaining", 0, 120, 36)
        mc_wickets = st.slider("Wickets Down", 0, 9, 3)
        mc_runs = st.number_input("Runs (Required / Scored)", 0, 250, 48)
        run_mc = st.button("🚀 Execute Monte Carlo Simulation", use_container_width=True)

    with col_m2:
        if run_mc:
            with st.spinner(f"Simulating {mc_trials:,} stochastic matches…"):
                if mc_mode.startswith("Chase"):
                    s = State(mc_balls, mc_wickets, mc_runs)
                    res = simulate_chase(model, s, n=mc_trials, seed=mc_seed)
                    p_exact = win_model.probability(s)

                    st.markdown(f"""
                    <div class="cric-card">
                        <div class="metric-box-label">SIMULATION RESULTS ({mc_trials:,} TRIALS)</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #10B981; margin: 6px 0;">{res['win_probability']*100:.2f}% Win Probability</div>
                        <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                            Standard Error: <b style="color: #F8FAFC;">±{res['standard_error']*100:.2f}%</b> | 
                            95% Confidence Interval: <b style="color: #F8FAFC;">[{res['conf_interval_95'][0]*100:.2f}%, {res['conf_interval_95'][1]*100:.2f}%]</b><br>
                            Analytical Exact DP Probability: <b style="color: #38BDF8;">{p_exact*100:.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    res = simulate_first_innings(model, mc_balls, mc_wickets, mc_runs, n=mc_trials, seed=mc_seed)
                    st.markdown(f"""
                    <div class="cric-card">
                        <div class="metric-box-label">FIRST INNINGS PROJECTION ({mc_trials:,} TRIALS)</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #0EA5E9; margin: 6px 0;">{res['win_probability']*100:.2f}% Win Probability</div>
                        <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                            Expected Final Total: <b style="color: #F8FAFC;">{res['mean_final_score']:.1f} runs</b> (Std Dev: ±{res['std_final_score']:.1f})<br>
                            Median Score: <b style="color: #F8FAFC;">{res['score_quantiles']['median']:.0f}</b> | 
                            80% Density Interval: <b style="color: #F8FAFC;">{res['score_quantiles']['p10']:.0f} – {res['score_quantiles']['p90']:.0f}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Benchmark consistency table
    st.markdown("### 🔬 Analytical Dynamic Programming vs Monte Carlo Verification")
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
    st.markdown("## 🏟️ Bayesian Venue & Team Ground Analysis")
    st.caption("Empirical Bayesian Beta-Binomial updating with shrinkage priors to model venue chasing bias and team venue records.")

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
            <div class="cric-card">
                <h3 style="color: #F8FAFC; margin: 0 0 6px 0;">{sel_venue}</h3>
                <div class="metric-box-label">Bayesian Posterior Chase Win Rate</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #10B981; margin: 4px 0;">{post_mean*100:.1f}%</div>
                <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                    Matches Analyzed: <b style="color: #F8FAFC;">{n}</b> | Chases Won: <b style="color: #F8FAFC;">{k}</b> ({k/n*100:.1f}%)<br>
                    95% Credible Interval: <b style="color: #FBBF24;">[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]</b><br>
                    Prior Specification: <b>Beta(10, 10)</b> (Shrinkage toward league mean)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with vc2:
            x = np.linspace(0.2, 0.8, 200)
            prior_y = beta_dist.pdf(x, 10, 10)
            post_y = beta_dist.pdf(x, 10 + k, 10 + n - k)

            beta_fig = go.Figure()
            beta_fig.add_trace(go.Scatter(x=x, y=prior_y, mode='lines', name='Prior Beta(10,10)', line=dict(dash='dash', color='#64748B')))
            beta_fig.add_trace(go.Scatter(x=x, y=post_y, mode='lines', name='Posterior Distribution', line=dict(color='#10B981', width=2.5), fill='tozeroy', fillcolor='rgba(16,185,129,0.12)'))
            beta_fig.add_vline(x=post_mean, line_color="#FDA4AF", line_dash="dot", annotation_text="Posterior Mean", annotation_font=dict(color="#FDA4AF"))
            beta_fig.update_layout(
                **PLOTLY_THEME,
                title=f"<b>Beta-Binomial Density Function</b>",
                xaxis_title="Chase Win Probability θ",
                yaxis_title="Probability Density"
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
                color='#10B981'
            ),
            marker=dict(color='#FDA4AF', size=8, line=dict(color='#F43F5E', width=1.5)),
            name="95% Credible Interval"
        ))
        forest_fig.add_vline(x=50, line_dash="dash", line_color="#64748B", annotation_text="50% Equilibrium")
        forest_fig.update_layout(
            **PLOTLY_THEME,
            xaxis_title="Chasing Win Probability (%)",
            height=500
        )
        st.plotly_chart(forest_fig, use_container_width=True)

        # Integrated Feature: Team-Venue Priors Matrix
        if 'team_venue_priors' in reports:
            st.markdown("### 🏟️ Franchise Ground Records & Bayesian Performance Matrix")
            st.caption("Bayesian posterior team win expectations per ground from historical empirical match data.")
            tv_df = pd.DataFrame(reports['team_venue_priors'])
            if not tv_df.empty:
                f_team, f_ground = st.columns(2)
                with f_team:
                    team_list = ['All Teams'] + sorted(tv_df['team'].unique().tolist())
                    sel_team = st.selectbox("Filter Franchise", team_list, index=0)
                with f_ground:
                    ground_list = ['All Grounds'] + sorted(tv_df['venue'].unique().tolist())
                    sel_ground = st.selectbox("Filter Ground", ground_list, index=0)

                sub_tv = tv_df.copy()
                if sel_team != 'All Teams': sub_tv = sub_tv[sub_tv['team'] == sel_team]
                if sel_ground != 'All Grounds': sub_tv = sub_tv[sub_tv['venue'] == sel_ground]

                sub_tv['Win Expectation'] = (sub_tv['posterior_mean'] * 100).map("{:.1f}%".format)
                sub_tv['95% Credible Interval'] = sub_tv['credible_interval_95'].map(lambda ci: f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
                sub_tv['Record'] = sub_tv.apply(lambda r: f"{r['wins']}/{r['matches']} ({r['wins']/max(1, r['matches'])*100:.0f}%)", axis=1)

                st.dataframe(sub_tv[['team', 'venue', 'matches', 'wins', 'Record', 'Win Expectation', '95% Credible Interval']].rename(
                    columns={'team': 'Team', 'venue': 'Ground', 'matches': 'Matches', 'wins': 'Wins'}
                ).sort_values('Matches', ascending=False), use_container_width=True)

# =========================================================================
# MODULE 8: TOSS HYPOTHESIS TESTING
# =========================================================================
elif nav_section == "🪙 Toss Hypothesis Testing":
    st.markdown("## 🪙 Toss Advantage Statistical Hypothesis Testing")
    st.caption("Hypothesis tests (Proportion Z-Test, Exact Binomial Test, Chi-Square Independence Test) to determine if coin toss outcome creates a statistically significant advantage.")

    toss_scope = st.radio("Dataset Evaluation Scope", ["2024 Holdout Season (70 matches)", "All Historical IPL Matches (1,034 matches)"], horizontal=True)

    if 'toss_analysis' in reports:
        t_data = reports['toss_analysis']['test_season'] if toss_scope.startswith("2024") else reports['toss_analysis']['full_dataset']

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Toss Winner Match Win Rate</div>
                <div class="metric-box-value" style="color: #FBBF24;">{t_data['observed_proportion']*100:.1f}%</div>
                <div class="metric-box-sub">{t_data['wins']} Wins / {t_data['matches']} Matches</div>
            </div>
            """, unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Proportion Z-Test</div>
                <div class="metric-box-value" style="color: #0EA5E9;">Z = {t_data['z_statistic']:.3f}</div>
                <div class="metric-box-sub">Two-Sided p-value: <b>{t_data['z_test_p_value']:.4f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with tc3:
            sig_col = "#F43F5E" if t_data['is_statistically_significant'] else "#10B981"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Statistical Significance (α = 0.05)</div>
                <div class="metric-box-value" style="color: {sig_col}; font-size: 1.6rem;">
                    {'REJECT H0' if t_data['is_statistically_significant'] else 'FAIL TO REJECT H0'}
                </div>
                <div class="metric-box-sub">Exact Binomial p = {t_data['p_value']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Chi-Square Analysis
        if 'toss_decision_chi_square' in t_data:
            chi_data = t_data['toss_decision_chi_square']
            st.markdown("### 📋 Chi-Square Test of Independence: Toss Decision (Bat vs Field)")
            st.markdown(f"""
            Contingency table testing if electing to bat first vs field first is statistically independent of winning the fixture:
            - **Chi-Square Statistic (χ²):** `{chi_data['chi2_statistic']:.4f}` (df = 1)
            - **p-value:** `{chi_data['p_value']:.4f}` ({'Statistically significant at α=0.05' if chi_data['is_statistically_significant'] else 'Not significant at α=0.05'})
            - **Odds Ratio (Fielding vs Batting):** `{chi_data['odds_ratio_field_vs_bat']:.2f}x`
            - **Cochran Condition Met:** `{'Yes (Minimum cell count >= 5)' if chi_data['cochran_condition_met'] else 'No'}`
            """)

# =========================================================================
# MODULE 9: MODEL VALIDATION AND CALIBRATION
# =========================================================================
elif nav_section == "📊 Model Validation and Calibration":
    st.markdown("## 📊 Model Validation & Decile Calibration")
    st.caption("Empirical scoring rule benchmarks (Brier Score, Log Loss, ECE) evaluated on the 2024 holdout test season.")

    if 'adjusted_metrics' in reports:
        m_rep = reports['adjusted_metrics']
        raw = m_rep['raw_markov']
        adj = m_rep['calibrated_hybrid']
        base = m_rep['constant_50_percent_baseline']

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Brier Score (Lower is Better)</div>
                <div class="metric-box-value" style="color: #10B981;">{adj['brier_score']:.4f}</div>
                <div class="metric-box-sub">Raw Markov: {raw['brier_score']:.4f} | Baseline: {base['brier_score']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with vc2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Log Loss / Cross-Entropy</div>
                <div class="metric-box-value" style="color: #0EA5E9;">{adj['log_loss']:.4f}</div>
                <div class="metric-box-sub">Raw Markov: {raw['log_loss']:.4f} | Baseline: {base['log_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with vc3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-label">Expected Calibration Error (ECE)</div>
                <div class="metric-box-value" style="color: #FBBF24;">{adj['expected_calibration_error']:.4f}</div>
                <div class="metric-box-sub">Accuracy: {adj['accuracy']*100:.1f}% | Skill Score: +{adj['brier_skill_score']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Reliability diagram
    if 'adjusted_calibration' in reports:
        cal_df = reports['adjusted_calibration']
        raw_cal = reports.get('calibration', cal_df)

        rel_fig = go.Figure()
        rel_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#64748B'), name='Perfect Calibration (y = x)'))
        rel_fig.add_trace(go.Scatter(
            x=raw_cal['mean_prediction'], y=raw_cal['observed_win_rate'],
            mode='lines+markers', name='Raw Markov Model', line=dict(color='#FDA4AF', dash='dot')
        ))
        rel_fig.add_trace(go.Scatter(
            x=cal_df['mean_prediction'], y=cal_df['observed_win_rate'],
            mode='lines+markers', name='Calibrated Hybrid Model', line=dict(color='#10B981', width=2.5)
        ))
        rel_fig.update_layout(
            **PLOTLY_THEME,
            title="<b>Decile Reliability Diagram (Holdout Season Evaluation)</b>",
            xaxis_title="Mean Predicted Win Probability",
            yaxis_title="Empirical Observed Win Proportion"
        )
        st.plotly_chart(rel_fig, use_container_width=True)

        with st.expander("📋 View Decile Reliability Table"):
            st.dataframe(cal_df, use_container_width=True)

# =========================================================================
# MODULE 10: HISTORICAL MATCH ANALYSIS
# =========================================================================
elif nav_section == "📜 Historical Match Analysis":
    st.markdown("## 📜 Historical Match Database & Turning Points Analysis")
    st.caption("Replay any IPL chase from 2008–2024 with ball-by-ball win probability swings and key momentum shifts.")

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
            <div class="cric-card">
                <h3 style="color: #F8FAFC; margin: 0 0 6px 0;">{m_row['team1']} vs {m_row['team2']}</h3>
                <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                    Date: <b style="color: #F8FAFC;">{m_row['date']}</b> | Ground: <b style="color: #F8FAFC;">{m_row['venue']}</b><br>
                    Toss: <b style="color: #F8FAFC;">{m_row['toss_winner']}</b> elected to <b style="color: #F8FAFC;">{m_row['toss_decision']}</b><br>
                    Result: <b style="color: #10B981;">{m_row['winner']} WON</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Analyze chase trajectory
        match_balls = full_df[full_df['match_id'].astype(str) == str(m_row['match_id'])]
        traj = analyze_chase_trajectory(match_balls, win_model, calibrator, context)

        if not traj.empty:
            st.plotly_chart(create_worm_chart(traj, m_row['team2'], m_row['team1']), use_container_width=True)

            st.markdown("### ⚡ Top 5 Match Turning Points (Win Probability Added)")
            top_swings = traj.reindex(traj['wpa_calibrated'].abs().sort_values(ascending=False).index).head(5)
            for _, sw in top_swings.iterrows():
                sign = "+" if sw['wpa_calibrated'] > 0 else ""
                swing_col = "#10B981" if sw['wpa_calibrated'] > 0 else "#F43F5E"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border-left: 4px solid {swing_col}; padding: 10px 16px; margin-bottom: 8px; border-radius: 6px;">
                    <b>Over {sw['over_label']}</b> — <b>{sw['event']}</b> | 
                    Score: <b>{sw['cumulative_runs']}/{sw['cumulative_wickets']}</b> (Required: {sw['runs_required']} off {sw['balls_left']} balls)<br>
                    Win Prob: {sw['calibrated_probability']*100:.1f}% → {sw['next_calibrated_probability']*100:.1f}% 
                    (<b style="color: {swing_col};">{sign}{sw['wpa_calibrated']*100:.1f}% WPA</b>)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Chase trajectory data could not be computed for this fixture.")
    else:
        st.info("No historical matches found in dataset.")
