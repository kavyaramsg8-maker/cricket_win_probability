# 🏏 Live Win-Probability Engine for T20 Cricket

An end-to-end, statistically rigorous, broadcast-grade **T20 Cricket Win-Probability Engine** powered by **Discrete-Time Absorbing Markov Chains**, **Exact Backward-Induction Dynamic Programming**, **Vectorized 10,000-Trial Monte Carlo Simulations**, and **Hierarchical Bayesian Feature Context with Monotone Logistic Calibration**.

Trained on over 230,000 deliveries from IPL 2008–2023 and rigorously evaluated on the held-out 2024 IPL season (70 matches), this application features a modern sports-broadcast dashboard with 10 interactive analytics modules.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Data Ingestion & Pipeline Execution
```bash
# 1. Convert raw Cricsheet IPL JSON data (2008–2024) to canonical CSV
python convert_cricsheet.py --zip data/raw/ipl_male_json.zip

# 2. Run the complete training, evaluation, and statistical analysis pipeline
# (Trains on 2008–2023, evaluates exclusively on held-out 2024 season)
python run_pipeline.py --input data/raw/ipl_deliveries.csv
```

### 3. Launch Interactive Analytics Dashboard
```bash
# Launch the premium Streamlit application
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run Automated Test Suite
```bash
# Execute all 34 automated unit and integration tests
python -m unittest discover -s tests -v
```

---

## 🧠 Mathematical Foundations & Algorithms

### 1. Discrete-Time Absorbing Markov Chains
A second-innings chase is modeled as a discrete stochastic process over match states $S = (b, w, r)$, where:
- $b \in [0, 120]$: Legal deliveries remaining in the chase.
- $w \in [0, 10]$: Wickets lost by the chasing team.
- $r \in \mathbb{Z}$: Runs required to achieve victory ($r = \text{Target} - \text{Runs Scored}$).

#### Absorbing Boundary Conditions
- **Victory Condition:** $r \le 0 \implies W(b, w, r) = 1.0$ (chasing team achieves target).
- **Defeat Condition:** $b = 0 \lor w = 10$ with $r > 0 \implies W(b, w, r) = 0.0$ (balls exhausted or all out).

#### Backward Induction Dynamic Programming Recursion
The exact probability of the chasing team winning from nonterminal state $S$ satisfies the Bellman expectation equation:
$$W(b, w, r) = \sum_{o \in \Omega} P(o \mid b, w) \cdot W(b - l_o, \min(10, w + w_o), r - r_o)$$
where $o = (r_o, w_o, l_o)$ represents a joint delivery outcome comprising runs scored $r_o$ (including extras), dismissal indicator $w_o \in \{0, 1\}$, and legal ball flag $l_o \in \{0, 1\}$.

Every transition strictly reduces $b$ (if legal $l_o = 1$) or strictly reduces $r$ (if illegal $l_o = 0$, which awards at least 1 extra run). This strictly precludes cycles. The model builds a precomputed 3D lookup table $V[b, w, r]$ enabling $O(1)$ microsecond lookups without recursive stack overhead.

### 2. MLE Transition Estimation with Laplace Smoothing
Transition distributions $P(o \mid b, w)$ are conditioned on 16 state bands:
- **4 Match Phases:** Powerplay ($b \in [91, 120]$), Middle 1 ($b \in [61, 90]$), Middle 2 ($b \in [31, 60]$), Death ($b \in [0, 30]$).
- **4 Wicket Bands:** Top order ($w \in [0, 2]$), Middle order ($w \in [3, 5]$), Lower order ($w \in [6, 8]$), Tail ($w \in [9, 10]$).

Laplace smoothing with parameter $\alpha = 0.5$ guarantees full probability support across all valid joint delivery events:
$$P(o \mid \text{band}) = \frac{c(\text{band}, o) + \alpha}{\sum_{o'} c(\text{band}, o') + \alpha K}$$

### 3. Fast Vectorized 10,000-Trial Monte Carlo Engine
For real-time simulation, the engine precomputes discrete cumulative density arrays per state band and executes 10,000 independent match trajectories simultaneously using vectorized NumPy batch transitions.
$$\hat{p}_{\text{MC}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Chase } i \text{ won}), \quad \text{SE} = \sqrt{\frac{\hat{p}_{\text{MC}}(1 - \hat{p}_{\text{MC}})}{N}}$$
10,000 independent trials execute in **under 35 milliseconds**.

### 4. Delivery Leverage Index (LI) & Win Probability Added (WPA)
- **Win Probability Added (WPA):** For delivery $t$, $\text{WPA}_t = W(S_t) - W(S_{t-1})$.
- **Delivery Leverage Index (LI):** Measures the volatility/criticality of the delivery state:
$$\text{LI}(S) = \frac{\sqrt{\sum_o P(o \mid S) \left[ W(\text{advance}(S, o)) - W(S) \right]^2}}{\sigma_{\text{baseline}}}$$
Normalized such that $\text{LI} = 1.0$ represents an average chase situation. $\text{LI} > 2.0$ highlights extreme clutch deliveries.

### 5. Hierarchical Bayesian Team & Venue Context
Team strengths and venue chase advantages are updated via Beta-Binomial conjugacy:
- **Global Team Prior:** $\text{Beta}(10, 10)$. Historical win/loss updates yield posterior mean $\hat{\theta}_{\text{team}}$.
- **Venue Hierarchical Shrinkage:** 20 pseudo-matches centered at the team posterior shrink small venue samples:
$$\hat{\theta}_{\text{venue}} = \frac{k_{\text{venue}} + 20 \hat{\theta}_{\text{team}}}{n_{\text{venue}} + 20}$$
- **Context Feature:** Odds differential between chasing and bowling teams:
$$\text{context} = \frac{1}{2} \left[ \text{logit}(\hat{\theta}_{\text{chase}}) - \text{logit}(\hat{\theta}_{\text{bowl}}) \right]$$

### 6. Monotone Logistic Calibration (Platt Scaling)
To eliminate raw model overconfidence, predictions undergo bounded L2-regularized logistic calibration:
$$q = \sigma\left( a + b \cdot \left[ \text{logit}(p_{\text{Markov}}) + \text{context} \right] \right)$$
with slope constraint $b > 0$, preserving monotonic ordering and boundary guarantees ($0.0$ and $1.0$).

### 7. Toss Hypothesis Testing
- **Proportion Z-Test:** Tests $H_0: P(\text{Toss Winner Wins}) = 0.5$:
$$Z = \frac{2k - n}{\sqrt{n}}, \quad p = 2 \cdot (1 - \Phi(|Z|))$$
- **Exact Binomial Test:** Two-tailed Clopper-Pearson / Wilson test.
- **Chi-Square Test of Independence:** $2 \times 2$ contingency table of Toss Decision ($\text{Bat}$ vs $\text{Field}$) $\times$ Match Outcome ($\text{Win}$ vs $\text{Loss}$).

---

## 📊 Empirical Holdout Results (2024 IPL Season, 70 Matches)

| Metric | Raw Markov Model | Calibrated Hybrid Model | 50% Constant Baseline |
|---|:---:|:---:|:---:|
| **Brier Score** (lower is better) | `0.1554` | **`0.1319`** | `0.2500` |
| **Brier Skill Score** (vs 50%) | `+37.8%` | **`+47.2%`** | `0.0%` |
| **Log Loss** (lower is better) | `0.5883` | **`0.4254`** | `0.6931` |
| **Classification Accuracy** | `80.3%` | **`82.5%`** | `50.0%` |
| **Expected Calibration Error (ECE)** | `0.1596` | **`0.0924`** | `—` |
| **Analytical vs Monte Carlo Agreement** | `100%` | **`100%`** (8/8 states inside 95% CI) | `—` |

---

## 🖥️ Interactive GUI Modules (Streamlit)

1. **🏏 Live Match Dashboard:** Broadcast scorecard header, dual-colored radial win-probability gauge, RRR vs CRR comparison, match momentum ticker.
2. **⚡ Ball-by-Ball Match Simulator:** Interactive delivery keypad (`0`, `1`, `2`, `3`, `4`, `6`, `Wicket`, `Wide`, `No-ball`), single/over/rest auto-simulation, Undo/Reset controls, and live delivery log with WPA swings.
3. **🎯 Win Probability Predictor:** Team and venue selection, custom state inputs, instant calculation of Exact DP, Calibrated Hybrid, and 10,000-trial Monte Carlo with 2D sensitivity lookup heatmap.
4. **📈 Probability Worm Chart:** Dynamic Plotly worm tracking win probability across all 120 deliveries, highlighting wickets, boundaries, and sixes with unified hover cards.
5. **⚖️ Delivery Leverage Analysis:** Delivery Leverage Index (LI), bar chart of WPA swings across all possible outcomes, and maximum potential swing spread.
6. **🎲 Monte Carlo Simulation Laboratory:** 10,000-trial simulator for both second-innings chases and first-innings targets, with convergence curves and score quantile distributions.
7. **🏟️ Bayesian Venue Analysis:** All 58 IPL venues analyzed via Beta-Binomial updating; interactive prior vs posterior density curves, 95% credible intervals, and league-wide forest plot.
8. **🪙 Toss Hypothesis Testing:** Proportion Z-test, exact binomial test, and Chi-square test of independence on toss decision with interactive contingency tables.
9. **📊 Model Validation & Calibration:** Decile reliability diagram, calibration error bars, ECE/MCE metrics, and baseline comparisons.
10. **📜 Historical Match Analysis:** Database of 2008–2024 IPL matches, full ball-by-ball chase trajectory replay, and automated identification of the Top 5 pivotal match turning points.

---

## 📁 Repository Structure

```
cricket_win_probability/
├── app.py                      # Premium Streamlit broadcast dashboard (10 modules)
├── convert_cricsheet.py        # Cricsheet JSON-to-CSV parser (2008-2024 filter)
├── run_pipeline.py             # End-to-end data pipeline, model training & validation
├── improve_model.py            # Calibrated model reproduction script
├── requirements.txt            # Tested Python dependencies (numpy, pandas, scipy, streamlit, plotly)
├── cricket/                    # Core Python engine package
│   ├── __init__.py             # Package exports
│   ├── state.py                # State (b, w, r), Outcome, advance() transition logic
│   ├── markov.py               # TransitionModel (Laplace MLE) & WinModel (DP recursion)
│   ├── simulation.py           # Vectorized 10,000-trial Monte Carlo simulation engine
│   ├── bayes.py                # Beta-Binomial posteriors, Toss Z-test & Chi2 independence
│   ├── adjustment.py           # BayesianContext & ChaseCalibrator (Platt scaling)
│   ├── evaluation.py           # Brier score, Log loss, decile calibration, ECE/MCE
│   ├── preprocessing.py        # Delivery CSV cleaning, state derivation, train/test split
│   ├── historical.py           # Historical match queries & ball-by-ball WPA trajectories
│   ├── cricsheet.py            # Cricsheet JSON converter & conservative match filters
│   └── synthetic.py            # Synthetic match generator for pipeline smoke testing
├── data/
│   ├── raw/                    # ipl_male_json.zip and generated ipl_deliveries.csv
│   └── processed/              # train.csv, test.csv, model.pkl, adjustment.pkl, metadata.json
├── reports/                    # Diagnostic JSON summaries, calibration CSVs, and plots
└── tests/                      # 34 automated unit and integration tests
    ├── test_project.py         # Markov DP, boundaries, state advance correctness
    ├── test_app.py             # Official Streamlit AppTest simulated UI checks
    ├── test_cricsheet.py       # Cricsheet converter and exclusion filtering checks
    ├── test_adjustment.py      # Bayesian context, monotonicity, and calibration bounds
    ├── test_real_pipeline.py   # Training distribution support & refit isolation
    ├── test_leverage.py        # Delivery Leverage Index & outcome swing monotonicity
    └── test_advanced_stats.py  # Venue posteriors, toss Z/Chi2 tests, vectorized MC
```
