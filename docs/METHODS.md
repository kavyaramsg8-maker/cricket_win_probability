# Methods and limitations

## Markov chain
State: (balls left, wickets lost, runs required). A joint delivery event is
(runs, wicket indicator, legal-ball indicator). Runs include extras; an illegal
delivery consumes no legal ball. Score and wicket changes can occur together.
Transition probabilities are fit in four 30-ball phase bands and four wicket
bands. Runs required is an exact recursion coordinate but not a conditioning
feature of outcome estimation. This reduces sparsity at the cost of ignoring
required-rate effects on batting strategy.

For observed count c_o in a band, p(o)=(c_o+alpha)/(sum(c)+alpha*K).
alpha defaults to 0.5; supported events combine common outcomes and all joint
events observed in training. Empty bands use pooled training counts.
The stored model is sparse; it does not allocate a huge dense transition matrix.

W(b,w,R)=sum_o p(o|b,w)*W(advance(state,o)). Terminal target reached: W=1.
Otherwise zero balls or ten wickets: W=0. Target reached is checked first.
Memoization solves shared states once. Every event reduces b or R, since an
illegal ball must award runs. This prevents cycles in this simplified model.
Runs awarded on an event containing a wicket are treated as contributing to the
score before the terminal check; rare dismissal/law-specific cases are not
modelled separately. No free-hit flag, batter identity, bowling identity,
strike rotation, pitch, player matchup or time-varying strategy is represented.

## Monte Carlo
Sample each event from the same fitted transition distribution until terminal.
Repeat N times. Estimate = wins/N. SE=sqrt(p*(1-p)/N). SE quantifies simulation
noise, not bias or model uncertainty. Boundary estimates can have SE=0; this
is not a claim of certainty about the real match. First innings completes the
batting team's total and then simulates a new full chase for the opponent.
Both teams use the same pooled model. Tied simulations count as target not
reached; no Super Over winner is predicted.

## Bayesian inference and toss study
A uniform Beta(1,1) prior on a match-level win rate updates to
Beta(1+wins,1+losses). Report its mean and central 95% credible interval.
Team/venue rates use training matches only; no delivery-level fusion is claimed.
Toss study deduplicates matches, tests H0: toss-winner win rate=0.5 using an
exact two-sided binomial test and reports the exact frequentist interval.
An optional chi-square test compares toss decision and outcome. Small expected
cell counts make its approximation unreliable. These observational comparisons
are not adjusted causal estimates of the toss effect; venue, team and decision
selection can confound them. Tiny synthetic samples are not research findings.

## Evaluation
Train on earlier seasons, test on latest season. Fit all transition counts and
team/venue priors on train only. Targets and final winner on test are used to
construct legitimate in-match chase states and evaluation labels, never fitted.
Brier=mean((p-y)^2); log loss uses clipping for numerical safety.
Calibration groups predictions into ten equally spaced probability bins.
Multiple predictions from a match are correlated. Scores are averages over
sampled states, so longer matches can carry greater weight. No independent-ball
confidence intervals are claimed. For research, add match-level bootstrap
intervals and compare against an appropriate baseline on a much larger holdout.

## Scaling
Exact recursion may take noticeable time for large required scores and caches
up to 500,000 states. The app caps runs input at 400. First-innings simulation
is slower than chase calculations; use 500-1,000 simulations for quick demos.

## References
Project design: supplied Review 1 slide deck.
Cricsheet delivery formats: https://cricsheet.org/format/csv_original/
Streamlit app testing: https://docs.streamlit.io/develop/concepts/app-testing/get-started
Brier (1950), Verification of forecasts expressed in terms of probability.
Gelman et al., Bayesian Data Analysis, 3rd edition.
Norris (1997), Markov Chains.
