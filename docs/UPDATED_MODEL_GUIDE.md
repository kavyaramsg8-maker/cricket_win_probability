# Bayesian context and calibration update

## What changed
Second-innings predictions now have an additional adjusted probability.
Choose the chasing team, bowling team and venue to use historical context.
Raw Markov and raw Monte Carlo remain visible to demonstrate the original
algorithm. Compare these two raw numbers with each other; the adjusted number
is a different forecast, not something the raw simulation should match.
First-innings simulation remains pooled and uncalibrated.

## Separation of fitting data
- 2008–2024 (1,034 matches): fit delivery transitions and team/venue estimates.
- 2025 (68 matches): fit the adjustment intercept and positive shared slope.
- 2026 (70 matches): report the resulting test performance.
2025 does not enter the transition or prior counts in this version.
2026 was already inspected in earlier project development, so this is a REUSED
holdout and not new independent confirmation. Do not tune further using 2026.
The calibration design was developed using 2025, which is not an untouched
validation set. A future-season evaluation is needed for fresh confirmation.

## Bayesian context
For each team, use a Beta(10,10) prior on match win rate. Update with historical
wins/losses. For a team/venue pair use a Beta prior with 20 equivalent matches
centred on the global team posterior mean, then update with venue wins/losses.
This shrinks tiny venue samples toward the team estimate instead of trusting
unstable proportions. Unknown venues fall back to global team rates. Unknown
teams have rate 0.5. With either team unspecified, context is neutral (0).
Selected known teams can still affect predictions at an unspecified venue.
A few historical franchise name changes are harmonized in adjustment.py.
Venue aliases are not harmonized, so some histories remain fragmented.

context = 0.5 * [logit(chasing_team_rate) - logit(bowling_team_rate)]
q = sigmoid(a + b * [logit(raw_Markov_probability) + context])

The two calibration coefficients a and b are estimated on 2025 only, with b
constrained positive and a fixed small L2 penalty. Each match has equal total
weight in fitting. This keeps predictions monotone in raw probability and
ensures selected team/venue context can affect a nonterminal forecast.
The reported context coefficient equals b by design; it is NOT a separately
learned claim that team history predicts outcomes independently.

This is a hybrid odds adjustment with Bayesian feature estimates plus logistic
calibration. It is NOT an exact Bayesian posterior for a match state: the
historical state model and context evidence overlap. Do not claim that the
raw Markov transition distribution is now team-specific. The adjustment is
applied to the output probability, and raw MC standard error applies only to
raw MC. Boundary states are enforced: target reached=1; exhausted chase=0.

An exploratory separately weighted context fit assigned context zero weight
on 2025. Therefore the fixed prior combination is a modelling choice, and no
independent accuracy benefit from team/venue context has been established.
The main expected improvement is calibration of the overconfident state model.
See adjusted_metrics.json for measured scores, not a promise of real accuracy.

## Files and three-person ownership
Person 1: BayesianContext in cricket/adjustment.py, historical counts, priors,
franchise aliases and data separation.
Person 2: Markov engine, shared state interfaces and explaining how raw odds
combine with the Bayesian feature.
Person 3: ChaseCalibrator, improve_model.py, evaluation, UI and report plots.

## Running
Precomputed model/results are included. From the project folder:
python -m streamlit run app.py

To reproduce the improved pipeline:
python improve_model.py --input data/raw/ipl_deliveries.csv

To run all checks:
python -m unittest discover -s tests -v

Running run_pipeline.py builds the ORIGINAL unadjusted baseline and switches
the app back to that mode. Run improve_model.py to restore adjusted mode.

## Sensible checks
Set both teams to Unspecified: context is neutral but calibration still applies.
Choose different teams and compare q while keeping state fixed. Swap teams to
reverse the context sign (not necessarily complementary full probabilities,
because the current chase state still matters). Try 0 runs required and 0 balls
remaining to confirm boundaries. First innings deliberately has no adjustment.
This app has no live API. A good log loss does not establish causal accuracy,
player-specific validity, applicability to DLS games or calibration in new eras.
