# Three-person implementation and integration

Ownership describes module responsibilities. This generated starter does not
establish that three people independently authored it. Each member should run,
explain, verify and adapt their part, and follow their course's AI-use rules.

## Person 1: data and statistical inference
- Own cricket/preprocessing.py, cricket/synthetic.py, cricket/cricsheet.py, convert_cricsheet.py and cricket/bayes.py.
- Obtain the real data separately, map it using DATA_FORMAT.md, filter unsupported
  matches and verify legal delivery/wicket definitions.
- Prepare states BEFORE each event and a chronological train/test split.
- Explain Beta(1,1) priors, posterior counts, intervals and toss hypothesis test.
- Deliver canonical CSV, train/test CSV and JSON statistical summaries.
- Verify one match's target, score, wickets and legal-ball count by hand.

## Person 2: state transition and Markov engine
- Own cricket/state.py and cricket/markov.py.
- Maintain State and Outcome interfaces used by both other members.
- Fit outcome counts by phase/wicket bands with Laplace smoothing.
- Explain rare-state fallback, normalized rows and absorbing boundaries.
- Explain backward recursion and memoization, including why illegal deliveries
  reduce required runs even though they do not reduce balls left.
- Deliver TransitionModel.fit/distribution and WinModel.probability.
- First-innings prediction is owned jointly with Person 3: complete innings and
  chase via simulation rather than claiming exact first-innings DP.

## Person 3: simulation, evaluation and application
- Own cricket/simulation.py, cricket/evaluation.py, app.py and run_pipeline.py.
- Sample from Person 2's distribution and reuse advance for chase transitions.
- Compare MC with DP on known cases; report MC standard error.
- Build holdout predictions, Brier/log-loss/calibration, and manual-entry UI.
- Run the complete pipeline and record limitations of the real-data results.

## Shared interfaces (freeze these before separate edits)
- State(balls_left, wickets_lost, runs_required): integer counts BEFORE a ball.
- Outcome(runs, wicket, legal): joint event, not three independent samples.
- advance(state, outcome) -> State.
- TransitionModel.fit(training_dataframe) -> fitted model.
- TransitionModel.distribution(state) -> list of (Outcome, probability).
- WinModel(model).probability(state) -> chasing team's probability in [0,1].
- simulate_chase(model,state,n,seed) -> probability, standard error, simulations.
- Canonical data schema in DATA_FORMAT.md.

## Integration order
1. All members run the synthetic demo before changing anything.
2. Person 1 supplies canonical real data. Person 2 can meanwhile use synthetic
   data; Person 3 can use the explicit fallback model for development only.
3. Person 2 fits on training seasons only. Rebuild WinModel if the fitted model
   changes, because old cached probabilities must not survive a refit.
4. Person 3 evaluates held-out matches and integrates reports in the app.
5. All three run tests and check one historical chase together.

The workloads differ: Person 2 carries the hardest algorithm; Person 3 owns more
integration/UI work; Person 1 owns real-data cleaning and statistical inference.
Do not edit another member's interface without telling the group.


For this updated edition, the adjustment ownership and new split in
UPDATED_MODEL_GUIDE.md supersede the original baseline workflow above.
