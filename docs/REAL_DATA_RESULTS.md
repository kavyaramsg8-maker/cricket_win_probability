# IPL baseline: real-data evaluation

Input: user-supplied ipl_male_json.zip (Cricsheet).
1,243 JSON matches; 1,172 retained; 71 excluded; 282,000 delivery rows.
The conversion audit includes SHA-256 and every excluded match/reason in
reports/data_conversion.json. Date coverage ends 2026-05-31.
Calendar match year is used for split; original season labels are preserved.
This makes IPL 2007/08 map to 2008 and 2009/10 map to 2010.

Training: 2008–2025. Holdout: 2026, 70 matches.
Evaluation samples every 24 delivery events, starting before the first ball
of each chase: 374 state predictions. This is not evaluation after every ball.

| Score | Markov model | Constant 50% baseline |
|---|---:|---:|
| Brier score (lower is better) | 0.212997 | 0.250000 |
| Log loss (lower is better) | 0.746746 | 0.693147 |

The model improves squared probability error but has WORSE log loss than the
constant baseline. Some incorrect predictions are too confident. Do not call
this a highly accurate or calibrated model. This is a useful baseline and an
honest starting point for improvement. The model pools across teams, venues,
required rates and historical eras; more recent data or richer conditioning
may help, but must be developed on training/validation data, not tuned on the
2026 holdout and then reported as independent test performance.

Toss study (2026 holdout matches, separate from fitting): 37/70 toss winners
won. Beta(1,1) posterior mean 52.78%, central 95% credible interval
41.29–64.12%. Exact binomial p-value vs 50% is 0.7202. This sample does not
provide evidence against 50% for the toss-winner win rate.
The optional toss-decision chi-square is an observational association, not a
causal claim; decision, team and conditions can be confounded.

Exclusion counts (first applicable reason):
- Tie/no-result/eliminator: 25
- Adjusted result method: 23
- Shortened target: 10
- Miscounted overs: 7
- Unusual innings ending: 6

Conservative exclusion of miscounted overs and absent-hurt endings preserves
the fixed 120-ball/10-wicket model. It reduces coverage and can bias selection.
Names of historical teams/venues remain as recorded, without franchise or
venue alias harmonization. This affects the separate Bayesian grouping tables.
Team/venue posterior estimates do not modify delivery probabilities.
First-innings probability remains a two-innings Monte Carlo approximation.
No external live-score feed, rain model or Super Over model is included.

For presentation: explain both scores and their disagreement, demonstrate the
state changes, show a real held-out chase curve and describe the limits.

Source format: https://cricsheet.org/format/json/
Data downloads: https://cricsheet.org/downloads/
