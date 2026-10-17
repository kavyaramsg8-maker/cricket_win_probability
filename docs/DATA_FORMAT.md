# Real data integration contract

This updated package includes converted IPL deliveries from the supplied Cricsheet ZIP.
To convert a new ZIP, use convert_cricsheet.py; exclusions are logged in
reports/data_conversion.json.
The pipeline reads one canonical CSV. The supplied converter handles Cricsheet
JSON archives. Arbitrary Kaggle/CSV schemas still need a separate mapping. Keep delivery order chronological within innings.

Required columns:

| Column | Meaning |
|---|---|
| match_id | Unique match identifier |
| season | Integer season year, e.g. 2025 |
| innings | 1 or 2 |
| total_runs | Batter runs plus ALL extras on the delivery |
| wicket | 1 only when this event counts as a team wicket, otherwise 0 |
| legal | 0 for wide/no-ball, 1 otherwise |
| batting_team | Team batting during this innings |
| bowling_team | Team bowling during this innings |
| winner | Official winning team, repeated across this match |
| toss_winner | Toss-winning team, repeated across this match |

Recommended: venue, toss_decision (bat/field), delivery_index.
The preprocessing function needs the first six columns; the full pipeline
also needs the match metadata above to label outcomes and analyse the toss.
Venue is needed to produce team/venue prior estimates.

Map source fields explicitly. Do not use extras>0 to identify an illegal ball:
byes and leg-byes can occur on legal balls. Retired hurt does not count as a
team wicket. A run-out can occur on an illegal delivery. Preserve joint
outcomes (runs, wicket, legal) rather than assuming they are exclusive.

Filter BEFORE conversion: retain completed, uninterrupted, standard 20-over
matches with two regulation innings and a non-tied regulation result. Exclude
DLS/shortened games, no-results, abandoned games, ties and Super Overs.
The program cannot reliably infer all these exclusions from canonical fields.
Winner labels must agree with the regulation chase result. Ties are excluded
from training/evaluation; during hypothetical simulation a tie is a failure to
reach the winning target (no Super Over prediction is attempted).

Season split: latest season is test; all earlier seasons are training.
Never fit transition counts or priors to the holdout matches.

Sources for format mapping:
https://cricsheet.org/format/csv_original/
Review your particular download's format; Cricsheet provides multiple formats.
