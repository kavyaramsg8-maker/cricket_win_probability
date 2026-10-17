# Updated results (reused 2026 holdout)

| Model | Brier score | Log loss |
|---|---:|---:|
| Original delivered baseline (training through 2025) | 0.212997 | 0.746746 |
| Current raw Markov (training through 2024) | 0.219821 | 0.784152 |
| Current calibrated hybrid | 0.127057 | 0.397573 |
| Constant 50% baseline | 0.250000 | 0.693147 |

All scores use the same 374 sampled states from 70 included 2026 matches.
Lower scores are better. The updated forecast improves both metrics on this
reused sample. The original and current raw model have different training
cutoffs: 2025 is reserved for calibration in the update.

Exploratory independently weighted context calibration assigned team/venue
context zero weight; that calibration-only model scored Brier 0.126389 and
log loss 0.396162 on the same reused 2026 sample. Thus the context-linked
hybrid does NOT establish incremental predictive gain from team/venue
information. The improvement is predominantly calibration. The final hybrid
was specified after exploring 2025 calibration data; its context contribution
is a fixed prior modelling choice, not a separately learned positive effect.

Because 2026 had already been inspected, these figures are development results
on a reused holdout. Fresh confirmation needs a new future season or a separate
predeclared backtesting protocol. Do not repeatedly tune on these scores.
See UPDATED_MODEL_GUIDE.md for equations, split, usage and limitations.
