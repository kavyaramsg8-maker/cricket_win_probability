# Calibrated edition verification

- Full improved pipeline ran with transitions and Bayesian context trained only
  through 2024; calibration fitted on 2025; test scoring on reused 2026.
- 25 automated tests passed, including first/chase UI forms and a check that
  choosing teams changes the displayed adjusted prediction.
- Tests cover monotonicity, preserved terminal states, neutral unknown context,
  posterior direction, calibration softening and the previous converter tests.
- Additional fitted-artifact checks: Bayesian context changes a nonterminal
  probability; swapped context reverses its direction; terminal outputs remain
  exactly 0/1; all three match partitions are disjoint.
- Fitted delivery counts match the pre-2025 training partition.
- New Brier 0.1270568247 and log loss 0.3975728781 on 374 states from 70 matches.
- See UPDATED_RESULTS.md for raw, original, calibration-only and constant
  baseline comparisons. Team-context incremental accuracy was not established.
- Python syntax compilation passed.

Streamlit AppTest simulates UI; no manual browser visual inspection was done.
The standard bare-mode ScriptRunContext warning appeared without app errors.
Linux/Python 3.12 tests; no local Mac execution here.
The test season is reused, so this is not independent confirmation.
