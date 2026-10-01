# Task1 independent review before main training

Read task-1-brief.md in this directory as requirements, plus ledger rulings.
Review review_cycle/training.py, training_data.py and test_training.py against
immutable core/model.py. Main has not started. Source frozen only after review.

Resolved runtime decision: existing DaxDream .venv Python (torch2.12.0+cu130) is
read-only runtime. Small conditions run CPU, development/confirmation run CUDA,
all frozen explicitly. Calibration found small CPU faster than GPU. No installs.

Check scientific identifiability and paired inputs/initial states; exact 36 small
+ 9 dev + 3 confirm schedule; sparse label RNG; five-state and blocking oracle;
correct TBPTT route; development-only LR selection and final-data gate; exact
optimizer counts and resumability; missing/stale/ambiguous checkpoint rejection;
source pins; actual evaluation of declared variants and raw metric evidence.
Pay attention to restart and partial condition state integrity. Tests must detect
actual errors, not mirror implementation. Run bounded tests only, no main or GPU.
Do not edit implementation. Report actionable evidence-backed findings promptly
to parent, full report at training-review.md in this same directory. No subagents.
