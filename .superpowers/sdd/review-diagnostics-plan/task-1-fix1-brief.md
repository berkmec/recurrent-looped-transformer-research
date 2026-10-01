# Task1 fix round1 — before any main freeze

Fix confirmed training-review findings. Preserve source scope and no main run.
1. A completed condition is immutable: run_one must validate its complete
   checkpoint/receipt/timing set then return without rewriting it. Deleted
   selected.pt, corrupt last.pt, altered train.json, coherent partial resume
   conflicting with completed last/receipt must all fail. No duplicate updates.
2. Cached evaluation summaries must be bound/integrity checked and raw-derived
   counts independently checked. A changed final_accuracy must not be accepted
   because raw predictions/checkpoint still match. Add meaningful tests.
3. Parent resource ruling: use checkpoint_every64, not every1. Every-update
   optimizer snapshots write at least447GB for R5, so this is unacceptable
   overhead for the modest model campaign. Maintain every-update finite
   loss/gradient/count/input/timing journal; persist full optimizer snapshots at
   every64, validation-selected points, equal-label64, orderly stop and finish.
   EXACT CLEAN RESUME is required. An interrupted uncheckpointed segment must
   remain explicitly marked and be rejected without silent repair/rollback.
   Preserve inflight/partial logs. A documented fresh attempt can be separately
   audited if failure ever happens; automatic recovery is out of scope.
   Snapshots may retain accumulated history for straightforward integrity.
   All terminal artifact checks also apply to coarser checkpoint boundaries.

This cadence was an implementation decision, not original scientific protocol;
data/steps/losses/models/seeds remain unchanged. Add cadence/resume tests and
update notes/report/calibration write-volume projection. No need for new long
pilots. Never touch old evidence. No subagents. Report fixes when ready so the
same reviewer can perform scoped re-review; do not freeze or launch main.
