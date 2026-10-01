# Independent Task 1 fix1 re-review

Verdict: PASS for the scoped fixes and cadence64 restart contract. No remaining load-bearing blocker found for parent-controlled freeze/main launch. This is implementation verification, not a main-campaign or scientific-effect result.

Reviewed task-1-fix1-brief.md, current training.py and test_training.py, updated tests, and immutable Core contracts. Runtime: C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe -B, CPU only. No implementation edits, GPU work, main runs or subagents. This report supplements and preserves training-review.md.

## Previous findings closed

P1 CLOSED: completed bundles now carry complete.json, which seals the required file inventory and byte hashes. run_one routes existing terminal evidence to _read_complete before model construction or training. The validator checks the seal, final/resume state equality, checkpoint provenance, receipt, journals and transition markers. Completed calls return the original receipt without recreating artifacts.

The original independent two-update reproducer was rerun on this implementation. Model width8, heads1, batch2, length4, validation4, equal-label step1. Each attempt patches train_batch to raise if any training occurs; compares every surviving artifact's bytes before and after the attempt.

| Case | Result | Existing evidence preserved |
|---|---|---|
| selected.pt deleted | REJECT: training incomplete | Yes |
| last.pt replaced with corrupt bytes | REJECT: completed bundle changed | Yes |
| train.json replaced with {} | REJECT: completed bundle changed | Yes |
| coherent step1 resume/timing/update journals restored alongside completed step2 artifacts | REJECT: completed bundle changed | Yes |
| Unchanged completed bundle | COMPLETE returned, no training invoked | Yes |

P2 CLOSED: cached metadata carries a digest, identity and exact row inventory are checked, declared source cases are regenerated, and raw predictions independently determine accuracy, class histograms and paired neutral changes. The old 1.2345 final_accuracy reproducer now rejects both (a) ordinary changed metadata and (b) changed metadata with a recomputed digest. The latter fails specifically because the summary disagrees with raw predictions. Missing lengths/neutral rows and changed condition/variant are covered by real tests. This is accidental-change/integrity protection; no cryptographic third-party authenticity claim is implied.

## Cadence64 review

Default checkpoint_every is64. Full state snapshots occur at cadence boundaries, newly selected validation points, the exact equal-label checkpoint, orderly stop and finish. Under the default validation schedule, validation boundaries128 are already multiples of64. Each update appends finite loss, gradient norm, update/label/input identity and a timing row to separate durable journals. Inflight and uncheckpointed markers remain until their relevant transition is safely completed. An interrupted segment is rejected and preserved; there is no automatic rollback/replay.

Passing behavioral test used steps7, cadence3, validation7, equal-label3. Uninterrupted checkpoint boundaries were [0,3,6,7]. An orderly noncadence stop at2 followed by continuation produced exactly equal final model and optimizer hashes to uninterrupted execution. A real interrupted second minibatch after one accepted update retained a zero-update resume plus one durable journal row and segment marker; continuation rejected without changing bytes. Default64 is the same branch with a different positive cadence value. No 2048-update campaign was run here.

Cost ruling is reasonable within this contract: cadence changes persistence frequency, not examples, optimizer update count, supervision or model. Using the recorded 18.19 MB model+optimizer pilot snapshot as a lower bound, the 32 scheduled update snapshots per2048-step condition imply about 6.99 GB across12 R5 conditions rather than about447 GB for every-update persistence. Initial, equal-label, final, selected states, retained selected models inside snapshots, history, journal/marker writes and filesystem overhead must be added. These are projected serialized bytes, not measured physical SSD writes.

Operational constraints:

- Never remove markers/journals or restore an older checkpoint to force continuation. Ambiguous interrupted segments require a separately audited attempt, as directed by the parent's brief.
- Freeze only after code edits finish. All three source files are pinned; later runner edits must reject reuse of the campaign.
- Continue using the fixed runtime and explicit CPU-small/CUDA-R5 stage assignments. GPU execution remains parent-serialized and untested by this review.
- Short old pilot projections are not full main wall-time measurements. At review time calibrate() still computes 2048*(update_time + snapshot_time), which is the historical every-update estimate; it does not reflect cadence64. Label it historical/conservative and use a separate corrected estimate, or update it before freeze if it will be reused.
- Per-step wall_seconds is captured before update/timing journal flush and marker cleanup, so it excludes that overhead and marker creation before the timer. invocation_seconds includes total measured invocation setup/finalization; parent process-level elapsed should be used for full campaign cost. In a multi-invocation resume, preserve each invocation duration separately if total elapsed is needed. Do not describe the summed per-step field as complete end-to-end cost.
- notes/training-implementation.md was being revised by the implementer during this review; the version first read still described cadence1. Final documentation should agree with the new ruling. This is a documentation/accounting follow-up, not an unresolved restart-integrity blocker.

## Verification

Independent reruns:

- python -B -m unittest review_cycle.test_training -v: 20 PASS, 14.892s.
- python -B -m unittest tests.test_core_contract -v: 9 PASS, 1.074s.
- Additional original counterexample script: 4 damage cases rejected; unchanged completed call returned without training; all bytes preserved.
- Additional cached metric script: both ordinary and rehashed impossible metric rejected.

Source hashes reviewed:

- review_cycle/training.py: df14af4ec6735abfc3a86b18d830f7682fb63df23d14334e7530b02aae64de0b
- review_cycle/training_data.py: 3095902caa7d79a6f977aa3f599ccd82baf4279ac3a559ef5b8868a1c2ed7c61
- review_cycle/test_training.py: e5059fbd4acaa6b70ed1d3d3d69f84ba91f5cbe81bed55daf99c074bce4de8e2
- core/model.py: 121f11fcb3c1308b7f67d898f2aa09c37226d3b0db27dd4e5bf1ebeaea281f43

Main outcomes, GPU behavior, long1024 evaluation resource costs and final scientific conclusions remain UNMEASURED by this review. Parent retains responsibility for complete-scope gating, raw-evidence independent audit, scheduling and final reporting.
