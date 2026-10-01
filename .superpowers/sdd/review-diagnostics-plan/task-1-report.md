# Task 1 report — paired R2–R5 runner

**Implementation status: fix round 1 implemented; scoped re-review pending. Main experiment status: NOT RUN. Main protocol: NOT FROZEN.**

Owned implementation delivered in `review_cycle/training.py`, `review_cycle/training_data.py`, `review_cycle/test_training.py`; detailed operational notes in `notes/training-implementation.md`. No Git staging or commits. No old evidence edits. No changes to immutable Core source. No subagents.

## Review fix round 1 (current state)

The initial independent review correctly reproduced two missing guarantees: completed results could be silently regenerated or rewound, and cached evaluation metrics were not checked against raw evidence. Both are now fixed in owned source. Parent's explicit resource ruling also changes snapshot cadence from every update to64 while retaining every-update evidence. No GPU/main run or subagent was used for this fix.

- Completed bundles are sealed with `complete.json` covering the exact required initial/resume/selected/last/equal-label, receipt and journal files. `run_one` validates the complete bundle and returns the existing receipt without rewriting bytes. A missing selected file, corrupt last file, changed receipt, or coherent old resume/timing/journal conflicting with terminal evidence is rejected without changing any existing bytes. Incomplete terminal transitions are marked with `finalizing.json` and require audit.
- Cached evaluation metadata is hashed and checked for exact condition/variant/checkpoint and complete length/neutral inventory. Raw source inputs, all targets and IDs are regenerated and compared; accuracies, histograms and paired changes are independently recomputed from raw predictions. Tests also rehash deliberately damaged metadata, proving the raw/identity/inventory checks themselves catch mismatches rather than relying only on the metadata hash.
- Default full snapshot cadence is64, with additional snapshots on a newly selected validation point, equal-label point, orderly stop and finish. Each optimizer update still appends finite loss, gradient, update count, source/input hashes, labels and timing journals. The checkpoint stores its boundary list and accumulated history. Exact clean resume is tested at an off-cadence orderly stop. A persistent uncheckpointed-segment marker plus per-update marker preserves/rejects interrupted segments; no rollback or replay is performed.

Fix-round RED: 4 new behavioral tests produced **11 failing subcases + 2 expected unsupported-cadence errors**, 6.538 s. Failures reproduce valid completed rerun rewriting, four terminal damage/rewind cases, and six cache metric/inventory/identity changes. Cadence tests initially errored because the prior code explicitly refused non-1 cadence. First full GREEN: **29 PASS = 20 training + 9 immutable Core**, 12.653 s. Added a further coherently rehashed wrong transition count and coherently rehashed inventory/identity tests; final combined rerun: **29 PASS**, 15.943 s. Current source SHA256: training.py `df14af4ec6735abfc3a86b18d830f7682fb63df23d14334e7530b02aae64de0b`; training_data.py `3095902caa7d79a6f977aa3f599ccd82baf4279ac3a559ef5b8868a1c2ed7c61`.

The original two pilot JSONs are unchanged. Revised regular snapshot-write projection: width256 model+optimizer18.19 MB × 32 cadence boundaries × 12 R5 runs ≈ **6.99 GB**, compared with prior ~447 GB. Selected state/history, journals and initial/companion/terminal writes add overhead. Default selected-validation boundaries128 align with cadence64. Using prior measured update/snapshot costs gives approximate compute+regular-snapshot estimates **286.66 s CPU32** and **531.85 s CUDA256** per condition, hence ideal small CPU2 queue1.43 h and serial R5 GPU queue1.77 h. These remain incomplete short-pilot estimates, not measured total campaign costs.

## Count and interface

36 unique small conditions verified by enumeration and independent arithmetic: 12 R2 + (18 R3 − 3 shared) + (12 R4 − 3 shared). R5 has 9 development + 3 fresh confirmation runs. APIs implemented: `--calibrate`, `--freeze`, `--run-one`, `--run-selection`, `--evaluate`, `--summarize`. Default stage devices: small CPU, development and confirmation CUDA when supported; an explicit device argument assigns all stages alike. The existing CUDA runtime is DaxDream's `.venv/Scripts/python.exe`; the initially supplied RLT runtime is CPU-only. No environment was modified.

## RED/GREEN evidence

Commands ran from the isolated worktree using `python -B -m unittest ... -v`.

1. Initial eight behavior tests before runner/data implementation: **8 FAIL**, each `AssertionError: unexpectedly None : paired runner is not implemented` (0.004 s). Tests specify independent oracle fixture, histogram/nonneutral-order/final preservation, input/init RNG pairing, 36 conditions and exact label budget, freeze/source drift, A/A real-update resume, incomplete checkpoint rejection, and gated raw matched evaluation.
2. Minimal implementation: **8 PASS**, 3.041 s. Exact uninterrupted/resumed last-state, selected-state, optimizer-state, initial-state and batch hashes agree.
3. Added snapshot-history/selected-weight tamper and actual wall-cost checks: **3 FAIL** (selected corruption accepted, batch receipt corruption accepted, wall cost missing), 4.717 s. Added selected/history hashes and hashed per-update timing ledger; **12 PASS**, 4.692 s. LR selection also ran 9 tiny development and 3 tiny fresh confirmation conditions with validation-only outputs.
4. Added complete-receipt timing-ledger mutation: **1 FAIL**, 1.758 s, because tampering was accepted. Added timing hash validation; **14 PASS**, 4.997 s. Gradient-routing test confirms actual label use for dense/sparse/final.
5. Added frozen stage-device assignment behavior: initial missing API error corrected to an explicit feature-presence assertion; **1 expected FAIL**, `stage device assignment missing`, .014 s. Implemented per-stage execution protocol and receipt device; **15 PASS** under CUDA-capable runtime, 5.635 s. Test execution itself uses bounded CPU fixtures.
6. Added deletion of exact equal-label companion before resume: **1 FAIL**, 1.717 s, because resume was accepted. Added companion provenance/update validation. Final combined command `python -B -m unittest review_cycle.test_training tests.test_core_contract -v`: **25 PASS = 16 runner + 9 immutable Core**, 5.758 s, no errors/warnings.

The preexisting source remained unchanged. Behavioral tests use temporary fresh output directories and real optimizer/inference behavior. No main-campaign artifact was reused as a test result.

## Cost and calibration evidence

- `training-calibration-01/calibration.json`: supplied CPU runtime, seed9117, 8 updates each width32/256, no final data. CPU32 mean .137725 s/update + .009247 s snapshot; CPU256 .355548 + .020654. Preliminary compute+snapshot serial total about 5.58 h for 36+12 conditions, excluding all unmeasured overhead.
- `training-calibration-02/calibration.json`: existing DaxDream CUDA runtime, seed9117, 8 updates each CPU/CUDA × width32/256, no final data. All losses and gradient norms finite, including early high LR losses; negative observations retained.
- Pilot02 CPU32 .139730 s/update + .015570 snapshot = 318.05 s/projected condition. CUDA32 .252657+.011981 = 541.98 s: GPU is slower at small width in this probe. CUDA256 .259227+.029671 = 591.66 s/projected condition. Raw CPU256 row also saved.
- Intended 36 small CPU runs at two processes: ideal 1.59 h; 12 serial GPU large runs: 1.97 h. They exclude validation, hashing, selected-model copies, long-horizon evaluation and contention. These short pilots cannot guarantee total elapsed time.
- CUDA pilot peak allocation: width32 26,729,472 bytes; width256 102,354,432 bytes. These are NOT R6 steady-memory results.
- Pilot model+optimizer checkpoint: width256 ~18.19 MB; width32 ~.33 MB. Main keeps only initial/resume/selected/last/equal-label snapshots, overwriting resume atomically. However R5 lower-bound cumulative writes are ~447 GB across 12 × 2048 updates before selected-state/history overhead. Retained disk size is much smaller. Parent should account for this cost before freeze.

GPU was explicitly coordinated with parent, both pilot calls completed, and GPU was released. No main campaign launched.

## Resume contract and limits

Every accepted update is in an atomic resume checkpoint with optimizer state and validated internal optimizer step. Orderly pause/continue is exact. In-flight marker, temporary write, stale/active lock, missing initial/equal-label evidence, altered selected/history/model/optimizer state, changed source/protocol, and incomplete timing ledger fail closed. An abrupt crash during an update cannot be blindly recovered because whether that update occurred is uncertain; automatic ambiguous-update reconciliation is deliberately not implemented. Evidence remains on disk for audit.

Checkpoint selection is common validation final-state CE, not each arm's different training loss. R5 LR selection uses all three development seeds and only validation. Final CLI evaluation requires every small/confirmation selection complete before test opening; development models are refused. The Python-only partial evaluation API remains explicitly PARTIAL and should not be used as whole-campaign evidence. Results retain raw predictions/targets/IDs, class counts, changes, costs and hashes. Equal-label step64 companions are separately evaluated with unequal updates/inputs explicitly reported.

Historical pre-fix concerns above are retained as an audit trail; the every-update snapshot design has been superseded by cadence64 as described at the top. Current remaining boundaries: exact recovery of an ambiguous interrupted segment is blocked rather than guessed; full evaluation and realistic multi-process contention costs remain unmeasured. Runner tests and timing probes do not establish scientific effect. Parent owns main freeze, scheduling, execution and final audit.
