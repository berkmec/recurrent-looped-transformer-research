# R2–R5 paired training runner implementation

Status: executable runner and bounded development calibration completed; review fix round 1 implemented and awaiting scoped re-review. Main campaign NOT RUN and main protocol NOT FROZEN by this worker. Existing `core/model.py` and `core/run.py` remain unchanged. The sealed 2026-09-16 evidence directory was not edited or used to supply training receipts.

## Scope and condition count

The runner is `review_cycle/training.py`; deterministic input/supervision sources are in `review_cycle/training_data.py`. Behavioral tests are `review_cycle/test_training.py`.

R2 has 2 tasks × 2 arrangements × 3 seeds = 12 conditions. R3 has 2 architectures × 3 supervision types × 3 seeds = 18, sharing 3 random/dense parity conditions with R2. R4 has 2 gradient routes × 2 supervision types × 3 seeds = 12, sharing 3 random/dense five-state conditions with R2. The unique small total is **12 + 18 − 3 + 12 − 3 = 36**. R5 adds 9 development conditions and 3 fresh confirmation conditions, giving 48 training runs.

Default budgets are 2048 updates, batch 32, input length 32. Small models have width 32 / 2 heads, learning rate .003. R5 has width 256 / 4 heads, LR grid .0003/.001/.003. AdamW weight decay .01, norm clip 1, float32. Primary seeds are 117/143/191; confirmation seeds are 217/243/291. All conditions exhaust their update budget unless operational failure prevents completion. A failed result is preserved, not silently replaced.

## Pairing and supervision

Training source identity depends on fixed split namespace, task, seed, update and input length. It does not depend on arrangement, supervision, LR, feedback switch or width. The underlying source uses token probabilities [.4375, .4375, .125]. Each step stores original and effective input hashes, source seed, finite loss, gradient norm, actual optimizer update and labels consumed. Model initialization is separately seeded and leaves global random state unchanged. Same task/seed/width/heads produces identical allocated initial state across compared arms, including no-feedback.

Blocked ordering preserves token counts and the order of all non-2 operations. Token 2 placement cycles prefix/middle/suffix by global example index modulo 3, which balances placements to within one example across the campaign. Middle splits the non-2 sequence at floor(n/2). All prefix oracle states are recomputed; final-state equality is checked for every training minibatch. The independent five-state oracle is covered by a hand-computed fixture.

Sparse supervision picks one uniform position per sequence from its own deterministic split namespace. The source generator and initialization do not consume this RNG. A gradient-routing test checks that dense loss supervises every position and sparse/final losses supervise exactly one position per sequence.

Each dense condition saves the exact step-64 snapshot. Its 64 × 32 × 32 = **65,536 labels** match 2048 × 32 one-target labels. The comparison changes optimizer updates (64 versus 2048), input sequences (2048 versus 65,536), and input tokens (65,536 versus 2,097,152). This is explicitly an equal-label-budget comparison, not an equal-input, equal-update or equal-compute comparison. The companion is checked on continuation and evaluated separately. Default final evaluation has 39 selected checkpoints (36 small + 3 confirmation), plus 21 dense equal-label companions = 60 evaluation variants.

TBPTT8 uses the unchanged Core contract: detach `s`, `dk`, `dv` every eight positions, with global encoder and projected memory routes retained. No claim of fully truncated end-to-end memory or pure activation-memory savings follows from this route.

## Freezing, checkpoint selection and final data

`frozen.json` stores the complete protocol, its canonical hash, hashes of `core/model.py`, `review_cycle/training.py`, `review_cycle/training_data.py`, and runtime metadata. A separate `frozen.sha256` seals the manifest. All operations validate the protocol and source hashes. A new freeze requires an empty output directory; the runner does not import prior campaign receipts.

Default CUDA-capable runtime assignment is `{small: cpu, development: cuda, confirm: cuda}` in the frozen protocol. An explicit `protocol(device='cpu')` or `--freeze --device cpu` assigns all stages to CPU. `execution_device` is recorded in each receipt. CPU intra-op and inter-op threads are both one. The parent scheduler must limit small-model training to two processes and serialize GPU jobs. Per-condition exclusive locks prevent duplicate launches of the same condition; they are not a GPU scheduler.

Validation runs every 128 steps and at the final update. Model selection uses minimum validation **final-state cross entropy**, with earliest step on ties. This is a predeclared common endpoint across supervision arms. R5 chooses the LR with highest mean selected-checkpoint validation final accuracy across all three development seeds; ties choose the smaller LR. Only then are the three fresh confirmation seeds run. Development models cannot access held-out evaluation through the evaluator. No no-feedback architecture-superiority claim is part of R5.

The normal `--evaluate` API checks all 36 small and 3 confirmation receipts and their checkpoint hashes before opening final data. Selection hash identities are written in a test gate and cannot change after opening. A Python-only `evaluate(..., condition_ids=[...])` exists for explicitly scoped diagnostics/tests; it still requires completion for every named condition, refuses development conditions, and writes PARTIAL-scoped results. Do not use this optional subset API as a full-campaign completion claim.

Final sources are fresh fixed namespaces, separate from training, sparse labels, and validation. Final evaluation uses lengths 32/128/1024 with 512 cases each. Matched neutral tests insert 0/8/32/128 token-2 operations into the same length-32 base cases, with prefix/middle/suffix for nonzero counts. Each saved compressed raw file contains inputs, predictions, all prefix targets, input IDs, source seed and condition. Summary rows contain final target/prediction class histograms, correct-to-wrong and wrong-to-correct changes, prediction changes, consumed tokens, batch size, wall cost and raw-file hash. Eval batch is 32 below length 512 and 8 at/above length 512, fixed across arms. On both creation and cache reuse, metadata hash, condition, variant, checkpoint and exact expected row inventory are validated; inputs/targets/IDs are regenerated from frozen sources and compared to raw files. Final/prefix accuracies, class histograms, and paired changes are independently recounted from raw predictions. Loss/timing fields are bound by metadata hash and checked finite/nonnegative; logits are not stored, so CE is not reconstructed from discrete predictions.

## Resume and operational boundaries

`resume.pt` is atomically replaced at every **64 updates**, every newly validation-selected point, the exact equal-label point, orderly stop, and finish. It includes model state, full optimizer state, selected model/step/validation, accumulated history, input/label receipts, hashes, provenance, update counter and actual checkpoint boundary list. Optimizer-internal step values must match the accepted update count. Every update still writes durable finite loss/gradient/count/source/input/label records in `updates.jsonl` and a timing entry in `timing.jsonl`. At any resumable boundary those journals must match the snapshot history/count exactly; a journal tail cannot be silently rolled back or replayed.

`initial.pt`, `selected.pt`, `last.pt` and `equal-label.pt` supply audit evidence. Checkpoints do not accumulate once per step; the resume file is overwritten atomically. `wall_seconds` includes snapshot I/O on checkpoint updates, whereas `train_seconds` is the compute subtotal. Timing is captured before appending the journal rows, so per-row fsync and finalization overhead are additional; `invocation_seconds` measures one invocation's setup/finalization and journal overhead as well.

Completed conditions have a `complete.json` seal covering the exact required checkpoint/receipt/journal bundle. A repeated `run_one` validates that bundle, checks terminal and resume state equality and journal history, and returns the existing receipt without changing file bytes. Missing selected files, corrupted last snapshots, changed receipts, or a coherent older resume/timing/journal paired with terminal evidence fail closed. `finalizing.json` marks the terminal transition; an interrupted terminal transition requires explicit audit, never routine regeneration.

Orderly `run_one(out, id, stop_after=k)` then `run_one(out, id)` continuation is tested to produce exact matching final and selected states, optimizer hash, initial hash and minibatch identities versus an uninterrupted run. The CLI normally runs a whole condition; the optional stop parameter is Python-only.

**Fail-closed boundary:** `uncheckpointed.json` is created before the first update after a checkpoint and remains until the next snapshot and both journals are durably consistent. An abrupt stop anywhere in that uncheckpointed segment is rejected. Per-update `inflight.json`, temporary write files, finalization markers and exclusive locks likewise block continuation. The runner does not guess whether an interrupted update happened, discard a journal tail, or replay an already executed segment. It also rejects missing initial/companion evidence, altered selected state/history, changed optimizer step, missing ledger entries, source drift and mismatched protocol. An audit/recovery procedure for an ambiguous interrupted segment is not implemented; the parent must not delete its evidence to force continuation. A documented fresh attempt would require separate audit.

## Verified tests and bounded cost pilot

2026-09-17 review-fix check in the existing CUDA-capable DaxDream environment: **20 runner tests + 9 immutable Core tests = 29 PASS** (first fix-round combined run 12.653 s). Tests exercise real optimizer updates, source files, output files and inference rather than matching source text. New checks reproduce all four completed-evidence defects, preserve file bytes on valid rerun, reject cached metadata/inventory/raw-metric inconsistencies, prove exact non-cadence orderly continuation, and simulate interruption after an uncheckpointed real update. RED/GREEN evidence is preserved in the task report.

First calibration used the initially supplied CPU-only environment. The second used the existing `C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe` with torch 2.12.0+cu130 and RTX 2060 SUPER, without changing either environment. Pilot seed 9117 is separate from development/confirmation seeds. Each device/width pair ran eight real optimizer updates; no final data was opened. Both calibration directories are preserved under `.superpowers/sdd/review-diagnostics-plan/`.

The following retained original pilot table used the former every-update snapshot projection; it is historical and superseded by the cadence-64 projection below. Raw pilot observations were not changed.

| Pilot 02 device/width | Mean update seconds, steps 2–8 | One snapshot seconds | Historical every-update projection |
|---|---:|---:|---:|
| CPU / 32 | 0.139730 | 0.015570 | 318.05 s |
| CUDA / 32 | 0.252657 | 0.011981 | 541.98 s |
| CUDA / 256 | 0.259227 | 0.029671 | 591.66 s |

These are short development timing probes, not main-run throughput claims. The CPU256 row is also retained in raw calibration JSON. With the new cadence, use `2048 × measured update time + 32 × one snapshot time`: approximately **286.66 s** per small CPU condition and **531.85 s** per large GPU condition. For the intended assignment, 36 small runs at two CPU processes ideally require about **1.43 h**, and 12 large GPU runs sequentially about **1.77 h**. This estimate excludes validation, hash computation, selected-state serialization overhead, per-update journal/fsync, initial/companion/terminal writes, final evaluation and resource contention. Running the queues concurrently does not make these independent measurements a guaranteed wall-clock bound. Long-horizon evaluation cost remains UNMEASURED. No new timing pilot was needed for the arithmetic projection.

Width256 pilot model+optimizer checkpoint is about 18.19 MB (decimal); width32 about .33 MB. A real resume snapshot adds selected state/history, so it is larger. The parent rejected the former every-update design's ~447 GB R5 lower-bound write projection. Cadence64 reduces the regular resume-write component to **18.19 MB × 32 × 12 ≈ 6.99 GB**. Initial, equal-label, terminal and selected-model files/history/journals add overhead; orderly stops or extra selected points can add writes. Default validation128 aligns with cadence64, so selected points add no extra regular snapshot boundaries. Retained disk count remains constant. Data, steps, losses, models and scientific seeds were unchanged by this resource decision.

## Run interface

From `E:/rlt-research/2026-09-17-diagnostics`, use the same CUDA-enabled DaxDream Python runtime for both CPU and CUDA stages:

```text
python -B -m unittest review_cycle.test_training tests.test_core_contract -v
python -B -m review_cycle.training --calibrate --pilot-updates 8 --out <fresh-pilot-directory>
python -B -m review_cycle.training --freeze --out <empty-campaign-directory>
python -B -m review_cycle.training --run-one <condition-id> --out <campaign-directory>
python -B -m review_cycle.training --run-selection --out <campaign-directory>
python -B -m review_cycle.training --evaluate --out <campaign-directory>
python -B -m review_cycle.training --summarize --out <campaign-directory>
```

Condition IDs come from `ident(c)` for `c in conditions(protocol())`. `--run-selection` sequentially executes all nine development runs, seals LR choice, then executes three confirmation runs. The parent owns scheduling, the main freeze, main execution, independent audit and final scientific interpretation.
