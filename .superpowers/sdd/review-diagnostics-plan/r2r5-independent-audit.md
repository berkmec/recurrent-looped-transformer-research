# R2–R5 paired diagnostics — independent raw audit

**Verdict: PASS — 24 checks PASS, 0 FAIL.** Every primary metric, every one of the 780 evaluation endpoints (60 variants × 13), all 71 prespecified contrasts and the 3 R5 confirmation rows were recomputed from the raw `.npz`/`.pt`/`.jsonl` files and agree with the producer files to the last bit (max |Δ| = 0.0 on every accuracy, difference, mean and interval bound).

| | |
|---|---|
| Worktree | `E:\rlt-research\2026-09-17-diagnostics` (branch `codex/rlt-review-diagnostics`) |
| Campaign root | `results/training/` — 48 condition directories + campaign JSON |
| Auditor | `r2r5_raw_audit.py` (this directory); read-only; CPU; `torch.set_num_threads(1)`; **no producer module imported, no model forward pass, no training/evaluation/GPU work**; one checkpoint at a time, npz loaded lazily |
| Runtime | Python 3.11.9, torch 2.12.0+cu130, numpy 2.3.5 — 40.9 s wall |
| Numbers | `r2r5-independent-audit.json` (all values below and per-condition/per-endpoint detail) |

Independent re-implementations used for recomputation (verified against the raw files, not trusted a priori): sha256 / canonical-JSON digest / producer `state_hash` algorithm; the 3-token source generator (`default_rng(seed).choice(3, p=[.4375,.4375,.125])`); the parity and five-state oracles written from the task semantics (token 0 → s+1 mod 5, token 1 → swap 0↔1, token 2 → identity); split seeds `sha256("{base}|{task}|{seed}|{step}|{length}")[:8]`; blocked rearrangement (global example index mod 3 → prefix / floor(n/2) / suffix of the non-neutral sequence); neutral insertion (prefix / floor(32/2)=16 / suffix); sparse positions; paired accuracy; the three-seed t-approximation with coefficient 4.302652729 (recomputed from the closed-form df=2 quantile: 4.3026527297).

## 1. Pins

| Object | Value | Result |
|---|---|---|
| `frozen.json` sha256 vs `frozen.sha256` | `7eaf57b7…7b464` | PASS |
| `protocol_hash` = sha256(canonical JSON of `protocol`, sort_keys, separators `,`/`:`) | `eed6fb30b7d6808c…ac56d836c` | PASS |
| `core/model.py` | `121f11fc…281f43` = frozen pin | PASS |
| `review_cycle/training.py` | `df14af4e…64de0b` = frozen pin | PASS |
| `review_cycle/training_data.py` | `3095902c…2ed7c61` = frozen pin | PASS |
| `docs/training-analysis-plan.md` | `1949d8d7ceda69a4f1a3bf8bf6b9b7ff9460e0f45d1ff6c3f7a7acc7ce421030` = ledger pin (progress.md line 46, recorded before the line-71 completion entry) = `scientific-analysis.json.analysis_plan_sha256` | PASS |
| `selection.json` selection_hash | `cbd25c7a…6d2ec` recomputes | PASS |
| Producer files hashed | evaluation.json `8fd839a3…`, test-gate.json `d152d564…`, selection.json `1a6a14c0…`, summary.json `d0c5ec6a…`, scientific-analysis.json `f5cb572d…` | recorded |

## 2. Check table

| # | Check | Key numbers | Result |
|---|---|---|---|
| 1 | frozen.sha256 vs frozen.json | equal | PASS |
| 1 | protocol_hash canonical JSON | equal, prefix `eed6fb30` | PASS |
| 1 | source pins match current files | 3/3 | PASS |
| 1 | analysis-plan pin | current hash = pin, ledger line 46 | PASS |
| 1 | protocol values as stated | steps 2048, batch 32, L 32, w32/h2 & w256/h4, LR grid {3e-4,1e-3,3e-3}, seeds 117/143/191 & 217/243/291, val 256 every 128, ckpt 64, equal-label step 64, test 512 × {32,128,1024}, neutral {0,8,32,128}, eval batch 32 / 8 (≥512), threads 1, float32 | PASS |
| 1 | workers: no FAILED, STARTED…COMPLETE | 5 event logs (small-0, small-1, one-dev-s117, selection, evaluate); 0 FAILED; 37 CONDITION_COMPLETE events (36 small + 1 dev), all `updates: 2048` | PASS |
| 1 | stderr logs empty, exit codes | 5/5 stderr logs 0 bytes; launch-02/03 logs `exited code=0` | PASS |
| 1 | selection.json | hash recomputes; mean selected-validation final accuracy by LR: 3e-4 → 1.0 (1.0/1.0/1.0), 1e-3 → 1.0 (1.0/1.0/1.0), 3e-3 → 0.5247 (0.4922/0.5508/0.5312); rule "highest mean, lowest LR on ties" → **3e-4** = declared; 9 development hashes = receipts; `final_data_opened: false` | PASS |
| 1 | receipts / seals / checkpoint hashes (48) | every `complete.json` file map recomputes (7 or 8 files), every seal_hash recomputes, `train.json` protocol/sources/condition match, `checkpoint_hashes` = seal = directory `*.pt` set; no stray lock/inflight/uncheckpointed/finalizing/tmp/failure files | PASS |
| 1 | no development evaluation files | 9/9 development dirs contain no `*-evaluation.json` / `*.npz` | PASS |
| 1 | summary.json | 48 rows, 48 COMPLETE, each row ≡ its train.json | PASS |
| 1 | test-gate.json | COMPLETE_SCOPE, 39 selections = 36 small + 3 confirm, each = receipt `selected_state_hash` = `state_hash(selected.pt)` | PASS |
| 2 | directory names parse & round-trip | 48/48 | PASS |
| 2 | condition census | small 36 (R2 12 + R3 18 + R4 12 − 6 shared rlt-random-dense), development 9 (3 LR × 3 seeds), confirm 3 at LR 3e-4 seeds 217/243/291; 0 missing, 0 unexpected | PASS |
| 3 | real training | 48/48: `updates.jsonl` 2048 rows, step = real_update = 1…2048, all loss/grad_norm finite; receipt `updates`=`finite_updates`=`timed_updates`=2048; labels 2,097,152 (dense, 21 cond.) / 65,536 (sparse+final, 27 cond.) both from receipts and from per-row sums; every row's `source_seed`, `base_input_hash`, `input_hash` (incl. blocked rearrangement) and `sparse_position_hash` regenerated exactly; `last.pt` and `resume.pt`: updates 2048, **AdamW `step` = 2048 in every optimizer entry**, model/optimizer/history/selected hashes recompute and equal the receipt, history ≡ journal; `equal-label.pt` is the exact step-64 snapshot (updates 64, AdamW step 64, history = first 64 journal rows); `initial.pt` hash = receipt | PASS |
| 3 | parameters changed | rlt: 35/35 tensors changed (24,226 / 24,325 / 1,512,706 params); nofeedback: 30/35 changed, exactly {`initial`,`pn.weight`,`gate.weight`,`gate.bias`,`merge.weight`} unchanged and only 30 optimizer entries (feedback route out of the graph → grad None → AdamW skips); tbptt8-dense: 35/35; **tbptt8-final: 34/35 — `initial` unchanged although it has an AdamW entry with step 2048: its moments are exactly zero (the last-position loss has no path to the t=0 state across the truncation; autograd still delivers a zero gradient via the stacked outputs; zero-init × decoupled weight decay stays 0)** | PASS |
| 3 | selected_step & selection rule | for all 48, `selected_step` = earliest argmin of validation `final_loss` over the 16 validation rows (steps 128…2048) and `selected_validation` equals that row; `selected == last` iff selected_step == 2048; 23 conditions selected before 2048 (listed in §7) | PASS |
| 4 | pairing | every group shares one `initial_state_hash` and the identical 2048-step `base_input_hash` sequence: parity s117/s143/s191 (rlt/nofeedback × dense/sparse/final + rlt-blocked), five s117/s143/s191 (rlt/tbptt8 × dense/final + rlt-blocked), development s117/s143/s191 (3 LRs); random arms have `input_hash == base_input_hash`; blocked arms differ from base at 2048/2048 steps while the recomputed blocked hash matches and the final target is unchanged; sparse arms share all 2048 `sparse_position_hash`; development w256 arms share the training inputs of the small parity arms of the same seed but have a different initial state | PASS |
| 4 | distinct seeds | 3 distinct initial states / first-step inputs across 117/143/191; 3 distinct across 217/243/291 | PASS |
| 5 | evaluation recomputation | 60 variants (39 selected + 21 equal-label), 780 endpoints: raw file sha256 = row sha256; `raw_file` name; `metadata_hash` recomputes; `checkpoint_sha256` = sha256(selected.pt / equal-label.pt); inputs ≡ regenerated declared source; targets ≡ independent oracle; `input_ids` = sha256(f"{seed}:{i}"); `source_seed`/`condition` strings; prediction dtype/range; n=512, consumed_tokens, input_hash, batch (8 for L1024 else 32), target & prediction histograms, final/prefix accuracy (max |Δ| 0.0), and for slots>0 correct_to_wrong / wrong_to_correct / prediction_changes vs the same variant's neutral0-base; neutral files = base inputs with the declared number of token-2 insertions at prefix / index 16 / suffix and identical final targets; `evaluation.json` entry ≡ per-condition file for all 60 | PASS |
| 5 | shared source | inputs/targets/ids identical across all variants of the same (task, seed, endpoint): 117 groups; **not** identical across seeds — by construction the source seed includes the training seed, and the plan requires pairing within a seed | PASS (note) |
| 5 | heldout ∩ training = ∅ | for each of the 9 evaluated (task, seed) streams: 65,536 training rows (all unique), 512 test-L32 rows, 512 neutral-base rows, 256 validation rows — 0 overlaps in every pairwise comparison | PASS |
| 6 | 71 contrasts | file set == prespecified set (R2 26, R3 39, R4 6), same order, 13 primary; per seed inputs/targets/ids asserted identical before differencing; n, candidate/baseline correct, gained, lost, difference, mean, t95 bounds and direction all recomputed with max |Δ| 0.0; `analysis_plan_sha256` = pin, selected_lr 3e-4, 48 complete, 39 evaluated | PASS |
| 6 | R5 confirmation | 3 rows × 13 endpoints: n, correct, accuracy, prediction/target histograms, observed_constant_prediction, selected_step, selected_lr — identical | PASS |
| 7 | statistical scope | no contrast pools seeds: every per-seed row has n=512, `seeds`=3, 3 seed effects; interval = mean ± 4.302652729·stdev(3 effects)/√3, i.e. df=2 (coefficient reproduced from closed form to 1e-9); `interval_scope` string present on all 71 | PASS |

## 3. Primary contrasts (candidate − baseline, final-state accuracy, percentage points)

| ID | Task | Contrast | Endpoint | Seeds 117 / 143 / 191 | Mean | ≈t95 (df=2) | Direction |
|---|---|---|---|---|---|---|---|
| R2 | parity | blocked − random | neutral-128-suffix | −44.73 / +0.20 / −3.32 | **−15.95** | [−78.01, +46.11] | INCLUDES_ZERO |
| R2 | five | blocked − random | neutral-128-suffix | +56.05 / −16.60 / +52.73 | **+30.73** | [−71.18, +132.64] | INCLUDES_ZERO |
| R3 | parity | rlt final − sparse | L128 | −54.88 / −53.71 / −54.30 | **−54.30** | [−55.75, −52.84] | NEGATIVE_INTERVAL |
| R3 | parity | rlt sparse − dense | L128 | 0 / 0 / 0 | **0.00** | [0, 0] | INCLUDES_ZERO |
| R3 | parity | rlt final − dense | L128 | −54.88 / −53.71 / −54.30 | **−54.30** | [−55.75, −52.84] | NEGATIVE_INTERVAL |
| R3 | parity | nofeedback final − sparse | L128 | −1.56 / 0 / −8.59 | **−3.39** | [−14.76, +7.99] | INCLUDES_ZERO |
| R3 | parity | nofeedback sparse − dense | L128 | +2.73 / −2.34 / +8.98 | **+3.13** | [−10.97, +17.22] | INCLUDES_ZERO |
| R3 | parity | nofeedback final − dense | L128 | +1.17 / −2.34 / +0.39 | **−0.26** | [−4.85, +4.33] | INCLUDES_ZERO |
| R3 | parity | rlt − nofeedback (dense) | L128 | +50.78 / +51.37 / +54.69 | **+52.28** | [+47.05, +57.51] | POSITIVE_INTERVAL |
| R3 | parity | rlt − nofeedback (sparse) | L128 | +48.05 / +53.71 / +45.70 | **+49.15** | [+38.93, +59.38] | POSITIVE_INTERVAL |
| R3 | parity | rlt − nofeedback (final) | L128 | −5.27 / 0 / 0 | **−1.76** | [−9.32, +5.81] | INCLUDES_ZERO |
| R4 | five | full − tbptt8 (dense) | L128 | +0.78 / −0.78 / 0 | **0.00** | [−1.94, +1.94] | INCLUDES_ZERO |
| R4 | five | full − tbptt8 (final) | L128 | 0 / 0 / 0 | **0.00** | [0, 0] | INCLUDES_ZERO |

All 71 contrasts (including the 58 secondary ones) are in the JSON under `contrasts.all`. Non-primary intervals excluding zero: R2 five L32 (−2.73 [−5.30, −0.17]), R2 five L1024 (−54.43 [−94.31, −14.54]), R2 five neutral-32-middle (+39.84 [+16.01, +63.68]); R3 rlt final−sparse and final−dense at L32/L1024 (≈ −49 to −50); R3 rlt sparse2048−dense64 at L32/L128/L1024 (+48.6 / +52.5 / +49.4, equal-label baseline at step 64 — different updates/inputs/compute, as caveated in the plan); R3 rlt−nofeedback dense and sparse at L32/L1024 (+48.7 to +52.6).

## 4. R5 confirmation (parity, w256, LR 3e-4 selected on validation only; selected checkpoints)

| Seed | selected_step | L32 | L128 | L1024 | neutral-128 prefix | middle | suffix |
|---|---|---|---|---|---|---|---|
| 217 | 2048 | 1.0000 (285/227) | 1.0000 (274/238) | 1.0000 (238/274) | 0.9316 (pred 259/253) | 0.5020 (279/233) | **0.4824 (pred 7/505; target 272/240)** |
| 243 | 2048 | 1.0000 (270/242) | 1.0000 (260/252) | 1.0000 (268/244) | 1.0000 | 0.9980 (271/241) | 1.0000 |
| 291 | 2048 | 1.0000 (254/258) | 1.0000 (252/260) | 1.0000 (248/264) | 1.0000 | 0.9961 (251/261) | 1.0000 |

Prediction histograms equal the target histograms at every length endpoint; `observed_constant_prediction` is false for all 39 rows. Seeds 243 and 291 also hold ≥0.998 at neutral-8/32 (all placements). Seed 217 degrades under long neutral runs: neutral-32-middle 0.9648, neutral-128-prefix 0.9316, neutral-128-middle 0.5020, neutral-128-suffix 0.4824 with 505/512 predictions in class 1 (correct_to_wrong 265, wrong_to_correct 0) — not constant by the producer's strict definition, but a near-collapse in one of three confirmation seeds. The equal-label (step-64) variants of the three confirmation models are at chance (L32 0.47–0.50).

## 5. Statistical scope

- Nothing pools seeds or neutral variations as independent examples: each contrast is three paired per-seed differences on the same 512 heldout problems, summarised by mean ± 4.302652729·s/√3 (t with df=2), exactly as prespecified; no multiplicity adjustment; 13 primary endpoints declared before the data. `INCLUDES_ZERO` is reported as direction uncertainty, not as evidence of no effect; where all three seed effects are exactly 0 the interval is degenerate [0, 0].
- L1024 evaluation batches are 8 (protocol `eval_long_threshold` 512, `eval_batch_long` 8) versus 32 for L32/L128 and all neutral endpoints; the rows declare this and it was verified as a declared field only. Batching should not change predictions in principle (per-example forward, no batch statistics), but **this audit did not re-verify batch-invariance** because no model was run.
- Contrasts with identical `INCLUDES_ZERO` verdicts differ greatly in information: the R3 final-supervision and R4 final-supervision rows compare two models that both failed to learn (see §7), so their zero differences are not evidence about supervision position or truncation.

## 6. Pairing and source summary

- Same-seed arms are exactly paired: one initial state per (task, seed[, width]) group, identical original training inputs at all 2048 updates, identical sparse label positions, and — for blocked — identical original inputs with a different effective input at every step, recomputed and matched. Heldout test/neutral/validation inputs are identical across all arms and variants of the same task and seed, are the declared fresh sources, and share no length-32 row with the 65,536 training rows or with each other.
- Development (w256) and confirm conditions are not paired with small ones (different width; confirm seeds are fresh). The three development LR arms of a seed are paired with each other.

## 7. Anomalies and observations (none is an integrity failure)

1. **selected_step ≠ 2048 in 23/48 conditions** (all consistent with the earliest-argmin rule): development LR 3e-3 s117/s143/s191 (1536/512/512, validation accuracy 0.49/0.55/0.53 — this LR collapsed and was correctly not selected); five rlt-blocked s117/s143/s191 (1664/640/768); five rlt-random-dense s143 (1408); five rlt-final s117/s191 (1024/1920); five tbptt8-dense s117 (1152); five tbptt8-final s117/s191 (1024/1920); parity nofeedback dense s117/s143/s191 (256/896/1408), final s117/s143/s191 (128/1280/768), sparse s117/s143 (1920/512); parity rlt-final s117/s143/s191 (896/1280/768).
2. **Constant-class predictions: 223 of 780 endpoints in 37 of 60 variants.** They are *not* confined to equal-label variants: **163 endpoints are in 18 selected models** — all six five-final arms (rlt and tbptt8, 13/13 endpoints each, all-class-0 or all-class-1), parity nofeedback-final s143/s191 and rlt-final s143/s191 (13/13 each), nofeedback-final s117 (10), nofeedback-sparse s143/s191 (7 each), nofeedback-sparse s117, nofeedback-dense s117/s191, rlt-final s117 (2 each), tbptt8-dense s143 (1, neutral-128-suffix) — and **60 endpoints are in 19 equal-label (step-64) variants** (0–9 per variant, mostly neutral-128 and suffix placements, including the three confirmation models).
3. **Final-supervision arms did not learn within the budget**: parity rlt-final and nofeedback-final have selected validation accuracy 0.49–0.55 and test accuracies 0.45–0.54 at every endpoint; all six five-final arms are at chance (0.15–0.22, chance 0.20) with a constant class at all 13 endpoints. For each seed the paired five rlt-final and tbptt8-final arms show the same selected_step (1024/2048/1920), the same validation accuracy (0.2344/0.2266/0.2070), the same first/last journal loss to three decimals and the same accuracy at every endpoint, while their selected_state_hashes differ. Consequently the R3 "final" contrasts (−54 pp vs sparse/dense) measure "learned vs not learned", and rlt−nofeedback (final) and R4 full−tbptt8 (final) compare two collapsed models.
4. **Equal-label (step-64) variants are at chance everywhere** (parity 0.44–0.55, five 0.17–0.24); the "sparse2048/final2048 − dense64" rows therefore reflect training progress at 64 updates, as the plan caveats.
5. **R2 heterogeneity**: parity blocked−random at neutral-128 is dominated by seed 117 (−44.7 pp suffix, −48.6 middle) with ≈0 for s143; five blocked−random at neutral-128-suffix is +56/−17/+53. Five rlt-blocked s143 collapses under long neutral runs (neutral-128 middle 0.180, suffix 0.197, correct_to_wrong 391) and all five blocked arms lose length generalisation (L1024 0.33–0.64 vs 0.98–1.0 random).
6. **R5 seed 217** near-collapse at neutral-128 middle/suffix (0.502 / 0.482, 505/512 in one class) while L32/L128/L1024 are perfect; seeds 243/291 are robust.
7. Pre-clip gradient norms reach 2002.7 (parity rlt-blocked s143), 1664.1 (five rlt-random-dense s191), 647.0 (five rlt-blocked s143); all finite, clipped to 1.0, training completed.
8. nofeedback arms keep 5 tensors untouched and carry 30 optimizer entries; tbptt8-final arms keep `initial` at exactly zero with zero AdamW moments (explained in §2) — both consistent with the declared ablations, the latter meaning the learned initial state is untrainable under tbptt8 + final supervision.
9. Wall time: small CPU conditions 266–400 s (nofeedback 43–46 s: the no-feedback route runs the parallel decoder); w256 CUDA conditions 536–699 s.

## 8. Files written

- `.superpowers/sdd/review-diagnostics-plan/r2r5_raw_audit.py` — the audit script (the version on disk is the one that produced the JSON).
- `.superpowers/sdd/review-diagnostics-plan/r2r5-independent-audit.json` — all numbers (checks, pins, 48 condition receipts, pairing groups, 780 endpoint recomputations, source identity, 71 contrasts, R5 rows, anomaly list).
- `.superpowers/sdd/review-diagnostics-plan/r2r5-independent-audit.md` — this report.

Nothing under `results/`, `review_cycle/`, `core/`, `notes/` or `docs/` was modified.
