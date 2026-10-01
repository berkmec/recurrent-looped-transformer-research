# Drift mini-experiment - independent raw audit

Root `E:\rlt-research\2026-09-18-drift` (branch codex/rlt-drift-mini, base f83b527). Read-only; no training or evaluation executed; CPU, 1 thread; torch 2.12.0+cu130, numpy 2.3.5. Audit runtime 6 s.

**Verdict: PASS** - 42 PASS / 0 FAIL over 42 checks.

Frozen protocol_hash `1496384366fc62f96da0d84abee3070c1d452d57f09017fbbd661890051a05e6`; frozen.json sha256 `6558c3d5e5535c6b02be6d2dc7c7601f0a1e80a9f869bc54a1c62ab71b365e61`; plan sha256 `bca4c455af2d684a5a48422bc170260fb53909af7b573c1a9df663b663895162`.

## Decisions (recomputed from raw NPZ, arm minus base final accuracy, pp)

| task/arm | status | n128-middle mean [t95] | n128-suffix mean [t95] | L1024 cost mean [t95] | seeds | negative trigger |
|---|---|---|---|---|---|---|
| five/win16 | MEASURED_NEGATIVE | -7.75 [-43.5, +28.0] INCLUDES_ZERO | -13.87 [-35.8, +8.1] INCLUDES_ZERO | -7.16 [-24.7, +10.4] | 3 | primary mean <= 0 |
| five/idinv | MEASURED_NEGATIVE | -1.63 [-49.4, +46.1] INCLUDES_ZERO | -6.38 [-59.8, +47.0] INCLUDES_ZERO | -8.14 [-26.9, +10.6] | 3 | primary mean <= 0 |
| five/reread | MEASURED_NEGATIVE (INCOMPLETE) | -11.43 [-171.5, +148.6] INCLUDES_ZERO | -18.65 [-39.7, +2.4] INCLUDES_ZERO | +1.17 [-11.2, +13.6] | 2 | primary mean <= 0 |
| parity/win16 | MEASURED_NEGATIVE | -45.77 [-53.1, -38.5] NEGATIVE_INTERVAL | -49.28 [-57.3, -41.3] NEGATIVE_INTERVAL | -6.25 [-27.7, +15.2] | 3 | primary mean <= 0 |
| parity/idinv | MEASURED_NEGATIVE | -2.15 [-26.6, +22.3] INCLUDES_ZERO | +0.07 [-0.2, +0.3] INCLUDES_ZERO | +0.00 [+0.0, +0.0] | 3 | primary mean <= 0 |
| parity/reread | MEASURED_NEGATIVE (INCOMPLETE) | -1.56 [-21.4, +18.3] INCLUDES_ZERO | +0.10 [-1.1, +1.3] INCLUDES_ZERO | +0.00 [+0.0, +0.0] | 2 | primary mean <= 0 |

Per-seed primary effects (pp): five/win16/neutral-128-middle: [3.32, -24.02, -2.54]; five/win16/neutral-128-suffix: [-20.7, -16.99, -3.91]; five/idinv/neutral-128-middle: [2.73, -22.66, 15.04]; five/idinv/neutral-128-suffix: [-20.51, -16.99, 18.36]; five/reread/neutral-128-middle: [1.17, -24.02]; five/reread/neutral-128-suffix: [-20.31, -16.99].

## Failures (reread arm, seed 191)

- `drift-five-reread-s191`: failure-step1632.json step 1632 = last journal step 1631 + 1 (1631 rows); loss at failure 1.8577277660369873, grad_norm inf; last finite grad_norms [(1626, 4.081), (1627, 3.514), (1628, 942.8), (1629, 381.9), (1630, 141500.0), (1631, 1477000000000.0)].
- `drift-parity-reread-s191`: failure-step222.json step 222 = last journal step 221 + 1 (221 rows); loss at failure 0.6991235613822937, grad_norm inf; last finite grad_norms [(216, 0.1522), (217, 0.6406), (218, 4.944), (219, 54.04), (220, 198000.0), (221, 125300000.0)].

## Base reproduction vs sealed diagnostics (2026-09-17)

| drift base | sealed dir | inputs | targets | predictions (all positions) | selected step (drift, sealed) |
|---|---|---|---|---|---|
| drift-five-base-s117 | small-five-w32-rlt-random-dense-lr0.003-s117 | 13/13 | 13/13 | 13/13 | (2048, 2048) |
| drift-five-base-s143 | small-five-w32-rlt-random-dense-lr0.003-s143 | 13/13 | 13/13 | 13/13 | (1408, 1408) |
| drift-five-base-s191 | small-five-w32-rlt-random-dense-lr0.003-s191 | 13/13 | 13/13 | 13/13 | (2048, 2048) |
| drift-parity-base-s117 | small-parity-w32-rlt-random-dense-lr0.003-s117 | 13/13 | 13/13 | 13/13 | (2048, 2048) |
| drift-parity-base-s143 | small-parity-w32-rlt-random-dense-lr0.003-s143 | 13/13 | 13/13 | 13/13 | (2048, 2048) |
| drift-parity-base-s191 | small-parity-w32-rlt-random-dense-lr0.003-s191 | 13/13 | 13/13 | 13/13 | (2048, 2048) |

## Checks


### 1-integrity

- **PASS** frozen.sha256 matches sha256(frozen.json) - 6558c3d5e5535c6b02be6d2dc7c7601f0a1e80a9f869bc54a1c62ab71b365e61 vs sidecar 6558c3d5e5535c6b02be6d2dc7c7601f0a1e80a9f869bc54a1c62ab71b365e61
- **PASS** protocol_hash recomputed from canonical JSON of frozen protocol - 1496384366fc62f96da0d84abee3070c1d452d57f09017fbbd661890051a05e6
- **PASS** protocol() in current drift/run.py digests to the frozen protocol_hash - 1496384366fc62f96da0d84abee3070c1d452d57f09017fbbd661890051a05e6
- **PASS** source pin core/model.py - 121f11fcb3c1308b7f67d898f2aa09c37226d3b0db27dd4e5bf1ebeaea281f43 vs pinned 121f11fcb3c1308b7f67d898f2aa09c37226d3b0db27dd4e5bf1ebeaea281f43
- **PASS** source pin review_cycle/training_data.py - 3095902caa7d79a6f977aa3f599ccd82baf4279ac3a559ef5b8868a1c2ed7c61 vs pinned 3095902caa7d79a6f977aa3f599ccd82baf4279ac3a559ef5b8868a1c2ed7c61
- **PASS** source pin drift/model.py - f03ed9cecc1a9f51c98f8a4fb05d39e62c359df2699ca53d458d636e3491a931 vs pinned f03ed9cecc1a9f51c98f8a4fb05d39e62c359df2699ca53d458d636e3491a931
- **PASS** source pin drift/run.py - f7f6cb0659c50fa62908da26f65c70ea70abbed453d9e261ef0b25952ae3f4c4 vs pinned f7f6cb0659c50fa62908da26f65c70ea70abbed453d9e261ef0b25952ae3f4c4
- **PASS** frozen manifest pins exactly the four declared sources - ['core/model.py', 'drift/model.py', 'drift/run.py', 'review_cycle/training_data.py']
- **PASS** core/model.py and review_cycle/training_data.py byte-identical to base commit f83b527 - core/model.py: worktree blob a664a3e39011 vs f83b527 a664a3e39011; review_cycle/training_data.py: worktree blob 344f68524610 vs f83b527 344f68524610
- **PASS** plan sha256 matches ledger record and analysis-tolerant.json - bca4c455af2d684a5a48422bc170260fb53909af7b573c1a9df663b663895162
- **PASS** test file sha256 matches ledger record - 81e02d0db45c3a8762c35f3c4f07b475ba1ce9a5284d2d5cad7853888628cb65
- **PASS** results/drift inventory is exactly 24 condition dirs + frozen.json/.sha256 + analysis-tolerant.json + processes/ + workers/ - extra=[] missing=[]
- **PASS** 22 COMPLETE + 2 FAILED conditions, both failures are reread seed 191 - complete=22 failed=['drift-five-reread-s191', 'drift-parity-reread-s191']
- **PASS** every COMPLETE dir holds exactly the 20 expected files (no lock/tmp leftovers) - 22/22
- **PASS** FAILED dirs hold only failure-step*.json + initial.pt + updates.jsonl (no evaluation/NPZ/checkpoints/seal) - drift-five-reread-s191: ['failure-step1632.json', 'initial.pt', 'updates.jsonl']; drift-parity-reread-s191: ['failure-step222.json', 'initial.pt', 'updates.jsonl']
- **PASS** train.json / evaluation.json / complete.json carry the frozen protocol_hash, frozen sources and the declared condition (22 conditions) - 22/22
- **PASS** complete.json: all 5 file sha256 recomputed and seal_hash recomputed (22 x 5 files) - 110/110 file hashes, 22/22 seal hashes
- **PASS** evaluation.json: 3+10 rows, raw NPZ sha256 recomputed, checkpoint_sha256 == sha256(selected.pt), metadata_hash recomputed - 286/286 raw hashes match
- **PASS** worker logs: 24 distinct CONDITION_STARTED, 22 CONDITION_COMPLETE == the 22 COMPLETE dirs - started=24 complete=22
- **PASS** worker logs: FAILED rows - FAILED rows=['drift-parity-reread-s191']; drift-five-reread-s191 has CONDITION_STARTED in slot-2.jsonl with no terminal row (worker crashed) - its failure is evidenced by failure-step1632.json and slot-2.stderr.log, not by a FAILED log row
- **PASS** worker logs: no states other than STARTED/CONDITION_STARTED/CONDITION_COMPLETE/FAILED/COMPLETE - {'STARTED': 4, 'CONDITION_STARTED': 24, 'CONDITION_COMPLETE': 22, 'COMPLETE': 3, 'FAILED': 1}
- **PASS** worker logs: slot queues == conditions[slot::3] in order, monotonic utc, STARTED after freeze, selected_step/wall_seconds/L1024 rows match receipts - slot-0: 8 started, ok=True; slot-1: 8 started, ok=True; slot-2: 4 started, ok=True; slot-2b: queue ['drift-parity-idinv-s117', 'drift-parity-win16-s143', 'drift-parity-base-s191', 'drift-parity-reread-s191'] == remaining slot-2 queue: True
- **PASS** train.json pid equals the worker pid for every condition trained by slots 0/1/2 - slot-0 pid 22696; slot-1 pid 16624; slot-2 pid 18856
- **PASS** stderr logs: exactly two non-empty, each a single nonfinite traceback from drift/run.py run_one - 2b-drift-parity-reread-s191.stderr.log: tracebacks=1 last="ValueError: nonfinite loss/gradient; failure preserved"; slot-2.stderr.log: tracebacks=1 last="ValueError: nonfinite loss/gradient; failure preserved"; empty stderr logs=8
- **PASS** stdout logs: slots 0/1 printed null (worker returns None), slot-2/2b-failed empty, 2b receipts equal train.json (minus history) and evaluation.json - slot-0.stdout='null'; slot-1.stdout='null'; 2b-drift-parity-idinv-s117: stdout receipt/evaluation == files: True; 2b-drift-parity-win16-s143: stdout receipt/evaluation == files: True; 2b-drift-parity-base-s191: stdout receipt/evaluation == files: True

### 2-training

- **PASS** updates.jsonl (22): 2048 rows, step==real_update 1..2048, finite loss/CE/grad_norm, aux null (non-idinv) or >=0 with loss==CE+aux (idinv), 16 validation rows n=256, selected_step == argmin val final_loss, history == journal, train/validation source seeds recomputed - 22/22
- **PASS** checkpoints (22): state hashes == receipts; every tensor differs initial vs last and vs selected; selected==last iff selected_step==2048; reread_* = 6176 params (5 tensors, W_o zero at init, nonzero after) only in reread; core params 24325 (five) / 24226 (parity) - reread extra params: [0, 6176]
- **PASS** FAILED (2): failure-step N == last journal step + 1 == rows + 1, grad_norm "inf", finite loss, journal tail shows exploding finite grad_norm - drift-five-reread-s191: failure step 1632 = last journal step 1631+1, tail grad_norm [4.081, 3.514, 942.759, 381.852, 141000.0, 1480000000000.0], loss at failure 1.8577277660369873; drift-parity-reread-s191: failure step 222 = last journal step 221+1, tail grad_norm [0.152, 0.641, 4.944, 54.042, 198000.0, 125000000.0], loss at failure 0.6991235613822937

### 3-pairing

- **PASS** per task/seed: input_hash, base_input_hash and source_seed sequences identical across arms (FAILED arms on their surviving prefix) - five/s117: journal rows compared base=2048,win16=2048,idinv=2048,reread=2048; five/s143: journal rows compared base=2048,win16=2048,idinv=2048,reread=2048; five/s191: journal rows compared base=2048,win16=2048,idinv=2048,reread=1631; parity/s117: journal rows compared base=2048,win16=2048,idinv=2048,reread=2048; parity/s143: journal rows compared base=2048,win16=2048,idinv=2048,reread=2048; parity/s191: journal rows compared base=2048,win16=2048,idinv=2048,reread=221
- **PASS** per task/seed: Core parameters in initial.pt identical (tensor-exact) across all four arms; train.json core_initial_hash equals recomputed hash - five/s117: 35 core tensors, 1 distinct core hash (e9707ab6f195); five/s143: 35 core tensors, 1 distinct core hash (ad26086316d1); five/s191: 35 core tensors, 1 distinct core hash (451fd4fe4fe4); parity/s117: 35 core tensors, 1 distinct core hash (56f0a34afd97); parity/s143: 35 core tensors, 1 distinct core hash (b27214b855af); parity/s191: 35 core tensors, 1 distinct core hash (7d8b014c95bb)
- **PASS** initial.pt (24) reproduces the seed-derived initial weights (Core seeded by seed; reread_* by seed+1000, W_o zero) - no training, construction only - 24/24

### 4-evaluation

- **PASS** NPZ (22 x 13): final/prefix accuracy and target/prediction histograms recomputed exactly; n=512; input_ids, input_hash, source_seed (frozen namespace) and inputs (rng regeneration) recomputed; targets recomputed with an independent oracle - 286/286 rows, 572 histograms exact
- **PASS** neutral NPZ: inputs == base inputs with the declared number of token-2 insertions at prefix(0)/middle(16)/suffix(32); final targets == base; correct_to_wrong/wrong_to_correct/prediction_changes recomputed - covered by the row loop above
- **PASS** inputs/targets/input_ids identical across arms for every task/seed/endpoint - five/s117: 4 arms x 13 endpoints identical inputs/targets/input_ids; five/s143: 4 arms x 13 endpoints identical inputs/targets/input_ids; five/s191: 3 arms x 13 endpoints identical inputs/targets/input_ids; parity/s117: 4 arms x 13 endpoints identical inputs/targets/input_ids; parity/s143: 4 arms x 13 endpoints identical inputs/targets/input_ids; parity/s191: 3 arms x 13 endpoints identical inputs/targets/input_ids

### 5-contrasts

- **PASS** analysis-tolerant.json header: protocol_hash frozen, complete=22, declared=24, status MEASURED_WITH_INCOMPLETE, condition_status == raw-derived status, absolute == evaluation.json final accuracies
- **PASS** all 78 contrasts recomputed from raw NPZ after pairing assertion: per_seed counts exact, seed_effects/mean/t95 within 1e-12, direction/df/seeds/missing/primary flags exact - 78/78
- **PASS** six decisions recomputed from the plan rule (>= +20 pp with POSITIVE_INTERVAL on both five primary endpoints and L1024 cost >= -5 pp -> POSITIVE; any primary mean <= 0 or cost < -5 pp -> NEGATIVE; else MIXED) - five/win16: MEASURED_NEGATIVE (middle -7.75 pp, suffix -13.87 pp, L1024 cost -7.16 pp, seeds 3) match=True; five/idinv: MEASURED_NEGATIVE (middle -1.63 pp, suffix -6.38 pp, L1024 cost -8.14 pp, seeds 3) match=True; five/reread: MEASURED_NEGATIVE (middle -11.43 pp, suffix -18.65 pp, L1024 cost +1.17 pp, seeds 2) match=True; parity/win16: MEASURED_NEGATIVE (middle -45.77 pp, suffix -49.28 pp, L1024 cost -6.25 pp, seeds 3) match=True; parity/idinv: MEASURED_NEGATIVE (middle -2.15 pp, suffix +0.07 pp, L1024 cost +0.00 pp, seeds 3) match=True; parity/reread: MEASURED_NEGATIVE (middle -1.56 pp, suffix +0.10 pp, L1024 cost +0.00 pp, seeds 2) match=True
- **PASS** all six decisions are MEASURED_NEGATIVE; the two reread decisions are flagged incomplete with seeds_used=2 (df=1, t=12.706)

### 6-base-reproduction

- **PASS** six base conditions vs sealed 2026-09-17 diagnostics small-{task}-w32-rlt-random-dense-lr0.003-s{seed}: inputs/input_ids/targets identical and predictions identical at ALL positions on 13/13 endpoints; selected steps equal; sealed NPZ sha256 verified against sealed selected-evaluation.json - drift-five-base-s117: inputs 13/13 targets 13/13 predictions 13/13 selected_step 2048 vs sealed 2048; drift-five-base-s143: inputs 13/13 targets 13/13 predictions 13/13 selected_step 1408 vs sealed 1408; drift-five-base-s191: inputs 13/13 targets 13/13 predictions 13/13 selected_step 2048 vs sealed 2048; drift-parity-base-s117: inputs 13/13 targets 13/13 predictions 13/13 selected_step 2048 vs sealed 2048; drift-parity-base-s143: inputs 13/13 targets 13/13 predictions 13/13 selected_step 2048 vs sealed 2048; drift-parity-base-s191: inputs 13/13 targets 13/13 predictions 13/13 selected_step 2048 vs sealed 2048

### 7-anomalies

- **PASS** no active.lock or *.tmp leftovers anywhere under results/drift - []
- **PASS** frozen_utc precedes every worker STARTED row and the first CONDITION_STARTED - freeze at 1789725982.8, first worker at 1789725985.8
- **PASS** drift/analyze_tolerant.py is not pinned by frozen.json (unpinned analysis path, as the ledger discloses) - sha256 now 2bf20ac0b1e1d34c

## Anomalies and observations (not PASS/FAIL items)

- git status: ?? docs/drift-mini-plan.md; ?? drift/; ?? notes/DRIFT_MINI_RAPORU.md; ?? results/drift/ (drift/, docs/drift-mini-plan.md and results/drift/ are untracked at audit time; the cycle is not yet committed).
- slot-2b.jsonl was written by a shell wrapper (slot-2b.sh), not by run.py worker(): integer utc, no pid, no selected_step/wall_seconds/L1024 in its CONDITION_COMPLETE rows, and STARTED/COMPLETE rows carry condition "-". Schema differs from slot-0/1/2 but the four conditions it ran are the exact remainder of the slot-2 queue.
- Pre-clip grad_norm spikes >10 are NOT unique to the reread arm. Per task/arm (seed: max, #updates>10): five/base: 117: 174, 23, 143: 136, 20, 191: 1.66e+03, 38; five/win16: 117: 70, 50, 143: 84.4, 30, 191: 114, 11; five/idinv: 117: 626, 85, 143: 955, 58, 191: 85.3, 27; five/reread: 117: 306, 78, 143: 5.63e+04, 74; parity/base: 117: 3.81, 0, 143: 2.35, 0, 191: 4.95, 0; parity/win16: 117: 22.8, 4, 143: 20.5, 1, 191: 4.78, 0; parity/idinv: 117: 17.5, 2, 143: 8.9, 0, 191: 8.44, 0; parity/reread: 117: 300, 19, 143: 145, 5. The ledger sentence "no other arm shows this" is overstated for spike counts: on the five task every arm spikes (base up to 1.66e3, idinv up to 955). What is unique to reread is the runaway to inf in 2/6 conditions and the largest surviving spike (5.63e4, five-s143); the ledger numbers for the surviving reread runs (306/78, 5.6e4/74, 300/19) are reproduced.
- Plan pre-registers df=2 (three seeds) only; the two-seed reread decisions (df=1, t=12.706) come from the unpinned drift/analyze_tolerant.py, disclosed in the ledger. Under either reading the reread verdict is negative: five/reread primary means are -11.4 and -18.7 pp on the two surviving seeds, so MEASURED_POSITIVE was unreachable even with seed 191.
- Sealed vs drift base receipts: selected validation final_loss identical for all six seeds; initial_state_hash strings differ by construction (the diagnostics runner hashes state dicts with a different byte layout) - predictions settle equivalence.
- Mechanism visible in the raw predictions (five task, 128 trailing neutrals): in every low-accuracy suffix row the prediction is a deterministic MANY-TO-ONE function of the true final state, i.e. the neutral run collapses the recurrent state onto 1-2 attractor states; accuracy equals the share of examples whose true final state happens to be an attractor. Rows (attractor classes, deterministic fraction, accuracy): drift-five-base-s117: [1, 4], 0.998, 0.402; drift-five-win16-s117: [0], 1.000, 0.195; drift-five-idinv-s117: [0], 0.982, 0.197; drift-five-reread-s117: [1], 0.998, 0.199; drift-five-base-s143: [2, 4], 0.998, 0.363; drift-five-win16-s143: [0], 1.000, 0.193; drift-five-idinv-s143: [0], 1.000, 0.193; drift-five-reread-s143: [0], 0.980, 0.193; drift-five-base-s191: [0, 2], 0.998, 0.447; drift-five-win16-s191: [0, 1, 4], 0.916, 0.408; drift-parity-win16-s143: [1], 1.000, 0.496; drift-parity-win16-s191: [0], 1.000, 0.480. With the middle placement the same collapsed state is then permuted by the trailing 16 real tokens, so predictions look scrambled (deterministic fraction 0.26, 0.28, 0.26, 0.27, 0.42, 0.28, 1.00, 0.26, 0.38, 0.36). Not an audit failure; a reading of the producer's own data that the report should state as observation, not as the pre-registered finding.
- drift-five-reread-s117/neutral-128-prefix: final accuracy 0.0000 with a spread prediction histogram {3: 107, 0: 97, 1: 114, 4: 91, 2: 103}: after 128 LEADING neutrals the state sits in attractor state 2 (explained-by-start-state {'0': 0, '1': 17, '2': 493, '3': 1, '4': 1}), and the 32 real tokens then permute it, so every example ends in a wrong-but-coherent state (0/512). Same attractor mechanism as the suffix rows, seen from the other side.
- Rows with neutral-128 final accuracy < 0.5 (22 complete conditions x 3 placements): 27 (all five-task rows plus parity/win16 s143/s191 suffix and s143 middle, where win16 collapses to a constant class at chance accuracy).
- Selection: 17/22 conditions selected the last checkpoint (step 2048); others: drift-five-win16-s117=1408, drift-five-idinv-s117=1408, drift-five-base-s143=1408, drift-five-reread-s143=1664, drift-parity-idinv-s191=1920.
- idinv auxiliary loss first->last: drift-five-idinv-s117: 1.62->8.97e-05; drift-five-idinv-s143: 1.66->0.000146; drift-five-idinv-s191: 1.13->0.00246; drift-parity-idinv-s117: 1.67->0.00119; drift-parity-idinv-s143: 1.63->8.35e-06; drift-parity-idinv-s191: 1.29->8.43e-05.

## Training summary (22 COMPLETE)

| condition | selected step | val final CE | first loss | last loss | max grad_norm | updates grad_norm>10 | aux first->last | wall s |
|---|---|---|---|---|---|---|---|---|
| drift-five-base-s117 | 2048 | 0.000441 | 1.671 | 0.000442 | 174 | 23 | - | 331 |
| drift-five-win16-s117 | 1408 | 0.000143 | 1.670 | 0.00018 | 70 | 50 | - | 332 |
| drift-five-idinv-s117 | 1408 | 0.00013 | 3.290 | 0.000251 | 626 | 85 | 1.62->8.97e-05 | 331 |
| drift-five-reread-s117 | 2048 | 0.00026 | 1.671 | 0.000273 | 306 | 78 | - | 438 |
| drift-five-base-s143 | 1408 | 0.000505 | 1.699 | 0.000602 | 136 | 20 | - | 325 |
| drift-five-win16-s143 | 2048 | 0.000134 | 1.699 | 0.000137 | 84.4 | 30 | - | 328 |
| drift-five-idinv-s143 | 2048 | 6.52e-05 | 3.361 | 0.000212 | 955 | 58 | 1.66->0.000146 | 330 |
| drift-five-reread-s143 | 1664 | 0.000164 | 1.699 | 0.000487 | 5.63e+04 | 74 | - | 422 |
| drift-five-base-s191 | 2048 | 0.000156 | 1.762 | 0.000167 | 1.66e+03 | 38 | - | 328 |
| drift-five-win16-s191 | 2048 | 0.000112 | 1.767 | 0.000107 | 114 | 11 | - | 323 |
| drift-five-idinv-s191 | 2048 | 5.78e-05 | 2.891 | 0.00251 | 85.3 | 27 | 1.13->0.00246 | 323 |
| drift-parity-base-s117 | 2048 | 1.44e-05 | 0.768 | 1.39e-05 | 3.81 | 0 | - | 326 |
| drift-parity-win16-s117 | 2048 | 8.2e-05 | 0.768 | 9.64e-05 | 22.8 | 4 | - | 328 |
| drift-parity-idinv-s117 | 2048 | 9.08e-06 | 2.438 | 0.0012 | 17.5 | 2 | 1.67->0.00119 | 329 |
| drift-parity-reread-s117 | 2048 | 2.05e-05 | 0.768 | 2.11e-05 | 300 | 19 | - | 417 |
| drift-parity-base-s143 | 2048 | 1.31e-05 | 0.694 | 1.34e-05 | 2.35 | 0 | - | 327 |
| drift-parity-win16-s143 | 2048 | 1.82e-05 | 0.694 | 1.94e-05 | 20.5 | 1 | - | 334 |
| drift-parity-idinv-s143 | 2048 | 1.02e-05 | 2.324 | 1.88e-05 | 8.9 | 0 | 1.63->8.35e-06 | 311 |
| drift-parity-reread-s143 | 2048 | 3.04e-05 | 0.694 | 3.17e-05 | 145 | 5 | - | 529 |
| drift-parity-base-s191 | 2048 | 1.55e-05 | 0.859 | 1.58e-05 | 4.95 | 0 | - | 329 |
| drift-parity-win16-s191 | 2048 | 8.74e-05 | 0.861 | 8.64e-05 | 4.78 | 0 | - | 300 |
| drift-parity-idinv-s191 | 1920 | 1.83e-05 | 2.152 | 0.000103 | 8.44 | 0 | 1.29->8.43e-05 | 297 |

Files: `drift_raw_audit.py` (this script), `drift-independent-audit.json`, `drift-independent-audit.md` under `.superpowers/sdd/drift-mini/`.
