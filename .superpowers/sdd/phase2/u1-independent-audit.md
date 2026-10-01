# U1 (t axis) — independent raw-data audit

Verdict: **PASS** — 328 PASS / 0 FAIL of 328 checks. Read-only, CPU, one thread, 24 s. Runtime python 3.11.9, torch 2.12.0+cu130, numpy 2.3.5 (same as frozen.json environment).

Method: no producer module under `horizon/` was imported. Oracles (five: explicit +1 mod 5 / swap 0-1 / identity rules; stack: Python list stack with pop-on-empty no-op and labels a=0, b=1, empty=2), the source-seed derivation (sha256 of `ns|task|seed|step|length`, first 8 bytes little-endian), input regeneration (numpy default_rng choice with the frozen probabilities), neutral insertion, accuracies, histograms, flips, effective horizon and the df=2 t summaries were reimplemented here. The pinned `core/model.py` was imported only to reproduce the seeded initial checkpoints and to spot-check selected.pt on 32 rows of L128 (no training, no full re-evaluation, GPU untouched).

## Key numbers

### five (fixed-state automaton)

| condition | acc 2048 | acc 4096 | horizon | neutral 128 middle | 128 suffix | 512 middle | 512 suffix | selected step |
|---|---|---|---|---|---|---|---|---|
| u1-five-w32-s117 | 1.000 | 1.000 | 4096 | 40.8 pp | 39.2 pp | 59.5 pp | 58.8 pp | 2048 |
| u1-five-w32-s143 | 1.000 | 1.000 | 4096 | 60.9 pp | 60.8 pp | 79.9 pp | 61.1 pp | 2048 |
| u1-five-w32-s191 | 0.987 | 0.947 | 2048 | 30.6 pp | 39.2 pp | 38.7 pp | 39.2 pp | 2048 |

Prefix insertions cost 0.0 / 0.0 / 0.5 pp (128) and 0.0 / 0.1 / 0.6 pp (512) for s117 / s143 / s191. Task status recomputed: **MIXED** (all-points >= 0.95: [True, True, False], neutral within 2 pp: [False, False, False]).

### stack (growing memory)

| condition | 128 | 256 | 512 | 1024 | 2048 | 4096 | horizon | neutral 128 suffix | 512 suffix | selected step |
|---|---|---|---|---|---|---|---|---|---|---|
| u1-stack-w32-s117 | 0.936 | 0.900 | 0.864 | 0.834 | 0.836 | 0.801 | 0 | 64.9 pp | 65.8 pp | 2048 |
| u1-stack-w32-s143 | 0.936 | 0.888 | 0.854 | 0.826 | 0.798 | 0.813 | 0 | 56.4 pp | 53.7 pp | 1792 |
| u1-stack-w32-s191 | 0.954 | 0.868 | 0.835 | 0.805 | 0.759 | 0.715 | 128 | 48.2 pp | 56.2 pp | 1792 |

Task status recomputed: **MEASURED_NEGATIVE**. Stack prefix/middle neutral losses are 2.8-12.1 pp (all above the 2 pp target).

### stack input depth (independent simulation of the test inputs)

| condition | L128 mean max depth | L256 mean max depth | L512 mean max depth | L1024 mean max depth | L2048 mean max depth | L4096 mean max depth | final empty share L128 / L4096 |
|---|---|---|---|---|---|---|---|
| u1-stack-w32-s117 | 11.98 | 17.27 | 25.03 | 35.89 | 50.18 | 74.04 | 0.092 / 0.013 |
| u1-stack-w32-s143 | 12.30 | 17.80 | 24.63 | 36.16 | 51.16 | 73.53 | 0.093 / 0.011 |
| u1-stack-w32-s191 | 12.32 | 17.31 | 25.70 | 35.31 | 51.87 | 72.58 | 0.084 / 0.022 |

Baselines at the final position (majority class / last-pushed-symbol heuristic / model):

- u1-stack-w32-s117: L128 0.457 / 0.698 / 0.936; L256 0.482 / 0.744 / 0.900; L512 0.489 / 0.739 / 0.864; L1024 0.507 / 0.705 / 0.834; L2048 0.510 / 0.750 / 0.836; L4096 0.495 / 0.734 / 0.801; neutral-base 0.491 / 0.702 / 0.946
- u1-stack-w32-s143: L128 0.458 / 0.682 / 0.936; L256 0.510 / 0.724 / 0.888; L512 0.491 / 0.733 / 0.854; L1024 0.492 / 0.737 / 0.826; L2048 0.497 / 0.751 / 0.798; L4096 0.509 / 0.761 / 0.813; neutral-base 0.482 / 0.718 / 0.940
- u1-stack-w32-s191: L128 0.462 / 0.742 / 0.954; L256 0.477 / 0.705 / 0.868; L512 0.496 / 0.736 / 0.835; L1024 0.501 / 0.730 / 0.805; L2048 0.509 / 0.738 / 0.759; L4096 0.503 / 0.750 / 0.715; neutral-base 0.466 / 0.707 / 0.938

### checkpoint spot check (32 rows of L128 through selected.pt)

- u1-five-w32-s117: position mismatches 0 / 4096, final mismatches 0 / 32
- u1-five-w32-s143: position mismatches 0 / 4096, final mismatches 0 / 32
- u1-five-w32-s191: position mismatches 0 / 4096, final mismatches 0 / 32
- u1-stack-w32-s117: position mismatches 0 / 4096, final mismatches 0 / 32
- u1-stack-w32-s143: position mismatches 0 / 4096, final mismatches 0 / 32
- u1-stack-w32-s191: position mismatches 0 / 4096, final mismatches 0 / 32

## Checks

| id | status | check | detail |
|---|---|---|---|
| 1.1 | PASS | frozen.sha256 matches sha256(frozen.json) | file=e98d7ef369276bfe2409017894855853d0c24dcf8e8a7bf193ae48cf31f4661e, pin=e98d7ef369276bfe2409017894855853d0c24dcf8e8a7bf193ae48cf31f4661e |
| 1.2 | PASS | protocol_hash == sha256(canonical protocol JSON) | recomputed=56ef56a18ce02dd0d66e19a82c3bd52d2805c29719300c5bf4895c6086447830, stored=56ef56a18ce02dd0d66e19a82c3bd52d2805c29719300c5bf4895c6086447830 |
| 1.3 | PASS | five source pins equal current files |  |
| 1.4 | PASS | protocol constants as ledgered |  |
| 1.5 | PASS | six predeclared conditions present | ['u1-five-w32-s117', 'u1-five-w32-s143', 'u1-five-w32-s191', 'u1-stack-w32-s117', 'u1-stack-w32-s143', 'u1-stack-w32-s191'] |
| 1.6 | PASS | worker logs: 3 slots, no FAILED, all end COMPLETE, 6 CONDITION_COMPLETE rows, stderr logs empty |  |
| 1.7/u1-five-w32-s117 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-five-w32-s117 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-five-w32-s117 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=78f2836c86842de33d839a6283d3226ea8433b790522b955a52235d2ad2c35ba, recomputed=78f2836c86842de33d839a6283d3226ea8433b790522b955a52235d2ad2c35ba |
| 1.10/u1-five-w32-s117 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-five-w32-s117 | PASS | initial.pt embedding has 3 rows (task five) | shape=(3, 32) |
| 2.1/u1-five-w32-s117 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.667777419090271, loss_last=0.00013921574281994253, mean_first100=1.5986118483543397, mean_last100=0.00015285060653695836, max_grad=17765 |
| 2.2/u1-five-w32-s117 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-five-w32-s117 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=2048, recomputed=2048 |
| 2.4/u1-five-w32-s117 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=e9707ab6f19520c0, last=8c660df9876500df, selected=8c660df9876500df, selected_step=2048 |
| 2.5/u1-five-w32-s117 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24325, receipt=24325 |
| 3.1/u1-five-w32-s117/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s117/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-five-w32-s117/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=8351119588096178727, stored=8351119588096178727, identical=True |
| 3.2/u1-five-w32-s117/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=9421411020931665096, stored=9421411020931665096, identical=True |
| 3.2/u1-five-w32-s117/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=4801386275044180721, stored=4801386275044180721, identical=True |
| 3.2/u1-five-w32-s117/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=51489863540104060, stored=51489863540104060, identical=True |
| 3.2/u1-five-w32-s117/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=11524103223087273468, stored=11524103223087273468, identical=True |
| 3.2/u1-five-w32-s117/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=5895119358736147793, stored=5895119358736147793, identical=True |
| 3.3/u1-five-w32-s117/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=17010480628976185854 |
| 3.4/u1-five-w32-s117/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s117/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s117/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s117/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s117/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s117/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-five-w32-s117 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-five-w32-s117/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s117/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0, correct_to_wrong=0, wrong_to_correct=0, prediction_changes=0, n=1000 |
| 4.1/u1-five-w32-s117/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.592, prefix_accuracy=0.71555078125, correct_to_wrong=408, wrong_to_correct=0, prediction_changes=408, n=1000 |
| 4.1/u1-five-w32-s117/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.608, prefix_accuracy=0.8235390625, correct_to_wrong=392, wrong_to_correct=0, prediction_changes=392, n=1000 |
| 4.1/u1-five-w32-s117/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0, correct_to_wrong=0, wrong_to_correct=0, prediction_changes=0, n=1000 |
| 4.1/u1-five-w32-s117/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.405, prefix_accuracy=0.5397546875, correct_to_wrong=595, wrong_to_correct=0, prediction_changes=595, n=1000 |
| 4.1/u1-five-w32-s117/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.412, prefix_accuracy=0.6174109375, correct_to_wrong=588, wrong_to_correct=0, prediction_changes=588, n=1000 |
| 5.1/u1-five-w32-s117 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.6, max_constant_rows=0 |
| 4.5/u1-five-w32-s117 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 1.7/u1-five-w32-s143 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-five-w32-s143 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-five-w32-s143 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=6177ba6d8f774b00113e11d0669cedabcc6aa04e48812055b0ec492e2b0d9f53, recomputed=6177ba6d8f774b00113e11d0669cedabcc6aa04e48812055b0ec492e2b0d9f53 |
| 1.10/u1-five-w32-s143 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-five-w32-s143 | PASS | initial.pt embedding has 3 rows (task five) | shape=(3, 32) |
| 2.1/u1-five-w32-s143 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.6881555318832397, loss_last=6.085564018576406e-05, mean_first100=1.6027546298503876, mean_last100=6.428986627724953e-05, max_grad=57.171 |
| 2.2/u1-five-w32-s143 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-five-w32-s143 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=2048, recomputed=2048 |
| 2.4/u1-five-w32-s143 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=ad26086316d19ec3, last=088e4b6227883fe9, selected=088e4b6227883fe9, selected_step=2048 |
| 2.5/u1-five-w32-s143 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24325, receipt=24325 |
| 3.1/u1-five-w32-s143/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s143/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-five-w32-s143/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=6139140102123655842, stored=6139140102123655842, identical=True |
| 3.2/u1-five-w32-s143/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=10955272995638461396, stored=10955272995638461396, identical=True |
| 3.2/u1-five-w32-s143/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=6558431060210500077, stored=6558431060210500077, identical=True |
| 3.2/u1-five-w32-s143/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=4937098017223897149, stored=4937098017223897149, identical=True |
| 3.2/u1-five-w32-s143/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=17496995066888770947, stored=17496995066888770947, identical=True |
| 3.2/u1-five-w32-s143/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=5408692131091907860, stored=5408692131091907860, identical=True |
| 3.3/u1-five-w32-s143/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=3969402530975258288 |
| 3.4/u1-five-w32-s143/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s143/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s143/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s143/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s143/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s143/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-five-w32-s143 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-five-w32-s143/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s143/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0, correct_to_wrong=0, wrong_to_correct=0, prediction_changes=0, n=1000 |
| 4.1/u1-five-w32-s143/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.391, prefix_accuracy=0.57021484375, correct_to_wrong=609, wrong_to_correct=0, prediction_changes=609, n=1000 |
| 4.1/u1-five-w32-s143/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.392, prefix_accuracy=0.7218671875, correct_to_wrong=608, wrong_to_correct=0, prediction_changes=608, n=1000 |
| 4.1/u1-five-w32-s143/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.999, prefix_accuracy=0.999996875, correct_to_wrong=1, wrong_to_correct=0, prediction_changes=1, n=1000 |
| 4.1/u1-five-w32-s143/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.201, prefix_accuracy=0.370703125, correct_to_wrong=799, wrong_to_correct=0, prediction_changes=799, n=1000 |
| 4.1/u1-five-w32-s143/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.389, prefix_accuracy=0.524184375, correct_to_wrong=611, wrong_to_correct=0, prediction_changes=611, n=1000 |
| 5.1/u1-five-w32-s143 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.808, max_constant_rows=0 |
| 4.5/u1-five-w32-s143 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 1.7/u1-five-w32-s191 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-five-w32-s191 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-five-w32-s191 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=16cb39ffc655d6d1aa7ca8297c5c4f65c025525bf5abbbd917ab84e735f208a8, recomputed=16cb39ffc655d6d1aa7ca8297c5c4f65c025525bf5abbbd917ab84e735f208a8 |
| 1.10/u1-five-w32-s191 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-five-w32-s191 | PASS | initial.pt embedding has 3 rows (task five) | shape=(3, 32) |
| 2.1/u1-five-w32-s191 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.7642643451690674, loss_last=0.00018802091653924435, mean_first100=1.6054788661003112, mean_last100=0.0002045672433450818, max_grad=1231. |
| 2.2/u1-five-w32-s191 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-five-w32-s191 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=2048, recomputed=2048 |
| 2.4/u1-five-w32-s191 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=451fd4fe4fe40d21, last=4bd579a11e257eb1, selected=4bd579a11e257eb1, selected_step=2048 |
| 2.5/u1-five-w32-s191 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24325, receipt=24325 |
| 3.1/u1-five-w32-s191/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-five-w32-s191/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-five-w32-s191/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=16474979348808038987, stored=16474979348808038987, identical=True |
| 3.2/u1-five-w32-s191/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=338505406845218122, stored=338505406845218122, identical=True |
| 3.2/u1-five-w32-s191/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=17645023366615197036, stored=17645023366615197036, identical=True |
| 3.2/u1-five-w32-s191/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=4750994190515392687, stored=4750994190515392687, identical=True |
| 3.2/u1-five-w32-s191/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=8004500107246919062, stored=8004500107246919062, identical=True |
| 3.2/u1-five-w32-s191/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=9720326653684314223, stored=9720326653684314223, identical=True |
| 3.3/u1-five-w32-s191/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=13309394350896264460 |
| 3.4/u1-five-w32-s191/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s191/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s191/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-five-w32-s191/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s191/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-five-w32-s191/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-five-w32-s191 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-five-w32-s191/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s191/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s191/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.998, prefix_accuracy=0.999201171875 |
| 4.1/u1-five-w32-s191/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=0.9999990234375 |
| 4.1/u1-five-w32-s191/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.987, prefix_accuracy=0.99646728515625 |
| 4.1/u1-five-w32-s191/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.947, prefix_accuracy=0.982098388671875 |
| 4.1/u1-five-w32-s191/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=1.0, prefix_accuracy=1.0 |
| 4.1/u1-five-w32-s191/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.995, prefix_accuracy=0.998671875, correct_to_wrong=5, wrong_to_correct=0, prediction_changes=5, n=1000 |
| 4.1/u1-five-w32-s191/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.694, prefix_accuracy=0.82478125, correct_to_wrong=306, wrong_to_correct=0, prediction_changes=306, n=1000 |
| 4.1/u1-five-w32-s191/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.608, prefix_accuracy=0.84741796875, correct_to_wrong=392, wrong_to_correct=0, prediction_changes=392, n=1000 |
| 4.1/u1-five-w32-s191/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.994, prefix_accuracy=0.9996203125, correct_to_wrong=6, wrong_to_correct=0, prediction_changes=6, n=1000 |
| 4.1/u1-five-w32-s191/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.613, prefix_accuracy=0.7118109375, correct_to_wrong=387, wrong_to_correct=0, prediction_changes=387, n=1000 |
| 4.1/u1-five-w32-s191/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.608, prefix_accuracy=0.7037671875, correct_to_wrong=392, wrong_to_correct=0, prediction_changes=392, n=1000 |
| 5.1/u1-five-w32-s191 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.423, max_constant_rows=0 |
| 4.5/u1-five-w32-s191 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 1.7/u1-stack-w32-s117 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-stack-w32-s117 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-stack-w32-s117 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=b76beeba5227bf5115aa3c0f2e36bd518ecac0c9322b75e5d7e2b2634043d2fd, recomputed=b76beeba5227bf5115aa3c0f2e36bd518ecac0c9322b75e5d7e2b2634043d2fd |
| 1.10/u1-stack-w32-s117 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-stack-w32-s117 | PASS | initial.pt embedding has 4 rows (task stack) | shape=(4, 32) |
| 2.1/u1-stack-w32-s117 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.2417378425598145, loss_last=0.07362928241491318, mean_first100=0.3831083606183529, mean_last100=0.07083807796239854, max_grad=2.92447423 |
| 2.2/u1-stack-w32-s117 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-stack-w32-s117 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=2048, recomputed=2048 |
| 2.4/u1-stack-w32-s117 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=38c074bfd965046b, last=fe83765f8a4348e0, selected=fe83765f8a4348e0, selected_step=2048 |
| 2.5/u1-stack-w32-s117 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24291, receipt=24291 |
| 3.1/u1-stack-w32-s117/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s117/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-stack-w32-s117/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=3172237505154113875, stored=3172237505154113875, identical=True |
| 3.2/u1-stack-w32-s117/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=6122802944284603370, stored=6122802944284603370, identical=True |
| 3.2/u1-stack-w32-s117/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=16284627597693921512, stored=16284627597693921512, identical=True |
| 3.2/u1-stack-w32-s117/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=10191514803759109044, stored=10191514803759109044, identical=True |
| 3.2/u1-stack-w32-s117/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=11723619455787773679, stored=11723619455787773679, identical=True |
| 3.2/u1-stack-w32-s117/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=11439542911141777328, stored=11439542911141777328, identical=True |
| 3.3/u1-stack-w32-s117/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=3563590824447929461 |
| 3.4/u1-stack-w32-s117/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s117/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s117/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s117/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s117/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s117/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-stack-w32-s117 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-stack-w32-s117/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.936, prefix_accuracy=0.9707890625 |
| 4.1/u1-stack-w32-s117/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.9, prefix_accuracy=0.94066015625 |
| 4.1/u1-stack-w32-s117/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.864, prefix_accuracy=0.90573046875 |
| 4.1/u1-stack-w32-s117/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.834, prefix_accuracy=0.8786123046875 |
| 4.1/u1-stack-w32-s117/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.836, prefix_accuracy=0.8493056640625 |
| 4.1/u1-stack-w32-s117/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.801, prefix_accuracy=0.82815673828125 |
| 4.1/u1-stack-w32-s117/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.946, prefix_accuracy=0.97121875 |
| 4.1/u1-stack-w32-s117/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.89, prefix_accuracy=0.95961328125, correct_to_wrong=63, wrong_to_correct=7, prediction_changes=96, n=1000 |
| 4.1/u1-stack-w32-s117/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.91, prefix_accuracy=0.74055078125, correct_to_wrong=45, wrong_to_correct=9, prediction_changes=76, n=1000 |
| 4.1/u1-stack-w32-s117/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.297, prefix_accuracy=0.72469921875, correct_to_wrong=663, wrong_to_correct=14, prediction_changes=700, n=1000 |
| 4.1/u1-stack-w32-s117/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.851, prefix_accuracy=0.977346875, correct_to_wrong=99, wrong_to_correct=4, prediction_changes=145, n=1000 |
| 4.1/u1-stack-w32-s117/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.907, prefix_accuracy=0.4336375, correct_to_wrong=60, wrong_to_correct=21, prediction_changes=98, n=1000 |
| 4.1/u1-stack-w32-s117/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.288, prefix_accuracy=0.415959375, correct_to_wrong=667, wrong_to_correct=9, prediction_changes=707, n=1000 |
| 5.1/u1-stack-w32-s117 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.541, max_constant_rows=0 |
| 4.5/u1-stack-w32-s117 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 1.7/u1-stack-w32-s143 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-stack-w32-s143 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-stack-w32-s143 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=4e7e10abdc973d7edea2c7d357fdc982cc2edd8111b54f92f2df271379474f6e, recomputed=4e7e10abdc973d7edea2c7d357fdc982cc2edd8111b54f92f2df271379474f6e |
| 1.10/u1-stack-w32-s143 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-stack-w32-s143 | PASS | initial.pt embedding has 4 rows (task stack) | shape=(4, 32) |
| 2.1/u1-stack-w32-s143 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.200477123260498, loss_last=0.06709690392017365, mean_first100=0.3916851454973221, mean_last100=0.08099413752555847, max_grad=4.046589374 |
| 2.2/u1-stack-w32-s143 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-stack-w32-s143 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=1792, recomputed=1792 |
| 2.4/u1-stack-w32-s143 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=b91e9da478a09e5a, last=5f63ea35a834e8d5, selected=62f2c6586fa9c638, selected_step=1792 |
| 2.5/u1-stack-w32-s143 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24291, receipt=24291 |
| 3.1/u1-stack-w32-s143/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s143/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-stack-w32-s143/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=5310598066536483402, stored=5310598066536483402, identical=True |
| 3.2/u1-stack-w32-s143/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=2261283684385878873, stored=2261283684385878873, identical=True |
| 3.2/u1-stack-w32-s143/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=5606713698370453043, stored=5606713698370453043, identical=True |
| 3.2/u1-stack-w32-s143/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=639679092705244582, stored=639679092705244582, identical=True |
| 3.2/u1-stack-w32-s143/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=10481342588186627646, stored=10481342588186627646, identical=True |
| 3.2/u1-stack-w32-s143/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=9907400369190757934, stored=9907400369190757934, identical=True |
| 3.3/u1-stack-w32-s143/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=9155919173787228108 |
| 3.4/u1-stack-w32-s143/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s143/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s143/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s143/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s143/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s143/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-stack-w32-s143 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-stack-w32-s143/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.936, prefix_accuracy=0.9643828125 |
| 4.1/u1-stack-w32-s143/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.888, prefix_accuracy=0.9321875 |
| 4.1/u1-stack-w32-s143/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.854, prefix_accuracy=0.898876953125 |
| 4.1/u1-stack-w32-s143/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.826, prefix_accuracy=0.8762216796875 |
| 4.1/u1-stack-w32-s143/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.798, prefix_accuracy=0.85313818359375 |
| 4.1/u1-stack-w32-s143/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.813, prefix_accuracy=0.82956787109375 |
| 4.1/u1-stack-w32-s143/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.94, prefix_accuracy=0.96228125 |
| 4.1/u1-stack-w32-s143/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.89, prefix_accuracy=0.95565625, correct_to_wrong=59, wrong_to_correct=9, prediction_changes=104, n=1000 |
| 4.1/u1-stack-w32-s143/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.892, prefix_accuracy=0.82466796875, correct_to_wrong=57, wrong_to_correct=9, prediction_changes=95, n=1000 |
| 4.1/u1-stack-w32-s143/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.376, prefix_accuracy=0.8058828125, correct_to_wrong=578, wrong_to_correct=14, prediction_changes=603, n=1000 |
| 4.1/u1-stack-w32-s143/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.829, prefix_accuracy=0.97429375, correct_to_wrong=113, wrong_to_correct=2, prediction_changes=170, n=1000 |
| 4.1/u1-stack-w32-s143/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.872, prefix_accuracy=0.5550765625, correct_to_wrong=80, wrong_to_correct=12, prediction_changes=121, n=1000 |
| 4.1/u1-stack-w32-s143/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.403, prefix_accuracy=0.5657171875, correct_to_wrong=551, wrong_to_correct=14, prediction_changes=576, n=1000 |
| 5.1/u1-stack-w32-s143 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.851, max_constant_rows=0 |
| 4.5/u1-stack-w32-s143 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 1.7/u1-stack-w32-s191 | PASS | directory holds exactly the 19 expected files (no lock/failure/pilot files) |  |
| 1.8/u1-stack-w32-s191 | PASS | train.json protocol_hash and sources equal frozen manifest |  |
| 1.9/u1-stack-w32-s191 | PASS | complete.json file hashes match, seal_hash recomputed, protocol_hash matches | seal_hash=28f220d587abfb7be319eedf98b5a4679b6d00c8a0d711ce59f1ff544a89698d, recomputed=28f220d587abfb7be319eedf98b5a4679b6d00c8a0d711ce59f1ff544a89698d |
| 1.10/u1-stack-w32-s191 | PASS | evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash | raw_sha_ok=13, rows=13 |
| 1.11/u1-stack-w32-s191 | PASS | initial.pt embedding has 4 rows (task stack) | shape=(4, 32) |
| 2.1/u1-stack-w32-s191 | PASS | updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls | rows=2048, loss_first=1.0877249240875244, loss_last=0.06708245724439621, mean_first100=0.385518541932106, mean_last100=0.07808951895684003, max_grad=4.259142875 |
| 2.2/u1-stack-w32-s191 | PASS | all 2048 training source seeds and input hashes regenerate from the frozen train namespace | seeds_ok=True, input_hash_ok=True, distinct_train_rows=65536 |
| 2.3/u1-stack-w32-s191 | PASS | validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees | selected_step=1792, recomputed=1792 |
| 2.4/u1-stack-w32-s191 | PASS | state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048 | initial=a8f15ea94d3a3bdc, last=23642b982ef361e7, selected=443ad6e9e654741a, selected_step=1792 |
| 2.5/u1-stack-w32-s191 | PASS | initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt | init_reproduced=True, parameters=24291, receipt=24291 |
| 3.1/u1-stack-w32-s191/selected-L128.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-L256.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-L512.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-L1024.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-L2048.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-L4096.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral0-base.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral128-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral128-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral128-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral512-prefix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral512-middle.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.1/u1-stack-w32-s191/selected-neutral512-suffix.npz | PASS | stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata | target_mismatches=0, metadata_ok=True |
| 3.2/u1-stack-w32-s191/L128 | PASS | test inputs regenerate from frozen test namespace seed | seed=11415209568714571909, stored=11415209568714571909, identical=True |
| 3.2/u1-stack-w32-s191/L256 | PASS | test inputs regenerate from frozen test namespace seed | seed=16224664157032727343, stored=16224664157032727343, identical=True |
| 3.2/u1-stack-w32-s191/L512 | PASS | test inputs regenerate from frozen test namespace seed | seed=14241800079996921528, stored=14241800079996921528, identical=True |
| 3.2/u1-stack-w32-s191/L1024 | PASS | test inputs regenerate from frozen test namespace seed | seed=13568240200517537393, stored=13568240200517537393, identical=True |
| 3.2/u1-stack-w32-s191/L2048 | PASS | test inputs regenerate from frozen test namespace seed | seed=12418028239672817831, stored=12418028239672817831, identical=True |
| 3.2/u1-stack-w32-s191/L4096 | PASS | test inputs regenerate from frozen test namespace seed | seed=1338425162603504858, stored=1338425162603504858, identical=True |
| 3.3/u1-stack-w32-s191/base | PASS | neutral base inputs regenerate from frozen neutral namespace seed (length 128) | seed=13769692429498787400 |
| 3.4/u1-stack-w32-s191/selected-neutral128-prefix.npz | PASS | 128 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s191/selected-neutral128-middle.npz | PASS | 128 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s191/selected-neutral128-suffix.npz | PASS | 128 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=256 |
| 3.4/u1-stack-w32-s191/selected-neutral512-prefix.npz | PASS | 512 neutral tokens at prefix (position 0) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s191/selected-neutral512-middle.npz | PASS | 512 neutral tokens at middle (position 64) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.4/u1-stack-w32-s191/selected-neutral512-suffix.npz | PASS | 512 neutral tokens at suffix (position 128) on the base inputs; identical final targets; same seed | consumed_tokens=640 |
| 3.5/u1-stack-w32-s191 | PASS | no length-128 test / neutral-base / validation row appears among the 65536 training rows | test-L128=0, neutral-base=0, validation=0 |
| 4.1/u1-stack-w32-s191/selected-L128.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.954, prefix_accuracy=0.9620703125 |
| 4.1/u1-stack-w32-s191/selected-L256.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.868, prefix_accuracy=0.93261328125 |
| 4.1/u1-stack-w32-s191/selected-L512.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.835, prefix_accuracy=0.896443359375 |
| 4.1/u1-stack-w32-s191/selected-L1024.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.805, prefix_accuracy=0.8556923828125 |
| 4.1/u1-stack-w32-s191/selected-L2048.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.759, prefix_accuracy=0.81243310546875 |
| 4.1/u1-stack-w32-s191/selected-L4096.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.715, prefix_accuracy=0.776306884765625 |
| 4.1/u1-stack-w32-s191/selected-neutral0-base.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.938, prefix_accuracy=0.959578125 |
| 4.1/u1-stack-w32-s191/selected-neutral128-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.91, prefix_accuracy=0.9645, correct_to_wrong=49, wrong_to_correct=21, prediction_changes=83, n=1000 |
| 4.1/u1-stack-w32-s191/selected-neutral128-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.889, prefix_accuracy=0.779578125, correct_to_wrong=60, wrong_to_correct=11, prediction_changes=83, n=1000 |
| 4.1/u1-stack-w32-s191/selected-neutral128-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.456, prefix_accuracy=0.78601953125, correct_to_wrong=514, wrong_to_correct=32, prediction_changes=555, n=1000 |
| 4.1/u1-stack-w32-s191/selected-neutral512-prefix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.817, prefix_accuracy=0.9728359375, correct_to_wrong=124, wrong_to_correct=3, prediction_changes=172, n=1000 |
| 4.1/u1-stack-w32-s191/selected-neutral512-middle.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.846, prefix_accuracy=0.5553296875, correct_to_wrong=105, wrong_to_correct=13, prediction_changes=146, n=1000 |
| 4.1/u1-stack-w32-s191/selected-neutral512-suffix.npz | PASS | final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly | final_accuracy=0.376, prefix_accuracy=0.5724890625, correct_to_wrong=590, wrong_to_correct=28, prediction_changes=635, n=1000 |
| 5.1/u1-stack-w32-s191 | PASS | no constant-class predictions (no file with a class share >= 0.99 or all-constant rows) | max_top_class_share=0.789, max_constant_rows=0 |
| 4.5/u1-stack-w32-s191 | PASS | checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions) | rows=32, final_mismatches=0, position_mismatches=0 |
| 4.2 | PASS | analysis.json protocol_hash and status |  |
| 4.3/u1-five-w32-s117 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=4096, stored=4096 |
| 4.3/u1-five-w32-s143 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=4096, stored=4096 |
| 4.3/u1-five-w32-s191 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=2048, stored=2048 |
| 4.3/u1-stack-w32-s117 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=0, stored=0 |
| 4.3/u1-stack-w32-s143 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=0, stored=0 |
| 4.3/u1-stack-w32-s191 | PASS | analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12) | effective_horizon=128, stored=128 |
| 4.4/five | PASS | analysis.json per-task seed summaries (df=2, t=4.302652729), horizons, flags and status recomputed (1e-12) | status=MIXED, stored=MIXED |
| 4.4/stack | PASS | analysis.json per-task seed summaries (df=2, t=4.302652729), horizons, flags and status recomputed (1e-12) | status=MEASURED_NEGATIVE, stored=MEASURED_NEGATIVE |
| 4.6 | PASS | ledger u1-analyze-output.json agrees with analysis.json statuses/horizons |  |

## Anomalies and observations

- u1-five-w32-s191: L512 prefix accuracy 0.999201 exceeds final accuracy 0.998 (errors accumulate toward the end; expected for a recurrent state, not a metric bug)
- u1-five-w32-s191: L1024 final accuracy 1.000 while prefix accuracy 0.9999990234 < 1: transient mid-sequence mispredictions that recovered (1 wrong positions of 1024000)
- u1-five-w32-s191: L2048 prefix accuracy 0.996467 exceeds final accuracy 0.987 (errors accumulate toward the end; expected for a recurrent state, not a metric bug)
- u1-five-w32-s191: L4096 prefix accuracy 0.982098 exceeds final accuracy 0.947 (errors accumulate toward the end; expected for a recurrent state, not a metric bug)
- u1-five-w32-s117: pre-clip gradient norm peaks at 17765.8 (step 272); 67 of 2048 updates exceed clip=1.0; all losses/gradients finite
- u1-five-w32-s143: prediction mass concentrates on class 0 in selected-neutral512-suffix.npz (share 0.808 of 1000 final predictions) — partial collapse toward a constant answer under the long neutral run; no file is a global constant-class predictor and no row is constant along the sequence
- u1-five-w32-s143: pre-clip gradient norm peaks at 57.2 (step 463); 83 of 2048 updates exceed clip=1.0; all losses/gradients finite
- u1-five-w32-s191: pre-clip gradient norm peaks at 1231.7 (step 855); 99 of 2048 updates exceed clip=1.0; all losses/gradients finite
- u1-stack-w32-s117: pre-clip gradient norm peaks at 2.9 (step 1148); 576 of 2048 updates exceed clip=1.0; all losses/gradients finite
- u1-stack-w32-s143: prediction mass concentrates on class 0 in selected-neutral512-suffix.npz (share 0.851 of 1000 final predictions) — partial collapse toward a constant answer under the long neutral run; no file is a global constant-class predictor and no row is constant along the sequence
- u1-stack-w32-s143: pre-clip gradient norm peaks at 4.0 (step 1323); 684 of 2048 updates exceed clip=1.0; all losses/gradients finite
- u1-stack-w32-s191: prediction mass concentrates on class 0 in selected-neutral128-suffix.npz (share 0.789 of 1000 final predictions) — partial collapse toward a constant answer under the long neutral run; no file is a global constant-class predictor and no row is constant along the sequence
- u1-stack-w32-s191: pre-clip gradient norm peaks at 4.3 (step 508); 743 of 2048 updates exceed clip=1.0; all losses/gradients finite
- five: validation final loss sits at chance level (ln 5 = 1.609) through step 256 (s117), 384 (s143), 1152 (s191) and then drops by more than an order of magnitude within one or two validation intervals — a late, abrupt transition; the latest learner (s191) is the seed whose accuracy degrades at 2048/4096 (0.987/0.947) and whose horizon is 2048.
- stack: validation final loss at steps 1536..2048 = s117 [0.1588, 0.1235, 0.105, 0.1178, 0.0905]; s143 [0.167, 0.1175, 0.1121, 0.1265, 0.1278]; s191 [0.1296, 0.1164, 0.1137, 0.1162, 0.1409] — still trending downward at the frozen 2048-update budget, so MEASURED_NEGATIVE is a statement at the frozen budget, not at convergence.
- u1-stack-w32-s117: stack baselines final position — majority class L128 0.457, L256 0.482, L512 0.489, L1024 0.507, L2048 0.510, L4096 0.495, neutral-base 0.491; last-pushed-symbol heuristic L128 0.698, L256 0.744, L512 0.739, L1024 0.705, L2048 0.750, L4096 0.734, neutral-base 0.702; model L128 0.936, L256 0.900, L512 0.864, L1024 0.834, L2048 0.836, L4096 0.801, neutral-base 0.946
- u1-stack-w32-s143: stack baselines final position — majority class L128 0.458, L256 0.510, L512 0.491, L1024 0.492, L2048 0.497, L4096 0.509, neutral-base 0.482; last-pushed-symbol heuristic L128 0.682, L256 0.724, L512 0.733, L1024 0.737, L2048 0.751, L4096 0.761, neutral-base 0.718; model L128 0.936, L256 0.888, L512 0.854, L1024 0.826, L2048 0.798, L4096 0.813, neutral-base 0.940
- u1-stack-w32-s191: stack baselines final position — majority class L128 0.462, L256 0.477, L512 0.496, L1024 0.501, L2048 0.509, L4096 0.503, neutral-base 0.466; last-pushed-symbol heuristic L128 0.742, L256 0.705, L512 0.736, L1024 0.730, L2048 0.738, L4096 0.750, neutral-base 0.707; model L128 0.954, L256 0.868, L512 0.835, L1024 0.805, L2048 0.759, L4096 0.715, neutral-base 0.938
- five neutral insertions: prefix blocks are harmless (state 0 is a fixed point of the identity token) but 128/512 neutral tokens in the middle/suffix cost 30-80 pp although the oracle final target is unchanged: long neutral runs (training neutral rate 0.125, geometric run lengths) are out of distribution and the recurrent state drifts.
- stack neutral insertions: prefix/middle cost 3-12 pp, suffix 48-66 pp — the stored top symbol decays over a long idle run; the prefix/middle asymmetry versus five is because the stack state at the insertion point is usually non-trivial while five starts from state 0 at the prefix.
- worker logs: the three STARTED rows carry an identical utc (1789735273.3394809) and identical CONDITION_STARTED utc across three different pids — consistent with the ~15.6 ms Windows time.time() tick when three processes are launched together, not with copied logs (pids, wall_seconds and completion times differ).
- stdout logs contain only "null" (worker() returns None and main prints json.dumps(None)); stderr logs are empty.
- selection: five s117/s143/s191 and stack s117 selected step 2048 (selected.pt == last.pt); stack s143/s191 selected step 1792.

Nothing was modified under results/, horizon/, core/, review_cycle/, language/, semantic_memory/, docs/ or notes/. Files written: E:\rlt-research\2026-09-18-phase2\.superpowers\sdd\phase2\u1_raw_audit.py, E:\rlt-research\2026-09-18-phase2\.superpowers\sdd\phase2\u1-independent-audit.json, E:\rlt-research\2026-09-18-phase2\.superpowers\sdd\phase2\u1-independent-audit.md
