EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against any pre-registered claim.

QUESTION: is the distance cliff of the M1 checkpoints a TRAINING-RANGE effect? They answer a needle perfectly up to the
longest fact-to-question distance they trained on (filler 0-64 bytes) and collapse within 8-16 bytes beyond it
(results/M1-diagnosis/distance-sweep-*.json). If so, a short fine-tune on a wider range should move the cliff out.

WHAT RUNS: mline.emergence from results/M1/m1-plain-w1152-s117/selected.pt (the vm-mirror copy, sha256 in probe.json),
500 steps, batch 32 of which 8 rows are needle rows (rate 0.25), needle filler uniform on 0-384 bytes, packed into 512-byte
rows as in M1, lr 3e-4 (half of M1), fresh language-model data (seed 217, M1 used 117), fresh AdamW state.

WHY THE PLAIN ARM: the M1 gate never closed (hold_rate 4.8e-6 over 4000 steps; 0.0000 on every needle segment of the
saved checkpoint; gate_open == full to the last digit), so hard_st IS the plain Core and the two arms are two training
runs of one function. results/M1-diagnosis/channel-*.json.

READ IT WITH: the same distance sweep as the M1 checkpoints (TRAIN_KEYS and EVAL_KEYS, fillers 0..2048), compared cell by
cell to results/M1-diagnosis/distance-sweep-plain.json. Expected if it is a training-range effect: the cliff moves to
about 384-420. Falsified if accuracy beyond 64 stays near zero after 500 steps with half the needle rows beyond 64.
