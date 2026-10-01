EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
B2 CONTROL: same recipe as ft384-plain-s217 but the distance range is NOT widened (0-64).
Recipe: mline.emergence from E:/rlt-research/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt, 500 updates, batch 32 (8 needle rows = rate 0.25, packed into 512-byte rows), needle
filler uniform on 0,64 bytes, lr 3e-4, fresh AdamW, fresh language-model data (seed 217), arm plain.
Read it with: sweep-train-keys.json and sweep-eval-keys.json, against ft384-plain-s217/ (the original 0-384 run) and
results/M1-diagnosis/distance-sweep-plain.json (the checkpoint it started from).
