EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
D1: does decoy-aware training (3 facts, random asked one) teach matching on the KEY? The nearest fact is right only 1 time in 3.
Recipe: mline.decoy_train from E:/rlt-research/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt, 500 updates, batch 32 (8 needle rows = rate 0.25, packed into 512-byte rows), 3
facts per example with the asked one chosen at random, total filler uniform on 0,128 bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed 217).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
