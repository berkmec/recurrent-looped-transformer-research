EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
D2: MQAR-style, 4 facts / 4 questions (each fact asked once, random order), from the original checkpoint. Denser key->value signal; does the key get used?
Recipe: mline.decoy_train from E:/rlt-research/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt, 1000 updates, batch 32 (needle rows at rate 0.5, packed into 512-byte rows), 4
facts per example with the asked one chosen at random, total filler uniform on 0,64 bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed 219).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
