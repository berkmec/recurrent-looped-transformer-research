EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
D1b: D1 continued for 1000 updates on fresh data with 50 % needle rows. D1 collapsed onto 'answer the first fact'; this asks whether more updates and twice the signal teach the key.
Recipe: mline.decoy_train from results/M1-finetune-distance/D1-decoy3-f128/latest.pt, 1000 updates, batch 32 (needle rows at rate 0.5, packed into 512-byte rows), 3
facts per example with the asked one chosen at random, total filler uniform on 0,128 bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed 218).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
