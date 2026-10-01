EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
D3: D2 with key length drawn 1-4 per example. Short keys are easier to match on; does the key get used, and does it carry to 4-letter keys?
Recipe: mline.decoy_train from __RLT_ROOT__/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt, 1000 updates, batch 32 (needle rows at rate 0.5, packed into 512-byte rows), 4
facts per example with the asked one chosen at random, total filler uniform on 0,64 bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed 220).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
