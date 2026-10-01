EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
D6: D5 then real keys (half of the examples key on words and names from the training text, 2000 distinct), filler 0-128, questions with replacement. Does key matching carry to real words and names it never keyed on?
Recipe: mline.decoy_train from results/M1-finetune-distance/D5-mqar4-broad/selected.pt, 1000 updates, batch 32 (needle rows at rate 0.5, packed into 512-byte rows), 4
facts per example with the asked one chosen at random, total filler uniform on 0,128 bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed 223).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
