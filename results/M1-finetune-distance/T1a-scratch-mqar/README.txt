EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1a: small Core from scratch on MQAR only (4 facts / 4 questions, key length 1-4, wide key alphabet A-S). Can this architecture learn key-addressed retrieval?
Pre-registered reading (written before any step ran):
- key matching LEARNED: asked-share >= 0.8 for BOTH asks on TRAIN_KEYS/4 at some eval step.
- GENERALISES to unseen letters: the same on HELD_KEYS/4 (T-Z never appear in a training key), threshold 0.6.
- If it never leaves the position-locked pattern in the given steps, that is a statement about budget and width, not a verdict on the architecture.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1a-scratch-mqar --steps 3000 --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301
