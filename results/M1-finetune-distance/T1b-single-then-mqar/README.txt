EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1b: T1s's weights, then exactly T1a's MQAR training (same seed, same batches). Does a single-fact start slow or block key matching?
Pre-registered reading (written before any step ran):
- key matching LEARNED: asked-share >= 0.8 for BOTH asks on TRAIN_KEYS/4 at some eval step.
- GENERALISES to unseen letters: the same on HELD_KEYS/4 (T-Z never appear in a training key), threshold 0.6.
- If it never leaves the position-locked pattern in the given steps, that is a statement about budget and width, not a verdict on the architecture.
- single-fact start HINDERS: at the last step T1b's decoy-asked share on TRAIN_KEYS/4 stays >= 0.3 below T1a's, or T1b reaches 0.5 at least twice as late.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1b-single-then-mqar --steps 3000 --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --init-from results/M1-finetune-distance/T1s-scratch-single/latest.pt
