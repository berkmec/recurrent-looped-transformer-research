EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1s: small Core from scratch on the single-fact needle task only - what every 30M checkpoint learned first. Its decoy probe says whether it is position-locked.
Pre-registered reading (written before any step ran):
- position-locked: first-asked share near 0 and decoy-asked share near 1 at 4-letter keys (it copies the nearest fact), as the 30M plain model did.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1s-scratch-single --steps 1500 --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4
