EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1r-a: small Core from scratch on MQAR with the questions drawn WITH replacement. Can this architecture learn key-addressed retrieval once elimination is not available?
Pre-registered reading (written before any step ran), on the FIRST question (no elimination is possible there) of the
training-shaped score mqar_eval writes into every eval row (train_k4_first etc.; chance for a first digit is 0.1, copying a
random fact scores about 0.33):
- key matching LEARNED: train_k4_first >= 0.8 at some eval step (with replacement every question needs the key).
- GENERALISES to unseen letters: held_k4_first >= 0.6 at the same step (T-Z never appear in a training key).
- if it stays near 0.33 the budget or width is too small, or the architecture cannot do it: that is the question this run asks.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1r-a-scratch-replace --steps 3000 --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --with-replacement
