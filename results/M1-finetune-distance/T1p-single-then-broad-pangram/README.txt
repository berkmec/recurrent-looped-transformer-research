EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1p: T1s2's weights, then MQAR with replacement, key length 4, the BROAD alphabet (53 symbols; T-Z never a key) and pangram filler. The fair small-Core version of the generalisation test: T-Z are trained letters that were never keys.
Pre-registered reading (written before any step ran):
- key matching LEARNED: train_k4_first >= 0.8 (T1g, the same without the pangram, reached 0.73 at 2500 updates).
- GENERALISES: held_k4_first >= 0.6 at the same step. A held-out score near 0.33 here is NOT explained by untrained embeddings.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1p-single-then-broad-pangram --steps 4000 --facts 4 --queries 4 --key-length 4 --key-alphabet broad --train-filler 0,64 --seed 311 --with-replacement --filler-kind pangram --init-from results/M1-finetune-distance/T1s2-scratch-single-pangram/latest.pt
