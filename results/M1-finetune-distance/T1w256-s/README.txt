EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. Small sealed Core, width 256, heads 8, CPU, one thread.
Stage 1: single-fact needle task from scratch (the position-locked start).
Pre-registered reading: the update at which train_k4_first first reaches 0.8 (T1r-b at width 128: between 2000 and 2500 MQAR updates after a 1500-update single-fact start).
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1w256-s --steps 1500 --width 256 --heads 8 --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4
