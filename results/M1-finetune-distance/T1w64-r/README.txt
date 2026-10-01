EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. Small sealed Core, width 64, heads 2, CPU, one thread.
Stage 2: MQAR with replacement over the wide key alphabet, from stage 1's weights.
Pre-registered reading: the update at which train_k4_first first reaches 0.8 (T1r-b at width 128: between 2000 and 2500 MQAR updates after a 1500-update single-fact start).
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1w64-r --steps 4000 --width 64 --heads 2 --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --with-replacement --init-from results/M1-finetune-distance/T1w64-s/latest.pt
