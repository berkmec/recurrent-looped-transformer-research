EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1s2: T1s again (small Core from scratch, single-fact needle task) but with a PANGRAM as filler, so every letter is a trained letter.
Pre-registered reading (written before any step ran):
- position-locked like T1s (first-asked share near 0, decoy-asked near 1 on the decoy probe); only the start for T1p.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1s2-scratch-single-pangram --steps 1500 --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 --filler-kind pangram --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4
