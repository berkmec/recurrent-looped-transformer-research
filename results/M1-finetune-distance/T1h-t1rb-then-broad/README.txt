EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1h: T1r-b's weights (key matching on 17 letters, first digit 0.98 on training letters, 0.27-0.39 on T-Z), then MQAR with
replacement, key length 4, over the BROAD alphabet (53 symbols; T-Z never appear in a training key), 2000 updates.
Pre-registered reading (written before any step ran), held_k4_first in the eval rows (the T-Z score):
- GENERALISES: held_k4_first >= 0.6 at some eval step.
- if train_k4_first stays high while held_k4_first stays near 0.33, a narrow-then-broad curriculum does not make matching general
  at this size and budget.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1h-t1rb-then-broad --steps 2000 --facts 4 --queries 4 --key-length 4 --key-alphabet broad --seed 331 --with-replacement --init-from results/M1-finetune-distance/T1r-b-single-then-replace/latest.pt
