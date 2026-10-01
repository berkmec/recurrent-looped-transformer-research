EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1g: the small-Core analogue of D5. T1s's weights, then MQAR (4 facts / 4 questions, questions drawn WITH replacement) with key
length 4 over the BROAD alphabet (53 symbols; T-Z never appear in a training key), 4000 updates.
Pre-registered reading (written before any step ran), on the training-shaped score in every eval row:
- key matching LEARNED: train_k4_first >= 0.8.
- GENERALISES: held_k4_first >= 0.6 at that step (T-Z were never in a key). T1b without the broad alphabet reached 0.93 / 0.57
  (training letters / T-Z, T-Z being elimination only), so a held-out score well above 0.57 would be the new result.
- if train_k4_first rises but held_k4_first stays near T1b's, a broad alphabet alone does not make matching general.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1g-single-then-broad-replace --steps 4000 --facts 4 --queries 4 --key-length 4 --key-alphabet broad --seed 311 --with-replacement --init-from results/M1-finetune-distance/T1s-scratch-single/latest.pt
