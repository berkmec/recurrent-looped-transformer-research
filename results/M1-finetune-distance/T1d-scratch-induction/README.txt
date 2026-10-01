EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1d: small Core from scratch on induction rows only (random string, filler, the same string; no key/value template, no positional shortcut). Does the match-then-copy circuit form, and how fast?
Pre-registered reading (written before any step ran):
- the circuit FORMS: copy_accuracy (exact next byte on the second copy after its first 2 bytes; chance ~0.016) >= 0.9 at some eval step.
- no rise by 2000 steps is a statement about budget, width or learning rate, not about the architecture.
Recipe: python -B -m mline.mqar_scratch --out results/M1-finetune-distance/T1d-scratch-induction --steps 2000 --source induction --seg-length 8-32 --gap 0,96 --lead 0,16 --skip 2 --seed 321
