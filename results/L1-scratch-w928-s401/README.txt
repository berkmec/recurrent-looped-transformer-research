L1 - THE LONG RUN. Exploratory, one seed, plain arm. Written before any update ran.

Question: does a from-scratch ~20.3M-parameter RLT (width 928, 8 heads), trained on 500M byte-tokens (about 2.4 epochs of cnn_dailymail) with
25 % synthetic rows (single-fact, MQAR with questions drawn with replacement over random and natural keys, distance), learn
  (a) key-addressed retrieval that holds on letters it never keyed on and on real words and names,
  (b) retrieval across at least 384 bytes,
while it keeps learning language?

It is NOT stopped for a plateau. A flat loss is not a reason to stop: transitions here arrive late and sharply.

Logged every update (probe.jsonl): loss, loss_lm, loss_synthetic, answer_loss (loss on the answer digits only), first_digit_accuracy, lr, grad norm.
Every 1000 updates (the 'eval' field): validation bits per byte; train_k4_first / exact (training letters), held_k4_first / exact (T-Z, never keys),
natural_first / exact (real validation words and names that are not in the training key list); single_fact_exact@{0,64,128,256,384,448};
attention_conditionality_mean / best. A model snapshot (snapshots/stepNNNNNN.pt) every 1000 updates.

Pre-registered thresholds (chance for a first digit is 0.1):
  key matching LEARNED  : train_k4_first >= 0.8
  GENERALISES           : held_k4_first >= 0.6 AND natural_first >= 0.6, at the same eval
  REACH                 : single_fact_exact@384 >= 0.9
  language              : bits per byte is reported, with no threshold - language quality is not the claim.
A TRANSITION is dated by the first update at which the 500-update moving average of first_digit_accuracy exceeds 0.5.
The final report gives every curve and, for each threshold, the update at which it was first crossed or 'never' - not only the last value.
