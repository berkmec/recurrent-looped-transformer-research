Z1-G: plain GRU under the sealed U1 / U2c neutral-run protocol (written BEFORE any condition was run; committed 2026-10-02 00:50 local as 00c6846, whose commit time is the authoritative record)

Why. The campaign's first positive architecture result, the learned HOLD/WRITE gate (U2c, hard_st), keeps the Core's state through runs of
neutral (identity) tokens. It never had a standard-recurrence baseline, and a GRU already has an update gate. If a plain GRU trained on the
same data does not drift under neutral runs, the gate reproduces a standard feature and the Core's drift is a property of its weak additive
feedback, not of learned recurrence in general.

Protocol. Identical to U1 (results/U1/frozen.json, hash 56ef56a1...): the numbers are READ from horizon/sweep.py protocol(), not copied:
2,048 updates, batch 32, train length 128, AdamW lr 0.003, weight decay 0.01, clip 1.0, validation every 128 updates on 256 problems, checkpoint
= minimum validation final cross-entropy (earliest step on ties), seeds 117 / 143 / 191, tasks five and stack, same source seeds (so every
training batch and every test problem is the one the Core arms saw; the comparison is paired by seed). Evaluation: clean lengths 128, 256,
512, 1024, 2048, 4096 (1,000 fresh problems each); neutral insertions of 128 and 512 slots at the prefix / middle / suffix of 1,000 length-128
problems; loss = base final accuracy minus inserted final accuracy, in percentage points. Arms: GRU hidden width 32 (about 6.7k parameters,
the Core arms' width) and width 64 (about 25k parameters, matched to the Core arm's 24k). CPU, one thread. Weights are saved next to each result.

Reference numbers (U1, Core without gate, three-seed rows 'five'): neutral loss +44.1 (128 middle), +46.4 (128 suffix), +59.4 (512 middle),
+53.0 (512 suffix) points; prefix about 0. U2c arms are compared from results/U2c.

Pre-registered readings (task 'five'; cells = {128, 512} slots x {middle, suffix}; a cell counts per seed):
 R1  GRU HOLDS: loss <= 2 points in all four cells, in >= 2 of 3 seeds, at both widths  ->  a plain GRU already holds state through neutral
     runs after length-128 training only; the HOLD/WRITE gate result is a repair of a defect specific to the Core's feedback, not a new
     capability, and the campaign headline must say so.
 R2  GRU DRIFTS: loss >= 10 points in >= 2 of the four cells, in >= 2 of 3 seeds, at width 64  ->  neutral-run drift is general to learned
     recurrences trained at length 128; the gate result stands as a useful addition.
 R3  anything else: reported as mixed, with the per-seed numbers; no headline.
Clean length generalisation (descriptive): effective horizon at accuracy >= 0.95 against the Core arms' 4096 / 4096 / 2048 (U1).
Task 'stack' (growing memory; the Core did not reach 0.95 even at length 128): descriptive only, in-range accuracy and neutral loss, no verdict.

Limits stated in advance: three seeds; training length 128 only; two widths; a GRU without attention (no LSTM arm; one would be added if R1
holds and the question becomes how standard the capability is); neutral tokens are exact identities inserted into otherwise normal problems.
