# Independent core evidence audit

**Status: PASS within the stated evidence scope.**

Completed 21 primary conditions, 6 final-only conditions, 9 diagnostics and 18 benchmark rows.
Independently reconstructed 258 raw prediction arrays and checked 15 actual initial-tensor pairings.

Parity and five-state targets were reconstructed without importing the experimental generator or model. Exact seed-defined inputs, prediction accuracy, true state changes, neutral suffix semantics, condition completeness, validation checkpoint selection, execution chronology, source preparation pins, snapshots and final evidence-binding hashes passed.

## Delayed-final supervision: observed accuracy

| Seed | Training mode | Length | Final accuracy |
|---|---|---:|---:|
| 17 | rlt | 32 | 0.48046875 |
| 17 | rlt | 128 | 0.51562500 |
| 17 | tbptt | 32 | 0.48046875 |
| 17 | tbptt | 128 | 0.51562500 |
| 43 | rlt | 32 | 0.47460938 |
| 43 | rlt | 128 | 0.49804688 |
| 43 | tbptt | 32 | 0.47460938 |
| 43 | tbptt | 128 | 0.49804688 |
| 91 | rlt | 32 | 0.47851562 |
| 91 | rlt | 128 | 0.48046875 |
| 91 | tbptt | 32 | 0.47851562 |
| 91 | tbptt | 128 | 0.48046875 |

## Benchmark process memory

| Loss | Mode | Mean peak working set after update (bytes) |
|---|---|---:|
| prefix | full | 620939946.67 |
| prefix | tbptt8 | 621143381.33 |
| prefix | checkpoint | 627978240.00 |
| final_only | full | 620948138.67 |
| final_only | tbptt8 | 620897621.33 |
| final_only | checkpoint | 628523008.00 |

## Limits

- No model training or tuning performed; recorded optimizer execution is not externally attested.
- Saved initial tensors and seed-defined input streams verified; every training batch was not saved.
- Cross entropy/validation CE not independently recomputed because raw logits were not saved.
- Diagnostic autograd/finite-difference values and checkpoint gradient/update differences checked from receipts; no new gradient run.
- Influence uses full untruncated forward maps of TBPTT-trained weights; not delivered training-time truncated gradients.
- TBPTT cuts hidden/decoder KV only; encoder/global-memory gradients remain.
- Primary prefix and delayed-final supervision use different predefined streams and unequal label budgets.
- Benchmark peak working set is whole process including interpreter/model/optimizer/native libraries, not activation-only memory.
- CPU timings include external contention and cannot establish isolated throughput or energy.
- Hash bindings and chronology are internal evidence consistency, not trusted external signatures.

Audit source SHA256: `741395a90c3dd9cb475ca394acdc380dd0dcce01ce3bffcba82dbfd802f5b560`.
Full machine-readable evidence: [core-independent-audit.json](E:/rlt-research/2026-09-16-e2e/audit/core-independent-audit.json).
