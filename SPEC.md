# RLT end-to-end research specification

User authorization: execute the agreed fourteen research packages end to end without stopping for routine approvals. This is local research, not publication, paid compute, or modification of OGMA. Preserve previous evidence.

## Scientific scope

1. Source-faithful decoder state/cache and matched no-feedback control.
2. Long horizon generalization at 32, 64, 128, 256, 512, 1024 operations.
3. Separate token time t, title refinement k, dependency depth d; measure useful computation and overthinking.
4. Long-distance learning, gradient transport, truncated/full backpropagation, memory/time tradeoffs.
5. Learned extraction of headings and typed links from text rather than oracle-only routing.
6. Directed/bidirectional/incorrect graph controls at matched title compute.
7. Selective revision preserving both affected and unaffected final answer quality.
8. Capacity-triggered title windows, same-title continuation and meaningful splitting.
9. Compaction, rereading, insufficiency detection and reopening with raw preservation.
10. Immutable raw archive separate from vector index; relevant and applicable retrieval including version checks.
11. Measured halting error AND coverage AND actual saved work.
12. Title feedback into future recurrent computation, not only answer readout; connect extracted text and memory controller to the learned model.
13. Transfer verified solution methods into model weights; evaluate with archive unavailable and novel held-out examples.
14. Scaling across model/task size, controlled-language transfer, actual time/memory including prefill/training/retrieval; report unmeasured energy/RL or large language-model scale honestly.

The twenty user ideas remain distinct requirements. The thirty supplied faults are hypotheses/risk categories, not all established defects. Finite tests cannot prove infinite correct reasoning. A completed experiment may have a negative or inconclusive hypothesis outcome. Never count a tiny-model or controlled-language pilot as unrestricted natural-language or large-scale validation.

## Binding constraints

- Write only inside E:/rlt-research/2026-09-16-e2e. Prior evidence folders on C: are read-only.
- Official source pin: cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39. Cite source and record deviations. It specifies H_t=(s_t,C_t^D), causal global encoder memory and layerwise SWA, one token update per consumed token.
- Raw observations, generated public experiment traces and old versions are append-only; do not claim to collect an assistant provider's private reasoning.
- Distinguish t/k/d in every relevant record. k does not append another token to SWA. Future information must not alter earlier token predictions.
- Freeze each experiment protocol and its SHA256 before opening final evaluation. Calibration may use training/validation only. If a protocol changes, preserve both versions and run fresh held-out tests.
- Keep paired examples/batches/initialization wherever comparisons allow. State remaining parameter/compute mismatches. Use seeds 17,43,91 for primary learned comparisons, and report per-seed results; never count correlated predictions as independent examples.
- Save per-example predictions and targets, split identities, initial/final checkpoint hashes, source hashes, actual wall times, protocol hash and environment. Statistical uncertainty must respect example/seed clustering.
- Negative controls must have the same available information except the mechanism deliberately removed. Oracle solvers/links are explicitly labeled and must not leak into learned inference or selection.
- No result deletion, test-set tuning, hidden fallback to a symbolic answer, or inflated closure status. A symbolic oracle is for labels/checks, not a claimed learned model output.
- At least one independent audit of code/spec and recomputation of primary metrics before scientific closure.
- Use existing environments read-only. CPU env: C:/Users/pc/Documents/Codex/2026-09-16/rlt-derin-arastirma/.venv/Scripts/python.exe. CUDA env: C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe. No installs into these environments.
- Four CPU cores and roughly 5 GiB free RAM at preflight. CPU workers use torch.set_num_threads(1). Only the core worker owns the GPU until released. Keep artifacts below 10 GiB total; this is a storage protection bound, not a reason to fabricate coverage.
- Independent workers own disjoint directories. Do not edit shared plan/ledger, commit other workers' files, spawn helpers, or modify OGMA. Root handles integration and commits.

## Shared artifact contract

Each worker writes protocol.json before test evaluation; execution.jsonl with actual events; report.md; tests and their execution evidence; results/metrics.json containing per-condition rows and package status; results/*.npz with raw predictions/labels; checkpoints as needed. Every result records actual scope, failures, controls and unresolved external validity.

Package status distinguishes IMPLEMENTED_TESTED, MEASURED_POSITIVE, MEASURED_NEGATIVE, MIXED, INCONCLUSIVE, UNMEASURED. A result report is not evidence that a model or connector has been built. Each package has a scoped result and a separately stated generalization boundary.
