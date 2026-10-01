# Core research measured report

Primary protocol SHA256: `51b39ee9debb8769148c0d18d92f2adf91be8ef5e052ee58fd9d93a04c808a3f`.

Scope: packages1–4 and the model-width portion of14. Independent tiny implementation, not an official accuracy reproduction. Pinned source commit cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39: [official README](https://raw.githubusercontent.com/yifanzhang-pro/recurrent-looped-tranformer/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md). All experimental models and oracle labels are local synthetic tasks. No infinite-depth or unrestricted-language conclusions.

## Execution and controls

All 27 trained conditions completed 2,000 optimizer updates each, batch 32, training length 32;21 prefix-supervised primary conditions and6 separate final-target-only conditions. Seeds 17, 43, 91. Validation-only checkpoint selection, no test-based tuning. Same initialization tensors and input streams are paired within task/width/seed. No-feedback keeps encoder, global cross-attention and SWA; only hidden feedback is disabled. All allocated merge parameters remain present but are inactive in no-feedback. FLOPs differ; no equal-FLOP claim. CPU 1 thread per training process; training seconds include validation/checkpoint overhead and concurrent system activity. Primary and companion later run concurrently on separate CPU threads, so their durations must not be added as elapsed wall time.

## Primary final-state accuracy

Each cell lists seed17 /43 /91 as percentages (512 independent examples per seed/condition). Length is actual consumed tokens and generated operations, including identity. Seeds are replication units; do not pool1536 examples as independent model replications.

| Task | Width | Mode |32|64|128|256|512|1024|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|parity|32|rlt|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|
|parity|32|nofeedback|48.44 / 47.27 / 49.02|52.34 / 49.22 / 51.37|45.31 / 48.44 / 47.66|47.07 / 51.17 / 48.44|51.37 / 45.51 / 49.22|52.93 / 48.05 / 48.63|
|parity|32|tbptt|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|
|five|32|rlt|100.00 / 100.00 / 100.00|99.41 / 100.00 / 100.00|99.61 / 100.00 / 100.00|99.22 / 100.00 / 100.00|97.46 / 100.00 / 100.00|96.09 / 99.80 / 100.00|
|five|32|nofeedback|19.34 / 20.12 / 18.36|23.83 / 21.09 / 20.51|19.53 / 19.53 / 19.73|17.58 / 17.77 / 21.48|19.14 / 18.16 / 18.95|21.29 / 18.36 / 22.66|
|parity|48|rlt|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|100.00 / 100.00 / 100.00|
|parity|48|nofeedback|51.56 / 48.05 / 48.44|53.12 / 49.22 / 49.61|44.92 / 47.46 / 50.00|46.09 / 52.15 / 51.37|49.02 / 48.24 / 48.24|49.02 / 52.15 / 52.15|

Metadata correction: raw `mean_nonidentity_count` was computed as input tokens other than designated neutral symbol2. For parity, bit0 is also mathematically an identity; therefore the correct raw-field interpretation is **non-neutral input count**. Supplemental recomputation separately records true state-changing operations by comparing each oracle target with the previous target, starting at state0. The five-state swap fixes states2,3,4, so token identity alone cannot count actual state changes. Operation counts are workload measures, not proved lower bounds on minimal algorithmic depth. Raw observations are preserved.

## Checkpoint selection, active parameters and actual training times

| Condition | Selected step | Best validation final accuracy | Allocated / active parameters | Training seconds |
|---|---:|---:|---:|---:|
|parity w32 rlt s17|2000|100.00%|24226 / 24226|326.56|
|parity w32 nofeedback s17|2000|48.05%|24226 / 21058|56.59|
|parity w32 tbptt s17|2000|100.00%|24226 / 24226|359.54|
|parity w32 rlt s43|2000|100.00%|24226 / 24226|383.60|
|parity w32 nofeedback s43|1100|51.95%|24226 / 21058|60.23|
|parity w32 tbptt s43|2000|100.00%|24226 / 24226|416.85|
|parity w32 rlt s91|1800|100.00%|24226 / 24226|417.91|
|parity w32 nofeedback s91|2000|54.30%|24226 / 21058|49.17|
|parity w32 tbptt s91|2000|100.00%|24226 / 24226|333.48|
|five w32 rlt s17|1100|100.00%|24325 / 24325|313.04|
|five w32 nofeedback s17|2000|22.66%|24325 / 21157|51.05|
|five w32 rlt s43|2000|100.00%|24325 / 24325|388.98|
|five w32 nofeedback s43|1600|20.31%|24325 / 21157|58.66|
|five w32 rlt s91|2000|100.00%|24325 / 24325|399.43|
|five w32 nofeedback s91|1900|18.36%|24325 / 21157|57.00|
|parity w48 rlt s17|2000|100.00%|54002 / 54002|519.67|
|parity w48 nofeedback s17|500|49.22%|54002 / 46946|86.54|
|parity w48 rlt s43|2000|100.00%|54002 / 54002|424.17|
|parity w48 nofeedback s43|1600|55.08%|54002 / 46946|61.78|
|parity w48 rlt s91|2000|100.00%|54002 / 54002|432.00|
|parity w48 nofeedback s91|1800|46.88%|54002 / 46946|75.10|

## Extra neutral-token computation (k=0)

Same base input and final answer; suffix symbols are identity operations. The useful base input remains unchanged. Output is one state classification; every added slot consumes another token. This measures token-clock recurrence, not a title-refinement loop. Each entry is seed-averaged final accuracy; individual seed rows and actual latency are in per-condition evaluation files.

| Task | Width | Mode | slots0 | slots8 | slots32 | slots128 |
|---|---:|---|---:|---:|---:|---:|
|parity|32|rlt|100.00%|93.75%|80.14%|66.67%|
|parity|32|nofeedback|51.76%|50.13%|49.41%|49.93%|
|parity|32|tbptt|100.00%|100.00%|100.00%|85.16%|
|five|32|rlt|100.00%|87.43%|47.59%|33.59%|
|five|32|nofeedback|20.44%|20.12%|19.40%|20.31%|
|parity|48|rlt|100.00%|90.95%|83.33%|83.33%|
|parity|48|nofeedback|49.80%|49.74%|48.96%|49.80%|

## Delayed-final-error companion

Separate frozen protocol; final-target-only CE and final-CE validation selection. TBPTT8 cuts only recurrent hidden and decoder SWA gradients; encoder and global memory remain differentiable. Prefix-supervised success does not establish learning from delayed errors. Within each supervision regime full/TBPTT share initialization, examples and2,000 updates. Across regimes there are different predefined example streams; both see64,000 programs and2,048,000 input tokens, but prefix training supplies2,048,000 targets versus64,000 final-only targets. This is not an equal-label-budget comparison.

| Mode | Seed | Selected step | Validation final accuracy | Test32 | Test128 | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
|rlt|17|600|50.00%|48.05%|51.56%|399.96|
|tbptt|17|600|50.00%|48.05%|51.56%|397.18|
|rlt|43|500|55.47%|47.46%|49.80%|476.80|
|tbptt|43|500|55.47%|47.46%|49.80%|454.96|
|rlt|91|1300|51.95%|47.85%|48.05%|396.04|
|tbptt|91|1300|51.95%|47.85%|48.05%|389.14|

## Influence and checkpoint diagnostics

Oracle final-state CE provides a meaningful loss. Recurrent-state and embedding lag norms, plus central finite differences of the first hidden state, are saved in diagnostics.json. No-feedback has zero sensitivity to explicitly supplied previous output, while early embeddings can still influence the final loss through the encoder/global-memory path. These influence diagnostics use the prefix-trained selected checkpoints. A nonzero gradient is not evidence of successful learning; magnitudes also depend on terminal loss/confidence and parameterization, so a small gradient at an already-correct checkpoint does not diagnose learning failure. Gaussian state-noise sweeps are diagnostics, not equivalent arithmetic precision experiments. Whole-forward activation checkpointing compares the same real loss and parameter gradients; peak activation memory is UNMEASURED.

Checkpoint 3 repeats: maximum gradient difference 0; standard seconds [0.1788, 0.1638, 0.1792]; recomputation seconds [0.3328, 0.3438, 0.4614]. Timing is scope-limited.

## Fresh-process real-step memory/time comparison

Each repetition starts a fresh Python process from the same frozen checkpoint and data, warms one inference pass, then performs one real AdamW update. Peak working set includes the interpreter/native libraries/model/optimizer; it is not isolated activation memory. External system load is not controlled.

| Loss | Mode | Mean seconds ± sample SD | Process peak MiB range |
|---|---|---:|---:|
|prefix|full|0.2180 ± 0.0189|592.1–592.2|
|prefix|tbptt8|0.2265 ± 0.0207|592.0–592.6|
|prefix|checkpoint|0.4696 ± 0.1385|598.6–599.4|
|final_only|full|0.2400 ± 0.0454|592.0–592.5|
|final_only|tbptt8|0.1878 ± 0.0117|591.9–592.6|
|final_only|checkpoint|0.3763 ± 0.0329|599.2–599.7|

Primary process observed peak working set: 602.9MiB. This includes Python, model, native libraries and allocations; it is not isolated activation memory and is not GPU VRAM.

## Scope and status boundaries

Package1 IMPLEMENTED_TESTED: causal state/cache contract, matched information control, serial/full encoder and parallel-control numerical+gradient parity. Package2 measured at finite length up to1024; interpret numerical tables, not an unbounded-reasoning claim. Package3 measured neutral-token computation with no new useful facts, k=0; title refinement remains outside this module. Package4 measured full/TBPTT learning and delayed-error companion, with path-specific gradient diagnostics; no complete removal of all long paths and no memory-saving claim. Package14 measures width32 versus48 on parity; not a model-law, natural-language, energy, RL or large-scale result.

Eighteen tests passed in the final core/green-final-18.log; earlier RED/GREEN logs are preserved. Independent root audit is recorded separately; scientific closure requires that external audit. Source, environment, checkpoint and prediction hashes are included in core artifacts. The original runner is unsafe for in-place partial restart; this uninterrupted run is attributed to its exact preserved source snapshots. Do not rerun legacy runners in this evidence directory. The separate safe_entrypoint verifies completed evidence, rejects partial or drifted receipts, and creates only a post-run consistency binding, not a signed execution receipt. Checkpoints contain model weights, not resumable optimizer state. CPU timings under shared load cannot be treated as isolated hardware throughput. The held-out examples differ across model/data seeds; input and target hashes are recorded per condition, and paired models within a seed share cases. Three model/data seeds provide limited precision; per-seed outcomes and paired deltas are retained instead of pooling correlated prefix predictions. Supplemental Wilson95 intervals use512 final outcomes conditional on each fixed checkpoint; they are not confidence intervals over the population of model seeds.

## Descriptive package outcomes

These labels describe the measured scope; they are not new significance tests. Raw metric files remain unchanged.

| Package | Status | Finding | Boundary |
|---|---|---|---|
|1|IMPLEMENTED_TESTED|Causal recurrent/cache contract and matched-information no-feedback control pass numerical and oracle-gradient checks.|Independent tiny implementation; no production state manager or official checkpoint reproduction.|
|2|MIXED|Finite long-horizon accuracy is strong, but task/seed-dependent errors occur; per-seed 1024-token values below.|Tests stop at1024 consumed tokens; no infinite-depth or arbitrary-task claim.|
|3|MEASURED_NEGATIVE|Extra neutral tokens do not establish a useful-computation benefit; five-state cases degrade while consuming additional measured time.|Token-clock recurrence with k=0; not the separate title-refinement mechanism.|
|4|MIXED|Dense-prefix learning succeeds; final-target-only full and TBPTT models remain near chance. Checkpoint gradient/update equivalence holds, without observed whole-process peak-memory saving.|TBPTT cuts recurrent output/decoder KV only; global encoder routes remain. Whole-process memory is not activation-only allocation.|
|14|INCONCLUSIVE|Widths32 and48 were actually trained/evaluated on parity; these two tiny widths do not establish a scaling law or a width advantage.|Core-width portion only; external module covers other package14 scopes. Energy, RL, large-model and unrestricted-language scaling unmeasured.|

At1024 tokens, RLT seed17/43/91 final accuracies: parity width32 [1.0, 1.0, 1.0]; five-state width32 [0.9609375, 0.998046875, 1.0]; parity width48 [1.0, 1.0, 1.0].

Five-state RLT neutral-suffix accuracy by slots, each list seed17/43/91: {'0': [1.0, 1.0, 1.0], '8': [0.810546875, 0.828125, 0.984375], '32': [0.419921875, 0.6171875, 0.390625], '128': [0.232421875, 0.34765625, 0.427734375]}. All slots preserve the base answer; degradation is observed, not attributed to an unmeasured causal explanation.

Final-only full/TBPTT accuracies across seeds and both lengths range from [0.474609375, 0.515625]. Failure to learn in both arms is not evidence of general equivalence between gradient methods.
