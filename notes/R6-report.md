# R6 memory and gradient-route diagnostics — development checkpoint

Status: implementation and bounded CPU calibration complete; GPU memory main UNMEASURED. No efficiency conclusion is authorized by CPU process RAM or saved-hook counts. Frozen legacy Core/training/R1 files were not changed.

## Mathematical route

Original Core computes `u=(e,mk,mv)=g_theta(x)` for the whole sequence. TBPTT8 detaches only decoder `s,dk,dv` at boundaries. Encoder and projected-memory routes remain attached across all boundaries.

For each chunk j, define `L_j=sum(CE for chunk j)/(batch*total_length)`. Segment execution detaches `u` into leaf views, backpropagates `L_j`, accumulates each leaf gradient, releases the segment decoder graph, and detaches only `s,dk,dv` before the next chunk. At the end one multi-output backward computes `J_g^T * sum_j dL_j/du`, accumulating into the direct decoder parameter gradients. This preserves the original TBPTT global routes; it is a different recurrence gradient from full BPTT.

Arms: full; detach-only TBPTT8; whole-forward nonreentrant checkpoint (full route); segment-backward TBPTT8; segment-backward with checkpoint of the complete encoder/projection function (TBPTT8 route). The last arm retains output e/mk/mv storage and leaf gradients, while recomputing internal encoder activations.

## Evidence and limits

Seven CPU tests PASS after recorded RED failures. Double precision tests use L19, batch2, width8, an uneven final segment and three AdamW updates. They compare losses, every parameter gradient, updated weights, and optimizer step/first/second moments. Negative controls establish full versus truncated gradients differ and removal of the global bridge removes embedding/mk gradients. Hook tests deduplicate aliased storage, release by saved-object lifetime, and prevent graph-retaining wrapper payloads. Worker test checks real parameter changes, two initialized optimizer warmups, three final optimizer steps and checkpoint output.

RED/GREEN logs: `.superpowers/sdd/review-diagnostics-plan/task-3-red.log`, `task-3-red-hook-graph.log`, `task-3-green.log`. These unit tests do not replace same-device fresh-process route validation before main measurement.

Fresh-process CPU float32 route validation at L32 also PASS: five separate workers, each two warmups plus three measured actual updates, with all three intended equivalent pairs matching final gradients, model state, all AdamW state and measured losses. Maximum absolute discrepancy across compared state tensors was2.62633e-7. Observed host CPU busy fractions were0.614–0.660 during these workers; timing is scoped to that load. Evidence: `.superpowers/sdd/review-diagnostics-plan/task-3-cpu-equivalence/equivalence.json`. Final test rerun: `task-3-green-final.log`,7PASS. CUDA equivalence remains required after freeze.

Initial development calibration used seven fresh CPU child processes and actual 2 warmup + 3 measured AdamW updates per condition, width32/heads2/batch8/float32. L32 phase-time means were approximately full0.17s, detach0.16s, checkpoint0.37s, segment0.18s, segment+encoder-checkpoint0.18s. L128 segment mean was0.77s. Separate L32 encoder-checkpoint hook run was0.22s. Whole-process lifetime peak working sets were approximately610–624MB. These values include the Python/PyTorch runtime and are not activation memory. The hook calibration preceded the graph-retention fix and is preserved as development-only evidence. No hook-memory comparison should be made from it.

All original calibration requests, PIDs, stdout/stderr, update journals, initial/final checkpoints and source pins are retained under `.superpowers/sdd/review-diagnostics-plan/task-3-calibration`. Its `source-before-hook-fix.py` preserves the old runner. New launcher invocations additionally preserve hash-named source snapshots.

Primary CUDA runs never enable saved-tensor hooks. CUDA rows report live allocated/reserved bytes and per-phase reset peaks; whole-update peak is max across its phase peaks. Phase time includes synchronization and excludes between-phase instrumentation overhead; it is not a claim of uninstrumented training throughput. RAM is reported separately as current working set, lifetime peak working set and private committed memory. System CPU counters bracket the worker and record observed host busy fraction; external CPU training may continue concurrently.

Diagnostic hooks report only backing storage referenced by autograd saved tensors, including parameters/inputs when saved and excluding tensors not saved for backward. Aliased storage is counted once while a saved reference is live. This is not total live allocation or resident VRAM. Nonreentrant checkpoint's internal hooks can shadow the outer diagnostic hooks, so checkpoint-interior saves are incompletely observed. Do not compare hook totals as if all arms had identical observability.

## Proposed main protocol — pending review and GPU calibration

Keep width32,heads2,batch8,float32, chunk8 and lengths32/128/512/1024. Each arm/length runs in one fresh child process, with two real AdamW warmup updates then three measured updates. lr0.003,weight_decay0.01,foreach=False,no clipping; same deterministic seed and batches across arms. No batch/precision changes on OOM. OOM is a retained condition outcome. The 20 primary conditions perform100 real updates in total. A separate hook ladder may run after primary measurements, explicitly as diagnostic data. Based on CPU L32/L128 calibration, the full ladder remains reasonable; exact GPU duration and OOM boundary remain unmeasured.

Freeze source and protocol only after review. Then fresh-process CUDA equivalence uses L32 and128, all five arms, and checks raw final gradients, final model weights, all AdamW states and measured losses for full/checkpoint and detach/each segment arm. Default tolerance rtol3e-4,atol3e-6 is predeclared for float32; CPU double tests use stricter rtol2e-9,atol2e-11. Main runner requires matching source/protocol-equivalence evidence.

Runtime: `__PY_ENV_DIR__/.venv/Scripts/python.exe`. Run commands from `__RLT_ROOT__/2026-09-17-diagnostics`:

```powershell
$py = '__PY_ENV_DIR__/.venv/Scripts/python.exe'
# Only after parent grants GPU slot:
& $py -m review_cycle.memory_bench pilot --device cuda --arm full --length 32 --out .superpowers/sdd/review-diagnostics-plan/task-3-gpu-dev-full32
& $py -m review_cycle.memory_bench pilot --device cuda --arm segment --length 32 --out .superpowers/sdd/review-diagnostics-plan/task-3-gpu-dev-segment32
# Only after review and protocol freeze decision:
& $py -m review_cycle.memory_bench freeze --device cuda --out results/review-diagnostics-r6
& $py -m review_cycle.memory_bench validate --out results/review-diagnostics-r6
foreach ($length in 32,128,512,1024) {
    foreach ($arm in 'full','detach','checkpoint','segment','segment_encoder_checkpoint') {
        & $py -m review_cycle.memory_bench run --out results/review-diagnostics-r6 --length $length --arm $arm
        if ($LASTEXITCODE -ne 0) { throw 'R6 worker failed; preserve evidence and inspect' }
    }
}
# Separate diagnostic example, not primary memory/timing:
& $py -m review_cycle.memory_bench run --out results/review-diagnostics-r6 --length 128 --arm segment --diagnostic
```

No main freeze or GPU execution was performed at this report checkpoint.


## Main measurements — 2026-09-17, frozen protocol 2eb5f530a8275b37082c076eefd11c4d5529a0f99157842e8fdfd95851877149

Status: **MEASURED**. 20/20 primary conditions PASS, 0 OOM, 0 FAIL, plus one hooks diagnostic (segment L128). GPU RTX 2060 SUPER, torch 2.12.0+cu130, float32, width32/heads2/batch8, chunk8, one fresh process per condition, two real AdamW warmups then three measured real updates. Driver log `.superpowers/sdd/review-diagnostics-plan/r6-driver.log`; frozen manifest `results/review-diagnostics-r6/frozen.json` (sha256 12b9ff7f76adb5bea3445ce6acf4f0b33644fb41792721e174cf5d1c9d60f55d); read-only summary `results/review-diagnostics-r6/summary.json` (sha256 145a0b34ceb426fd55f1c6d9ddb099a4338af9ef44763ffe0e286bbbc7a8d9c0) produced by `review_cycle/r6_report.py`.

Pre-freeze source change by the controller (after the independent pre-freeze review): `review_cycle/memory_bench.py` 3e5569e9… → 5830610d207c8dff2cdf238efa9f18c36167fac37174d1dcab2371773f804dde. Finding 1: the OutOfMemoryError handler now samples CUDA allocated/reserved and since-reset peaks. Finding 2: segment arms keep detached chunk losses on device and convert once after the loop, so segment latency no longer includes one implicit device synchronization per chunk; the reported segment loss is a float64 torch sum of the chunk losses. The four-hunk diff is `task-3-controller-fix.diff`; 7/7 unit tests `task-3-green-controller-fix.log`. No gradient route changed.

Fresh-process CUDA route validation before main (`validation/equivalence.json`, sha256 ca577a9a8705d3ebe96a93848966ef0a3d3cd0f7c881f0eb07191c764487fbd0): at L32 and L128, all five arms, 175 compared tensors each (model, AdamW step/exp_avg/exp_avg_sq, final gradients) plus the three measured losses, tolerance rtol 3e-4 / atol 3e-6. Maximum absolute differences: full vs checkpoint 0 (L32) and 3.13e-7 (L128); detach vs segment 1.43e-7 / 1.53e-7; detach vs segment_encoder_checkpoint 1.43e-7 / 8.20e-8. The segment implementation therefore reproduces the original detach-only TBPTT8 gradient route within float32 noise, and whole-forward checkpoint reproduces the full route.

### Results (mean of three measured updates; MiB = 2^20 bytes)

| Uzunluk | Kol | Durum | Güncelleme s (3 ort.) | Full oranı | CUDA tepe alloc MiB | CUDA tepe reserved MiB | Süreç WS tepe MiB | Host meşgul |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32 | full | PASS | 0.2318 | 1.000 | 19.0 | 24.0 | 1004.1 | 0.15 |
| 32 | detach | PASS | 0.2262 | 0.976 | 19.0 | 24.0 | 1004.1 | 0.15 |
| 32 | checkpoint | PASS | 0.4235 | 1.827 | 19.0 | 24.0 | 1012.5 | 0.16 |
| 32 | segment | PASS | 0.2570 | 1.109 | 18.1 | 22.0 | 999.5 | 0.19 |
| 32 | segment_encoder_checkpoint | PASS | 0.2422 | 1.045 | 17.6 | 22.0 | 998.7 | 0.18 |
| 128 | full | PASS | 0.8960 | 1.000 | 25.4 | 30.0 | 1028.6 | 0.16 |
| 128 | detach | PASS | 0.9076 | 1.013 | 25.4 | 30.0 | 1030.4 | 0.17 |
| 128 | checkpoint | PASS | 1.6560 | 1.848 | 25.4 | 30.0 | 1060.5 | 0.17 |
| 128 | segment | PASS | 0.9174 | 1.024 | 20.7 | 26.0 | 998.8 | 0.15 |
| 128 | segment_encoder_checkpoint | PASS | 0.9441 | 1.054 | 20.0 | 26.0 | 999.7 | 0.15 |
| 512 | full | PASS | 5.2989 | 1.000 | 51.6 | 56.0 | 1132.5 | 0.50 |
| 512 | detach | PASS | 3.7771 | 0.713 | 51.5 | 56.0 | 1128.1 | 0.20 |
| 512 | checkpoint | PASS | 6.7656 | 1.277 | 51.6 | 56.0 | 1223.4 | 0.19 |
| 512 | segment | PASS | 3.7412 | 0.706 | 31.2 | 36.0 | 1023.4 | 0.17 |
| 512 | segment_encoder_checkpoint | PASS | 3.7991 | 0.717 | 30.1 | 36.0 | 1023.4 | 0.57 |
| 1024 | full | PASS | 7.2807 | 1.000 | 86.6 | 108.0 | 1216.0 | 0.16 |
| 1024 | detach | PASS | 7.9046 | 1.086 | 86.5 | 108.0 | 1215.8 | 0.22 |
| 1024 | checkpoint | PASS | 14.2245 | 1.954 | 86.6 | 108.0 | 1408.9 | 0.21 |
| 1024 | segment | PASS | 7.2316 | 0.993 | 45.5 | 68.0 | 1023.8 | 0.16 |
| 1024 | segment_encoder_checkpoint | PASS | 8.3919 | 1.153 | 43.6 | 66.0 | 1024.9 | 0.15 |
| 128 | segment (hooks tani) | PASS | 1.0821 | - | 20.7 | 26.0 | 999.4 | saklanan depolama tepe 2.8 MiB |

Static allocation after the optimizer step (model, AdamW state, gradients, CUDA workspace) is 16.6–16.8 MiB in every arm, so activation-related peak is roughly peak minus 16.7 MiB.

### Reading

- **Detach-only TBPTT8 (original Core route) saves no memory.** Its CUDA peak equals the full route at every length (ratio 0.999–1.000). Detaching s/dk/dv shortens the gradient horizon, not the lifetime of saved activations; the whole-sequence graph still lives until the single backward. This confirms the review's source-level diagnosis with measured data.
- **Whole-forward nonreentrant checkpoint saves no peak memory here and costs 1.83–1.95× update time** (recompute of the whole forward inside backward rebuilds the same graph). It is not a memory tool for this model at these sizes.
- **Segment execution (backward per chunk, decoder graph released, one global bridge backward) lowers the CUDA peak monotonically with length:** 0.95× at L32, 0.81× at L128, 0.61× at L512, 0.53× at L1024 (86.6 → 45.5 MiB allocated; reserved 108 → 68 MiB). Subtracting the 16.8 MiB static part, the activation peak at L1024 falls from about 69.8 to 28.7 MiB (≈0.41×). The remaining segment peak grows with length because the retained e/mk/mv leaves, their gradients and the global encoder graph scale with the sequence.
- **Adding encoder checkpointing to segment execution gains only 1–4 percentage points more** (0.50× at L1024) at 1.05–1.15× the segment time: the encoder/projection saves are a small part of the peak for width 32.
- **Latency:** segment execution costs +11% at L32 and +2% at L128, and is 0.99× at L1024. The L512 row is not a clean latency comparison: the full and segment_encoder_checkpoint workers recorded host busy fractions 0.50 and 0.57 (other rows 0.15–0.22), so their times are inflated by concurrent host load; treat L512 timing ratios as UNRELIABLE and use L128/L1024 for latency statements. Memory peaks are unaffected by host load.
- **Process RAM:** lifetime peak working set 999–1024 MiB for segment arms versus 1004–1216 MiB for full/detach and up to 1409 MiB for checkpoint; these include the Python/PyTorch/CUDA runtime and pinned host structures, not activation memory, and are reported separately from device memory.
- **Hooks diagnostic (segment L128):** peak deduplicated saved-tensor backing storage 2.83 MiB with 10,540 save requests per measured update; live saved storage returns to 0 after each update. This is saved-for-backward storage lifetime, not resident VRAM, and checkpoint-interior saves are not observable this way.

### Scope and limits

The result is a **measured, matched-route demonstration** that the memory reduction the review asked about is real for this Core at width 32 and requires segment execution (releasing the decoder graph per chunk), not the original detach-only truncation. Three measured updates per fresh process, one model width, one batch size, one GPU; no OOM boundary was reached at L1024 (peak 108 MiB reserved of 8 GiB). Larger widths, batches, precision changes, the 1.51M-parameter model, multi-GPU or training-throughput claims were not measured. Timing includes explicit synchronization at each phase mark and is scoped to the recorded host busy fraction. Equivalence was validated at L32/L128 only; the L512/L1024 arms rely on the same frozen source and the identical per-update losses recorded in their result files. No efficiency claim extends to a different gradient route: segment arms compute the TBPTT8 gradient, not the full-BPTT gradient, exactly like the original detach arm.

### Independent raw audit

`.superpowers/sdd/review-diagnostics-plan/r6-independent-audit.md` (+ `.json`, script `r6_raw_audit.py`): fresh auditor, CPU only, no producer import. Verdict PASS_WITH_EXPLAINED_DEVIATIONS, zero hard failures across eight checks and all 31 processes; summary.json reproduced with zero mismatches; per-tensor equivalence reproduced (worst difference 1.5% of tolerance); real updates, paired batches and route separation confirmed (full vs detach losses differ at every length, 1.2e-3 at L32 down to 8.9e-5 at L1024, while paired equivalent arms agree within 5e-8). Explained deviation: every `*.launch.json` pid differs from the worker's `result.json` pid because the venv `python.exe` is CPython's redirector (byte-identical to `Lib/venv/scripts/nt/python.exe`) that starts the real interpreter as a child process; the worker-written pid chain is consistent, so measurement identity rests on source hashes, request/result protocol identity and the worker files, not on the launcher pid. Informational: at L128 the validation and main processes end with different `final_state_hash` while their losses agree within 7.5e-9 — cross-process CUDA float32 nondeterminism inside the declared tolerance.
