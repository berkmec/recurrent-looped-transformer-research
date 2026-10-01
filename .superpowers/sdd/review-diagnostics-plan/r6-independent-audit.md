# R6 independent raw audit

Generated 2026-09-17T19:37:59Z by `r6_raw_audit.py` (torch 2.12.0+cu130, CPU, 1 thread, read-only). Verdict: **PASS_WITH_EXPLAINED_DEVIATIONS** (0 FAIL).

Scope: 20 main + 1 hooks diagnostic + 10 validation conditions recomputed from raw files; summary.json used only as a comparison target. Three measured updates per process are timing repeats of one process, not independent samples; no intervals are computed and nothing is pooled across lengths.

## Verdict summary

- Hard failures: 0. Explained deviations: 1 (see below). Summary.json (sha256 145a0b34ceb4...) reproduced with 0 mismatches across all 21 rows.
- Segment vs full CUDA peak allocated ratio: L32 0.954 (18.1 vs 19.0 MiB), L128 0.813 (20.7 vs 25.4 MiB), L512 0.605 (31.2 vs 51.6 MiB), L1024 0.525 (45.5 vs 86.6 MiB); peak reserved ratio: L32 0.917, L128 0.867, L512 0.643, L1024 0.630.
- Segment+encoder-checkpoint vs full peak allocated ratio: L32 0.927, L128 0.786, L512 0.584, L1024 0.504.
- Detach vs full peak allocated ratio: L32 1.000, L128 1.000, L512 0.999, L1024 0.999 (no saving); checkpoint vs full: L32 1.000, L128 1.000, L512 1.000, L1024 1.000.
- Update-time ratio vs full at L128 / L1024: detach 1.013 / 1.086; checkpoint 1.848 / 1.954; segment 1.024 / 0.993; segment_encoder_checkpoint 1.054 / 1.153.
- Host busy fraction > 0.3 (timing for these rows is scoped to that load): full-L512@main 0.501, segment_encoder_checkpoint-L512@main 0.573. All other rows 0.147-0.257. Memory peaks are unaffected by host load.

## Check status

| # | Check | Status | Key numbers |
|---|---|---|---|
| 1 | integrity | PASS (explained deviation) | 3/3 source pins match=True; protocol_hash match=True; 31 request.json, 31 result.json checked; overlaps=[]; stderr tracebacks=0 |
| 2 | real_updates | PASS | 31 conditions: PASS/5 updates/[3,4,5]/AdamW step 5; distinct initial hashes=1 |
| 3 | paired_data | PASS | full vs detach max abs by L: {32: '1.22e-03', 128: '5.89e-04', 512: '1.10e-04', 1024: '8.88e-05'}; differ at every length=True |
| 4 | equivalence | PASS | L32 full/checkpoint 0.00e+00 (175 t); L32 detach/segment 1.43e-07 (175 t); L32 detach/segment_encoder_checkpoint 1.43e-07 (175 t); L128 full/checkpoint 3.13e-07 (175 t); L128 detach/segment 1.53e-07 (175 t); L128 detach/segment_encoder_checkpoint 8.20e-08 (175 t) |
| 5 | metrics | PASS | summary sha match=True; mismatches=0; counts={'planned': 20, 'pass_': 20, 'oom': 0, 'fail': 0, 'missing': 0} |
| 6 | host_load | PASS | flagged >0.3: [('full-L512@main', 0.501), ('segment_encoder_checkpoint-L512@main', 0.573)] |
| 7 | diagnostic | PASS | peak_unique_storage_bytes=[2962676, 2962676, 2962676], save_requests=[10540, 10540, 10540], end live=[0, 0, 0] |
| 8 | source_sanity | PASS | float() in chunk loop=[]; OOM handler samples=['memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved']; OOM in data=0 |

## Explained deviations (strict check failed; cause demonstrated from files on disk)

- integrity: launch.json pid != result.json pid in 31/31 conditions (all of them). Cause demonstrated: the launched executable C:\Users\pc\Documents\ChatGPT\DaxDream\.venv\Scripts\python.exe is byte-identical (sha256 21bb438c0d4a...) to CPython's venv redirector Lib/venv/scripts/nt/python.exe, which spawns the real interpreter as a child; Popen.pid is the redirector, os.getpid() is the child. launch.json therefore cannot bind the worker by pid on this host. The worker-side chain (started.json, updates.jsonl, stdout final JSON, result.json) is pid-consistent in every condition and launch.started <= result.started <= result.finished holds.
  - launched executable `C:\Users\pc\Documents\ChatGPT\DaxDream\.venv\Scripts\python.exe`: 274712 bytes, sha256 `21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082`; CPython redirector `C:\Program Files\Python311\Lib\venv\scripts\nt\python.exe` sha256 `21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082`; real interpreter `C:\Program Files\Python311\python.exe` sha256 `5f7b89a612c9b8af1d6456cdfcd1dbe5ca630849e79aebced9bee9a6694952ec`; is_redirector=True

## Recomputed headline (main measurements; MiB = 2^20 bytes; ratios vs full at the same length)

| L | arm | mean s | s ratio | peak alloc MiB | alloc ratio | peak reserved MiB | reserved ratio | static after opt MiB | host busy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.2318 | 1.000 | 18.95 | 1.000 | 24.0 | 1.000 | 16.65 | 0.153 |
| 32 | detach | 0.2262 | 0.976 | 18.95 | 1.000 | 24.0 | 1.000 | 16.65 | 0.154 |
| 32 | checkpoint | 0.4235 | 1.827 | 18.95 | 1.000 | 24.0 | 1.000 | 16.65 | 0.157 |
| 32 | segment | 0.2570 | 1.109 | 18.08 | 0.954 | 22.0 | 0.917 | 16.65 | 0.193 |
| 32 | segment_encoder_checkpoint | 0.2422 | 1.045 | 17.57 | 0.927 | 22.0 | 0.917 | 16.65 | 0.176 |
| 128 | full | 0.8960 | 1.000 | 25.43 | 1.000 | 30.0 | 1.000 | 16.66 | 0.162 |
| 128 | detach | 0.9076 | 1.013 | 25.42 | 1.000 | 30.0 | 1.000 | 16.66 | 0.169 |
| 128 | checkpoint | 1.6560 | 1.848 | 25.43 | 1.000 | 30.0 | 1.000 | 16.66 | 0.172 |
| 128 | segment | 0.9174 | 1.024 | 20.66 | 0.813 | 26.0 | 0.867 | 16.66 | 0.153 |
| 128 | segment_encoder_checkpoint | 0.9441 | 1.054 | 19.98 | 0.786 | 26.0 | 0.867 | 16.66 | 0.153 |
| 512 | full | 5.2989 | 1.000 | 51.56 | 1.000 | 56.0 | 1.000 | 16.71 | 0.501 |
| 512 | detach | 3.7771 | 0.713 | 51.53 | 0.999 | 56.0 | 1.000 | 16.71 | 0.205 |
| 512 | checkpoint | 6.7656 | 1.277 | 51.56 | 1.000 | 56.0 | 1.000 | 16.71 | 0.190 |
| 512 | segment | 3.7412 | 0.706 | 31.20 | 0.605 | 36.0 | 0.643 | 16.71 | 0.166 |
| 512 | segment_encoder_checkpoint | 3.7991 | 0.717 | 30.12 | 0.584 | 36.0 | 0.643 | 16.71 | 0.573 |
| 1024 | full | 7.2807 | 1.000 | 86.60 | 1.000 | 108.0 | 1.000 | 16.77 | 0.161 |
| 1024 | detach | 7.9046 | 1.086 | 86.54 | 0.999 | 108.0 | 1.000 | 16.77 | 0.222 |
| 1024 | checkpoint | 14.2245 | 1.954 | 86.60 | 1.000 | 108.0 | 1.000 | 16.77 | 0.213 |
| 1024 | segment | 7.2316 | 0.993 | 45.47 | 0.525 | 68.0 | 0.630 | 16.77 | 0.163 |
| 1024 | segment_encoder_checkpoint | 8.3919 | 1.153 | 43.64 | 0.504 | 66.0 | 0.611 | 16.77 | 0.147 |

## Paired losses (main, measured updates 3/4/5)

| L | full~checkpoint max abs | detach~segment | detach~seg_enc_ckpt | full vs detach abs diff per update | step-1 warmup spread (5 arms) |
|---:|---:|---:|---:|---|---:|
| 32 | 0.000e+00 | 2.980e-08 | 2.980e-08 | 5.84e-04, 6.76e-04, 1.22e-03 | 1.49e-08 |
| 128 | 0.000e+00 | 1.118e-08 | 1.118e-08 | 5.89e-04, 4.79e-04, 1.32e-04 | 3.73e-09 |
| 512 | 0.000e+00 | 3.725e-08 | 3.446e-08 | 1.10e-04, 6.38e-06, 5.67e-05 | 4.84e-08 |
| 1024 | 0.000e+00 | 4.983e-08 | 5.029e-08 | 8.88e-05, 7.69e-05, 8.68e-05 | 8.85e-09 |

Validation vs main reproducibility (same arm/length, separate processes): full-L32 loss max abs 0.0e+00 (final hash equal: True), detach-L32 loss max abs 0.0e+00 (final hash equal: True), checkpoint-L32 loss max abs 0.0e+00 (final hash equal: True), segment-L32 loss max abs 0.0e+00 (final hash equal: True), segment_encoder_checkpoint-L32 loss max abs 0.0e+00 (final hash equal: True), full-L128 loss max abs 0.0e+00 (final hash equal: False), detach-L128 loss max abs 0.0e+00 (final hash equal: False), checkpoint-L128 loss max abs 0.0e+00 (final hash equal: False), segment-L128 loss max abs 7.5e-09 (final hash equal: False), segment_encoder_checkpoint-L128 loss max abs 3.7e-09 (final hash equal: False)

Hooks run vs main segment-L128 loss max abs: 1.12e-08

## Equivalence (validation final.pt, recomputed; tolerance |a-b| <= atol + rtol*|b|, rtol 3e-4, atol 3e-6)

| L | base | candidate | tensors | max abs | worst abs/tol | violations | loss max abs | equivalence.json max abs | per-tensor match |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 32 | full | checkpoint | 175 | 0.000e+00 | 0.000 | 0 | 0.000e+00 | 0.000e+00 | True |
| 32 | detach | segment | 175 | 1.434e-07 | 0.015 | 0 | 2.980e-08 | 1.434e-07 | True |
| 32 | detach | segment_encoder_checkpoint | 175 | 1.434e-07 | 0.015 | 0 | 2.980e-08 | 1.434e-07 | True |
| 128 | full | checkpoint | 175 | 3.129e-07 | 0.010 | 0 | 0.000e+00 | 3.129e-07 | True |
| 128 | detach | segment | 175 | 1.527e-07 | 0.009 | 0 | 7.451e-09 | 1.527e-07 | True |
| 128 | detach | segment_encoder_checkpoint | 175 | 8.196e-08 | 0.005 | 0 | 7.451e-09 | 8.196e-08 | True |

## Host load (system-wide busy fraction over worker lifetime)

Flagged (> 0.3): full-L512@main = 0.501, segment_encoder_checkpoint-L512@main = 0.573

All rows: full-L32=0.153, detach-L32=0.154, checkpoint-L32=0.157, segment-L32=0.193, segment_encoder_checkpoint-L32=0.176, full-L128=0.162, detach-L128=0.169, checkpoint-L128=0.172, segment-L128=0.153, segment_encoder_checkpoint-L128=0.153, full-L512=0.501, detach-L512=0.205, checkpoint-L512=0.190, segment-L512=0.166, segment_encoder_checkpoint-L512=0.573, full-L1024=0.161, detach-L1024=0.222, checkpoint-L1024=0.213, segment-L1024=0.163, segment_encoder_checkpoint-L1024=0.147, segment-L128-hooks=0.239, full-L32(v)=0.165, detach-L32(v)=0.178, checkpoint-L32(v)=0.214, segment-L32(v)=0.156, segment_encoder_checkpoint-L32(v)=0.149, full-L128(v)=0.172, detach-L128(v)=0.186, checkpoint-L128(v)=0.257, segment-L128(v)=0.155, segment_encoder_checkpoint-L128(v)=0.152

## Hooks diagnostic (segment-L128-hooks)

diagnostic=True, status=PASS, peak_unique_storage_bytes per update=[2962676, 2962676, 2962676], save_requests per update=[10540, 10540, 10540], end-of-update live bytes=[0, 0, 0], phases with snapshots=['ready', 'encoder', 'segment_forward_backward', 'global_backward', 'optimizer']. Primary rows with snapshots: none.

Per-phase (phase, live bytes, peak bytes, save requests) for update 3: [('ready', 0, 0, 0), ('encoder', 2322832, 2322832, 55), ('segment_forward_backward', 2322832, 2962676, 10540), ('global_backward', 0, 2962676, 10540), ('optimizer', 0, 2962676, 10540)]

## Source sanity (frozen memory_bench.py)

sha256 match=True; snapshot identical=True; chunk loop lines 134-145: float() calls in loop=[], item/cpu calls=[], returns=['float(torch.stack(parts).sum(dtype=torch.float64))', 'float(value.detach())']; OOM handler line 317: assigns cuda_at_failure=True, samples=['memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'], recorded in result=True. OOM observed in data: 0 (handler is untested by data). Pre-fix file: {'sha256': '3e5569e94e32f21800ddd85cc09867a7ae1ed5968640943eb092e3d4118bc680', 'matches_memory_review_pin': True, 'had_float_in_loop': True, 'had_cuda_at_failure': False}.

## Notes (informational)

- validation result.json/request.json carry the frozen protocol with lengths=[32, 128] and repeats=3 (protocol_hash 947078310687...), as the producer's validate_routes constructs it; equivalence.json itself carries the frozen hash 2eb5f530a827.... All 21 measurement result.json files carry the frozen hash exactly.
- cross-process bitwise reproducibility (validation vs main, same arm/length, separate processes): final_state_hash identical for ['full-L32', 'detach-L32', 'checkpoint-L32', 'segment-L32', 'segment_encoder_checkpoint-L32'], different for ['full-L128', 'detach-L128', 'checkpoint-L128', 'segment-L128', 'segment_encoder_checkpoint-L128']; measured losses agree to <= 7.5e-09 in every case. This is CUDA kernel-level nondeterminism between processes, not a protocol difference; it is within the predeclared equivalence tolerance and does not affect any check.

## Files

- results: `.superpowers/sdd/review-diagnostics-plan/r6-independent-audit.json`
- script: `.superpowers/sdd/review-diagnostics-plan/r6_raw_audit.py`
