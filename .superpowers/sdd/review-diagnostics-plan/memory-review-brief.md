# R6 review before frozen main

Read task-3-brief.md and task-3-context.md, then review_cycle/memory_bench.py and
test_memory_bench.py. Legacy core/model.py is immutable. Source may gain a
host-load observation field before freeze; require final hash-specific review.

Check full/whole-checkpoint equivalence, detach-only/segment/encoder-checkpoint
equivalence, and negative route controls. Global e/mk/mv contribution must reach
original encoder/projections once with correct normalization; local s/dk/dv cuts
must match legacy TBPTT8. Real optimizer states/updates, not just logits, matter.

Check measurements: fresh process, initialized optimizer, synchronization,
allocation versus reserve versus processRAM, backward/live/peak and saved-hook
storage/lifetime dedup. Hook diagnostic runs must not contaminate primary CUDA
latency/memory; don't treat cumulative saves as peak. Check source/protocol pins,
OOM/failure accounting and no hidden batch changes. Do not assume GPU behavior
from CPU checks. CPU pilot files exist in task-3-calibration; GPU pilot pending.

Read-only reviewer; run bounded CPU tests if useful. No GPU, main runs, source
edits or subagents. Python C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe -B.
Return concrete findings promptly and save full report to memory-review.md in
this directory. Parent is running 2 CPU trainings + one GPU training condition;
measurement contention will be recorded, GPU exclusive ownership is mandatory.
