# Scheduler revision, before companion execution

Root authorized using a newly free fourth CPU thread on2026-09-16. Final-only training has no dependency on the primary model checkpoints: it initializes fresh models from paired seeds and uses its own frozen splits. The primary live process and its frozen source/protocol are untouched.

Only idle `core.followup` and `core.finish` schedulers were stopped, after confirming both output logs empty and no diagnostic/companion outputs. Their original source files/logs are preserved. `followup_v2` waits for primary completion and launches diagnostics only. The final-only companion is launched directly as its own CPU1thread process. `finish_v2` waits for BOTH complete learned experiment streams,9diagnostic rows and21evaluation seals before benchmarks/reports/binding.

The scientific budgets, model, selection rule, data streams and protocol hashes are unchanged. Training wall times include this additional CPU contention; they are not isolated hardware benchmarks.

## Finishing scheduler version3

Root audit requested quiescent writers before final binding. The still-idle finish_v2 scheduler is replaced by finish_v3, which runs benchmarks/plots/report generation and then exits without binding. The worker will verify that all core writer processes are stopped, bind manually, and only then refresh the derived manifest after output logs close. This avoids the finishing scheduler's own final output invalidating a preliminary log hash. Frozen scientific runners and protocols remain unchanged.
