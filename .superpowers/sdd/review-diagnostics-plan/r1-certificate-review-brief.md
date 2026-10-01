# R1 proof implementation review

Read notes/R1-proof.md as the specification, then review_cycle/certificate.py and
review_cycle/test_certificate.py. Baseline is memory/api.py::verify_candidate.
The parent has not frozen or run heldout R1 data yet. This review concerns only
the verifier contract and implementation, not the still-pending benchmark.

Check mathematical soundness, actual source consistency and scope/version
checks, floating point boundary behavior, edge direction/revisions, missing
off-path candidate handling, and whether independent certificate audit can be
fooled into accepting a wrong query. Identify concrete counterexamples or
substantive missing tests. Avoid cosmetic refactors. Do not edit implementation
or source files; tests may be run in memory or in this plan's scratch directory.
No subagents, no GPU, no training, no old evidence writes.

Use Python C:/Users/pc/Documents/Codex/2026-09-16/rlt-derin-arastirma/.venv/Scripts/python.exe
with -B. Write the full report to this directory/r1-certificate-review.md;
return status, evidence-backed findings with file/line, and test summary.
