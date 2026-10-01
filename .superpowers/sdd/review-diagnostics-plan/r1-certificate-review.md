# R1 certificate implementation review

Date: 2026-09-17. Reviewer: independent bounded R1 reviewer.

Verdict: **CONDITIONAL PASS — numerical soundness supported; one reproducible contract defect requires disposition.** No false `VERIFIED` or successful wrong-value certificate forgery was found. The report-only floating conversion violates the unrestricted integer numeric contract at extreme magnitudes. This is a completeness/status defect, not demonstrated false acceptance. Benchmark and real candidate performance remain UNMEASURED by this review.

## Scope and source identity

Reviewed `notes/R1-proof.md`, `review_cycle/certificate.py`, `review_cycle/test_certificate.py`, and baseline `memory/api.py::verify_candidate` / `constraints`. No implementation, training, GPU, historical evidence or source files were changed. Tests ran with `C:/Users/pc/Documents/Codex/2026-09-16/rlt-derin-arastirma/.venv/Scripts/python.exe -B`; only this requested report was written.

SHA256 values at review:

| File | SHA256 |
| --- | --- |
| notes/R1-proof.md | 2CDF186E47646AF396524BB97148D09A5EE8879961BC114E41D954A6B2DA70F0 |
| review_cycle/certificate.py | AA051BA17B0AA07D65331E8E21E861B67165A8D5106BAD1D4EE1B8D0EB638361 |
| review_cycle/test_certificate.py | 0B9F1ADBED58729C7D4FF51464735F65DC7BB8137FF025739AD158B19A4425EF |
| memory/api.py | 826E7A185B9DA272F7DA559A6C01FC93ED54124406D2145447D416942EE88A92 |

## Finding F1 — report conversion can discard a valid exact decision [P2]

Location: `review_cycle/certificate.py:170-175`, specifically `bound=float(bound)` at line 171. Contract: `notes/R1-proof.md:28-33`.

The implementation computes the rational bound and rational threshold correctly, but construction of its result still converts the bound to float inside the outer exception handler. A finite Python integer or an exact residual larger than the float range raises `OverflowError`; the handler then discards the already constructed certificate and returns `UNRESOLVED_BUDGET`. Thus report formatting affects both status and certificate availability, contrary to the stated contract.

Executed reproducer:

```python
from review_cycle import certificate as c
big = 10**400
events = [dict(kind='root', child='q', parent=None, value=0)]
result = c.verify_path(dict(q=big), events, 'q', big)
print(result)
```

Observed:

```text
{'status': 'UNRESOLVED_BUDGET', 'bound': None,
 'reason': 'integer division result too large for a float',
 'checked_edges': 0, 'certificate': None}
```

The exact bound and threshold are both `10**400`; the contract therefore requires `VERIFIED`. With the experiment's ordinary small threshold the same extreme candidate should produce `CONTINUE` with its exact certificate, but it also loses that result. The large-threshold positive case is outside the preregistered benchmark thresholds, so this does not demonstrate a false acceptance or bias in those pending benchmark runs.

Requested fix: isolate conversion of the display field from the exact decision; preserve `status`, `bound_exact` and the certificate when no finite float representation exists. Alternatively, explicitly narrow the supported magnitude contract and validate the range, but then the unrestricted Python integer claim must be removed. Add a regression for both accepted and rejected exact bounds beyond float range.

## Mathematical and implementation assessment

The proof is sound under its explicit source premises. For a selected anchor and an oriented path, subtracting the true affine constraints from the candidate differences gives the query error as the signed sum of the anchor error and edge errors. The triangle inequality bounds its magnitude by the sum of absolute residuals. Consistent anchored components have a uniquely determined query value; unanchored unrelated components do not change it.

The implementation matches these premises:

- `parse_evidence`, lines 71-81, preserves separate `absolute:child` and `edge:child` facets and installs reverse traversal with a negated offset. Last chronological write to a facet wins, including a change of parent.
- Lines 84-103 check every active component, not only the query component. Exact potential conflicts and conflicting anchor origins are rejected. Superseded constraints are not incorrectly treated as active contradictions.
- `number` and `threshold`, lines 18-29, implement the declared distinction between exact IEEE represented values and the threshold's decimal string. The `.1` versus `nextafter(.1, 0)` boundary test has the correct outcome.
- Lines 134-155 implement multi-source Dijkstra on nonnegative exact residual costs, with each available anchor initialized by its own anchor residual. Strict improvements and nonnegative costs avoid predecessor cycles. It minimizes complete simple-path bounds, not all possible certificates; the proof carefully makes only that narrower claim.
- All supplied candidate values are validated, while unavailable nodes merely remove candidate paths. Missing off-path candidates can therefore pass; missing all complete anchor paths cannot.
- `candidate_answer` directly returns `candidate[query]`. No solver value is substituted as the model answer.
- `audit_certificate`, lines 178-212, independently reconstructs the selected path from active source facets and recomputes exact residuals. Its success requires a real anchored path ending at the caller's query, matching source and candidate digests, correct bound and threshold, and an exact bound within tolerance. It does not trust the shortest-path routine. It need not prove that the supplied path is shortest to prove sound acceptance.
- `verify_envelope`, lines 220-231, rejects mismatched task scope, query and complete source history digest before numerical verification. A changed historical event invalidates the envelope even when the final active constraints happen to match.

Scope qualification: the standalone numerical certificate/auditor does not carry or check task scope; `verify_envelope` is the context gate. This is consistent with the current layering only if consumers always apply that gate before treating the numerical result as task-applicable. Passing `audit_certificate` alone must not be reported as a scope/version-envelope audit. No final benchmark caller existed in the reviewed usage search, so this integration property was not claimed as verified.

Relative to the baseline, the new verifier deliberately strengthens global consistency and exact arithmetic while replacing a sum over all query-component constraints with a selected path sum. Baseline `constraints` also rejects `continuation`, whereas the new parser treats it as a no-op; benchmark inputs should document this difference if such events occur. These are contract differences, not an observed mathematical error.

## Test evidence

1. Repository tests: `python.exe -B -m unittest review_cycle.test_certificate -v` — **12/12 PASS**, 0.192 seconds, including the existing 240 linear-algebra comparisons.
2. Additional in-memory independent enumeration: **600/600 PASS**, deterministic `random.Random(17391)`. Each case used 2-7 nodes with known integer true values, random anchors, at most one current parent facet per child, possible cycles/disconnected components, and independently missing/noisy candidates. Every available simple anchor-to-query path was recursively enumerated independently of Dijkstra. The implementation bound equaled the minimum enumerated exact bound. Query error never exceeded that bound. Counts: **256 VERIFIED, 163 CONTINUE, 181 incomplete**. All **419 complete-path certificates** agreed with the independent audit's expected acceptance. Candidate errors were drawn from `0, 0, .03125, .0625, .125, 1`; threshold `.1`.
3. Additional in-memory tampering probes: changed version, changed query, changed candidate digest, changed threshold, wrong edge facet, and an actually wrong query value with a recomputed candidate digest — **all six rejected**. The last case ensures a content hash alone cannot pass a stale zero-residual certificate for a changed answer.
4. Additional inverse/revision case: retained `edge b-a=1`, revised absolute `b` from 2 to 3, query `a`, candidate `{a:2,b:3}` and zero tolerance — **VERIFIED and independent audit PASS**. Absolute revision correctly preserves the edge facet and reverse direction.
5. Extreme exact-integer/report-overflow reproducer — **F1 reproduced**.

These are finite implementation checks. They do not prove arbitrary natural-language interpretation, unrestricted reasoning correctness, or benchmark performance.

## Substantive missing persistent regressions

The existing suite is useful but leaves several contract clauses weakly protected. The additional probes above passed; preserving focused regressions would prevent future regressions:

- A consistent cyclic/multi-anchor case compared with independent enumeration of all candidate paths, including a missing node on the superficially shortest route. The existing randomized suite creates only trees with a single anchor and complete candidates.
- Inverse traversal after an absolute revision, changed-parent edge revision, and a consistent unanchored off-query component. Only forward same-parent revision is currently explicit.
- Audit rejection when the candidate query is changed and its digest is legitimately recomputed; also direct candidate digest/query/version/threshold/edge-facet tampering. Existing persistent audit tests change only path, bound and source digest.
- F1's out-of-range display conversion regression.

No additional implementation blocker was demonstrated. Close F1 or document a justified contract restriction, ensure the benchmark uses the envelope gate, and keep the pending heldout experiment status separate from this verifier review.
