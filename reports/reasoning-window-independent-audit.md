# Independent reasoning-window companion audit

Status **PASS_WITH_SCOPE_LIMITS**. All12 complete conditions and6 paired comparisons audited:1536DBs,256 shared tasks. Protocol `54e37e67b107af932ceb2a22381547885944cc1139e83210a2bd6d30107c9de9`.

All source/checkpoint/data/prediction/DB hashes and raw/observation/compact payload hashes match. Independently propagated source labels differ by at most9.77e-15. All recomputed conditional candidate/certificate metrics match. Archive totals: {'raw_records': 94656, 'observations': 21504, 'trace_count': 70080, 'reasoning_snapshots': 8760, 'constraint_snapshots': 17868, 'archive_bytes': 778977280}.

All paired generated trace payloads are exactly identical, in addition to candidate values, certification statuses, budgets and consumed tokens. Every generated-state snapshot references exactly4 actual consecutive trace records, occurs at k4/8, stores the exact latest title latent state, preserves the queried heading identity, and has a guarded obligation digest. All enabled maxima are<=4; uncompacted maxima and serialized payload peaks independently reconstruct from raw traces. No raw trace was removed from observed append-only sequences.

| Seed | Split | Condition | Diagnostic accuracy | Certified correct yield | Max active generated traces | Mean snapshot/reread count | Mean archive bytes |
|---|---|---|---:|---:|---:|---:|---:|
| 17 | iid | uncompacted | 0.9922 | 0.2422 | 8 | 0.00 | 447392 |
| 17 | iid | window4 | 0.9922 | 0.2422 | 4 | 11.09 | 542304 |
| 17 | depth8 | uncompacted | 0.2891 | 0.0000 | 8 | 0.00 | 479744 |
| 17 | depth8 | window4 | 0.2891 | 0.0000 | 4 | 12.00 | 581632 |
| 43 | iid | uncompacted | 0.9922 | 0.3516 | 8 | 0.00 | 433696 |
| 43 | iid | window4 | 0.9922 | 0.3516 | 4 | 10.71 | 525472 |
| 43 | depth8 | uncompacted | 0.3594 | 0.0000 | 8 | 0.00 | 479744 |
| 43 | depth8 | window4 | 0.3594 | 0.0000 | 4 | 12.00 | 581632 |
| 91 | iid | uncompacted | 0.9688 | 0.3594 | 8 | 0.00 | 430816 |
| 91 | iid | window4 | 0.9688 | 0.3594 | 4 | 10.63 | 521952 |
| 91 | depth8 | uncompacted | 0.4922 | 0.0000 | 8 | 0.00 | 479744 |
| 91 | depth8 | window4 | 0.4922 | 0.0000 | 4 | 12.00 | 581632 |

## Limits

- Source/payload hashes, exact paired generated states and matching outputs independently verify observed fidelity. The complete runtime obligation fingerprint cannot be reconstructed solely from archived z; full KV/global-encoder/runtime state is not serialized. Guard completeness/order is supported by pinned source and prior mutation tests, not new full-runtime replay here.
- Reread counters match full boundary snapshots and source calls restore before candidate/sufficiency; no separate runtime reread event receipt exists.
- Active generated-history cap4 excludes model weights,encoder/KV,graph,live title state,source registry and reconstruction working memory. Serialized payload byte peaks are not measured process RAM.
- Uncompacted active history resets at everyquery, so comparison is8 versus4 local records, not unlimited cross-query history.
- Both arms certify only at4/8 boundaries; do not conflate these results with primary every-k adaptive stopping.
- 256 independent source tasks are shared across three seeds/two conditions;1536 executions are not1536 independent cases.
- No producer code, frozen tests, model parameters or prior results were modified.

Independent detailed output: `E:\rlt-research\2026-09-16-e2e\audit\reasoning-companion-independent-20260916T180847.json`. Streaming episode evidence: `E:\rlt-research\2026-09-16-e2e\audit\reasoning-companion-20260916T180847-episodes.jsonl`.

## Actual payload and timing costs

Producer wall time: 868.19s. Independent read-only audit wall time: 94.30s. These are different operations and are not summed into model inference cost.

| Seed | Split | Uncompacted episode seconds | Window4 episode seconds | Window4 / uncompacted | Mean peak serialized bytes: uncompacted | Mean peak serialized bytes: window4 |
|---|---|---:|---:|---:|---:|---:|
| 17 | iid | 0.554103 | 0.629047 | 1.135 | 54581.4 | 28012.3 |
| 17 | depth8 | 0.582414 | 0.583707 | 1.002 | 55868.2 | 28006.2 |
| 43 | iid | 0.487411 | 0.560525 | 1.150 | 54479.6 | 27962.7 |
| 43 | depth8 | 0.545086 | 0.664387 | 1.219 | 55833.4 | 27976.7 |
| 91 | iid | 0.549128 | 0.522339 | 0.951 | 54884.2 | 28040.3 |
| 91 | depth8 | 0.509839 | 0.541632 | 1.062 | 56001.2 | 28051.9 |

Window4 totals: 8760 full generated-history compactions and 8760 rereads. Same-heading continuation counter: 7992. This counter counts a later compact boundary under an already encountered queried heading within the same episode, including replay of earlier queries; it is not a count of new semantic discoveries or independent solved problems.

Serialized payload size counts retained generated trace JSON only. Model weights, current title state, graph, encoder/global-memory/KV caches and other working objects remain resident outside this buffer. The observed4-versus8 history bound is therefore not a total RAM reduction measurement. Episode timings include actual parsing/archive work and local contention; the added exact boundary persistence/restore checks have real overhead.

Cost details: `E:\rlt-research\2026-09-16-e2e\audit\reasoning-companion-cost-addendum.json`.
