# Independent connected experiment audit

Status: **PASS_WITH_SCOPE_LIMITS**. Audited24 complete groups,6144 episode executions,512 shared source tasks and three paired parser/model seeds. Protocol `0b61590618f799ecece783cac7d43c08521f7d2ae901228f293a1df1c5a4e2ac`.

All source/checkpoint pins, prediction-file hashes and archive-file hashes match. Every raw, source-observation and compact payload SHA was verified. Per-title contiguous versions, preserved source-observation references, heading semantic/history routing and compact raw-prefix hashes were checked. Archive totals: {'raw_records': 253746, 'observations': 86016, 'compacts': 53739, 'traces': 172082, 'archive_bytes': 1852268544}.

Original-source labels were independently derived with signed graph propagation, never evaluator least squares. Maximum difference: 7.11e-15. All intended queries, missing denominators, source extraction exactness, source-relative candidate errors and certificate failure counts including spurious queries independently match recorded metrics.

Every actual archived model-trace count equals reported executed refinement totals. Replay/current phases, contiguous k, invariant t during refinement, prior replay budgets and reported final candidates agree with archived node values.

| Seed | Split | Condition | Diagnostic candidate accuracy | Certified correct yield | Certificate coverage | Accepted source failures | Missing final |
|---|---|---|---:|---:|---:|---:|---:|
| 17 | iid | adaptive-replay-seen | 0.9766 | 0.3242 | 0.3242 | 0 | 0 |
| 17 | iid | fixed-replay-seen | 0.9766 | 0.1914 | 0.1914 | 0 | 0 |
| 17 | iid | adaptive-compact-seen | 0.9727 | 0.3281 | 0.3281 | 0 | 0 |
| 17 | iid | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |
| 17 | depth8 | adaptive-replay-seen | 0.2930 | 0.0000 | 0.0000 | 0 | 0 |
| 17 | depth8 | fixed-replay-seen | 0.2930 | 0.0000 | 0.0000 | 0 | 0 |
| 17 | depth8 | adaptive-compact-seen | 0.2227 | 0.0000 | 0.0000 | 0 | 0 |
| 17 | depth8 | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |
| 43 | iid | adaptive-replay-seen | 0.9883 | 0.4141 | 0.4141 | 0 | 0 |
| 43 | iid | fixed-replay-seen | 0.9883 | 0.2812 | 0.2812 | 0 | 0 |
| 43 | iid | adaptive-compact-seen | 0.9883 | 0.4492 | 0.4492 | 0 | 0 |
| 43 | iid | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |
| 43 | depth8 | adaptive-replay-seen | 0.3359 | 0.0000 | 0.0000 | 0 | 0 |
| 43 | depth8 | fixed-replay-seen | 0.3359 | 0.0000 | 0.0000 | 0 | 0 |
| 43 | depth8 | adaptive-compact-seen | 0.2266 | 0.0000 | 0.0000 | 0 | 0 |
| 43 | depth8 | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |
| 91 | iid | adaptive-replay-seen | 0.9805 | 0.4297 | 0.4297 | 0 | 0 |
| 91 | iid | fixed-replay-seen | 0.9805 | 0.2812 | 0.2812 | 0 | 0 |
| 91 | iid | adaptive-compact-seen | 0.9688 | 0.4609 | 0.4609 | 0 | 0 |
| 91 | iid | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |
| 91 | depth8 | adaptive-replay-seen | 0.5000 | 0.0039 | 0.0039 | 0 | 0 |
| 91 | depth8 | fixed-replay-seen | 0.5000 | 0.0000 | 0.0000 | 0 | 0 |
| 91 | depth8 | adaptive-compact-seen | 0.1680 | 0.0000 | 0.0000 | 0 | 0 |
| 91 | depth8 | adaptive-replay-heldout | 0.0000 | 0.0000 | 0.0000 | 0 | 256 |

## Conditional depth curves

Curves use actual fixed-replay-seen final-query traces for k1..8; earlier queries remain8,8. They describe diagnostic candidate quality, not certification or a whole-system computation-budget sweep. Missing outputs remain in accuracy denominators and are excluded from emitted-only MSE with counts reported.

| Seed | Split | k | Candidate accuracy | Emitted MSE | Missing |
|---|---|---:|---:|---:|---:|
| 17 | iid | 1 | 0.9609 | 0.0020165519215599066 | 0 |
| 17 | iid | 2 | 0.9766 | 0.001976656023630515 | 0 |
| 17 | iid | 3 | 0.9727 | 0.0019644071622344555 | 0 |
| 17 | iid | 4 | 0.9727 | 0.0019161826996599081 | 0 |
| 17 | iid | 5 | 0.9766 | 0.0018433231653550996 | 0 |
| 17 | iid | 6 | 0.9766 | 0.0017762515686028514 | 0 |
| 17 | iid | 7 | 0.9766 | 0.001732691797576511 | 0 |
| 17 | iid | 8 | 0.9766 | 0.0017241733414795628 | 0 |
| 17 | depth8 | 1 | 0.2656 | 0.31296230615053916 | 0 |
| 17 | depth8 | 2 | 0.2852 | 0.32073482842946777 | 0 |
| 17 | depth8 | 3 | 0.3125 | 0.3097563212126782 | 0 |
| 17 | depth8 | 4 | 0.3164 | 0.3082521624938354 | 0 |
| 17 | depth8 | 5 | 0.2539 | 0.313369255410175 | 0 |
| 17 | depth8 | 6 | 0.2812 | 0.31433671183773315 | 0 |
| 17 | depth8 | 7 | 0.2969 | 0.31358708371367466 | 0 |
| 17 | depth8 | 8 | 0.2930 | 0.31230002869791385 | 0 |
| 43 | iid | 1 | 0.9961 | 0.0010342168414298467 | 0 |
| 43 | iid | 2 | 0.9883 | 0.0012882846607098885 | 0 |
| 43 | iid | 3 | 0.9883 | 0.0012413176090403828 | 0 |
| 43 | iid | 4 | 0.9922 | 0.0011674045591412519 | 0 |
| 43 | iid | 5 | 0.9922 | 0.0011387611005259331 | 0 |
| 43 | iid | 6 | 0.9922 | 0.0011232452607610442 | 0 |
| 43 | iid | 7 | 0.9922 | 0.0011235601095275398 | 0 |
| 43 | iid | 8 | 0.9883 | 0.0011343296112253855 | 0 |
| 43 | depth8 | 1 | 0.3008 | 0.30699408415060647 | 0 |
| 43 | depth8 | 2 | 0.2969 | 0.3052535819038806 | 0 |
| 43 | depth8 | 3 | 0.4258 | 0.2964884135193118 | 0 |
| 43 | depth8 | 4 | 0.3516 | 0.29671103058342346 | 0 |
| 43 | depth8 | 5 | 0.2969 | 0.29490206860161133 | 0 |
| 43 | depth8 | 6 | 0.2578 | 0.29424571332209004 | 0 |
| 43 | depth8 | 7 | 0.2891 | 0.2941163510150945 | 0 |
| 43 | depth8 | 8 | 0.3359 | 0.29173643299209584 | 0 |
| 91 | iid | 1 | 0.9727 | 0.0017351187655474875 | 0 |
| 91 | iid | 2 | 0.9688 | 0.0016952376274255681 | 0 |
| 91 | iid | 3 | 0.9688 | 0.0016700514916011832 | 0 |
| 91 | iid | 4 | 0.9727 | 0.0016351505688928325 | 0 |
| 91 | iid | 5 | 0.9766 | 0.0016094454688939388 | 0 |
| 91 | iid | 6 | 0.9805 | 0.001595121577899051 | 0 |
| 91 | iid | 7 | 0.9805 | 0.0015960739495500006 | 0 |
| 91 | iid | 8 | 0.9805 | 0.0015992331498561696 | 0 |
| 91 | depth8 | 1 | 0.3203 | 0.30825288535399864 | 0 |
| 91 | depth8 | 2 | 0.3125 | 0.31627695750440105 | 0 |
| 91 | depth8 | 3 | 0.4102 | 0.3055292646728375 | 0 |
| 91 | depth8 | 4 | 0.3906 | 0.3072133747614624 | 0 |
| 91 | depth8 | 5 | 0.4570 | 0.30356594568939627 | 0 |
| 91 | depth8 | 6 | 0.5039 | 0.3007044458746431 | 0 |
| 91 | depth8 | 7 | 0.5078 | 0.2994505834054916 | 0 |
| 91 | depth8 | 8 | 0.5000 | 0.30001154420155474 | 0 |

## Limits

- Read-only final-state integrity checks support append-only history but cannot prove no privileged actor rewrote complete DBs and all hash receipts.
- Three joint parser/model seeds;6144 executions share512 tasks and are not6144 independent examples.
- A diagnostic numeric candidate can be correct without being certified; source-certified yield requires both.
- All certificate risk includes spurious query positions as source alignment failures.
- k1..8 curve varies final-query refinement only with earlier query refinements fixed8,8.
- No training or model execution performed by this audit.

Independent JSON: `E:\rlt-research\2026-09-16-e2e\audit\connected-independent-20260916T175138.json`. Streaming per-episode checks: `E:\rlt-research\2026-09-16-e2e\audit\connected-independent-20260916T175138-episodes.jsonl`. Curves: `E:\rlt-research\2026-09-16-e2e\audit\connected-independent-20260916T175138-depth-curves.json`.
