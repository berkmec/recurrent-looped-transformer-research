# Learned extraction and non-destructive title memory: measured scope

Packages 5, 8, 9, 10, 11 have runnable implementations and scoped measurements. This worker did not implement RLT decoder internals; the official cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39 decoder pin applies to the separate core worker. The extractor is a deliberately separate small BiGRU, not a source-faithful RLT language model.

## Formal contracts

- Stable title identity is `(task,title)` supplied by the caller for a genuine subproblem. Versions append monotonically. Learned title segmentation is not claimed.
- Event facets `(absolute,child)` and `(edge,child)` are independently versioned: an absolute observation on a non-root node must preserve its parent relation. This supports forward and inverse affine evidence.
- Raw event records, source-text observations, parser provenance, and compact snapshots are append-only SQLite rows with canonical JSON SHA256. Hashing provides detection, not filesystem access control or protection against a privileged attacker rewriting both payload and hash.
- The active capacity counts resident raw assertion records. Compaction occurs only when the next append finds the window full. Semantic compact payloads are disk-backed; title registry metadata and transient reconstruction buffers are separately unbounded. This is not a total-RAM bounded algorithm.
- Structured compaction retains all latest affine facets and raw-history hash/version obligations. Late evidence invalidates the scope; rereading verifies raw hashes and appends a replacement compact snapshot. No raw row is overwritten by the API.
- Vector retrieval is deterministic 128-dimensional signed hashed bag-of-words cosine. Applicability requires task, title, entity and latest facet version. Exact-key access is a separate oracle upper bound.
- For consistent affine evidence, summing absolute candidate residuals over the anchored connected component upper-bounds query error along any anchor-to-query path. Relative graph potentials check evidence consistency only; they never replace the candidate. Unsupported, inconsistent, unanchored or incomplete candidates cannot certify.
- `CONTINUE`, `VERIFIED`, `UNRESOLVED_BUDGET`, `COMPACTION_REJECTED` have distinct meanings. VERIFIED concerns extracted constraints only; it does not certify faithfulness to original text. Root integration must assess source-relative false acceptances separately.

## Protocol and evidence preservation

Extractor protocol v1 was frozen before training/test evaluation. The last of 18 epochs and confidence threshold 0.65 were predeclared; no test-based selection or subsequent retuning occurred. v1/v2/v3 protocols, API/measurement source snapshots, archives, outputs and execution history remain preserved. v2 fixed independent absolute/edge facets after integration review, using fresh seed20260917 tasks. v3 moved compact semantic payload to disk and made retrieval distractors numerically wrong, using fresh seed20260918 tasks. These are declared engineering corrections, not three confirmations of one frozen experiment.
Final engineering protocol SHA256: `cce658466765424b59af28da85638ea2f67ba33c83a1d696ff4384ac36fd35d8`. Extractor protocol SHA256: `0450467e0dc5f66e723506ff0b3720e92ef28f418faaafff073e08e839f2d23c`.

## Package 5: MIXED, genuine learned extraction

Supervised BiGRU learns token roles and event classes; exact entity/number copying occurs only after learned selection. Regex is limited to tokenization/type validation. Each seed trains on 3072 sentences, validates on384 and evaluates768 disjoint-identity sentences (576 positive,192 rejection challenges). Entity IDs are disjoint across splits. Lexical/layout challenge templates are withheld. Oracle extracted events score1.0 by definition and are separately labeled.

| Seed | Seen-layout exact | Heldout paraphrase exact | All positive exact | Edge exact | Abstain rate | Negative false acceptance |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 1.0000 | 0.2839 | 0.5226 | 0.4598 | 0.5117 | 0.0000 |
| 43 | 1.0000 | 0.1797 | 0.4531 | 0.3156 | 0.6211 | 0.0000 |
| 91 | 1.0000 | 0.1094 | 0.4062 | 0.3532 | 0.6523 | 0.0000 |

The main negative result is weak transfer to withheld wording/order, despite perfect seen-layout exactness. Rejection challenges share finite negative construction families with training; this is not arbitrary negation comprehension. Do not pool the three seeds as independent model populations. Checkpoints and per-example predictions are retained. Training wall seconds per seed: 17=21.13, 43=17.28, 91=20.50.

## Packages 8 and 9: structured controller engineering

Fresh192 tasks:96 forward and96 inverse, lengths8/16/32/64, capacities4/8/16, three competing stable titles, and delayed internal revisions. There were1216 actual capacity-triggered compactions, zero premature compactions and zero resident raw-record budget violations. All192 raw histories restored exactly; all192 stale scopes were detected and repaired by reopening. There were10304 raw records and1584 compact snapshots, all hash-verified in a fresh process.

| Condition | Mean absolute final error |
|---|---:|
| full_raw | 0.000000 |
| boundary_compact | 0.000000 |
| answer_only | 2.000000 |
| mean | 2.240209 |
| forced_stale | 2.000000 |

Boundary-preserving compact/raw bytes averaged 0.9430; this is only modest compression because most constraints remain necessary. Mean raw records read per example=225.0. All answer-quality values in this table are explicit symbolic engineering checks, not learned-model predictions. Learned controller integration is a separate root experiment.

## Package 10: MEASURED_NEGATIVE retrieval

| k | Unfiltered recall | Filtered precision | Filtered coverage | Unfiltered top1 numeric damage |
|---|---:|---:|---:|---:|
| 1 | 0.0000 | None | 0.0000 | 1.0000 |
| 3 | 0.0260 | 1.0 | 0.0260 | 1.0000 |
| 5 | 0.0417 | 1.0 | 0.0417 | 1.0000 |

Filtering prevents returned stale/cross-task/entity-invalid records, but very low coverage means this is not successful retrieval. Exact-key oracle coverage is1.0 and is not used to hide vector misses. Missing-reference coverage is0.0. Final decoys use99/-99 while valid values are2; earlier v1/v2 decoys coincidentally shared value2, so their zero numeric damage was not informative and remains preserved.

## Package 11: actual sequential halting, synthetic candidates

| Maximum steps | Accepted coverage | Wrong acceptances | Mean executed steps | Saved steps |
|---|---:|---:|---:|---:|
| 1 | 0.0000 | 0 | 1.0000 | 0.0000 |
| 2 | 0.0000 | 0 | 2.0000 | 0.0000 |
| 4 | 0.0000 | 0 | 4.0000 | 0.0000 |
| 8 | 0.7500 | 0 | 5.7500 | 0.2812 |

Candidates are explicitly synthetic shared-offset residual trajectories, including25% stable-wrong examples; labels are not learned inference. At budget8, coverage0.75 with zero wrong acceptances and28.125% actual saved calls. Budgets1/2/4 have zero coverage and are unsuccessful certification settings. This does not establish learned-model halting gains; root owns that integration.

## Public integration API

```python
from memory.api import Archive, TitleController, load_parser, render_event, verify_candidate, halt_candidates
parser = load_parser("memory/checkpoints/extractor-17.pt")
parsed = parser.parse(text)          # or parser.parse_batch(list_of_texts)
archive = Archive(path)
controller = TitleController(capacity=4, archive=archive)
controller.observe_text(task, title, text, parsed)  # preserves abstentions and source text
raw_events = controller.reconstruct(task, title)
compact_events = controller.compact_events(task, title)
check = verify_candidate(candidate_by_original_entity, compact_events, query, tolerance=0.1)
result = halt_candidates(lazy_candidate_iterator, compact_events, query, tolerance=0.1, max_steps=8)
```

`render_event(event, style="seen"|"heldout", variant=0)` provides fixed public finite-language templates. Canonicalize entity names to numeric IDs per problem outside the parser; map learned node predictions back to original names for verification. Events contain observations only, never answer targets. A fresh-process smoke test preserves four accepted observations and two abstentions including original text.

## Verification and limits

16 unit tests pass, with preserved RED/GREEN evidence. Controls include raw/compact tampering, stale applicability, competing titles, multiple full compactions, late internal revisions, naive-summary counterexample, false stable candidates, inverse facet preservation, inconsistent anchors, source-text/abstention persistence, and no execution after early halt. Recomputed saved per-example extractor/memory/retrieval/halting metrics match. This worker recomputation is not an independent audit; root review remains a distinct closure gate.

Energy, unrestricted natural-language understanding, learned title segmentation, general reasoning summarization, production vector retrieval, total-RAM bounds and large-model scale remain UNMEASURED. Main experiment artifacts are `memory/results/metrics.json`, `memory/results/extractor-*.jsonl`, `memory/results-v3/*.jsonl`, `.npz`, checkpoints, archives, protocol files, execution.jsonl and source/environment manifests.
