# Connected evaluation independent launch preflight

Verdict: PASS TO LAUNCH for the declared intended-query estimands. No launch-blocking leakage, pairing, inverse-source-label or missing-query-denominator defect found. This is authorization to measure, not a positive result. Do not open datasets until source/protocol freeze is complete.

Reviewed integration/evaluate_system.py, current integration/pipeline.py and its adapter correction, test_connected_evaluation.py, pipeline regression tests, learning/combined-protocol.json, and latest core/learning integrity guards. Existing16-test GREEN evidence accepted; unchanged suites were not rerun. Fresh final seeds104731/104738 were not generated or opened. Only one manually specified score fixture was executed; no model/evaluation dataset was needed.

The512 original tasks are shared across3 joint parser/model seeds and4 predefined conditions (6144 episode runs). Checkpoint arm/step are fixed combined600, not selected on connected test outcomes. Rounded original facts are labeled through an independent full-rank linear least-squares system. Rank/residual checks reject ambiguous or inconsistent labels. Label fields remain outside run_text_episode inputs. Inverse absolute/edge facets coexist. All model/parser hashes are recorded before fresh test creation and checked again at completion.

Missing/rejected/misparsed intended queries remain in the denominator. Missing predictions remain None, never zero loss. A certificate for the wrong requested entity counts as a source failure even if its number coincides with the intended answer. Confidence intervals are restricted to independent final cases for one fixed model; correlated within-episode queries receive no binomial interval. Shared cases/seeds must remain paired in downstream contrasts; no pooled independent-sample claim is justified. Emitted-only MSE must always accompany missing count and all-intended accuracy.

The adapter correction at pipeline.py:94-101 checks batch cardinality before opening an archive, normalizes contradictory valid/abstained or multiple-event outputs into rejection, and preserves raw_adapter_output in the archived observation. Thus the prior pipeline adapter P2 finding is closed for this entrypoint. Direct memory.api.observe_text retains its earlier permissive predicate, but connected execution now sanitizes before calling it.

Latest integrity guards supersede earlier findings: learning/safe_run.py now authenticates protocol SHA, unique original frozen receipt, source index and source bytes before delegation. core/safe_entrypoint.py now validates condition/seed/budget/length completeness, raw evaluation receipt hashes, first-observation seals, checkpoint hashes, shapes and prediction metrics before binding. These fixes close the previously identified missing checks at code level; completion still requires actually running the guards on complete artifacts. Neither post-run binding nor self-hashes represent a privileged tamper-proof signature.

Nonblocking reporting gap in reviewed evaluator: score_episode only counts spurious_query_outputs for outputs at positions where original text requested no query. Those outputs are excluded from accepted_count/accepted_failures over intended queries. A manually constructed certified output at a source fact position produced spurious_query_outputs=1, accepted_count=0, accepted_failures=0. Current intended-query risk is correctly scoped, but must not be described as risk over every system certificate. Prefer recording spurious_accepted_outputs and treating them as source-request failures in a separate all-emitted certificate summary before freeze. All raw outputs are retained, so this gap does not invalidate intended-final accuracy or require new labels.

Other required boundaries are already declared: observed/query-only transfer omits the training no-write continuation tokens; canonical IDs use first appearance; heldout templates are a known prior evaluation family with fresh entities/numbers; compact resets neural trajectory while replay reproduces earlier actual budgets; capacity bounds resident raw records rather than total RAM; title-round savings are not total wall-time savings; keyed replay does not test vector retrieval. Timing includes archive/verification/replay and local contention.

Reviewed SHA256 identifiers:
- integration/evaluate_system.py: a69a7c7f111b0c0c69376b7da55e03073b0aadf98c721a046b0b43e5a9084394
- integration/pipeline.py: 875a956224e545daf57f4a876f8260d2644b8c403cef406aa7015eb8f266d589
- core/safe_entrypoint.py: a6108c4c03896a605992ed060f50efbce0baa79a6716f7abf75c94e72fb742d7
- learning/safe_run.py: f8a05386edd5fa7cfab05f111d6bf90f8a963455ac4e5755d1748bd6aaf9a068

Any subsequent evaluator metric amendment should be frozen and tested before final opening. Preserve this review as the assessment of the exact hashes above.

## Final pre-freeze amendment review — supersedes metric-gap note above

Root added spurious_accepted_outputs and all_certificate_outputs. Re-reviewed evaluator SHA256 3d8933dd0cfba7400250cbe2ff5b2ed0c1f2fe2eea27c4f760ad0f0007019cac. Every spurious certificate now enters both the all-certificate denominator and source-failure numerator. Intended final-case estimands remain unchanged; the combined correlated-certificate summary correctly omits a binomial interval. New regression explicitly checks count1/failure1/rate1 for a certified output at a source fact position. Root reports RED1 then GREEN17; test source reviewed, no redundant rerun.

Final verdict: PASS TO FREEZE AND LAUNCH. No remaining launch blocker. Earlier metric-gap recommendation is closed by this amendment. Fresh final seeds were not opened by this independent reviewer.

## Operative-heading final delta — current launch authority

Reviewed stable-heading storage routing before final dataset opening. Each accepted event is archived under the HeadingRegistry-returned heading ID. Global replay merges each title history by raw archive record ID, preserving cross-title chronology rather than concatenating title histories. Compact mode rebuilds each heading's current boundary and resets the neural runtime explicitly. Model traces bind named slots to stable heading IDs. Missing headings quarantine inference. Absolute/edge facets remain together under the same child heading, so inverse evidence is retained.

Independent regression execution: tests/test_integration_pipeline.py and tests/test_connected_evaluation.py, CPU one thread,18 passed in0.49s. Evidence: audit/review/operative-heading-independent.xml. Existing fixture seeds990001/990002 only; final seeds104731/104738 were not opened. No source changes by reviewer.

Current approved SHA256:
- integration/pipeline.py: 0c64c1df469db5a38f3c235c8efadd486268db7fe731ee9aabf4f8768f4327aa
- integration/evaluate_system.py: 0f3de3abc5600ac9ae3d014f0ecfde1c83dc6c2a26dd0f0010c00003e18589b9

Final verdict remains PASS TO FREEZE AND LAUNCH with no remaining blocker. Precise reporting boundary: the single TitleController has one global resident-record capacity shared across headings; heading-owned version histories and compact boundaries are now operational, but separate per-heading capacity quotas are not implemented. Compact mode iterates headings in stable-ID order, so its token ordering/canonical mapping can differ from chronological replay; it remains an explicitly different memory/execution policy, not a causal test of numeric compression alone. Previous timing fixtures preceded this routing delta and should retain that scope.
