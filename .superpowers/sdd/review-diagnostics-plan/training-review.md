# Independent Task 1 training review

Verdict: CHANGES REQUIRED before freezing/main campaign. One P1 restart/evidence-integrity blocker and one P2 evaluation-cache integrity finding. The defined scientific schedule and ordinary training path otherwise match the brief on inspected evidence.

Reviewed 2026-09-17 in E:/rlt-research/2026-09-17-diagnostics. Read training-review-brief.md, task-1-brief.md, progress.md rulings, docs/review-diagnostics-plan.md, review_cycle/training.py, training_data.py, test_training.py, immutable core/model.py and core/run.py. No implementation changes, GPU work, full campaign, or subagents. Bounded CPU behavioral checks only, using C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe -B.

## P1: Restart silently repairs corrupt completed evidence or rewinds completed training

Location: review_cycle/training.py:323-348 and 393-402. run_one reads and validates resume.pt, initial.pt, equal-label.pt and the timing step sequence. It does not validate a pre-existing completion receipt or terminal artifacts before overwriting last.pt, selected.pt and train.json. _read_complete already detects several such problems, but run_one does not call it before these mutations.

Independent real two-update CPU reproduction: protocol cpu, steps2, batch2, train_length4, validation_n4, validation_every1, equal_budget_step1, width8, heads1. Run step1 and save its resume/timing bytes; finish step2. Independently mutate one artifact and invoke run_one again:

| Mutation | Actual result |
|---|---|
| Delete selected.pt | Accepted, COMPLETE, updates2; file silently recreated |
| Replace last.pt with invalid bytes | Accepted, COMPLETE, updates2; evidence overwritten |
| Replace train.json with {} | Accepted, COMPLETE, updates2; receipt silently recreated |
| Restore coherent step1 resume.pt plus timing.jsonl, leaving completed last.pt/train.json | Accepted, COMPLETE, updates2; update2 runs again and contradictory terminal evidence is overwritten |

All four cases reproduced; no exceptions. The fourth case directly undermines claimed exact real update accounting because the process executes an already certified update again without flagging stale state. Even if deterministic replay gives the same weights, it loses the distinction between accepted optimizer updates and actual execution/cost. The first three also violate the explicit missing/stale/ambiguous checkpoint rejection and evidence-preservation requirements. run_selection resumes every development condition through run_one, so this path will be routinely reachable.

Required correction: when terminal completion evidence exists, validate the entire completed condition and return the existing receipt without rewriting it. Reject contradictory or partial terminal bundles, including terminal artifacts paired with a shorter resume history. For legitimately interrupted finalization, define an explicit auditable state transition rather than silently repairing by normal run-one. Add behavior tests for all four cases and confirm unchanged completed reruns preserve hashes/receipt bytes. Preserve the existing valid interrupted-partial resume A/A test.

## P2: Cached evaluation accepts modified or impossible metrics without raw consistency validation

Location: review_cycle/training.py:509-517 (cached ep.exists branch). Cached evaluation validates protocol, checkpoint bytes and each listed raw file hash, but trusts cached metric fields and completeness of lengths/neutral rows. A modified summary can therefore become the aggregate evaluation again despite raw data showing a different result.

Independent CPU reproduction: one-update width8 run with test_n4, test_lengths[4], neutral_slots[0]. Evaluate partial scope, edit selected-evaluation.json lengths[0].final_accuracy to 1.2345, call evaluate again. Returned metric is 1.2345, accepted unchanged. The raw .npz and checkpoint were not changed.

Correction: authenticate saved evaluation metadata and validate expected variant/condition/row inventory; preferably recalculate accuracy, histograms and paired changes from raw arrays on cache reuse, or have the mandatory independent auditor reject any mismatch before status COMPLETE can be claimed. Tests should mutate an accuracy field, remove a declared length/neutral placement, and alter the variant/condition. This can be addressed before final evaluation rather than expensive training, but should not remain in the final evidence path.

## Verified behavior and scope

- Existing behavioral suite: 16 tests PASS, 5.449 seconds test execution. Command: Python -B -m unittest review_cycle.test_training -v. Tests are actual small forward/backward/optimizer and data checks, not textual-source comparisons.
- Conditions: loops deduplicate exact R2/R3/R4 overlaps into 36 small conditions; 9 development LR/seed combinations and 3 fresh confirmatory seed conditions follow frozen LR choice.
- Initial state pairing: Core initialization seed depends on condition seed, not supervision/arrangement/LR; no-feedback uses the same allocated parameter set and changes feedback behavior. Train data seeds omit arrangement/supervision/mode/LR, preserving paired inputs within task/seed. Sparse positions have a separate RNG namespace.
- Five-state recurrence agrees with immutable core oracle and hand-computed test. Blocking keeps token histogram and nonneutral order; prefix labels recomputed and final equality checked. Placement cycles globally prefix/middle/suffix; counts can differ by one because total examples are not divisible by three.
- Dense64 label budget is 64*32*32 = 65,536, equal to 2048*32 one-target labels. Receipt reports input sequences/tokens and updates so this does not imply equal examples/compute. Dense companions are actually included in evaluation variants, not merely saved.
- TBPTT8 calls immutable Core.forward(tbptt=8), whose boundary detaches s/dk/dv while preserving global encoder/projected-memory gradient routes. It is not full-route memory truncation; reporting must retain the frozen route description. This review inspected the actual route but did not independently numerically certify TBPTT gradient equivalence.
- Supervision loss routing has actual gradient-support tests; sparse RNG/input/initialization independence and valid resumed/uninterrupted final+optimizer state equality pass.
- Validation uses separate source namespace. R5 LR selection averages selected validation final accuracy over all three development seeds, chooses lowest LR on ties, then fresh seeds217/243/291. Main evaluation excludes development conditions.
- Default complete evaluation requires small and confirmation completion plus LR selection. Explicit Python condition_ids permits PARTIAL_SCOPE before other selections; this exception is declared in the frozen protocol and used by tests, but parent should use complete CLI evaluation for scientific main results.
- Evaluation saves predictions, targets, inputs, IDs, source seeds, hashes, final class histograms, and neutral correct-to-wrong/wrong-to-correct transitions using matched base cases. Expected final selected conditions: 39; dense equal-label variants are additional outputs.
- Source/protocol drift gates and per-update finite checks, optimizer-step checks, in-flight marker refusal, timing ledgers and checkpoint hashes are implemented and ordinary tests pass. Their existence does not close the P1 completed-restart loophole.
- Device ruling matches code: default small stage CPU, development/confirmation CUDA when available; stage device assignment frozen and actual receipt field tested. No CUDA training/calibration was run in this review.

## Remaining limitations / delivery status

- No full 2048-update campaign, long length1024 GPU evaluation, scientific effect or cost validation is claimed by this review.
- Runtime versions are recorded in frozen manifest but verify_frozen checks protocol/source identity rather than revalidating runtime/environment equality. Keep the exact chosen executable/environment unchanged during campaign.
- notes/training-implementation.md was absent at review time; implementer may still be preparing it. Existing RED evidence and resource calibration are not reproduced by this review.
- Fix P1 and run focused regression tests before freezing source hashes. P2 must be closed or explicitly caught by the independent raw-evidence audit before final COMPLETE/report claims.
