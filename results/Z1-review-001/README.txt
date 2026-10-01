Independent review of the Z1 code, swarm-bridge task 001 (reviewer: jcode, a separate agent; task written by Claude; 2026-10-02).

Files: task-001.md (the task as given: scope, questions, output format, rules), ../../notes/Z1_INCELEME_001_jcode.md (the reviewer's report, unmodified),
jcode-tests/*.py.txt (the reviewer's 15 tests, renamed so that pytest does not collect them; run with PYTHONPATH=. after renaming back).

Audit protocol: every test was read before it was run; all 15 were run by the requester, single process, one thread, from the repository root, on the code
as it stood before the fixes: 11 failed (the findings) and 4 passed (the pins), exactly as reported. Each finding was then judged on its merits.

B   reviewer  reproduced  verdict                                   action (tests/test_z1_review_001.py pins the fixed behaviour)
B1  medium    yes         real: learned_in_range failed open        the trained-range point is always evaluated; the flag is always measured
B2  medium    yes         doc/partition: the claim "vanish from both metrics" is overstated (those positions are in `deep`); the docstring said "everything"
                                                                    docstring fixed; partition test (deep / reentry / resync cover every post-exceedance position once)
B3  medium    yes         real: U1 aborts on a non-finite step, gru_control skipped it. No effect on recorded results (Z1-G: 0 skipped steps, 0 non-finite validation losses in all 12 runs)
                                                                    abort like U1, failure record written, no result for that condition
B4  medium    yes         real test gap (tautology)                 U1 numbers pinned literally; a drifting sweep protocol is refused
B5  medium    yes         latent provenance trap                    depth is part of the training seed; depth=None keeps the old derivation (every recorded pilot stays reproducible)
B6  medium    yes         latent, API level (not reachable in the call graph)   heldout refuses the 'train' split
B7  low       pin passes  naming/doc: `deep` is saturated-label emission     comment + pin test (a constant-cap model ties a perfect counter on `deep`)
B8  low       yes         real test gap                             losses_from_gru checks the stored loss_pp against the accuracies
B9  low       yes         latent (also in the sealed U1 sweep.py, which is not changed)   a NaN/inf validation candidate is never selected; no finite candidate is an error
B10 low       yes         real                                      steps < 1 is a clean ValueError
B11 low       yes         by design (n is not in the seed; permutation draws nest)    documented in heldout, pinned by a test, not changed
B12 low       yes         real doc error                            docstring says 64x / 32x; default scales pinned

Hit rate: 12 of 12 findings reproduced; 11 accepted (8 code changes, 3 documentation), 1 documented as design; 0 false findings; 2 overstated (B2, B6).
Question 3 (is Z1-G really paired with U1?): confirmed by the reviewer and by the requester: same batches, validation/test problems and selection rule; the only
semantic difference was the non-finite handling (B3), now removed. Question 6 (the counter generator): confirmed independently by pins that were added to the suite.
