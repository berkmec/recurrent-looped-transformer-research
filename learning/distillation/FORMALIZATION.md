# Actual successful trajectory distillation contract

This is a separate later-training experiment. It does not change the earlier oracle-supervision, primary or combined experiments.

For seed s the fixed teacher is combined-s-step600 and both students start from feedback-s-step600. Their original architecture and initial coordinates are identical; teacher weights have subsequently changed through supervised training. The teacher itself was trained earlier using oracle-derived intermediate supervision. The new distillation targets, however, are actual teacher predictions and title states, not solver targets.

On each of2048 previously designated TRAIN tapes, execute the teacher's real Runtime. For each of three query positions q, archive all six actual refinement states z[q,k,i,:], numeric node readouts u[q,k,i], and its final requested answer a[q]. Runtime state/cache semantics are the existing tested semantics; capture does not insert extra token steps. No target array is accessed by collection. Source TRAIN file hash, every teacher/student checkpoint hash and implementation hashes are frozen.

Let C[q] denote the observed affine connected component containing the requested node, found using topology only. At k=6, the existing memory verifier checks the candidate's root and edge residual sum against0.05. It does not return a repaired answer. Acceptance mask M[q] is1 only when this candidate is VERIFIED. A separate whole-episode test asks whether both components pass at all three query positions. The frozen eligibility rule prefers sufficiently represented whole episodes, otherwise falls back to verified query components. Actual collection produced zero whole episodes, so only the fallback is used. Every rejected full trace is retained.

For the fallback, node mask N[q,i]=M[q] times the indicator that i is in C[q]. Both arms use the same eligible episode indices, sampled batches, final targets a[q], and query mask M. Their final-answer loss is the masked mean squared difference from the actual teacher answer. The trajectory arm additionally uses0.1 times:

- masked mean squared difference between student and archived teacher latent title state z over q,k,i,dimensions;
- masked mean squared difference between student and archived teacher numeric node readout u over q,k,i.

Masked means divide by the actual number of accepted elements, not by the number of discarded elements. No uncertified node enters auxiliary loss. Both arms execute the same student forward trajectory; auxiliary backward work adds compute.300 AdamW updates, batch16, learning rate0.0005, weight decay0.0001, gradient clip1, all three seeds and fixed final checkpoints are declared before any new held-out opening.

Certification applies to the final numeric map, not to the truth of every intermediate candidate or to semantic correctness of latent vectors. The supervised trajectory is the actual path used by a teacher that ultimately succeeded on the certified component. Calling this universal thinking-style transfer would be unsupported. Shared coordinate architecture also limits portability to other models.

Filtering strongly favors shallow successes; some deeper inverse/affected cells have only1–3 eligible episodes. No rebalancing, tolerance widening or oracle repair follows this observation. Coverage and selection counts are reported by observed depth, direction and affected-component membership; those categories never enter the model or the certificate.

New IID and true depth8 datasets are generated only after all student training is complete. Their labels are used exclusively for evaluation. The inference export contains model/API/checkpoints and observation inputs only, with generator and trace modules absent from its isolated import path. This is an import-path/export boundary, not an operating-system security sandbox.

The scientific path tested is:

fixed trained teacher → actual generated states and candidates → observation-only final certification → immutable trace archive → paired later student training → fresh held-out, archive-off student inference.

A negative or mixed gain still demonstrates the implemented path; it does not demonstrate useful generalization. Final metrics must separate that implementation fact from the measured learning effect.
