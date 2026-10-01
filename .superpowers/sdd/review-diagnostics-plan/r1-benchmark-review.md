# R1 pre-freeze benchmark review and F1 closure

Date: 2026-09-17. No heldout evaluation or GPU work was performed.

Final verdict after the branch addendum below: **PASS for pre-freeze R1 evaluation.** F1 and both lineage findings are closed. The three original chain distributions remain explicitly labeled equality/negative controls. A separately specified fourth `branched` distribution now tests off-path conservatism under new topology; it must remain separately reported and cannot establish repair of original chain false abstentions. No heldout performance is asserted.

Reviewed pre-branch benchmark SHA256: `0ADE95D3DF07FF7F4FE3BF08012191906F2720FAFC3E8FA70EF4F649AE8851A8`. Final branch-inclusive SHA256 is recorded in the addendum.

Certificate SHA256: `774E60AA814CFF28706FB4478CC53731294BCB88B4EE3E0FB9246F36A2E182C2`.

Certificate test SHA256: `FECFADAAE2F5092818DE72A8DE76C2EC9C9B8F95F881DD8DA464DC4BDE02CB92`.

## Closed findings

**F1 CLOSED — display overflow no longer changes the exact decision.** At `review_cycle/certificate.py:170-179`, the display-only conversion now catches overflow locally, reports `bound=None` and `display_bound_overflow=True`, and preserves the exact bound, decision and certificate. The new test exercises `10**400` accepted at the same exact threshold, independent certificate audit, and rejection with the ordinary small threshold. All **13 certificate tests PASS** in 0.203 seconds.

**B1 CLOSED — transitive inference source pin.** The initial freeze implementation omitted `learning/token_core.py`, which `learning/model.py` imports and invokes throughout inference. The reviewed final `freeze` source list includes it. A test copying the current sources to a temporary fixture, freezing, then modifying the imported token core produces `SOURCE_DRIFT`. The token core itself imports only PyTorch runtime modules; no additional local inference import was found there.

**B2 CLOSED — result lineage and condition-set checks.** The initial resume/summary branches checked output artifact bytes but did not bind receipt protocol/source identity to the current context; the early completed-run return also skipped those checks. The final code adds `verify_receipt` at benchmark lines 78-89 and `verify_completion` at lines 92-103. It validates protocol and source-file hashes, split/seed identity, the exact expected artifact filenames, all artifact hashes and the exact completed condition set. Both resume/early-completion and summarize paths use these checks. Eight independent perturbations were rejected: receipt protocol hash, source hash, split, seed; completion protocol hash; duplicate condition; wrong artifact hash; missing artifact.

An attempted development fixture probe crossed a concurrent parent edit and first stopped with a missing development protocol file in the newly added checker. This was a fixture setup failure, not a benchmark defect; after adding the fixture protocol, the explicit rejection cases above all passed.

## Experimental-design limitation — no off-path gain in generated chains

Location: `review_cycle/certificate_bench.py::make_data`, final lines approximately 106-130. Both components are simple chains. The queried component has exactly one current anchor at one endpoint and asks for the other endpoint. Its unique anchor-query path therefore contains every active edge and anchor of that component. The second chain is disconnected, and the baseline already excludes disconnected components.

Consequently, for every finite complete candidate on these generated inputs, the exact component residual sum equals the exact path residual sum. In an additional development-only probe, this equality held for **72/72 random candidate vectors**, covering 24 sources each from `iid`, `depth5` and `depth8`, seed 810237 (candidate seed 810239). This equality also follows directly from the graph topology, beyond the finite probe.

The `offpath_wrong` challenge demonstrates the desired pruning behavior on a manually constructed branch, but it is separate from the learned-model coverage distribution. Any path/component differences in the current main distributions can arise from arithmetic semantics or unrelated implementation behavior, not pruning a branch that those distributions do not contain.

This is not a soundness defect. Keep chain results explicitly scoped as a matched-domain baseline/no-gain diagnostic, or introduce a distinct preregistered branched condition before freezing if the experimental claim needs path-selection coverage benefit. Do not silently change existing distribution semantics or tune after heldout results.

## Scoring, pairing and uncertainty assessment

- `reference_values` uses an independent exact Gaussian elimination over latest absolute/edge facets. It does not call the certificate graph routine, learned model or baseline solver. Scoring uses the float32-serialized source values actually shown to the model, interpreted as exact represented rationals; it does not score against pre-rounding generator internals. A development check independently propagated the retained affine constraints and agreed exactly with Gaussian elimination for all **72 sources**.
- Generated problems use independently drawn numeric values and node permutations, with balanced depth/direction/affected strata. The reference values match the revision semantics; all 72 development problems changed the query value if and only if the saved `affected` flag was true. All used 10 nodes and 16 event rows. Depth-8 leaves one valid separate anchored node.
- The evaluator generates one frozen trace and gives the same `candidate` dictionary to both verifiers at each problem/round. Reference values appear only in correctness scoring. They are not passed to model inference or certificate construction. The new path verifier is invoked through `verify_envelope` with per-problem scope; its numerical audit is additionally applied to every primary-threshold accepted certificate.
- `correct[i,k]` uses exact query error and exact `1/10`. The `acceptance` tensor stores the original baseline floating comparison and the path verifier's rational comparison against decimal thresholds. Summary reads this boolean tensor, avoiding a later lossy rethresholding of reported float bounds. The three fixed thresholds are `.025`, `.05`, `.099999`; the primary threshold is the largest. Therefore auditing all primary acceptances also covers every acceptance at the stricter thresholds.
- The primary fixed-final-round comparison holds the selected model candidate constant. Adaptive selection is separately labeled and chooses the earliest accepting round, or the final round when none accepts. It reports the accuracy of that actually selected candidate; it does not copy the correct answer into a rejected prediction.
- Each summary row is per source distribution and model seed. Rounds are not counted as independent examples. Seeds sharing generated sources are not pooled as independent observations. The primary contrast is a per-source paired difference in acceptance; bootstrap resamples the difference vector, preserving the pairing of methods. Reusing a deterministic bootstrap seed across rows does not change an individual interval's interpretation.
- Coverage, risk, precision and recall denominators are consistent. Zero acceptance is reported as null selective risk/precision, and zero observed accepted errors receives a finite upper uncertainty bound. These are finite sample summaries, not proof of zero population error. The intervals are per comparison, not simultaneous or adjusted for multiple comparisons.
- Lazy replay measures a separate batch-size-one subset and checks its node values against the frozen batched trace with a numerical tolerance. It is correctly described as a separately observed execution subset; approximate numerical parity alone does not establish identical threshold crossing arbitrarily near a boundary. The primary exact same-candidate comparison does not depend on this replay. Lazy timing includes real inference/verifier execution and is not a demonstrated production latency improvement.

## Executed checks

All commands used the specified Python executable with `-B`.

1. `unittest review_cycle.test_certificate -v`: **13/13 PASS**.
2. `unittest review_cycle.test_certificate_bench -v`: **6/6 PASS**, 0.173 seconds. Covers reproducible development generator/reference checks, exact inverse reference and contradiction rejection, metric denominators, imported-source drift, receipt identity/artifact-set checks and seven context/soundness challenges.
3. Additional source probes using seed **810237**, never the frozen proposed seeds: **72/72 exact Gaussian/reference propagation comparisons**, affected-flag checks, oracle-valued development certificate audits. These synthetic correct values were used only as development fixtures, not as measured model candidates.
4. Seven benchmark challenge cases: **7/7 expected outcomes**.
5. Development-only result context fixture under `.superpowers/sdd/review-diagnostics-plan/r1-review-dev-context`: **8/8 context/artifact perturbations rejected**. The trusted protocol loader was stubbed only to isolate the receipt/summary consumer; actual hash checks, raw-array loading, summary logic and bootstrap ran normally.
6. Synthetic identical-method summary fixture: expected **12 rows and one paired contrast**, full coverage, zero selective risk, zero paired difference and `[0,0]` bootstrap interval — **PASS**. This is software-test evidence, not experimental performance.
7. Exact chain component/path equality on random candidate vectors: **72/72 PASS**, documenting the design limitation above.

The previously reviewed 600-case independent path-enumeration evidence remains in `r1-certificate-review.md`; this turn did not repeat it because the mathematical path algorithm did not change. No implementation file was edited by the reviewer. Only this report and explicitly isolated development fixture outputs were written. Any future source change requires a new hash-specific review/freeze check. Main heldout performance remains **UNMEASURED** by this reviewer.

## Final branch addendum — pre-freeze design limitation resolved by separate condition

Final reviewed `review_cycle/certificate_bench.py` SHA256: `f8bfc4a79f39f3c0171bcc3e185830efb825603bc3f3941d4ed5eb9015df8bbc`.

Final reviewed `review_cycle/test_certificate_bench.py` SHA256: `d16709b7101fb15594a4d30dbe96a1e179fc2c00674ee8fa1bebe3107fb064f2`.

The protocol now retains `iid`, `depth5`, `depth8` as equality/negative controls and adds `branched` with its own proposed seed 2026091714, 512 independent generated problems, and explicit OOD-topology labeling. No proposed heldout seed was executed during review. The total expected completion set becomes 4 distributions times 3 model seeds = 12 conditions; the generic completion validator derives this from the protocol rather than hard-coding the former 9.

At `make_data`, the main query chain contains 5 nodes / 4 edges. A 4-node spur attaches at the interior node `chain[1]`; a tenth node is an isolated independently anchored component. Forward and inverse queries retain opposite endpoint anchors and queries, so both directions have a 4-edge main path and the spur remains off that path. Absolute revision affects either the queried component's endpoint anchor or the isolated anchor, retaining the affected/unaffected interpretation. The original chain generation branches are unchanged. No signed-residual cancellation certificate, changed answer, oracle substitution or post-outcome selection was added.

Executed branch-specific checks:

- Current complete certificate and benchmark suites: **20/20 PASS**, 0.372 seconds.
- Independent development seed **810241**, **16 sources**: each query component had exactly 9 nodes, exactly one anchor, a 5-node certified path and exactly 4 off-path nodes.
- Perturbing each of those four off-path candidate values by +10 produced **64/64 baseline CONTINUE / path VERIFIED contrasts**, with the query candidate unchanged and independent certificate audit passing every time. This proves the added condition can distinguish the mechanisms; it does not establish that learned-model candidates will realize this gain.

All scope, exact scoring, acceptance-tensor, paired comparison and statistical-unit conclusions above continue to apply. Final review has no demonstrated outstanding implementation blocker. Freeze and the subsequent heldout run remain the parent's next operations.
