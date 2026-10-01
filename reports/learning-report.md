# Learning experiments6,7,12,13

Protocol SHA256: `2647a34ff4a0c23fe2f275d1ac99e76906046a69f96c5f6f1eedcefb43e1b62f`.
Full evaluated-scope classification: **packages6/7/12/13 are MIXED**. There are21 corrected trained models including the combined follow-up. Positive primary graph results, positive depth8 method-transfer results, failed generalization and uncertainty are separated below.

Primary runs:18 learned models, seeds17/43/91,600 updates each,9600 examples seen/run, CPU one thread. Training/evaluation wall time 1509.5s.
Primary comparisons are on512 IID and512 query-depth-shift examples. These share example identities across seeds/arms; uncertainty respects paired seeds and common example clusters.

## Main final-answer results

| Arm | IID final MSE | Depth5 final MSE | IID error<=0.1 | Depth5 error<=0.1 |
|---|---:|---:|---:|---:|
| forward | 0.163407 | 0.243963 | 0.2611 | 0.1836 |
| bidir | 0.002358 | 0.156965 | 0.9570 | 0.1927 |
| wrong | 0.274353 | 0.379534 | 0.1628 | 0.1055 |
| flat | 0.253799 | 0.366219 | 0.1510 | 0.1178 |
| feedback | 0.002359 | 0.156954 | 0.9570 | 0.1921 |
| method | 0.001936 | 0.112162 | 0.9746 | 0.5306 |

##6: Graph direction

Status: **MEASURED_POSITIVE**. Paired bidirectional-minus-forward inverse-task final-MSE differences:
- iid: mean -0.285401; per-seed [-0.2850373948890308, -0.2828007543324702, -0.2883648100581695]; seed-t95 [-0.29235605216547206, -0.27844592068764157].
- shift: mean -0.178034; per-seed [-0.2043989274186515, -0.12668835202839546, -0.203015535190616]; seed-t95 [-0.28850946523255266, -0.06755907785922266].

These tasks genuinely observe a downstream absolute value and query an upstream unknown. One-step forward-title propagation cannot read reverse edges. Global causal token attention still sees all observations, so direction superiority is not a proof that the complete forward arm cannot solve inverse tasks. Wrong and flat controls have identical dense title shapes and budgets.

##7: Revision

Status: **MIXED**. Validation-selected local policy passed affected-improvement plus unaffected-FINAL-error<=0.01 gate in 3/6 seed/split cells. First and subsequent query budgets remain six in every policy.

| Split | Policy | Final MSE | Affected final MSE | Unaffected final MSE | Delta MSE |
|---|---|---:|---:|---:|---:|
| iid | retain | 0.002358 | 0.003281 | 0.001436 | 0.001230 |
| iid | all | 0.002365 | 0.003313 | 0.001416 | 0.002027 |
| iid | local | 0.002283 | 0.003131 | 0.001436 | 0.002328 |
| shift | retain | 0.156965 | 0.202107 | 0.111823 | 0.103765 |
| shift | all | 0.155714 | 0.202426 | 0.109002 | 0.082907 |
| shift | local | 0.158413 | 0.205003 | 0.111823 | 0.097478 |

Per-seed policy radii and validation selection are frozen in revision-selection.json; held-out gates are in results/paired-analysis.json. Retain refreshes the directly written title, while retaining other states. Local is a validation-selected graph-radius policy, not an oracle effect mask. No claim extends to edge-offset revisions, which are supported but not in this primary training distribution.

##12: Title feedback into future recurrent computation

Status: **MIXED**. Future causal intervention contract passed: **True**. Readout-only and recurrent-feedback models share initialization/batches/budgets. The source-style core uses causal encoder memory, gated recurrent hidden merge, global cross-attention and local KV cache.
Feedback is injected after query readout into the next token hidden merge. Zero and permuted title projections preserve all already-produced first-query predictions exactly. Refinement does not increment token clocks or duplicate KV cache entries. Intervention magnitudes and changed-example counts are reported without equating a tiny change to useful reasoning.
- iid feedback-minus-readout final MSE: 0.000001; seed-t95 [-8.8319639839471e-06, 1.08104734887347e-05].
- shift feedback-minus-readout final MSE: -0.000011; seed-t95 [-6.087286422630376e-05, 3.9294104573367035e-05].

##13: Verified intermediate supervision and archive-off transfer

Status: **MIXED**. Method traces are exact procedural all-node supervision from TRAIN ONLY. Inference imports no dataset/solver/archive. Novel graph IDs and values, plus a held-out query-depth5 shift, are tested; genuine unseen structural depth is evaluated separately below. This is not evidence for unrestricted thinking-style learning.

| Updates | Baseline IID MSE | Method IID MSE | Baseline depth5 MSE | Method depth5 MSE |
|---:|---:|---:|---:|---:|
| 150 | 0.007227 | 0.009073 | 0.140144 | 0.091605 |
| 300 | 0.004952 | 0.003651 | 0.147657 | 0.090246 |
| 600 | 0.002358 | 0.001936 | 0.156965 | 0.112162 |
- iid method-minus-baseline final MSE -0.000422; per-seed [0.000894601069755363, 0.00040081035724903735, -0.0025614974828460646]; seed-t95 [-0.005065407953682297, 0.004221350583121187].
- shift method-minus-baseline final MSE -0.044803; per-seed [-0.01978067570257558, -0.07200922413393585, -0.0426190254011658]; seed-t95 [-0.10984434899357928, 0.020238398835127812].

Observed mean training-plus-validation wall time: baseline 79.76s, method 79.85s, ratio 1.001. Auxiliary loss weight0.3 adds backward supervision; shared final examples/steps are matched. Timing is CPU wall time, not energy or pure kernel throughput.

## Verification, failed attempt and artifacts

- Recomputed 84 raw prediction files and checked their hashes against metrics.json.
- Nine test contracts passed before primary training; sixteen test cases passed in the final expanded suite: forward direction isolation, future feedback/prefix immutability, token/cache clocks, final-error gate, policy inputs, independent linear-solve labels, inference without oracle imports, validated IDs/query alignment, and causal encoder streaming parity (some contracts share a test).
- RED evidence: red-tests.txt (missing implementation) and red-core-parity.txt (source-core gap). GREEN evidence: green-tests.txt, green-core-parity.txt.
- The first GRU diagnostic was stopped by root audit before its150-step checkpoint. No test set was opened. Protocol/source/initial checkpoint remain in diagnostic-v1; execution.jsonl preserves the attempted run. This failed architecture attempt is not counted among18 corrected primary models.
- Data receipts: results/data-{train,validation,iid,shift}.npz and execution.jsonl. Training traces: verified-method-traces-TRAIN-ONLY.npz plus trace-provenance.json. Checkpoints include initialization and150/300/600 snapshots. source-hashes.json and protocol.sha256 bind the run.
- Public API: learning/api.py and Runtime in model.py. FORMALIZATION.md describes valid event codes, version semantics, incremental refinement/commit contracts and limits.

## Remaining limits

Independent root audit is required before scientific closure; this worker recomputation is not substituted for that independent audit. Three-seed uncertainty is limited; hierarchical bootstrap intervals are exploratory. Small synthetic affine forests do not establish large language-model scalability, unrestricted text extraction, arbitrary graph families, measured energy, RL, physical performance, or general thought transfer. Root integration separately measures parser/controller/verifier use. Validation-selected policy usefulness is judged by the reported final-error gate, never by prediction drift alone.

## Supplemental scope correction and genuinely unseen depth8

The main depth5 evaluation is held-out query depth only. TRAIN method targets include the other chain up to depth7. A separately frozen protocol, created before any primary test opening, evaluates512 new depth8 examples per arm/seed. Every dependency path in this supplemental query exceeds all supervised train depths. The model, training and six-round-per-query budget are unchanged.

| Arm | Depth8 final MSE | Depth8 error<=0.1 |
|---|---:|---:|
| forward | 0.515980 | 0.1198 |
| bidir | 0.643125 | 0.0990 |
| wrong | 0.525029 | 0.1185 |
| flat | 0.509114 | 0.1113 |
| feedback | 0.643177 | 0.0990 |
| method | 0.295622 | 0.3346 |

Method-minus-final-only depth8 MSE: -0.347502; per seed [-0.3735936274873808, -0.39873553288748553, -0.2701778546784368]; seed-t95 [-0.5167582858327413, -0.17824639086946084]. Final package13 classification including the genuine structural shift: **MIXED**.

Bidirectional-minus-forward depth8 MSE: 0.127145; per seed [0.04700461614447704, 0.14832052380023697, 0.18610961130927706]; seed-t95 [-0.05153789563764555, 0.3058277298069729]. The primary inverse-task benefit does not establish uniform deep generalization. Package6 classification over the full evaluated scope: **MIXED**.

## Combined feedback and method supervision: complete2×2

Three additional models were predeclared before test opening: feedback enabled plus0.3 intermediate supervision, same initialization/batches/600updates. This gives21 corrected trained models overall. Combined seed17 is the predeclared root full-system integration artifact; no test-best selection is used. All three combined checkpoints were fixed before this follow-up opened test arrays.

| Split | Combined final MSE | Combined error<=0.1 |
|---|---:|---:|
| iid | 0.002001 | 0.9661 |
| shift | 0.110723 | 0.5391 |
| depth8 | 0.297244 | 0.3418 |

The interaction contrast on final squared error is combined−method−feedback+bidir. Negative values indicate an additional reduction beyond the sum of isolated error differences; with three seeds intervals remain uncertain.
- iid: interaction 0.00006396, per-seed [-9.616349497143517e-06, -0.0002049122089580191, 0.00040639807880182843], seed-t95 [-0.0007116547438332412, 0.0008395677573976852].
- shift: interaction -0.00142771, per-seed [0.00010485155974393479, -0.001948019166860604, -0.002439975167931152], seed-t95 [-0.004780907898211574, 0.0019254793815130262].
- depth8: interaction 0.00156895, per-seed [-0.00028733179765083605, 0.003206912500445469, 0.0017872728840302443], seed-t95 [-0.0027964811032771012, 0.005934383494493686].

## Effective parameters, full trace verification and evidence preservation

All primary arms allocate29475 parameters. They are **not fully matched in active parameters or effective compute**. Observed nonzero-gradient entries on the same first training batch are:

- forward: 27170; gradient-present allocation 28386.
- bidir: 28322; gradient-present allocation 28386.
- wrong: 28322; gradient-present allocation 28386.
- flat: 28322; gradient-present allocation 28386.
- feedback: 29346; gradient-present allocation 29410.
- method: 28322; gradient-present allocation 28386.

The forward arm masks reverse inputs; feedback enables an extra1024-parameter projection; auxiliary training adds backward work. These are the mechanisms being isolated, and equal allocation alone does not make their costs equal.

Every TRAIN trace was independently checked: 8192 examples, 24576 query states and 245760 node targets. Maximum linear-solve discrepancy 1.62e-07, tolerance1e-6. The verifier imports no generator oracle. A corruption regression verifies rejection.

Archive-off export check: a separate inference-only package, containing model/API/checkpoint and observation inputs only, ran512 IID examples in isolated Python import mode. Neither learning.data nor the training runner was importable. Predictions matched the primary method checkpoint outputs exactly (max difference 0.0). This is an isolated import-path/export check, not an OS filesystem sandbox. The first export attempt could not import Torch because it lives in the user dependency site; that failed sandbox is preserved, and the corrected attempt adds only the dependency site, not the research source path.

A second failed attempt, v2, had the corrected recurrent core but omitted activation of unobserved parents mentioned only by an edge. It was stopped before test access. Three completed600-step diagnostic models and one partial arm remain in diagnostic-v2, alongside all sources/data/protocols. v3 fixed this with a regression and fresh test seeds. Both the v1 and v2 failures are retained, not counted as successful primary models.

The original frozen runner could overwrite evidence on restart. It is preserved as historical execution source; use only the tested successor safe_run.py, which fails closed on nonempty output/checkpoint destinations. The combined runner also rejects existing combined artifacts. Never restart the original runner against this evidence directory.

External protocol integrity guards also verify protocol bytes against the SHA256 sidecar, source-hash index against the original frozen execution receipt, actual source hashes, and combined-to-primary protocol pins. Protocol/model-plus-hash-index tampering regressions fail closed. The frozen primary and combined training sources were not edited.

Maximum OS-reported primary-process peak working set observed by5-second sampling: 285.72MiB. This is not total machine RAM or energy. Primary training/evaluation wall 1509.5s; combined follow-up wall 216.5s. Supplemental evaluation has per-run wall times in its metrics.

Final worker audit recomputed **111** held-out raw prediction files, verified their hashes, confirmed all seven arms share initialization tensors within each seed, checked unchanged frozen source hashes, and verified supplemental protocols predate first test access. Root independent audit and connected extraction/memory measurements remain separate.
