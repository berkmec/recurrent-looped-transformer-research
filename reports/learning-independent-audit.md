# Independent learning evidence audit

Status: **PASS_WITH_SCOPE_LIMITS**. Independently recomputed111 held-out prediction files without importing `learning.metrics`, `learning.analyze`, its solver or model. Read scope was learning primary/depth8/combined evidence only. New integration seeds104731/104738 were not opened.

Checks passed: 39/39. Failures: [].

## Evidence integrity

All111 raw SHA256 values match receipts. All predictions pair exactly with their split's target arrays, affected/inverse/depth masks and row IDs. Independent float64 metric recomputation differs from recorded float32 by at most6e-08; all are inside declared audit rounding tolerance. All reported subgroup metrics were also checked.

Actual initialization tensors, including every state_dict key, are equal across all seven arms within each seed. Initial checkpoint hashes and63 saved150/300/600 snapshot hashes match. Dataset hashes match logs; event-content overlap across train/validation/IID/depth5/depth8 is zero. All stored node targets satisfy observed affine constraints with max residual2.38e-07, query-label mismatch0. Maximum supervised dependency depth: {'train': 7, 'validation': 7, 'iid': 7, 'shift': 5, 'depth8': 8}.

Primary protocol/source freeze and both follow-up protocols precede first primary test opening at2026-09-16T17:04:27.158600+00:00. Source index equals the original freeze receipt; all frozen primary and supplemental source hashes match current bytes. Selection/training receipts match before test. All three combined final checkpoints were fixed before their own evaluation. These are recorded-log chronology checks, not external timestamp attestation.

## Stricter revision gate

Root gate requires >=10% affected final-error reduction, positive affected improvement in every seed, unaffected final MSE<=1.05 baseline, and overall improvement. It is evaluated separately from the worker absolute0.01 gate.

| Split | Policy | Affected gain | All seeds positive | Unaffected ratio | Overall gain | Strict pass |
|---|---|---:|---|---:|---:|---|
| iid | local | 0.045800 | False | 1.000000 | 0.031858 | False |
| iid | all | -0.009839 | False | 0.986166 | -0.002633 | False |
| shift | local | -0.014329 | False | 1.000000 | -0.009225 | False |
| shift | all | -0.001581 | False | 0.974781 | 0.007966 | False |

Original local-policy absolute gate passes3/6 seed/split cells, matching the report. Strict local-policy aggregate gates pass0/2 splits. Original pass counts therefore do not establish the stricter package7 requirement.

## Paired graph and method effects

Positive values below mean baseline-minus-candidate final-MSE improvement. Bootstrap uses2000 crossed draws of three seed rows and common example columns, seed170043, matching the root statistical contract. Individual seeds remain explicit in JSON; no flattened seed-example inference.

| Split | Candidate vs baseline | Subset | Mean improvement | Crossed95% interval | Per-seed improvement |
|---|---|---|---:|---|---|
| iid | bidir vs forward | inverse | 0.285401 | [0.244411820044435, 0.3294696656108818] | [0.28503739354189145, 0.28280075284640016, 0.2883648059936058] |
| iid | method vs bidir | all | 0.000422 | [-0.0009459087102368166, 0.0023859085673690935] | [-0.000894601048399454, -0.00040081035124037655, 0.0025614975074314296] |
| shift | bidir vs forward | inverse | 0.178034 | [0.11058246056550662, 0.24621090409707783] | [0.20439892687797626, 0.12668835236757156, 0.20301553525931518] |
| shift | method vs bidir | all | 0.044803 | [0.017511322546068315, 0.07234632521369763] | [0.01978067545746639, 0.07200922323291722, 0.04261902448896386] |
| depth8 | bidir vs forward | all | -0.127145 | [-0.2097066343733434, -0.035404152992077435] | [-0.04700461640917466, -0.14832052866545542, -0.18610960944424354] |
| depth8 | bidir vs forward | inverse | 0.046678 | [-0.057325711758777854, 0.1445888368912072] | [0.11625172417927154, 0.061000662517004076, -0.03721803658946856] |
| depth8 | method vs bidir | all | 0.347502 | [0.25342670791699956, 0.4507920777734585] | [0.3735936311618323, 0.39873553393787037, 0.27017785564990104] |

## Scientific report assessment

Reported primary/depth8/combined MSE table values independently match. Primary inverse-graph benefits and depth8 method-supervision benefits are supported within this finite task family. Bidirectionality is not uniformly superior at depth8. The report correctly separates depth5 heldout query depth from genuine depth8 structural extrapolation, acknowledges weak/tiny feedback differences, retained diagnostic failures, synthetic finite-domain scope and effective-parameter mismatch. Overall MIXED labels are consistent; package7 should additionally state failure of the root strict revision gate.

Limitations and cautions:
- The original affected-improvement plus absolute0.01 unaffected gate is weaker than root >=10% and <=1.05 ratio gate; do not transfer original pass counts to strict status.
- Three training seeds; crossed bootstrap intervals are descriptive. This audit keeps one shared example axis, never flattens1536 predictions as independent examples.
- example_id is0..511 in every split; identity must include split/protocol namespace. Event-content hashes show no split collisions.
- Combined protocol was frozen before primary test opening, but combined training finished after primary tests were available. Source hashes and fixed procedure support no tuning; logs cannot prove absence of human inspection.
- Stored source hashes and UTC log receipts demonstrate internal consistency, not externally notarized chronology.
- Equal allocated tensors/initialization does not imply equal effective gradient parameters or FLOPs; report already discloses forward/feedback differences.

Full checks, all111 metric recomputations, hashes, per-seed effects and all crossed comparisons: `audit/learning-independent-audit.json`. No training, result changes, or commits were performed.
