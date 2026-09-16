# Actual successful teacher traces to later student weights

Path implementation/evaluation: **IMPLEMENTED_TESTED**. Paired transfer result: **MIXED**.
This separate experiment uses actual teacher-generated refinement states/numeric candidates stored before student training. It does not reuse oracle intermediate labels as distillation targets. Teacher combined600 checkpoints themselves originate from the earlier supervised experiment; this is not unsupervised emergence or universal thinking-style learning.

Protocol SHA256 `ba435010d7df25e5e43e1f125e4b5e16ed6254e60c3db599964b19d280a6c1c9`. Three seeds17/43/91, two student arms,300 updates each; both start from the same seed-matched feedback600 weights and receive identical actual teacher final targets, example batches and query masks. The trajectory arm additionally minimizes0.1 times hidden-state MSE plus numeric trajectory MSE on certified components.

## Collection, successful-final criterion and selection bias

Each fixed combined600 teacher generated actual six-round traces at all three query states on2048 TRAIN-only tapes. A0.05 residual-sum tolerance was frozen before collection. No full ten-node, three-query episode met that certificate. The predeclared queried-component fallback was used in every seed. Certification concerns the final candidate only; intermediate latent/numeric steps are preserved actual states and are not mathematical proofs.

| Seed | Eligible episodes /2048 | Certified query fraction | Certified node fraction |
|---|---:|---:|---:|
| 17 | 313 | 0.07910 | 0.01634 |
| 43 | 418 | 0.12907 | 0.02679 |
| 91 | 333 | 0.09570 | 0.02074 |

All rejected traces and decisions remain in immutable-by-writer archives. No target is solved, repaired, substituted or widened after inspection. Filtering is strongly biased toward shallow successful components; some deep/direction/affected cells contain only1–3 eligible episodes. This is a material external-validity limit, not balanced proof of a general method.

| Seed | Observed depth | Inverse | Affected | Eligible / total | Certified queries / all queries |
|---|---:|---:|---:|---:|---:|
| 17 | 1 | 0 | 0 | 84/256 | 140/768 |
| 17 | 1 | 0 | 1 | 96/256 | 129/768 |
| 17 | 1 | 1 | 0 | 45/256 | 96/768 |
| 17 | 1 | 1 | 1 | 73/256 | 105/768 |
| 17 | 3 | 0 | 0 | 3/256 | 3/768 |
| 17 | 3 | 0 | 1 | 6/256 | 7/768 |
| 17 | 3 | 1 | 0 | 3/256 | 3/768 |
| 17 | 3 | 1 | 1 | 3/256 | 3/768 |
| 43 | 1 | 0 | 0 | 129/256 | 259/768 |
| 43 | 1 | 0 | 1 | 120/256 | 198/768 |
| 43 | 1 | 1 | 0 | 65/256 | 154/768 |
| 43 | 1 | 1 | 1 | 77/256 | 152/768 |
| 43 | 3 | 0 | 0 | 9/256 | 11/768 |
| 43 | 3 | 0 | 1 | 12/256 | 13/768 |
| 43 | 3 | 1 | 0 | 3/256 | 3/768 |
| 43 | 3 | 1 | 1 | 3/256 | 3/768 |
| 91 | 1 | 0 | 0 | 83/256 | 194/768 |
| 91 | 1 | 0 | 1 | 90/256 | 164/768 |
| 91 | 1 | 1 | 0 | 47/256 | 93/768 |
| 91 | 1 | 1 | 1 | 65/256 | 88/768 |
| 91 | 3 | 0 | 0 | 28/256 | 29/768 |
| 91 | 3 | 0 | 1 | 17/256 | 17/768 |
| 91 | 3 | 1 | 0 | 1/256 | 1/768 |
| 91 | 3 | 1 | 1 | 2/256 | 2/768 |

## Fresh held-out outcomes

| Split | Final-answer-only MSE | +Actual trajectory MSE | Final-answer-only accuracy0.1 | +Actual trajectory accuracy0.1 |
|---|---:|---:|---:|---:|
| iid | 0.002615 | 0.011706 | 0.9414 | 0.7611 |
| depth8 | 0.697890 | 0.557985 | 0.0970 | 0.1048 |

| Seed | Student arm | IID final MSE | Depth8 final MSE |
|---|---|---:|---:|
| 17 | answer_only | 0.004109 | 0.772102 |
| 17 | trajectory | 0.010628 | 0.546938 |
| 43 | answer_only | 0.002167 | 0.697722 |
| 43 | trajectory | 0.008482 | 0.542681 |
| 91 | answer_only | 0.001569 | 0.623845 |
| 91 | trajectory | 0.016009 | 0.584336 |

The seed-t intervals are the primary uncertainty summary. Hierarchical bootstrap intervals are exploratory with only three independently trained seeds; they do not override the mixed primary conclusion.
- iid: trajectory-minus-final-only MSE 0.009091; per-seed [0.006518975116504677, 0.006315045809976105, 0.014440302118511106]; seed-t95 [-0.0024184923069088798, 0.02060137433690347]; hierarchical bootstrap95 [0.005312246821034905, 0.014305239903796407].
- depth8: trajectory-minus-final-only MSE -0.139905; per-seed [-0.2251639825935854, -0.15504063575428972, -0.03950890565511145]; seed-t95 [-0.3727886580828309, 0.09297964208083986]; hierarchical bootstrap95 [-0.23867720921077476, -0.043096884319800216].

New512-example IID and512-example true-depth8 sets were first generated only after all student checkpoints were fixed. Common example/seed clustering is preserved. With three seeds, intervals remain limited. A result can show no benefit or harm while still demonstrating the actual trace-to-later-weights path.

## Provenance, archive-off inference and costs

- All actual teacher latent/numeric traces replayed exactly: maximum difference0 across all6144 collected episodes. All18432 final-query acceptance decisions rechecked from observations. Candidate readouts match archived teacher final targets exactly.
- Independently recomputed 12 raw held-out prediction files and verified hashes. Each paired student starts with identical weights.
- Isolated inference-only student export reproduced 3072 example prediction vectors with maximum difference 0.0; no generator or distillation/trace package was importable and no archive/targets were exported. This is not an OS filesystem sandbox.
- Frozen source, teacher/student checkpoint pins, train receipt, and additional evaluation dependency sidecar checked before training and after evaluation. No frozen primary/combined source was changed.
- Student training/evaluation wall time: 262.50s on CPU one thread. Teacher collection receipts record separate per-seed costs. Energy is UNMEASURED.
- Both arms compute the same forward trajectories. Auxiliary backward supervision adds compute; equal examples/updates do not imply equal effective cost.
- The initially reported interop-import concern was independently disproved by live source and a post-work import reproduction; the reviewer retracted it. No scientific protocol amendment or source mutation was made.

Six contract tests cover false-candidate no-repair, TRAIN/hash-only loading, immutable archive writes, actual Runtime trace parity, exclusion of uncertified nodes from auxiliary loss and rejection of incomplete/empty evaluation before classification. Selection query/node masks and eligible indices were independently recomputed from the frozen full/fallback rule. Independent root review remains separate from worker recomputation.
