# Independent actual-trajectory distillation audit

Status **FAIL**; 108/111 checks pass. Failures: ['independent16_episode_actual_teacher_replay_17', 'independent16_episode_actual_teacher_replay_43', 'independent16_episode_actual_teacher_replay_91'].

Frozen protocol/source/checkpoint and transitive dependency pins match. TRAIN-only collection precedes student training; all six300-update student runs and selection receipts precede first fresh heldout opening. Both arms use the exact regenerated300×16 index batches and source-feedback initialization tensor hash for each seed. Twelve raw prediction SHA values, target/mask/ID pairing and metrics match independent recomputation.

All18432 final-query certificate decisions and topology masks were independently recomputed using observed graph residuals; all three seeds use the predeclared queried-component fallback. No full episode meets the all-component certificate. Six archive-off student checkpoints and6×512 prediction vectors match their actual heldout outputs exactly.

## Original paired effect

| Split | Trajectory minus answer-only final MSE | Seed-t95 | Crossed bootstrap95 |
|---|---:|---|---|
| iid | 0.009091 | [-0.0024184921252185278, 0.020601374189503995] | [0.005309084505994649, 0.014077824411401194] |
| depth8 | -0.139905 | [-0.37278866222554885, 0.09297964497536504] | [-0.23684193913526413, -0.042541529783118065] |

Positive means worse. IID is worse and depth8 better in all three seeds, while both original seed-t95 intervals include0. This supports MIXED, not a general improvement claim.

## Separate post-hoc pre-adaptation reference

The original protocol has no heldout initial-model reference. Before additional inference this audit saved a new descriptive protocol, then evaluated all three frozen feedback600 initial checkpoints on the same existing IID/depth8 observations, six cells total. There was no training, selection or tuning. This is not a preregistered third arm.

| Split | Adaptation vs initial feedback600 | Candidate-minus-initial MSE | Per-seed differences | Seed-t95 |
|---|---|---:|---|---|
| iid | answer_only | 0.000608 | [0.0028800923787800805, 0.0011697347146126663, -0.002225988166297663] | [-0.005848288555929076, 0.007064181173992466] |
| iid | trajectory | 0.009699 | [0.009399067524823067, 0.007484780605196181, 0.012214313893504032] | [0.0037895588411281786, 0.015609215841220676] |
| depth8 | answer_only | -0.051458 | [-0.024461225171857906, -0.06105160394588691, -0.06886214761838189] | [-0.11034254099414711, 0.007425889836729309] |
| depth8 | trajectory | -0.191363 | [-0.24962521229881326, -0.21609223634095961, -0.10837105397162948] | [-0.3746989597386764, -0.008026708668925131] |

## Limits

- The original two-arm comparison establishes trajectory versus answer-only adaptation, not improvement over pre-adaptation weights. The separately frozen initial-reference analysis here is explicitly post hoc.
- 300 updates per student are supported by frozen loop,300 finite loss rows,100/200/300 progress receipts,4800 paired batch entries and changed final weights; optimizer step counters are not persisted, so no independent training replay is claimed.
- Final queried-component certificate uses observed constraints only. Intermediate states are genuine generated trajectories but not all certified correct. Teacher weights previously received oracle intermediate supervision.
- All18432 query certificate decisions and masks independently recomputed. Actual teacher state reproduction independently sampled16 fixed TRAIN episodes per seed; producer full6144 replay receipt is a separate result.
- Three seed t95 intervals are the primary original uncertainty summary. Crossed percentile bootstrap resamples three seeds and shared512 example columns and is descriptive; it can exclude0 while df2 t95 includes0.
- Archive-off verifies six actual student models and512 observation-only inputs each against exported prediction artifacts; no OS filesystem sandbox or rerun by this audit.
- Shared row IDs require split namespace; generated archives retained rejected examples; selection is strongly shallow-component biased.

Post-hoc raw predictions/protocol: `E:\rlt-research\2026-09-16-e2e\audit\distillation-initial-reference`. Full audit: `audit/distillation-independent-audit.json`.
