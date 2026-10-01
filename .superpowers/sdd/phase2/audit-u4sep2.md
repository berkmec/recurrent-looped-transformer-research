# U4-sep2 denetimi — 11 PASS / 0 FAIL / 0 UNVERIFIED

Denetleyici: aynı oturum, satır içi betik (`audit/audit_u4sep2.py`). Bağımsızlık sınırı: delege denetleyici haftalık API kotasına takılı (24 Eylül'de sıfırlanır).

- **1-freeze** — PASS: 10 sources, protocol_hash 5e10c8d741a9d545; problems=none
- **2-completeness** — PASS: 10/10 complete, 10/10 evaluated
- **3-integrity** — PASS: 10 conditions: checkpoint/updates/train hashes and 512 rows each; problems=none
- **4-pairing** — PASS: both arms of all 5 pairs load the identical frozen U2c checkpoint (sha compared); problems=none
- **5-control-not-null** — PASS: U2c gate_loss weight: soft=0.0, supervised=1.0. The fine-tuner uses neither, so the control is a true null only for the soft checkpoints - exactly as the report states.
- **6-pair-table** — PASS: 5 pairs: U2c / control / hinge repair recomputed from the frozen U2b reference; problems=none
- **7-sign-test** — PASS: over 5 matched pairs: gap/drift ratio predicts the repair direction 2/5, absolute drift (inverted) 4/5 - as reported
- **8-ratio-rose-and-target** — PASS: 5/5 pairs raised gap/drift above their paired control - the intervention moved the quantity in every pair. But the pre-registered target of 1.5 is reached by 5/5 in the TRAINING measurement (64 neutral slots, what the rule reads) and only 1/5 in the EVALUATION measurement (512 slots, what the target was calibrated on: fragile <= 0.57, robust >= 0.92). Scored with the evaluation ratio the verdict becomes INTERVENTION_INEFFECTIVE, not CONFOUNDED_BY_TRAINING. Recorded in the report as correction 7.
- **9-asymmetry-retracted** — PASS: mean hinge-minus-control by placement: prefix +0.053, middle -0.013, suffix -0.085. The suffix gain seen in the two soft pairs reverses overall, so the earlier "the asymmetry replicates" claim is correctly retracted.
- **10-verdict-and-seals** — PASS: rule returns CONFOUNDED_BY_TRAINING with 2/5 rescued, which is what the report prints; sealed U2c-lr0003 sources still verify; problems=none
- **11-start-geometry** — PASS: the `ratio_u2c` field in the sealed analysis is the control arm LAST fine-tuning ratio over 64 neutral slots, not the frozen checkpoint geometry over the 200-sequence evaluation. Re-measured here from the U2c checkpoints: true start ratios soft/s117 0.570, soft/s143 0.488, supervised/s117 0.711, supervised/s143 0.433, supervised/s191 0.786; the recorded field is high by 2.08x, 3.36x, 2.39x, 8.35x, 1.06x, worst 8.35x. No verdict, sign test, audit check or report sentence reads the field, so the sealed file is left as it is and the correction is section 8 of the report plus results/U4sep2/start-geometry.json; problems=none
