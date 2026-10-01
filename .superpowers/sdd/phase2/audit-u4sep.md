# U4-sep denetimi — 10 PASS / 0 FAIL / 0 UNVERIFIED

Denetleyici: aynı oturum, satır içi betik (`audit/audit_u4sep.py`). Bağımsızlık sınırı: delege denetleyici haftalık API kotasına takılı (24 Eylül'de sıfırlanır), bu yüzden aynı model kendi döngüsünü denetledi.

- **1-freeze** — PASS: 2 frozen directories, 9 sources each; problems=none
- **2-completeness** — PASS: counted, not asserted: 5 of 6 conditions left an initial.pt, 3 reached complete.json, 2 are quarantined with a REASON.txt, 1 never started. Started-but-unquarantined directories: none
- **3-integrity** — PASS: 3 conditions: checkpoint/updates/train hashes and 2048 rows each; problems=none
- **4-report-numbers** — PASS: validation / separation ratio / 512-suffix / clean-4096 recomputed for 3 conditions; problems=none
- **5-chance-level** — PASS: ln(5)=1.6094 and both low-lr validations sit within 0.02 of the 0.200 chance level
- **6-regrouping** — PASS: grouped by measured repair: 3 fragile, 9 robust. gap/spread max(fragile)=4.77 > min(robust)=4.70 -> OVERLAP as reported. gap max(fragile)=3.55 > min(robust)=3.43 -> OVERLAP as reported. gap/drift max(fragile)=0.57 < min(robust)=0.92 -> separates as reported.
- **7-u2d-grouping-correction** — PASS: 3 conditions repair >= 50 % while sitting at lr 0.0003, which U2d grouped as fragile: u2c-hard_st-five-w32-s117@lr0.0003, u2c-hard_st-five-w32-s143@lr0.0003, u2c-hard_st-five-w32-s191@lr0.0003
- **8-quarantine** — PASS: 2 quarantine directory with REASON.txt and no completion receipt inside, so the runner cannot silently adopt it
- **9-verdict** — PASS: mean separation ratio of the U4-sep low-lr arm 0.49 against the U2c soft baseline 2.93 = x0.17 < the pre-registered 1.25 threshold, so the rule returns INTERVENTION_INEFFECTIVE, as the report states
- **10-sealed-untouched** — PASS: both sealed U2c frozen manifests still verify against the current sources; problems=none
