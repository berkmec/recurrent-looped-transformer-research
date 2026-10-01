| Uzunluk | Kol | Durum | Güncelleme s (3 ort.) | Full oranı | CUDA tepe alloc MiB | CUDA tepe reserved MiB | Süreç WS tepe MiB | Host meşgul |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32 | full | PASS | 0.2318 | 1.000 | 19.0 | 24.0 | 1004.1 | 0.15 |
| 32 | detach | PASS | 0.2262 | 0.976 | 19.0 | 24.0 | 1004.1 | 0.15 |
| 32 | checkpoint | PASS | 0.4235 | 1.827 | 19.0 | 24.0 | 1012.5 | 0.16 |
| 32 | segment | PASS | 0.2570 | 1.109 | 18.1 | 22.0 | 999.5 | 0.19 |
| 32 | segment_encoder_checkpoint | PASS | 0.2422 | 1.045 | 17.6 | 22.0 | 998.7 | 0.18 |
| 128 | full | PASS | 0.8960 | 1.000 | 25.4 | 30.0 | 1028.6 | 0.16 |
| 128 | detach | PASS | 0.9076 | 1.013 | 25.4 | 30.0 | 1030.4 | 0.17 |
| 128 | checkpoint | PASS | 1.6560 | 1.848 | 25.4 | 30.0 | 1060.5 | 0.17 |
| 128 | segment | PASS | 0.9174 | 1.024 | 20.7 | 26.0 | 998.8 | 0.15 |
| 128 | segment_encoder_checkpoint | PASS | 0.9441 | 1.054 | 20.0 | 26.0 | 999.7 | 0.15 |
| 512 | full | PASS | 5.2989 | 1.000 | 51.6 | 56.0 | 1132.5 | 0.50 |
| 512 | detach | PASS | 3.7771 | 0.713 | 51.5 | 56.0 | 1128.1 | 0.20 |
| 512 | checkpoint | PASS | 6.7656 | 1.277 | 51.6 | 56.0 | 1223.4 | 0.19 |
| 512 | segment | PASS | 3.7412 | 0.706 | 31.2 | 36.0 | 1023.4 | 0.17 |
| 512 | segment_encoder_checkpoint | PASS | 3.7991 | 0.717 | 30.1 | 36.0 | 1023.4 | 0.57 |
| 1024 | full | PASS | 7.2807 | 1.000 | 86.6 | 108.0 | 1216.0 | 0.16 |
| 1024 | detach | PASS | 7.9046 | 1.086 | 86.5 | 108.0 | 1215.8 | 0.22 |
| 1024 | checkpoint | PASS | 14.2245 | 1.954 | 86.6 | 108.0 | 1408.9 | 0.21 |
| 1024 | segment | PASS | 7.2316 | 0.993 | 45.5 | 68.0 | 1023.8 | 0.16 |
| 1024 | segment_encoder_checkpoint | PASS | 8.3919 | 1.153 | 43.6 | 66.0 | 1024.9 | 0.15 |
| 128 | segment (hooks tani) | PASS | 1.0821 | - | 20.7 | 26.0 | 999.4 | saklanan depolama tepe 2.8 MiB |
{"counts": {"planned": 20, "pass_": 20, "oom": 0, "fail": 0, "missing": 0}, "equivalence": "PASS", "summary_sha256": "145a0b34ceb426fd55f1c6d9ddb099a4338af9ef44763ffe0e286bbbc7a8d9c0"}
