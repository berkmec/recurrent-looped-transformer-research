# RLT ve başlık belleği — ham veri, sonuçlar, kod ve testler

Bu depo, *Recurrent Looped Transformer* (RLT) ve çevresinde kurulan "başlık belleği" üzerine yürütülen yerel araştırmanın **ham sonuçlarını, istatistiklerini, kodunu ve testlerini** içerir. Planlar, devir notları, ajan defterleri ve `audit/` altındaki mühür kayıtları bu sürümde yoktur.

**English summary.** Raw results, statistics, code and tests of a small-scale research study on the Recurrent Looped Transformer (RLT) and a "heading memory" built around it. Plans, hand-off notes, agent ledgers and the seal records under `audit/` were removed; machine-specific paths were replaced by placeholders. The code is kept as a readable record: parts of it read protocol digests from `*.sha256` files that no longer exist (see below). Licence: MIT; the upstream project, third-party models and datasets are credited in the section "Lisans ve atıflar / License and credits" and in the English section.

## Klasörler

| Klasör | İçerik |
|---|---|
| `results/` | Ham sonuçlar (JSON, JSONL, log): R1, U1–U5, B1/B2, D1, G0, M0/M1, `M1-finetune-distance` (D1–D6 ve T1 serisi), drift, `review-diagnostics-r6`, `training` |
| `core/`, `learning/`, `memory/` | Birinci aşama paketlerinin kodu, günlükleri ve sonuçları (`core/results`, `learning/results`, `learning/distillation`, `memory/results*`, `memory/heading-results`) |
| `horizon/`, `language/`, `semantic_memory/` | Paket kodları |
| `integration/` | Bağlı sistem, akıl yürütme eşlikçisi ve ölçek sonuçları (`connected/`, `reasoning-companion/`, `scale/`) ve onları üreten kod |
| `mline/` | M hattı (needle / MQAR / decoy) kodu |
| `drift/`, `review_cycle/`, `protocols/`, `archive/`, `scripts/` | Kod ve çalıştırma betikleri |
| `tests/` | Testler (ayrıca `drift/test_drift.py` ve `review_cycle/` içindeki testler) |
| `audit/` | Bağımsız denetim sonuçları (JSON/JSONL) ve denetim betikleri |
| `reports/`, `notes/` | Sonuç raporları, şekiller ve ana belge (`reports/RLT_SONUC_RAPORU.md`, `notes/ANA_BELGE_RLT.md`) |

## Bilmeniz gerekenler

- **Yollar nötrleştirildi.** Dosyalardaki yerel makine yolları yer tutuculara çevrildi: `__RLT_ROOT__` (çalışma kökü), `__PY_ENV_DIR__` (Python ortamının bulunduğu dizin), `__WORKSPACE__` ve `__USER_HOME__`. Betikleri çalıştırmadan önce bunları kendi dizinlerinizle değiştirin. Sonuç dosyalarındaki sayılara dokunulmadı, yalnızca yol yazıları değişti.
- **Kaldırılanlar:** planlar ve tasarım belgeleri, devir notları, ajan defterleri, ön-inceleme ve denetim yazıları; `audit/` altındaki mühür/manifest/doğrulama kayıtları ve kırmızı/yeşil test günlükleri; tüm `*.sha256` yan dosyaları. Paket klasörlerindeki çalıştırma kayıtları (`frozen.json`, `receipt.json`, `manifest.json`, `*-seals.jsonl`) ve konsol/test günlükleri sonuç verisi olarak kaldı; içlerinde yazılı SHA-256 değerleri artık değişen ya da silinen dosyalarla eşleşmez.
- **Çalışmayan kod yolları:** protokol özetini silinen `*.sha256` dosyalarından okuyan kod (`core/benchmark.py`, `core/diagnostics.py`, `core/run_final_only.py`, `core/safe_entrypoint.py`, `learning/distillation/postprocess.py`) çalıştırılırsa `FileNotFoundError` verir. Kod okunabilir kayıt olarak durur.
- **Depoda olmayanlar:** model ağırlıkları (`.pt`), tahmin dizileri (`.npz`) ve SQLite arşivleri Git'e hiç girmedi. Bunlara dayanan hesaplar yeniden üretilemez; yalnızca özet JSON/JSONL sonuçlar vardır.
- **Testler:** `python -m pytest tests review_cycle drift` (Python 3.11, numpy, torch; CPU). Bu sürümde 660 testin 640'ı geçer. 13 test başarısız olur: 8'i bir ayrıştırıcı kontrol noktasına, 3'ü ham metin derlemine, 1'i bir değerlendirme kontrol noktasına, 1'i (`tests/test_sep_finetune.py`) bir kontrol noktasına ve silinen bir `*.sha256` dosyasına bağlıdır. 7 test, ilgili dosya yoksa atlanır.

## Lisans ve atıflar / License and credits

- **Lisans / Licence:** Bu deponun kodu, metinleri ve sonuçları MIT lisanslıdır (`LICENSE`). / The code, texts and results of this repository are released under the MIT licence (`LICENSE`).
- **Yukarı akış proje / Upstream project:** *Recurrent Looped Transformer* — Yifan Zhang, Jichen Feng, Shihan Qin, <https://github.com/yifanzhang-pro/recurrent-looped-tranformer> (Apache-2.0). Bu çalışma, projenin `cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39` commit'ine dayanan bağımsız ve küçük ölçekli bir uygulamadır; resmî bir yeniden üretim değildir. Yukarı akış belgeleri bu depoda yoktur. / An independent, small-scale implementation based on commit `cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39` of that project; not an official reproduction. Upstream documents are not included.
- **Hazır modeller / Pretrained models** (depoda yok / not included; sürümler / revisions: `protocols/G0-pins.json`): `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-4B` — Apache-2.0.
- **Veri kümeleri / Datasets:** MuSiQue (`dgslibisey/MuSiQue`; kayıtlarda lisans bilgisi yok, veri kümesinin kartına bakın / no licence recorded here, see the dataset card), LongMemEval-cleaned (`xiaowu0162/longmemeval-cleaned`, MIT), BigCodeBench (`bigcode/bigcodebench`, Apache-2.0). `results/B1*`, `results/B2` ve `results/D1*` bu kümelerden soru, cevap veya kimlik gibi kısa parçalar içerebilir; bunlar özgün lisanslarına tabidir. / These result folders may contain short excerpts of those datasets, which remain under their original licences.
- **Eğitim metni / Training text:** M hattı (`mline/`) modelleri `cnn_dailymail` (3.0.0, `article` alanı) metninden üretilmiş bir bayt derlemiyle eğitildi; derlem depoda yok. / The M-line models were trained on a byte corpus built from `cnn_dailymail` (3.0.0, `article` field); the corpus is not included.

## English

**What this is.** Raw results (JSON, JSONL, logs), statistics, code and tests from a set of small-scale experiments on the Recurrent Looped Transformer (RLT) and a heading-based memory. The reports in `reports/` and `notes/` are written in Turkish; the numbers in `results/` are language-independent.

**Layout.** `results/` raw results · `core/`, `learning/`, `memory/`, `integration/` package code, logs and results · `mline/` needle / MQAR / decoy experiments · `tests/` tests · `audit/` independent audit results and scripts · `reports/` and `notes/` result reports and the master document (`reports/RLT_SONUC_RAPORU.md`, `notes/ANA_BELGE_RLT.md`).

**Caveats.**

- Local machine paths were replaced by `__RLT_ROOT__`, `__PY_ENV_DIR__`, `__WORKSPACE__` and `__USER_HOME__`; the numbers in the result files are unchanged.
- Model weights (`.pt`), prediction arrays (`.npz`) and SQLite archives were never committed, so computations that depend on them cannot be reproduced; only the summary JSON/JSONL results are here.
- Seal files (`*.sha256`, the manifests under `audit/`) were removed. Code that reads protocol digests from them (`core/benchmark.py`, `core/diagnostics.py`, `core/run_final_only.py`, `core/safe_entrypoint.py`, `learning/distillation/postprocess.py`) fails with `FileNotFoundError`, and SHA-256 values recorded in other files no longer match.
- Tests: `python -m pytest tests review_cycle drift` (Python 3.11, numpy, torch; CPU). 640 of 660 tests pass; 13 fail and 7 are skipped because files that are not in the repository (checkpoints, a text corpus) are missing, and one of the failing tests also depends on a removed seal file.
