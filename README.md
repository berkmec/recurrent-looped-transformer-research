# RLT ve başlık belleği — ham veri, sonuçlar, kod ve testler

Bu depo, *Recurrent Looped Transformer* (RLT) ve çevresinde kurulan "başlık belleği" üzerine yürütülen yerel araştırmanın **ham sonuçlarını, istatistiklerini, kodunu ve testlerini** içerir. Planlar, devir notları, ajan defterleri ve `audit/` altındaki mühür kayıtları bu sürümde yoktur.

**English summary.** Raw results, statistics, code and tests of a local research study on the Recurrent Looped Transformer (RLT) and a "heading memory" built around it. Plans, hand-off notes, agent ledgers and the seal records under `audit/` were removed; machine-specific paths were replaced by placeholders. The code is kept as a readable record: parts of it read protocol digests from `*.sha256` files that no longer exist (see below).

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
| `audit/` | Bağımsız denetim sonuçları (JSON/JSONL), denetim betikleri ve yukarı akış projenin kaynak belgeleri (`audit/sources/`) |
| `reports/`, `notes/` | Sonuç raporları, şekiller ve ana belge (`reports/RLT_SONUC_RAPORU.md`, `notes/ANA_BELGE_RLT.md`) |

## Bilmeniz gerekenler

- **Yollar nötrleştirildi.** Dosyalardaki yerel makine yolları yer tutuculara çevrildi: `__RLT_ROOT__` (çalışma kökü), `__PY_ENV_DIR__` (Python ortamının bulunduğu dizin), `__WORKSPACE__` ve `__USER_HOME__`. Betikleri çalıştırmadan önce bunları kendi dizinlerinizle değiştirin. Sonuç dosyalarındaki sayılara dokunulmadı, yalnızca yol yazıları değişti.
- **Kaldırılanlar:** planlar ve tasarım belgeleri, devir notları, ajan defterleri, ön-inceleme ve denetim yazıları; `audit/` altındaki mühür/manifest/doğrulama kayıtları ve kırmızı/yeşil test günlükleri; tüm `*.sha256` yan dosyaları. Paket klasörlerindeki çalıştırma kayıtları (`frozen.json`, `receipt.json`, `manifest.json`, `*-seals.jsonl`) ve konsol/test günlükleri sonuç verisi olarak kaldı; içlerinde yazılı SHA-256 değerleri artık değişen ya da silinen dosyalarla eşleşmez.
- **Çalışmayan kod yolları:** protokol özetini silinen `*.sha256` dosyalarından okuyan kod (`core/benchmark.py`, `core/diagnostics.py`, `core/run_final_only.py`, `core/safe_entrypoint.py`, `learning/distillation/postprocess.py`) çalıştırılırsa `FileNotFoundError` verir. Kod okunabilir kayıt olarak durur; deneyleri yeniden üretmek için eski commit'lere dönmek gerekir.
- **Depoda olmayanlar:** model ağırlıkları (`.pt`), tahmin dizileri (`.npz`) ve SQLite arşivleri Git'e hiç girmedi. Bunlara dayanan hesaplar yeniden üretilemez; yalnızca özet JSON/JSONL sonuçlar vardır.
- **Testler:** `python -m pytest tests review_cycle drift` (Python 3.11, numpy, torch; CPU). Bu sürümde 660 testin 640'ı geçer. 13 test başarısız olur: 8'i bir ayrıştırıcı kontrol noktasına, 3'ü ham metin derlemine, 1'i bir değerlendirme kontrol noktasına, 1'i (`tests/test_sep_finetune.py`) bir kontrol noktasına ve silinen bir `*.sha256` dosyasına bağlıdır. 7 test, ilgili dosya yoksa atlanır.
