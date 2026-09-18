# G0 — ortak hazırlık kaydı (2026-09-18)

**Durum:** G0 çekirdek maddeleri tamamlandı; deney başına kalan maddeler (split/negatif kontrol/A-A) ilgili deneyin protokolünde.

| Madde | Sonuç |
|---|---|
| Eski manifest ve kaynak otoritesi | Faz 1 manifesti `146eb693…` ve önceki döngülerin manifestleri her mühürde yeniden doğrulanır; plan dosyaları `docs/phase2-plan-snapshot/` altına bayt-bayt kopyalandı (SHA256SUMS; kapanış makbuzlarıyla eşleşir). |
| Yeni kök | `E:/rlt-research/2026-09-18-phase2`, dal `codex/rlt-phase2` (taban `c01639f`). |
| Model pinleme | Qwen3-Embedding-0.6B `97b0c614…`, Qwen3-Reranker-0.6B `e61197ed…`, Qwen3-4B `1cfa9a72…` (Apache-2.0); her LFS dosyası indirme sonrası sha256 ile doğrulandı (`protocols/G0-pins.json`, `.superpowers/sdd/phase2/model-pins.json`). |
| Veri pinleme | MuSiQue `dgslibisey/MuSiQue@c8f4f8c9…` (train+dev, 259 MiB); LongMemEval-cleaned `98d7416c…` (yalnız S sürümü + oracle, 279 MiB; M sürümü 2,6 GB indirilmedi); BigCodeBench `b74c0d0b…` (16 MiB). |
| Ortam | Yeni venv `E:/rlt-research/.venv-phase2`: torch 2.14.0+cu130 (ilk kurulum WinError 32 ile CPU torch'a düşmüştü; E:'de geçici dizinle yeniden kuruldu), transformers 5.17.0, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.0.1 (`venv-freeze.txt`). Mevcut DaxDream venv'i (torch 2.12.0+cu130, transformers 5.8.1) salt-okunur; indirmeler ve 0.6B çıkarımı orada. |
| Cevap modeli kalibrasyonu | Qwen3-4B 8-bit: 4,11 GiB, **2,0 tok/s**; nf4: 2,67 GiB, **7,2 tok/s**; 8 sabit TR/EN istemde ikisi de doğru (düşünme kapalı, greedy). **Karar:** nf4. fp16'ya karşı nicemleme sadakati ölçülemedi (sığmıyor). |
| 32 + 200 kalibrasyon | B1: 200 geliştirme sorgusu (train, 2-hop) ile yapılandırma seçimi; U1: zaman pilotu; D1: 32 → 200 geliştirme aşaması protokolde. |
| Negatif kontroller / sayaçlar | Deney başına protokolde (U1 nötr ekler; B1 hash/BM25 tabanları; D1 programatik pertürbasyonlar). |
| A/A oynaklık | B1'de batch-vs-single kodlama farkı ölçülür; U1'de tohum dağılımı; D1 greedy (determinist). |
| Dondurma | Her deney final veri açılmadan `frozen.json` + kaynak hash'leriyle dondurulur; geliştirme seçimleri final dosyalarına bakılmadan yapılır. |

**Sınırlar:** Türkçe dilimleri yok (MuSiQue/BigCodeBench İngilizce; çeviri = model yazımı, ayrı etiketlenmeli); LongMemEval-V2 indirilmedi; insan anotasyonu yapılmadı; 100k merdiven basamağı train paragraflarıyla kuruldu (çeldirici havuzu).
