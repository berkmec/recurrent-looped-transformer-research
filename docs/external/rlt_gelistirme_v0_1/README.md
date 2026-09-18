# RLT geliştirme paketi — v0.1.0

**Başlangıç:** `RLT_64_MADDE_COZUM_TASARIMI.md`.

Bu paket özel RLT deponuza uygulanmış bir yama veya eğitilmiş model değildir. Kaynak problem havuzunun 64 maddesine çözüm/kabul testi eşleştirmesi ve bağımsız mekanizma kodu içerir. Orijinal deneyler yeniden çalıştırılmadı. 46 bağımsız test geçti; CPU üzerinde yapılan dört mekanizma deneyi ayrı kaydedildi.

## İçerik

- `source/problem_havuzu.md`: orijinal dosyanın değiştirilmemiş kopyası.
- `RLT_64_MADDE_COZUM_TASARIMI.md`: mimari, 64 madde, kabul testleri, riskler ve kaynaklar.
- `problem_mapping.json`: 64/64 eşleştirmenin makine-okunur sürümü.
- `INTEGRASYON_DEVIR.md`: mevcut araştırma ajanı/kodlayıcı için somut entegrasyon sırası.
- `integration_contract.json`: tam durum, cache, yazma, arşiv ve metrik sözleşmesi.
- `experiment_plan.template.json`: kapsamı azaltmayan, henüz çalıştırılmamış kampanya şablonu.
- `rlt_lab/core.py`: korunan yazma, kesintisiz segment remat ve affine scan.
- `rlt_lab/memory.py`: raw archive, hash zinciri, başlık sürümleri ve bağımlılık kapanımı.
- `rlt_lab/protocol.py`: atomik JSON, güvenilir makbuz, tam-log sayımı ve istatistik yardımcıları.
- `run_tests.py`, `tests/`: tek test girişi; JSON ve metin raporu.
- `run_experiments.py`: 65.536 hold geçişi, terminal gradyan yolu, affine parite ve arşiv testleri.
- `results/`: gerçekten çalıştırılan test ve mekanizma sonuçları.
- `sources.json`: dış teknik kaynaklar ve kullanım sınırları.
- `MANIFEST.json`, `verify_package.py`: teslim dosyalarının hash doğrulaması.

## Çalıştırma

Paket kökünde, PyTorch bulunan yalıtılmış Python ortamını kullanın. Mevcut CUDA PyTorch kurulumunuzun üzerine varsayılan CPU wheel kurmayın. Denenen ortam Python 3.13.5 ve torch 2.10.0+cpu; başka sürüm/Windows/CUDA paritesi burada ölçülmedi. Ek vektör veritabanı, Symbolica, ağ hesabı veya API anahtarı gerekmiyor; bunlara gerçek entegrasyon henüz uygulanmadı.

İlk olarak `python verify_package.py` teslim hash’lerini denetler.

`python run_tests.py` bütün 46 bağımsız testi çalıştırır ve results/tests.json üretir.

`python run_experiments.py` dört mekanizma probunu çalıştırır ve results/mechanisms.json üretir.

Testleri yeniden çalıştırmak sonuç dosyalarını değiştirir; bu durumda eski teslim manifestinin sonuç hash’leri doğal olarak farklı olur. Kendi yeni kampanyanız için yeni manifest üretin; eski manifesti geçmiş sonuçların üzerine yazmayın.

## Ne kanıtlandı, ne kanıtlanmadı?

Koruma kapısına doğru karar DIŞARIDAN verildiğinde kayıt sabit kalır; modelin doğru HOLD kararını öğrendiği kanıtlanmadı. Son kayıptan ilk duruma gradyan ulaştı ve remat tam referansla eşleşti; uzak krediyle öğrenme başarısı kanıtlanmadı. Affine alt yol seri referansla eşleşti; nonlinear RLT paralelleştirilmedi ve GPU hızlanması ölçülmedi. Arşiv kayıpsız ve append-only davranır; veritabanını sahibinin silmesini engelleyen WORM donanımı değildir.

`protected_update` doğruluk odaklı referans fonksiyondur: sonluluk kontrolü GPU üzerinde senkronizasyon oluşturabilir. Üretim kernel’i veya hız benchmarkı diye kullanılmamalıdır.

`recurrent_scan` adımı SAF olmalıdır. Cache’in in-place değiştirilmesi, dış global sayaçlar, BatchNorm güncellemeleri veya replay sırasında ağırlık değişmesi eşliği bozabilir. Tensor tuple bütün recurrence bağımlılıklarını taşımalıdır; integer konum/metaveri de taşınabilir. Girdi/bellek büyümesini sihirli biçimde ortadan kaldırmaz.

SQLite yedeği aynı diskte ise fiziksel disk kaybından korumaz. Paket kullanıcı bilgisayarına veya uzak servise herhangi bir veri yazmadı.
