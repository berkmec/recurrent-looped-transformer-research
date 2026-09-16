# RLT ve başlık belleği: yerel uçtan uca araştırma

Bu bağımsız çalışma alanı, RLT çekirdeği ile kullanıcının yirmi maddelik başlık/ağ/pencere/ham bellek önerisini on dört sonlu araştırma paketinde inceler. Resmî kaynak sabitlemesi `cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39`. Yazarların büyük modelleri yeniden üretilmiş değildir; küçük bağımsız uygulamalar, kontrollü metin ve sayısal görevler kullanılır. Olumsuz ve karışık sonuçlar korunur.

Ana sonuç raporu: [RLT_SONUC_RAPORU.md](reports/RLT_SONUC_RAPORU.md). Kullanıcının yirmi fikrinin ayrı mekanizmalarla eşlemesi: [gereksinim haritası](reports/user-requirement-map.md). Otuz risk, token/yerel tur/problem derinliği ve matematiksel sınırlar: [yorumlama sözleşmesi](reports/mathematical-interpretation.md).

## Kanıt düzeni

- `core/`: küçük RLT/no-feedback/tam-kesilmiş gradyan eğitimleri, uzun ufuk, gürültü, gerçek backward/optimizer ve bellek-zaman ölçümü.
- `learning/`: öğrenilmiş başlık grafiği, yönler, revizyon, gerçek token geri beslemesi ve ara gözetim; `learning/distillation/` gerçek üretilmiş izlerden sonraki öğrenci eğitimi.
- `memory/`: öğrenilmiş çıkarıcı, başlık kimliği/sürümü, kalıcı ham arşiv, vektör erişimi, özet ve yeterlilik denetleyicisi.
- `integration/connected/`: metinden aday ve sertifikaya bağlı ana deney; `reasoning-companion/` gerçek üretilmiş durum penceresi; `scale/`1,51M parametre duyarlılığı.
- `audit/`: üreticilerden ayrı ölçüm yeniden hesaplamaları, kaynak karmaları, önceki kanıtların korunması ve kapanış/manifest.
- `reports/`: açıklamalar, bağımsız incelemeler, PNG/SVG araştırma şekilleri.

Model ağırlıkları, tahmin dizileri ve SQLite arşivleri Git dışında tutulur; **nihai dosya manifestine dahildir**. Kaynak kodunun bir Git kopyası tek başına bütün deney teslimatı değildir. Başarısız ilk tanı sürümleri ayrı korunur; nihai protokollerle karıştırılmamalıdır.

## Mevcut teslimatı doğrulamak

PowerShell'de bu kökte çalıştırılır. Kurulu ortamlar değiştirilmez:

```powershell
Set-Location -LiteralPath 'E:\rlt-research\2026-09-16-e2e'
& 'C:\Users\pc\Documents\Codex\2026-09-16\rlt-derin-arastirma\.venv\Scripts\python.exe' -m audit.delivery_integrity verify
& 'C:\Users\pc\Documents\Codex\2026-09-16\rlt-derin-arastirma\.venv\Scripts\python.exe' -m pytest -q --basetemp=.pytest_cache/recheck-temp
```

Manifest doğrulaması yeni sonuç üretmez. Testlerin geçici dosyaları teknik önbellekte kalır. Kapanışın makinece okunabilir kaydı `audit/campaign-closure.json`; dosya bütünlüğü `audit/final-manifest.json` ve `.sha256` ile kontrol edilir. Bu karmalar haricî imza veya her bilimsel hipotezin başarılı olduğu anlamına gelmez.

## Yeni deneyin sınırı

Korunan sonuç dizininde eğitim/değerlendirme başlatılmamalıdır. Bazı eski iç üreticiler mevcut sonuçları yeniden yazabilir; güvenli girişler ve bağımsız denetimler bu riski ayrıca kontrol eder. Yeni bir bilimsel tekrar için ayrı bir çalışma kökü, yeni kaynak/protokol karmaları ve bütçe gerekir. Mevcut mutlak checkpoint/veri yolları ve bazı kaynak dondurma kayıtları bu kampanyaya aittir; tek komutla taşınabilir eğitim paketi iddiası yoktur.

Doğruluk, kabul kapsamı, koşullu hata, süre ve bellek ayrı ölçülerdir. Sıfır kabulde koşullu risk tanımsızdır. Gerçek çalışma geçmişinin korunması veya yerel durum eşdeğerliği, serbest doğal dilde sınırsız doğru düşünme kanıtı değildir.
