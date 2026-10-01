# Sürüklenme mini deneyi — sonuç raporu (2026-09-18)

**Durum:** MEASURED_NEGATIVE (üç müdahalenin üçü de). 22/24 koşul tamamlandı, 2 koşul (reread kolu, tohum 191, iki görevde de) sonsuz gradyanla ıraksadı ve kanıtı korunarak INCOMPLETE bırakıldı. Ön-kayıtlı plan: [`docs/drift-mini-plan.md`](../docs/drift-mini-plan.md) (sha `bca4c455…`, final veriden önce ledger'a yazıldı). Kök: `__RLT_ROOT__/2026-09-18-drift`, dal `codex/rlt-drift-mini`, taban `f83b527`. Mühürlü kökler değişmedi.

## Soru ve cevap

**Soru:** R2/R5'te ölçülen durum sürüklenmesi — durum kurulduktan sonra gelen uzun nötr (kimlik) koşularında son-durum doğruluğunun çökmesi — eğitim verisi değiştirilmeden, aynı girişler ve aynı başlangıç ağırlıklarıyla, yalnız (a) daha geniş yerel KV penceresi (4→16), (b) nötr girişte durum-değişmezlik kaybı (λ = 1) veya (c) durum-geçmişine attention (reread) ile azalır mı?

**Cevap:** Hayır — bu protokolde hiçbiri azaltmadı; üçü de bir yerde daha kötüleştirdi.

## Üç bulgu

1. **Yerel pencereyi büyütmek sürüklenmeyi artırıyor.** Pencere 16 ile beş-durum modelleri nötr-32 orta/son ekinde −30/−35 puan, nötr-128 ön ekinde −27 puan, temiz L1024'te −7 puan; paritede — tabanın 3/3 dayanıklı olduğu görevde — nötr-32 ve nötr-128 orta/son ekleri **−45…−49 puan, t95 aralıkları sıfırı dışlıyor**, L1024 −6 puan. Mekanistik okuma (hipotez, ölçüm değil): uzun bir nötr koşuda 16'lık yerel attention penceresi tamamen nötr tokenlarla dolup yinelemeli durumu bastırıyor; pencere 4 modeli durumu taşımak için yinelemeye daha çok yaslanıyor.
2. **Değişmezlik kaybı yerel olarak işe yarıyor, uzakta kırılıyor.** λ = 1 ile beş-durumda nötr-32 orta/son +9/+7 puan (aralıklar sıfırı kapsar), ama nötr-128 **ön** ekinde taban 1,00/0,99/1,00 iken 0,21/0,19/0,99'a düşüş (−53 puan) ve L1024 −8 puan. Eğitimde görülen kısa nötr koşularda "durum değişmesin" öğrenilmiş; 128'lik koşuda durum yine de başka bir sabit noktaya kayıyor ve sonraki işlemler bozuk durumdan başlıyor. Paritede etkisi yok (taban zaten dayanıklı).
3. **Durum-geçmişine attention, uygulandığı biçimiyle kararsız ve zararlı.** 6 koşuldan 2'si ıraksadı (gradyan normu birkaç güncellemede 0,6 → 54 → 2e5 → 1e8 → inf; kayıp sıçramadan önce sonlu), hayatta kalan koşularda da normlar 300–5,6e4'e sıçradı (onlarca adım > 10). Bağımsız denetimin düzeltmesi: gradyan sıçraması reread'e özgü değil — beş-durumda taban dahil her kol >10'a sıçrıyor (maksimum: base 1.7e+03, win16 1.1e+02, idinv 9.5e+02, reread 1.5e+12); reread'e özgü olan, kırpılmış güncellemelere rağmen normun inf'e kaçması. Hayatta kalan beş-durum tohumlarında nötr-32 −28 puan, nötr-128 ön ek −50 puan (tohum 117'de %0,0). Temiz uzunluk genellemesi ise mükemmel (L1024 1,000/0,998). Kusur benim kol tasarımımda: anahtar/değerler normalize edilmemiş durum ve gömme vektörlerinden alındığı için attention logit'leri durum normuyla büyüyor. Bu, "durum-geçmişi attention" fikrinin genel özelliği değil, bu spesifikasyonun sonucu; normalize-anahtarlı sürüm yeni bir dondurma gerektirir ve bu döngüde koşulmadı.

## Mutlak son-durum doğrulukları (seçilmiş checkpoint, 512 taze örnek/küme)

| Koşul | L32 | L128 | L1024 | n8 orta/son | n32 orta/son | n128 ön | n128 orta | n128 son |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| five base s117/143/191 | 1,00/1,00/1,00 | 1,00/0,99/1,00 | 1,00/0,98/1,00 | 0,99/0,80/1,00 · 1,00/0,79/1,00 | 0,67/0,42/0,55 · 0,72/0,36/0,76 | 1,00/0,99/1,00 | 0,16/0,42/0,39 | 0,40/0,36/0,45 |
| five win16 | 1,00/1,00/1,00 | 1,00/1,00/0,99 | 0,94/0,97/0,85 | 0,76/0,91/0,98 · 0,78/0,95/1,00 | 0,20/0,18/0,35 · 0,20/0,19/0,40 | 0,20/1,00/1,00 | 0,19/0,18/0,36 | 0,20/0,19/0,41 |
| five idinv | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 1,00/0,89/0,85 | 0,84/1,00/1,00 · 0,80/1,00/1,00 | 0,77/0,58/0,56 · 0,80/0,64/0,63 | 0,21/0,19/0,99 | 0,19/0,19/0,54 | 0,20/0,19/0,63 |
| five reread s117/143 (s191 ıraksadı) | 1,00/1,00 | 1,00/1,00 | 1,00/1,00 | 0,63/0,78 · 0,63/0,84 | 0,18/0,35 · 0,20/0,32 | 0,00/1,00 | 0,17/0,18 | 0,20/0,19 |
| parity base | 1,00 ×3 | 1,00 ×3 | 1,00 ×3 | 1,00 ×3 · 1,00 ×3 | 1,00 ×3 · 1,00 ×3 | 1,00 ×3 | 1,00/0,94/1,00 | 1,00/1,00/1,00 |
| parity win16 | 1,00/1,00/1,00 | 1,00/0,98/1,00 | 0,99/0,84/0,99 | 0,98/0,98/0,99 · 0,99/1,00/1,00 | 0,51/0,48/0,57 · 0,54/0,50/0,48 | 1,00/0,79/1,00 | 0,51/0,48/0,57 | 0,54/0,50/0,48 |
| parity idinv | 1,00 ×3 | 1,00 ×3 | 1,00 ×3 | 1,00 ×3 · 1,00 ×3 | 1,00 ×3 · 1,00 ×3 | 1,00 ×3 | 0,87/1,00/1,00 | 1,00 ×3 |
| parity reread s117/143 | 1,00/1,00 | 1,00/1,00 | 1,00/1,00 | 1,00/1,00 · 1,00/1,00 | 1,00/1,00 · 1,00/1,00 | 1,00/1,00 | 0,97/0,94 | 1,00/1,00 |

Beş-durumda şans düzeyi 0,20; paritede 0,50. Tablo `results/drift/analysis-tolerant.json` → `absolute` alanından; her hücre ham `.npz` tahminlerinden yeniden sayılabilir.

## Ön-kayıtlı karar kuralına göre sonuç (kol − base, puan; t95, df = tohum−1)

| Görev / kol | n128 orta | n128 son | L1024 bedeli | Karar |
|---|---:|---:|---:|---|
| five / win16 | −7,7 [−43,5; +28,0] | −13,9 [−35,8; +8,1] | −7,2 | MEASURED_NEGATIVE |
| five / idinv | −1,6 [−49,4; +46,1] | −6,4 [−59,8; +47,0] | −8,1 | MEASURED_NEGATIVE |
| five / reread (2 tohum) | −11,4 [−171,5; +148,6] | −18,7 [−39,7; +2,4] | +1,2 | MEASURED_NEGATIVE + UNSTABLE, INCOMPLETE |
| parity / win16 | −45,8 [−53,1; −38,5] | −49,3 [−57,3; −41,3] | −6,2 | MEASURED_NEGATIVE |
| parity / idinv | −2,1 [−26,6; +22,3] | +0,1 [−0,2; +0,3] | 0,0 | MEASURED_NEGATIVE (fark yok) |
| parity / reread (2 tohum) | −1,6 [−21,4; +18,3] | +0,1 [−1,1; +1,3] | 0,0 | MEASURED_NEGATIVE (fark yok), INCOMPLETE |

Kural: MEASURED_POSITIVE için iki birincil son noktada ≥ +20 puan ve sıfırı dışlayan aralık ve L1024 bedeli ≤ 5 puan; herhangi bir birincil ortalama ≤ 0 veya bedel > 5 puan → MEASURED_NEGATIVE. Aralıklar keşif amaçlı, çokluk düzeltmesiz; iki-tohumlu satırlar df = 1 ile hesaplandı ve üç-tohum iddiasına yükseltilmedi.

## Sağlamalar

- **Taban tekrar üretimi:** altı `base` koşulu, mühürlü tanı kampanyasının aynı tohumlu tabanlarını **78/78 son noktada tahmin-tahmin özdeş** yeniden üretti (seçilen adımlar da aynı). Decode-döngüsü eğitimi `Core.forward` ile bit-eşdeğer; mühürlü kampanya CPU'da tekrar üretilebilir. (`base-vs-sealed.txt`)
- **Eşleme:** aynı görev/tohumdaki kollar aynı Core başlangıç ağırlıklarını ve 2048 özdeş minibatch'i kullandı (test + makbuz hash'leri); reread kolu Core'a 6.176 ek parametre ekler (Core 24.325 beş durum / 24.226 parite; ~%25), sıfır-başlangıçlı W_o ile ilk ileri geçişi tabanla aynı.
- **Testler:** RED 5/5 başarısız (uygulama yokken) → GREEN 5/5 (`red.log`, `green.log`): decode döngüsü = forward (1e-9), reread başlangıçta kimlik ve öğreniyor, değişmezlik kaybı yalnız nötr konumlarda, 24 koşul/eşleme, uçtan uca çalıştırma ve dondurma reddi.
- **Çökmeler:** runner sonsuz gradyanda kanıtı yazıp durdu (`failure-step*.json`, kısmi `updates.jsonl`); kesilen dizin yeniden kullanılmadı, yeniden koşulmadı. Slot 2 işçisi ilk çökmede çıktı; kalan 4 parite koşulu dondurulmuş CLI ile tek tek koşuldu (`workers/slot-2b.jsonl`), biri (parity reread s191) yine ıraksadı.
- **Dondurma:** protokol `1496384366fc62f9…`, kaynak pin'leri `core/model.py`, `review_cycle/training_data.py`, `drift/model.py`, `drift/run.py`; test kaynak tohum uzayı tanı kampanyasıyla aynı (test girişleri özdeş; burada seçimde kullanılmadı). Dondurulmuş `analyze()` eksik küme yüzünden tasarım gereği reddetti; `analyze_tolerant.py` (pin dışı, salt analiz) kullanıldı.

## Bağımsız ham denetim

`.superpowers/sdd/drift-mini/drift-independent-audit.md` (+ `.json`, betik `drift_raw_audit.py`; taze denetçi ajan, CPU tek iş parçacığı, salt-okunur, üretici kodu içe aktarmadan, 6 s): **PASS, 42/42 kontrol.** Dondurma/protokol/kaynak pin'leri, 110 mühür hash'i ve 22 seal_hash, 286 değerlendirme satırının ham `.npz` sha256 ve metrikleri (son/prefix doğruluk, histogram, girdi kimlikleri, RNG'den yeniden üretilen girdiler, bağımsız oracle hedefleri, nötr ekleme konumları 0/16/32), 22×2048 gerçek güncelleme, kollar arası özdeş giriş hash dizileri ve tensör-tensör özdeş Core başlangıçları (24/24 initial.pt tohumdan yeniden üretildi), 78 karşılaştırma 1e-12 içinde, altı karar plandaki kuraldan yeniden türetildi (her birinde tetikleyici "birincil ortalama ≤ 0"), taban tekrar üretimi 6/6 koşul 78/78 son noktada bütün konumlarda özdeş, iki çökmenin günlük kuyruğu ve adım numaraları doğrulandı. Denetçinin düzeltmeleri bu rapora işlendi: gradyan sıçraması her beş-durum kolunda var (yukarıda), `slot-2b.jsonl` kabuk yazımlı farklı şemada, iki-tohumlu df = 1 kuralı dondurulmamış (ledger'da açıklanmış). Denetçinin ham veriden gözlemi: 128'lik nötr son ek **her kolda** durumu 1–2 çekici duruma çökertıyor (deterministik çoktan-bire, ≥ 0,98) — sürüklenme rastgele gürültü değil, sabit noktaya çekilme; reread s117'nin ön ekte 0/512 sonucu, çekici durum 2'nin sonraki işlemlerle permüte edilmesi.

## Faz 2'ye (U1/U2) girdi

- Nötr-koşu sürüklenmesi bir hiperparametre (pencere) veya basit bir düzenleyici (yerel değişmezlik kaybı) meselesi değil; U1/U2'de **koşunun konumu** (durum kurulmadan önce / sonra) ve **uzunluğu** ayrı faktörler olmalı, ölçüt "sabit nokta kararlılığı" olarak tanımlanmalı (uzun nötr koşuda durumun yörüngesi doğrudan izlenmeli — bu deneyde yalnız son çıktı ölçüldü).
- Reread fikri denenecekse anahtarlar normalize edilmeli ve gradyan/norm günlükleriyle kararlılık kapısı önceden konmalı; bu spesifikasyon geçersiz.
- Beş-durum modellerinde nötr-32 orta/son bile tabanı 0,36–0,76'ya düşürüyor: sürüklenme 128'e özgü değil, 32'de başlıyor. Parite (iki durum) çok daha dayanıklı — durum sayısı arttıkça kırılganlık artıyor gibi görünüyor (iki görev, hipotez).

## Hesap bütçesi

24 koşul, 3 CPU yuvası (tek iş parçacığı), koşul başına ~5,3 dk eğitim (reread ~7 dk) + ~2 dk değerlendirme; toplam ~70 dk duvar süresi. GPU kullanılmadı. Enerji ölçülmedi.

## Süreç notları

- Kontrolör: Claude Code (Fable 5.1), kullanıcı yetkisi "yapıştır ilerle" ile otonom; kararlar `.superpowers/sdd/drift-mini/progress.md`'de `Ruling:` satırları. Ledger'ın bir satırı fırlatmadan hemen sonra kabuk hatası (Python REPL başlığı sızması) nedeniyle yeniden yazıldı; deney dosyası etkilenmedi.
- Bağımsız ham denetim: PASS 42/42 (bölüm yukarıda).
- Sınırlar: iki sentetik görev, w32, üç tohum, tek λ ve tek pencere değeri, tek reread tasarımı; hiçbir sonuç doğal dile veya büyük modele genellenmez. Olumsuz sonuç, sorunun bu üç ucuz araçla çözülmediğini gösterir; çözülemez olduğunu göstermez.

## Kanıt yolları

- Protokol/kaynak: [`results/drift/frozen.json`](../results/drift/frozen.json); plan: [`docs/drift-mini-plan.md`](../docs/drift-mini-plan.md); kod: `drift/model.py`, `drift/run.py`, `drift/analyze_tolerant.py`, testler `drift/test_drift.py`.
- Koşullar: `results/drift/drift-*/` (initial/last/selected `.pt`, `updates.jsonl`, `train.json`, `complete.json`, `evaluation.json`, 13 `.npz`); çökmeler `results/drift/drift-{five,parity}-reread-s191/`.
- Analiz: [`results/drift/analysis-tolerant.json`](../results/drift/analysis-tolerant.json); işçi olayları `results/drift/workers/`; ledger ve test logları `.superpowers/sdd/drift-mini/`.
- Kapanış manifesti: `audit/drift-mini-manifest.json` + `.sha256` + verification (önceki döngülerin manifestleri dokunulmadı).
