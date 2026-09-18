# RLT tanı döngüsü R1–R6 — sonuç raporu

**Durum:** MEASURED_BOUNDED_DIAGNOSTICS. 16 Eylül incelemesinin önerdiği altı tanı kontrolünün (R1–R6) tamamı yerel olarak yürütüldü, ayrı ayrı dondurulmuş protokollerle ölçüldü ve bağımsız ham denetimlerden geçti. Bu döngü Faz 2'nin on iki deneyini (D1–D4, B1–B4, U1–U4) **yerine geçmez**; yalnız onların ön koşulu olan belirsizlikleri çözer. Serbest doğal dil, anlamsal bellek ve sınırsız doğru düşünme iddiaları bu döngüde ölçülmedi.

Tarih: 17–18 Eylül 2026. Kök: `E:/rlt-research/2026-09-17-diagnostics` (git worktree, dal `codex/rlt-review-diagnostics`, taban `3c118c7`). Mühürlü Faz 1 kökü (`2026-09-16-e2e`, manifest `146eb693…`) ve Faz 2 plan dosyaları değiştirilmedi; kapanışta kimlikleri yeniden doğrulandı.

## Beş bulgu

1. **Bellek tasarrufu gerçek ama yalnız parça-parça geri yayılımla (R6).** Orijinal detach-only TBPTT8 ve bütün-ileri checkpoint kolları CUDA tepe belleğini hiç düşürmedi (oran 1,00; checkpoint 1,8–1,95× yavaş). Her sekiz adımda decoder grafiğini serbest bırakan segment yürütmesi aynı gradyanı (fark ≤1,5e-7) verirken tepe belleği L1024'te 86,6→45,5 MiB'ye (0,53×; aktivasyon payı ≈0,41×) indirdi, süre 0,99×. *Kapsam:* genişlik 32, batch 8, tek GPU, üç ölçülü güncelleme; ölçek yasası değil.
2. **Öğrenme sorunu etiket yoğunluğu değil, etiketin konumu ve geri besleme yolu (R3).** Dizi başına rastgele konumda **tek** etiket, 32 etiketli yoğun gözetimle **aynı** sonucu verdi (parite, üç tohum, 32/128/1024 uzunluklarda %100; fark 0,00 puan). Her zaman son konumda tek etiket ise şansta kaldı (−54,3 puan, aralık [−55,8; −52,8]). Geri beslemesiz mimari üç gözetim düzeninde de şansta kaldı. "Gözetim yoğunluğu mimariden belirleyici" tezi bu biçimiyle desteklenmedi.
3. **Faz 1'deki "büyük modelde 1/3 tohum öğrendi" sonucu öğrenme oranı duyarlılığıydı (R5).** Genişlik 256'da sabit lr 0,003 üç geliştirme tohumunda da başarısız (doğrulama %49–55); lr 0,0003 ve 0,001 üçünde de %100. Yalnız doğrulamayla seçilen lr 0,0003 ile üç **taze** tohum 32/128/1024 uzunlukta %100. Nötr-koşu dayanıklılığı yine tohuma bağlı: 3 onay tohumundan 1'i 128 nötr orta/son ekinde %50'ye düştü.
4. **Bloklu nötr yerleşimle eğitmek uzun nötr koşulara dayanıklılık sağlamadı (R2).** Paritede 128 nötr son ekinde −16,0 puan (tohumlar −44,7 / +0,2 / −3,3), beş durumda +30,7 puan ama tohumlar +56 / −17 / +53 ve temiz L1024 doğruluğunda −54,4 puan bedelle. Rastgele yerleşimle eğitilen parite modelleri bu kampanyada 128 nötr son ekine 3/3 dayandı (Faz 1'de 1/3); beş-durum modelleri ise 32–128 nötr **orta/son** eklerinde çöktü, **ön** ekler zararsızdı.
5. **Yol sertifikası dar bir kazanç verdi, zincir görevlerinde beklenen sıfır etki (R1); TBPTT8 ile tam BPTT beş durumda ayırt edilemedi (R4).** R1: dallanmış dağılımda 512 örnekte 6 / 0 / 22 ek kabul, sıfır yanlış kabul; depth8 zincir abstention sorunu çözülmedi. R4: yoğun gözetimde L128 farkı 0,00 puan ([−1,9; +1,9]); son-etiket gözetiminde iki kol da şansta.

## Sözlük

| Terim | Bu raporda |
|---|---|
| Nötr işlem | Görevin doğru durumunu değiştirmeyen token 2; eğitimde olasılığı 0,125 |
| Rastgele / bloklu yerleşim | Eğitim dizisinde nötr tokenların dağınık kalması / ön-orta-son bloklarda toplanması (nötr sayısı ve diğer işlemlerin sırası aynı) |
| dense / sparse / final | Her konumda etiket / dizi başına rastgele bir konumda etiket / yalnız son konumda etiket |
| nofeedback | RLT'nin gizli durum geri beslemesi kapalı, dikkat yolları açık kontrol |
| tbptt8 | s/dk/dv gradyanı sekiz adımda kesilir; global encoder ve mk/mv yolları açık |
| segment | Aynı tbptt8 gradyanını chunk başına backward ile hesaplayıp decoder grafiğini serbest bırakan yürütme |
| Eşit-etiket kopyası | Yoğun koşulun tam 64. adım kaydı: 64×32×32 = 65.536 etiket = 2048×32 tek-etiket |
| Onay tohumu | LR seçiminden **sonra** eğitilen, seçimde kullanılmayan taze tohum (217/243/291) |
| t95 | Üç tohum farkının ortalaması ± 4,3027·s/√3; keşif amaçlı, çokluk düzeltmesiz |

## Yöntem ve dondurma

- Eğitim protokolü `results/training/frozen.json` (hash `eed6fb30…`; kaynak pin'leri `core/model.py 121f11fc…`, `review_cycle/training.py df14af4e…`, `training_data.py 3095902c…`), analiz sözleşmesi `docs/training-analysis-plan.md` (pin `1949d8d7…`, final veri açılmadan önce ledger'a yazıldı). 2048 güncelleme, batch 32, uzunluk 32, AdamW (wd 0,01, clip 1), küçük modeller w32/2 baş lr 0,003 CPU tek iş parçacığı; büyük w256/4 baş CUDA. Birincil tohumlar 117/143/191. Checkpoint seçimi doğrulama son-durum CE minimumu.
- Karşılaştırılan kollar aynı görev/tohumda **aynı başlangıç ağırlıklarını ve aynı eğitim girişlerini** kullanır (yerleşim yalnız sırayı, gözetim yalnız etiketlenen konumu değiştirir). Toplam 36 küçük + 9 geliştirme + 3 onay = 48 koşul; hepsi `complete.json` ile mühürlü, FAILED yok.
- Final test: seçimler sabitlendikten sonra tek seferde açıldı (`test-gate.json` COMPLETE_SCOPE, 39 seçim). Her model için taze 32/128/1024 uzunlukta 512'şer örnek ve aynı 32'lik tabana 0/8/32/128 nötr ön/orta/son ekleme; 60 varyant (39 seçilmiş + 21 eşit-etiket) × 13 küme, ham tahminler `.npz`. Geliştirme koşulları test göremedi.
- R1 protokolü `results/R1/protocol.json` (`16e16221…`), R6 protokolü `results/review-diagnostics-r6/frozen.json` (`2eb5f530…`). Her ölçüm için üretici hash'leri, taze kaynak tohumları ve makbuzlar kayıtlı.

## R1 — Yol sertifikası (MEASURED; dallanmışta sınırlı kazanç, zincirde sıfır)

2.048 taze kaynak problemi (iid / depth5 / depth8 zincir kontrolleri + yeni dallanmış OOD), üç dondurulmuş Faz 1 modeli, sekiz tur; 6.144 model-problem koşusu. Doğruluk eşiği 0,1; kabul eşiği 0,099999; tam rasyonel aritmetik.

| Dağılım | Tohum | Doğru aday/512 | Bileşen kabul | Yol kabul | Yanlış kabul |
|---|---:|---:|---:|---:|---:|
| iid | 17/43/91 | 498/503/495 | 108/149/165 | 108/149/165 | 0 |
| depth5 | 17/43/91 | 270/288/289 | 18/37/34 | 18/37/34 | 0 |
| depth8 | 17/43/91 | 130/185/252 | 0/0/0 | 0/0/0 | 0 |
| branched | 17/43/91 | 166/137/265 | 0/0/0 | **6/0/22** | 0 |

Zincir dağılımlarında tek çapa–sorgu yolu bileşenin bütün kenarlarını kapsadığı için iki sınır matematiksel olarak eşittir; oradaki eski retler değişmedi. Dallanmış grupta ortalama +1,82 puan kapsam, üç-tohum t95 [−3,69; +7,34]. 6 ve 22 kabulde sıfır hatanın tek taraflı %95 üst hata sınırı ≈%39 ve ≈%13 — az kabulden güçlü güvenilirlik çıkmaz. Ayrıntı: [R1-report.md](R1-report.md), [matematiksel sözleşme](R1-proof.md), bağımsız ham denetim [r1-main-audit.md](../.superpowers/sdd/review-diagnostics-plan/r1-main-audit.md) (49.152 aday turu, 294.912 kabul kararı, 3.943 sertifika yeniden hesaplandı, PASS).

## R2 — Nötr yerleşimi (parite: MEASURED_NEGATIVE; beş durum: MIXED)

Birincil son nokta: eğitimde görülmemiş 128'lik nötr **son ek**. Değerler seçilmiş checkpoint son-durum doğruluğu, üç tohum.

| Görev / kol | L32 | L128 | L1024 | nötr0 | 32 son | 128 ön | 128 orta | 128 son |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parite rastgele | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 1,00 | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 1,00/0,94/1,00 | 1,00/1,00/1,00 |
| parite bloklu | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 1,00/0,96/1,00 | 1,00 | 1,00/1,00/1,00 | 1,00/1,00/1,00 | 0,51/1,00/0,64 | 0,55/1,00/0,97 |
| beş rastgele | 1,00/1,00/1,00 | 1,00/0,99/1,00 | 1,00/0,98/1,00 | 1,00/1,00/1,00 | 0,72/0,36/0,76 | 1,00/0,99/1,00 | 0,16/0,42/0,39 | 0,40/0,36/0,45 |
| beş bloklu | 0,98/0,96/0,98 | 0,89/0,82/0,94 | 0,37/0,33/0,64 | 0,97/0,95/0,98 | 0,96/0,94/0,98 | 0,93/0,53/0,89 | 0,96/0,18/0,92 | 0,96/0,20/0,98 |

Birincil fark (bloklu − rastgele, 128 son): parite −15,95 puan, t95 [−78,0; +46,1]; beş durum +30,73 puan, t95 [−71,2; +132,6]; ikisi de sıfırı kapsar. Beş durumda bloklu eğitimin bedeli: L32 −2,73 [−5,3; −0,2], L1024 −54,43 [−94,3; −14,5] (negatif aralıklar). Sonuç: bloklu yerleşim genel bir dayanıklılık mekanizması değil; iki tohumda uzun son eke dayanıklılık kazandırırken temiz uzun-ufuk doğruluğunu düşürdü, bir tohumda çöktü. Faz 1'in "rastgele dizilerde başarı nötr eklere dayanıklılık garanti etmez" bulgusu duruyor; buna ek olarak nötr koşunun **konumu** belirleyici: durum kurulmadan önceki (ön) koşular zararsız, kurulduktan sonraki (orta/son) koşular beş-durumda 32 uzunluktan itibaren zararlı. Parite modelleri bu kampanyada 128 son eke 3/3 dayandı; Faz 1'de aynı genişlik ve görevde 1/3 dayanmıştı (farklı tohumlar ve protokol pin'i; iki kampanya birleştirilmez).

## R3 — Gözetim yoğunluğu × mimari (MEASURED; yoğunluk tezi desteklenmedi)

Parite, L128 birincil, aynı girişler/güncellemeler; sparse ve final birer etiket taşır.

| Karşılaştırma | Ort. puan | t95 | Yön |
|---|---:|---|---|
| rlt sparse − dense | **0,00** | [0; 0] | fark yok (üçü de %100) |
| rlt final − dense | −54,30 | [−55,8; −52,8] | negatif |
| rlt final − sparse | −54,30 | [−55,8; −52,8] | negatif |
| rlt − nofeedback (dense) | +52,28 | [+47,1; +57,5] | pozitif |
| rlt − nofeedback (sparse) | +49,15 | [+38,9; +59,4] | pozitif |
| rlt − nofeedback (final) | −1,76 | [−9,3; +5,8] | ikisi de şansta |
| nofeedback sparse/final − dense | +3,12 / −0,26 | sıfırı kapsar | hepsi şansta |

Rastgele konumda tek etiket, dizinin ortalama 16 adım gerisine kredi atar ve bazen erken konumları etiketler; son konumda tek etiket her zaman 32 adım geriye atar ve bu sabit 2048 güncellemede öğrenilmedi. Eşit-etiket kopyaları (dense 64. adım, 65.536 etiket) bütün görevlerde şans düzeyinde (parite 0,46–0,55; beş durum 0,17–0,24): eşit etiket sayısı 64 güncellemede yetmez; bu karşılaştırma yoğunluğun tek başına etkisini **izole etmez**, yalnız güncelleme sayısının belirleyici olduğunu gösterir. Geri beslemesiz kontrol üç düzende de şansta: mimari (durum geri beslemesi) gerekli koşuldur.

Bağımsız denetimin saptadığı önemli ayrıntı: **son-etiket kolları öğrenmemiş, sabit sınıfa çökmüş modellerdir** — altı beş-durum son-etiket koşulu ile parite son-etiket s143/s191 seçilmiş modelleri 13 son noktanın 13'ünde tek sınıf tahmin ediyor (s117'de 2/13); %45–54 "doğruluk" sınıf önselidir, kısmi öğrenme değildir. Geri beslemesiz kollar da 2–10/13 son noktada sabit sınıfa düşüyor. Dolayısıyla R3/R4'teki "final" karşılaştırmaları çökmüş modeller arasındadır. Ayrıca tbptt8-final koşullarında başlangıç-durum parametresi tam sıfırda ve AdamW momentleri sıfır: yalnız son konumdaki etiketin gradyanı sekiz adımlık kesilmiş yoldan 32 adım geriye, başlangıç durumuna hiç ulaşmıyor — kredi atama mesafesi yorumuyla tutarlı.

## R4 — Kesilmiş gradyan, beş durum (MEASURED; fark saptanmadı)

| Karşılaştırma (beş durum, L128) | Ort. puan | t95 |
|---|---:|---|
| full − tbptt8, dense | 0,00 | [−1,94; +1,94] (tohumlar +0,78 / −0,78 / 0) |
| full − tbptt8, final | 0,00 | [0; 0] (iki kol da şansta, 0,15–0,22) |

L1024'te full 1,00/0,98/1,00, tbptt8 0,95/1,00/1,00. Kesim yalnız s/dk/dv yollarında; global encoder ve mk/mv yolları açık. Sonuç bütün yolları kesen bir modele veya başka görevlere genellenmez; Faz 1'in parite bulgusu beş duruma taşındı.

## R5 — Büyük model (w256, 1,51M parametre) öğrenme oranı (MEASURED_POSITIVE: duyarlılık; dayanıklılık tohuma bağlı)

Geliştirme (yalnız doğrulama, seçilmiş checkpoint son-durum doğruluğu): lr 0,0003 → 1,000/1,000/1,000; lr 0,001 → 1,000/1,000/1,000; lr 0,003 → 0,492/0,551/0,531 (seçilen adımlar 1536/512/512, kayıp ≈0,69). Seçim kuralı beraberlikte en düşük LR → **0,0003**. Onay tohumları (seçimden sonra, taze):

| Tohum | L32 | L128 | L1024 | nötr0 | 128 ön | 128 orta | 128 son | Sabit sınıf? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 217 | 1,000 | 1,000 | 1,000 | 1,000 | 0,932 | 0,502 | 0,482 | hayır |
| 243 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,998 | 1,000 | hayır |
| 291 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,996 | 1,000 | hayır |

Faz 1'in sabit lr 0,003 ile "3 tohumdan 1'i öğrendi, ikisi %48,44" sonucu, bu ölçekte protokolün LR'ye duyarlılığıdır; kapasite sınırı değildir. Üç onay modelinin 39 son noktasının hiçbirinde tam sabit-sınıf tahmin yok; s217 128 nötr orta/son ekinde tek sınıfa yakın (505/512) — çöküşe yakın ama tam değil. Genişlik üstünlüğü testi yapılmadı (küçük ve büyük modeller farklı tohum/LR ile eğitildi).

## R6 — Bellek ve gradyan yolu (MEASURED_POSITIVE: segment; detach/checkpoint: etki yok)

Dondurulmuş protokol: w32/2 baş, batch 8, float32, chunk 8, uzunluk 32/128/512/1024, kol başına taze süreç, 2 ısınma + 3 ölçülü gerçek AdamW güncellemesi; 20/20 koşul PASS, OOM yok. Taze-süreç CUDA yol eşdeğerliği (175 tensör/çift: model, AdamW durumu, gradyan): full↔checkpoint 0 / 3,1e-7; detach↔segment 1,4e-7 / 1,5e-7; detach↔segment+encoder-ckpt 1,4e-7 / 8,2e-8 (L32 / L128, tolerans 3e-4).

| Uzunluk | Kol | Güncelleme s | Full oranı | CUDA tepe alloc MiB | Tepe oranı |
|---:|---|---:|---:|---:|---:|
| 128 | full / detach / checkpoint | 0,896 / 0,908 / 1,656 | 1,00 / 1,01 / 1,85 | 25,4 / 25,4 / 25,4 | 1,00 / 1,00 / 1,00 |
| 128 | segment / +encoder ckpt | 0,917 / 0,944 | 1,02 / 1,05 | 20,7 / 20,0 | 0,81 / 0,79 |
| 1024 | full / detach / checkpoint | 7,281 / 7,905 / 14,225 | 1,00 / 1,09 / 1,95 | 86,6 / 86,5 / 86,6 | 1,00 / 1,00 / 1,00 |
| 1024 | segment / +encoder ckpt | 7,232 / 8,392 | 0,99 / 1,15 | 45,5 / 43,6 | 0,53 / 0,50 |

Optimizer sonrası sabit tahsis her kolda 16,6–16,8 MiB; aktivasyon tepe payı L1024'te ≈69,8 → 28,7 MiB. L512 zaman satırları güvenilmez (full ve segment+encoder-ckpt işçilerinde host meşguliyeti 0,50/0,57; diğerleri 0,15–0,26); bellek tepeleri host yükünden etkilenmez. Süreç çalışma kümesi (Python/PyTorch/CUDA çalışma zamanı dahil, aktivasyon değil) segment kollarında ≈1,0 GiB, full/detach 1,0–1,2 GiB, checkpoint 1,4 GiB'e kadar. Hooks tanısı (segment L128): backward için saklanan tekilleştirilmiş depolama tepe 2,83 MiB, 10.540 kayıt isteği/güncelleme. Ayrıntı ve sınırlar: [R6-report.md](R6-report.md); bağımsız ham denetim [r6-independent-audit.md](../.superpowers/sdd/review-diagnostics-plan/r6-independent-audit.md) (PASS, açıklanmış tek sapma: venv `python.exe` yönlendiricisi nedeniyle launcher pid ≠ işçi pid; işçi tarafı pid zinciri tutarlı).

## Hesap bütçesi (gerçek)

| İş | Cihaz | Süre |
|---|---|---|
| R1 ana koşu (12 makbuz, 6.144 koşu × 8 tur) | — | ≈1 dk (3,6 s/makbuz) |
| 36 küçük eğitim (2048 güncelleme) | CPU, 1 iş parçacığı, 2 paralel yuva | koşul ort. 231 s; toplam 2,31 saat işçi-zamanı (yuvalar 65 + 78 dk) |
| 9 geliştirme + 3 onay (w256) | RTX 2060 SUPER | ort. 586 s / 545 s; 1,46 + 0,45 saat (seçim işçisi 6.330 s) |
| Final değerlendirme (60 varyant × 13 küme) | CUDA | 6.307 s (1,75 saat) |
| R6 pilot + doğrulama + 20 koşul + tanı | CUDA | ≈10 dk |
| R2–R5 analiz ve özet | CPU | < 1 dk |

Eşzamanlı işlerin süreleri toplanıp geçen süre olarak sunulmadı. Enerji ölçülmedi. Ek eğitim süresi (checkpoint I/O, günlük fsync) makbuzlarda `wall_seconds` − `train_seconds` olarak ayrıdır.

## Bağımsız denetimler

- **R1:** `r1-main-audit.md/json`, `r1_raw_audit.py` — PASS (üretici çözücüsünü içe aktarmadan bağımsız rasyonel çözüm, 6 kaynak pin'i, 384 gerçek yeniden oynatma).
- **R6:** `r6-independent-audit.md/json`, `r6_raw_audit.py` — PASS_WITH_EXPLAINED_DEVIATIONS (0 sert hata; summary.json 21 satırda 0 uyuşmazlık; eşdeğerlik tensör-tensör yeniden üretildi).
- **R2–R5:** `r2r5-independent-audit.json`, `r2r5_raw_audit.py` — PASS (24/24; 780 son nokta ve 71 karşılaştırma farksız yeniden hesaplandı); ayrıntı aşağıdaki bölümde.

## Bağımsız R2–R5 denetimi

`r2r5-independent-audit.json` (betik `r2r5_raw_audit.py`; taze denetçi ajan, CPU tek iş parçacığı, salt-okunur, üretici kodu içe aktarılmadan, model ileri geçişi yapılmadan; 40,9 s): **verdict PASS, 24 kontrol geçti, 0 başarısız.** Yeniden hesaplananlar: `frozen.sha256` ve kanonik protokol hash'i; üç kaynak pin'i; analiz sözleşmesinin final veri öncesi ledger pin'i (`1949d8d7…`); 48 makbuz/mühür/checkpoint hash'i; 48 koşulda 2048 gerçek güncelleme ve optimizer adımı, en az 30 parametre tensörünün değiştiği; seçim hash'i ve kuralı (`selected_lr` 0,0003 yeniden türetildi); test kapısındaki 39 seçim; geliştirme dizinlerinde değerlendirme dosyası olmadığı; aynı görev/tohumdaki kolların aynı başlangıç ağırlığı ve aynı özgün eğitim girişlerini paylaştığı, farklı tohumların paylaşmadığı; 60 varyant × 13 küme = 780 son noktada son-durum/prefix doğruluğu, histogramlar ve eşli değişimlerin ham `.npz`'den yeniden hesaplanıp `evaluation.json` ile **farksız** (0,0) eşleştiği; taze test girişlerinin görev/tohum içinde özdeş ve L32 eğitim kaynaklarından ayrık olduğu; 71 önceden tanımlı karşılaştırmanın ve 3 onay satırının **farksız** yeniden üretildiği; t katsayısının df=2 kapalı biçimle (4,3027) uyuştuğu ve tohumların birleştirilmediği. Denetçinin anomali listesi (556 satır; bütünlük hatası değil, gözlem): 223 sabit-sınıf son nokta satırının **163'ü 18 seçilmiş modelde** (bütün son-etiket kolları ve bazı geri beslemesiz kollar; bkz. R3), 60'ı 19 eşit-etiket kopyasında; 23/48 koşulda seçilmiş checkpoint 2048'den erken (kural doğrulandı); lr 0,003 geliştirme koşulları ≈0,5'te çökmüş; geri beslemesiz kollarda 5 geri besleme tensörü beklendiği gibi hiç güncellenmemiş; L1024 batch-8 değişmezliği denetçi tarafından yeniden doğrulanmadı. Denetçi ajanın `.md` raporu API kota kesintisi nedeniyle JSON'dan sonra teslim edildi ([r2r5-independent-audit.md](../.superpowers/sdd/review-diagnostics-plan/r2r5-independent-audit.md)).

## Faz 2 planına geri eşleme (Faz 2 tamamlanmadı)

| Tanı | Etkilediği paket | Karar |
|---|---|---|
| R1 | D2, B2, U3 | Zincir abstention sorunu daha dar sınırla çözülmez; U3 için kabul kapsamı artırmak ara gözetim/adaylık kalitesine bağlı. Dallanmış topolojide yol sınırı kullanılabilir, kazanç tohuma bağlı. |
| R2 | U1, U2 | Nötr-koşu dayanıklılığı bir eğitim-dağılımı hilesiyle (bloklu yerleşim) alınamaz; U1/U2 nötr koşunun konumunu ve durumun kurulma zamanını ayrı faktör olarak ölçmeli. |
| R3 | D4, U2 | D4'te "etiket yoğunluğu" kolu yerine "kredi atama mesafesi" ve "geri besleme yolu" kolları; tek rastgele etiket pahalı yoğun etiketlemenin yerine ucuz bir taban olabilir. |
| R4 | U1, D4 | Sekiz adımlık kesim bu görevlerde kayıp vermez; D4'te TBPTT kullanılabilir. Bütün yolların kesilmesi ölçülmedi. |
| R5 | D4 | Büyük modele geçişte LR geliştirme taraması zorunlu; sabit LR sonuçları kapasite sonucu sayılmaz. |
| R6 | G0, D4 | Uzun bağlam eğitiminde segment yürütmesi tepe belleği yarıya indirir; detach-only veya bütün-ileri checkpoint bellek aracı değildir. Ölçüm aracı (taze süreç, faz tepeleri, eşdeğerlik kapısı) G0'a hazır. |

## Süreç ve dürüstlük notları

- Döngü Codex çok-ajanlı SDD oturumunda başladı (Task 1–2 tamam, Task 3 ön-inceleme, eğitimler çalışıyor) ve kontrolör değişimiyle Claude Code (Fable 5.1) tarafından aynı ledger'dan sürdürüldü; devir noktası kullanıcının notu ve `events.jsonl` ile doğrulandı. Bütün kararlar `progress.md`'de `Ruling:` satırları olarak kayıtlı.
- R6 kaynak kodu ön-inceleme bulguları için dondurmadan **önce** kontrolör tarafından düzeltildi (OOM anında CUDA sayaçları; segment kolunda chunk başına örtük senkronizasyonun kaldırılması; 4 hunk, gradyan yolu değişmedi); ilk yazım dosyayı CRLF'ye çevirdi, LF'ye geri alındı; olay ledger'da.
- Faz 1'in `tests/` paketi bu worktree'de 101 geçti / 7 başarısız: yedisi de worktree'de bulunmayan git-ignored `memory/checkpoints/extractor-17.pt` dosyasını arıyor; aynı paket mühürlü kökte 108/108 geçti. Hiçbir test silinmedi veya değiştirilmedi; bu döngünün kendi 49 testi geçti.
- Hiçbir koşulda OOM, FAILED veya boş olmayan stderr yok. Kesilen segment yeniden yazılmadı; kesinti olmadı.
- Kontrolörün denetim JSON'undaki anomali listesini ilk okuması ilk 16 satırla sınırlıydı ve "sabit sınıf yalnız eşit-etiket kopyalarında" biçiminde yanlış genellenmişti; denetçinin raporu bunu düzeltti ve bu rapor buna göre değiştirildi (163 sabit-sınıf satırı seçilmiş modellerde).
- Sınırlar: küçük modeller ve iki sentetik görev; üç tohum; keşif aralıkları çokluk düzeltmesiz; Faz 1 ile bu kampanya farklı tohum/protokol pin'leriyle koştuğu için birleştirilmedi; hiçbir sonuç doğal dile, büyük dil modeline veya sonsuz doğru akıl yürütmeye taşınmaz.

## Kanıt yolları

- Ledger ve karar kaydı: [`progress.md`](../.superpowers/sdd/review-diagnostics-plan/progress.md); plan: [`review-diagnostics-plan.md`](../docs/review-diagnostics-plan.md); analiz sözleşmesi: [`training-analysis-plan.md`](../docs/training-analysis-plan.md).
- R1: [`results/R1/`](../results/R1/) (protokol, 12 makbuz, ham `.npz`, sertifikalar, `summary.json`).
- R2–R5: [`results/training/`](../results/training/) (`frozen.json`, 48 koşul, `selection.json`, `test-gate.json`, `evaluation.json`, `summary.json`, `scientific-analysis.json`, ham `.npz`, işçi olayları, launcher kayıtları).
- R6: [`results/review-diagnostics-r6/`](../results/review-diagnostics-r6/) (`frozen.json`, `validation/equivalence.json`, 21 ölçüm dizini, `summary.json`); pilotlar ve tanı kalibrasyonu SDD dizininde.
- Kapanış manifesti: `audit/review-diagnostics-manifest.json` + `.sha256` + `-verification.json` (Faz 1 manifesti `audit/final-manifest.json` dokunulmadı).
