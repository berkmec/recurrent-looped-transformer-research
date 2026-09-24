# U4-sep2 — aralık/sürüklenme menteşesiyle müdahale (U2d'nin nedensel testi, ikinci deneme)

**Durum: ÖNCEDEN KAYITLI NEDENSEL İDDİA REDDEDİLDİ.** 5 eşleşmiş çiftin 2'si kurtarıldı; ikisinde müdahale modeli
belirgin şekilde **kötüleştirdi**. Önceden yazılmış kuralın çıktısı `CONFOUNDED_BY_TRAINING`.

Tarih: 2026-09-23 · Dondurulmuş protokol `results/U4sep2` (`5e10c8d7…`) · Kod commit `a5feb74`, `7bdcbb1`
· 10 koşul (5 kontrol noktası × 2 kol), tamamı tamamlandı

---

## 1. Ana sonuç

512-suffix uç noktasında onarım (%, oracle tavanına göre):

| çift | U2c başlangıç | kontrol | menteşe | menteşe − kontrol | aralık/sürüklenme |
|---|---|---|---|---|---|
| soft/s117 | −40,7 | −40,7 | **+54,4** | **+95,1** | 0,60 → 0,96 |
| soft/s143 | +16,6 | +27,5 | **+52,4** | **+24,9** | 0,52 → 0,81 |
| supervised/s117 | −40,7 | **+61,8** | −44,2 | **−106,0** | 1,03 → 1,22 |
| supervised/s143 | −30,7 | **+87,5** | +84,7 | −2,9 | 2,62 → **5,71** |
| supervised/s191 | −99,5 | −51,8 | −103,6 | **−51,8** | 0,87 → 1,15 |

Hedef büyüklük **beş çiftin beşinde de yükseldi** — menteşe her seferinde işini yaptı. Onarım ikisinde arttı, üçünde
düştü. Dolayısıyla *"aralık/sürüklenme oranını dayanıklı aralığa taşı, dayanıklılık peşinden gelir"* iddiası, en
temiz haliyle kurulmuş bir müdahale deneyinde **tutmadı**.

En keskin karşı örnek supervised/s117: aynı başlangıç ağırlıkları, aynı bütçe, aynı veri, tek fark geometri terimi.
Menteşeli kol oranı **daha yükseğe** çıkardı (1,22 vs 1,03) ve onarımı **−106 puan** düşürdü.

Doğruluk hiçbir kolda bozulmadı (doğrulama 0,984–1,000), yani bu bir "onarımı doğrulukla satın alma" vakası değil.

---

## 2. Hangi büyüklük yönü öngörüyor

Eşleşmiş çiftlerde işaret testi — hangi değişim, onarım değişiminin yönünü doğru söylüyor:

| çift | Δonarım | Δoran | Δmutlak sürüklenme | oran | sürüklenme |
|---|---|---|---|---|---|
| soft/s117 | +95,1 | +0,36 | −1,78 | ✓ | ✓ |
| soft/s143 | +24,9 | +0,29 | −1,63 | ✓ | ✓ |
| supervised/s117 | −106,0 | +0,19 | +1,83 | ✗ | ✓ |
| supervised/s143 | −2,9 | +3,09 | −0,85 | ✗ | ✗ |
| supervised/s191 | −51,8 | +0,28 | +0,39 | ✗ | ✓ |

**Oran 2/5 · mutlak sürüklenme (ters yönde) 4/5.**

Oran beş çiftte de arttığı için, onarımın yönü hakkında hiçbir şey söyleyemez. Mutlak sürüklenme daha iyi bir aday
ama n=5 ve bir ıskayla (rastgele işaret varsayımında 4/5 ≈ 0,19) **kanıtlanmış değil**. Dürüst ifade: müdahale verisi
oran hipotezini reddediyor, sürüklenme hipotezini *öneriyor*, kanıtlamıyor.

Ön kayıtlı dış-örnek tahmini (pair 3'ten sonra, koşmamış iki çift üzerine yazıldı): s143 **ıskaladı**, s191 tuttu →
**1/2**. Yani o tahmin de tutmadı. İşaret testini yazarken asgari etki büyüklüğü koymamıştım; s143'ün Δonarımı −2,9
puan, yani gürültü seviyesi. Bu ön kaydın kusuru, ama eşiği sonradan eklemek kale direğini oynatmak olur — kayıtta
ne varsa o geçerli.

---

## 3. Tasarım kusuru: üç çiftte kontrol kolu boş değil

`supervised` kontrol noktaları U2c'de `görev kaybı + kapı denetimi` ile eğitilmişti; U4-sep2'nin **her iki kolu da**
o denetimi düşürüyor. Yani bu üç çiftte "kontrol" bir boş kontrol değil, kendisi bir müdahale: *denetimi kaldır*.
Ölçülen etkisi büyük — supervised/s117 kontrolü, hiçbir geometri terimi olmadan onarımı −40,7 → +61,8 taşıdı.

**Neyi bozar:** (d) tahmini ("kontrol kolu +10 puandan az oynar") bu üç çiftte yorumlanamaz, ve hükmü
`CONFOUNDED_BY_TRAINING`'e sürükleyen kriter de budur. (d) için temiz kanıt yalnızca iki `soft` çiftinden gelir:
kontroller 0,0 ve +10,9 puan oynadı.

**Neyi bozmaz:** eşli karşılaştırmayı. Her iki kol da denetimi düşürüyor, aralarındaki tek fark menteşe terimi;
dolayısıyla "menteşe − kontrol" farkı beş çiftte de geometri kaybını izole etmeye devam eder. Ve o fark üç çiftte
**negatif** — yani menteşe, denetimi kaldırmanın sağladığı iyileşmenin üstüne koymak bir yana, onu geri almış.

Kök sebep: seçim kuralım "görevi çözüyor mu, kırılgan mı" diye sordu; "ince ayarın düşüreceği bir kayıp terimiyle mi
eğitilmişti" diye sormadı. Bir karşılaştırma ancak taban eğitimi ile ince ayar **yalnızca hedeflenen terimde**
ayrışıyorsa eştir. (Bu, çiftin menteşeli kolu koşmadan önce deftere yazıldı.)

---

## 4. Geri çekilen ara iddia

İlk iki çiftten sonra "uç nokta asimetrisi tekrarlandı — sonek ve kısa-orta kazanıyor, önek kıpırdamıyor" demiştim.
**Beş çiftte tekrarlanmıyor.** Menteşe − kontrol farkının yerleşim ortalamaları: önek **+0,053**, orta −0,013,
sonek **−0,085**. Sonek kazancı iki `soft` çiftine özgüydü ve `supervised` çiftlerinde tersine döndü (−0,682 ve
−0,410). İki koşuldan çıkan deseni "tekrarlandı" diye sunmak erken bir genellemeydi.

---

## 5. Ayakta kalanlar

- **Müdahale, korelasyondan daha çok şey söylüyor.** U2d'nin verisinde aralık ve sürüklenme birbirine dolanmıştı ve
  ayrıştırılamıyordu. Eşli müdahale ikisini ayırdı ve oran hipotezini reddetti. Bu, döngünün asıl kazancı.
- **Eşli kontrol kolu kendini fazlasıyla ödedi.** Sadece "biraz daha eğitim" açıklamasını elemek için eklenmişti;
  sonunda üç çiftte müdahalenin *zararlı* olduğunu gösteren şey oldu. Kontrolsüz bir tasarımda soft/s117'nin +95
  puanı tek başına "başarı" diye raporlanırdı.
- **Menteşe mekanizması teknik olarak çalışıyor:** hedefe varınca duruyor, çökmeye yol açmıyor (sürüklenme tabanı
  sayesinde), görevi bozmuyor. Sorun mekanizmada değil, hedef büyüklüğün seçiminde.

---

## 6. Sıradaki

- **U4-drift** (havuza): doğrudan mutlak sürüklenmeye müdahale eden, kendi ön kaydı olan bir döngü. 4/5 işaret uyumu
  bunu hak ediyor ama kendi başına kanıt değil.
- Kontrol noktası seçim kuralına yeni şart: *taban eğitimi ile ince ayar yalnızca hedeflenen terimde ayrışmalı.*
- Mekanizma bilinmeden de taşınabilir olan sonuç U2c'nin kapısıdır (tam bit-düzeyi koruma); ölçek merdiveninin bir
  sonraki basamağı için aday odur, bu döngünün geometri hattı değil.

---

## 7. DÜZELTME (2026-09-24, bağımsız denetimden sonra) — manşet fazla iddialıydı

Bağımsız bir denetçi, bu döngünün hükmünün **hangi ölçüme bakıldığına bağlı olduğunu** buldu ve haklı. Düzeltme
raporun gövdesine dokunmadan buraya ekleniyor; mühürlü metin olduğu gibi duruyor, ikinci bir manifest çıkarılıyor.

**Sorun: hedef bir birimde kalibre edildi, başka birimde denetlendi.**

`aralık/sürüklenme` iki farklı şekilde ölçülüyor ve ikisi aynı sayı değil:

| ölçüm | nötr yuva | nerede kullanılıyor |
|---|---|---|
| eğitim-zamanı (`sep_finetune.geometry_terms`) | **64** | kuralın (a) maddesi bunu okuyor |
| değerlendirme (`u2d_compare.measure`) | **512** | §1 tablosu bunu gösteriyor |

Sürüklenme koşu uzunluğuyla büyüdüğü için eğitim oranı sistematik olarak 2–3 kat yüksek çıkıyor. Hedef 1,5 ise
protokolün kendi `target_rationale` cümlesinde **512 yuvalık** bantlarla (kırılgan ≤0,57, dayanıklı ≥0,92)
gerekçelendirilmiş. Yani **64 yuvada denetlenen bir eşik, 512 yuvada kalibre edilmiş.**

| çift | eğitim oranı | değerlendirme oranı | hedef 1,5'e ulaştı mı |
|---|---|---|---|
| soft/s117 | 2,54 | **0,96** | eğitimde evet, değerlendirmede **hayır** |
| soft/s143 | 3,09 | **0,81** | eğitimde evet, değerlendirmede **hayır** |
| supervised/s117 | 1,80 | **1,22** | eğitimde evet, değerlendirmede **hayır** |
| supervised/s143 | 5,57 | 5,71 | evet |
| supervised/s191 | 2,07 | **1,15** | eğitimde evet, değerlendirmede **hayır** |

Kural değerlendirme oranıyla yeniden koşulduğunda hüküm **`CONFOUNDED_BY_TRAINING` → `INTERVENTION_INEFFECTIVE`**
oluyor; yani *"menteşe hedefine ulaşmadı, nedensel soru test edilmedi"*.

**Manşetin düzeltilmiş hâli.** Raporun başlığındaki *"ÖNCEDEN KAYITLI NEDENSEL İDDİA REDDEDİLDİ"* fazla iddialıdır.
İnmeyen bir müdahaleyle bir iddia reddedilemez — U4-sep için tam olarak bu mantığı kabul etmiştim ve burada kendi
kuralımı kendime uygulamadım. Doğru ifade:

> **Müdahale yalnızca kısmen indi** (doğru birimde 5 çiftin 1'inde hedefe ulaştı) ve indiği kadarıyla oran,
> sonucun yönünü öngörmedi.

**Buna rağmen ayakta kalan, birimden bağımsız argüman.** Oran **5/5 çiftte yükseldi** — her iki birimde de, menteşe
kolu kontrolünün üstünde (değerlendirme biriminde farklar: +0,36 / +0,29 / +0,19 / +3,09 / +0,28). Onarım ise
ikisinde arttı, üçünde düştü. **Her yerde aynı yöne giden bir büyüklük, zıt sonuçları açıklayamaz.** Bu akıl
yürütme hangi birimi seçtiğinizden bağımsızdır ve oran hipotezini zayıflatmaya devam eder. Değişen şey, bunu
"reddedildi" diye ilan etme hakkıdır.

**M-hattına etkisi:** plandaki 7. madde (*"geometri düzenleyicisi KOYMA"*) her iki hükümde de ayakta — biri "oran
dayanıklılığı açıklamıyor", diğeri "müdahale inmedi" diyor; ikisi de geometri terimini modele koymak için gerekçe
vermiyor.

**Ek kusur, aynı denetimden:** `ratio_last` eğitimin **son** satırından, yani `final.pt` modelinden okunuyor; oysa
değerlendirilen model `selected.pt`. 10 koşunun 3'ünde `selected_step ≠ steps` (384, 256, 256), dolayısıyla o
koşullarda (a) maddesi **hiç değerlendirilmemiş bir modeli** tarif ediyor.
