# U2c — Öğrenilmiş HOLD/WRITE kapısı (2026-09-22/23)

## Bulgular (önce)

1. **Sürüklenme, modelin kendi öğrendiği bir kapıyla kapanıyor — oracle olmadan.** U2b'de nötr tokeni dışarıdan söyleyerek elde ettiğimiz kayıpsız onarım tavanına, eğitilmiş kapı üç kolda da yaklaşıyor ya da ulaşıyor. 512 nötr adımlık son ek için onarım (oracle'ın kapattığı açığın yüzdesi, tohum başına):

| kol | beş-durum | yığın | temiz 4096 farkı (beş / yığın) |
|---|---|---|---|
| `soft` | **+%100 +%100 +%100** | +%31 +%5 **−%45** | +0,014 / +0,060 |
| `hard_st` | +%100 +%84 +%98 | **+%102 +%99 +%95** | +0,014 / −0,005 |
| `supervised` | +%64 +%100 +%49 | +%99 +%98 +%100 | **−0,385** / +0,049 |

2. **Her iki görev ailesinde de işe yarayan tek kol `hard_st`** (tam bit-bit kopya): beş-durumda ortalama %94, yığında %99, temiz dizilerde kayda değer bedel yok. `soft` sabit-durumlu görevde kusursuz ama yığında hiçbir şey yapmıyor (ortalama −%3, bir tohumda zarar). `supervised` yığında iyi, beş-durumda hem zayıf hem yıkıcı (temiz 4096'da ortalama −0,385).
3. **Mekanizma ayrımı ölçüldü.** Yumuşak karışımda tutulan adımda durum yine kısmen yazılıyor; bu sızıntı permütasyon kodlu durumda (beş-durum) zararsız, içerik taşıyan durumda (yığın) birikip bozuyor. Bit-bit kopya ikisinde de sağlam. Tasarım paketinin P1 iddiası — *koruma yaklaşık değil tam olmalı* — bu karşıtlıkla doğrudan destekleniyor.
4. **Kapı, kimse söylemeden doğru kuralı bulabiliyor — ama "doğru kural" her zaman en iyisi değil.** Yığında `hard_st`'in temiz dizilerdeki tutma oranı üç tohumda 0,149 / 0,152 / 0,153; görevin doğal nötr oranı 0,1515, yanlış-yazma 0,000. Buna karşılık beş-durumda en iyi sonucu veren kollar doğal oranın (0,125) üç katı kadar tutuyor (~0,38). Yani optimal kural göreve bağlı ve bizim önselimizle örtüşmüyor.
5. **Önsel dayatmak geri tepti.** `supervised` kolu kapıya U2b'nin k=2 kuralını öğretiyor ve kapı bunu kusursuz öğreniyor (yanlış-tutma 0,000, temizde tutma 0,003). Ama bu kural *dondurulmuş modele test anında* uygulanınca optimaldi; *birlikte eğitilen* kapıya hedef olarak verilince model tutmayı hiç kullanmıyor, uzun dizilerde sürüklenme geri geliyor (beş-durumda temiz 4096: 1,000 → 0,284 / 0,608 / 0,912).
6. **Kararlılık ile kalite ayrışıyor.** Denetim gradyan patlamasını gerçekten çözüyor (tepe norm 4,7e2, `hard_st`'te 1,6e20) ama sonucu kötüleştiriyor. Aynı şekilde düşük öğrenme oranı kararlı ama kırılgan çözümler üretiyor (aşağıda).

## Öğrenme oranı: ayrı bir eksen, ayrı bir ders

Gece `hard_st` kolu 513. adımda çöktü (ayrıntı aşağıda). Sebebi anlamak için ayrı bir **geliştirme probu** koşuldu (`results/U2c-lr`, 2 kol × 4 lr, yalnız eğitim kararlılığı ve doğrulama — test/nötr uçlarına dokunmadan):

| lr | `hard_st` sıçrama / tepe norm / tutma | `soft` sıçrama / tepe norm / tutma |
|---|---|---|
| 0,003 | **6** / 1,6e20 / 0,368 | 0 / 3,0e4 / 0,373 |
| 0,001 | 0 / 9,3e3 / 0,353 | 0 / 1,0e2 / 0,386 |
| 0,0003 | 0 / 8,5e2 / 0,121 | 0 / 9,2e3 / 0,379 |
| 0,0001 | 0 / 6,2e3 / 0,120 | 0 / 1,9e4 / 0,275 |

Kayıp sıçraması **yalnız** `hard_st` × 0,003 hücresinde: kararsızlık kol ile lr'nin **etkileşimi**, lr'nin tek başına etkisi değil. Patlamanın imzası da ölçüldü: önce kapı kararları toptan değişiyor (tutma oranı bir adımda 0,31 → 0,56), kayıp 0,026 → 3,27 sıçrıyor, **sonra** gradyan patlıyor — ayrık karar optimizasyonunun klasik kararsızlığı.

Prob "en iyi kararlı lr = 0,0003" dedi. Tam ızgara (18 koşul, `results/U2c-lr0003`) bunu **çürüttü**: o lr'de `soft` kolunun altı koşulunun altısı da düz çekirdeğin altına düşüyor, biri görevi hiç öğrenemiyor, `hard_st` beş-durumda 512-orta 0,920 → 0,218'e iniyor ve bir tohumda temiz 4096 çöküyor — **oysa bu koşulların neredeyse hepsinde doğrulama doğruluğu 1,000'di.**

**Yöntem dersi (havuz C16):** eğitim dağılımındaki doğrulama, ölçtüğümüz özelliğe (sürüklenme dayanıklılığı, uzunluk genellemesi) **kör**. lr'yi doğrulamaya bakarak seçmek — yani standart ML refleksi — sonucu sessizce yok ederdi.

### İki lr seviyesi yan yana (36 koşul tamam)

| kol | görev | onarım @ lr 0,003 | onarım @ lr 0,0003 | temiz 4096 farkı (yüksek / düşük lr) |
|---|---|---|---|---|
| **`hard_st`** | **yığın** | **+%99** | **+%92** | −0,005 / −0,001 |
| `hard_st` | beş-durum | +%94 | +%79 | +0,014 / **−0,471** |
| `soft` | beş-durum | +%100 | **−%43** | +0,014 / −0,524 |
| `soft` | yığın | −%3 | −%46 | +0,060 / +0,031 |
| `supervised` | beş-durum | +%71 | **−%57** | −0,385 / −0,796 |
| `supervised` | yığın | +%99 | **−%39** | +0,049 / −0,003 |

**Sonuç: tam kopya kapısı, eğitim rejimi bozulduğunda ayakta kalan tek kol.** On kat düşük öğrenme oranında `hard_st` zarif biçimde geriliyor (94→79, 99→92) ama pozitif kalıyor; `soft` ve `supervised` negatife düşüyor, yani düz çekirdekten de kötü oluyor. Her iki eksende de (onarım ve temiz maliyet) her iki lr'de birden sağlam olan tek hücre **`hard_st` × yığın**. Bu, P1 iddiasını "en iyi ayarda en iyi"den **"ayar bozulduğunda da çalışan"a** yükseltiyor — ayarın göreve göre elle seçilemediği durumlarda önemli olan özellik budur.

### U2d: dayanıklılığı ne belirliyor (tanı, `results/U2d/compare.json`)

12 kapılı model (soft ve hard_st × 3 tohum × 2 lr) için durum kodunun geometrisi ölçüldü. **Tek bir nicelik dayanıklıyı kırılgandan örtüşmesiz ayırıyor:** sınıf çapalarının minimum aralığı (dayanıklı 13,6–33,8; kırılgan 0,3–5,2) ve bunun nötr koşu boyunca gözlenen sürüklenmeye oranı (dayanıklı ≥ 2,6; kırılgan ≤ 2,5). **Mutlak sürüklenme iki grupta da aynı** (ortalama 4,47 ve 4,47), kapının tutma oranı da ayırmıyor.

Yani sorun "durum kayıyor" değil, **"durum, sınıfları ayırt edemeyecek kadar sıkışık bir kodda kayıyor"**. Bu üç gözlemi birden açıklıyor: (a) yüksek lr, bütçe içinde geniş ve ayrık bir kod geliştiriyor (durum normu 2,0–4,9), düşük lr kodu küçük ve sıkışık bırakıyor (0,54–0,83) — yani etkin olarak yetersiz eğitim (bkz. C17); (b) tam kopya her yerde kazanıyor çünkü sürüklenme sıfırken oran kodun geometrisinden bağımsız olarak sonsuz; (c) hiçbir "tutma oranı" hikâyesinin tutmamasının sebebi. **Sınır:** n=12, tek görev ailesi, korelasyonel — kod geometrisine müdahale edilmedi. Nedensel testi `U4-sep` olarak havuza yazıldı.

**Geri çekilen açıklama:** "daha çok tutmak = daha kısa etkin özyineleme = daha iyi genelleme" demiştim; veri çürüttü. `soft` lr 0,003'te tutma 0,389 → 512-orta 1,000; lr 0,0003'te tutma 0,394 → 0,182. Neredeyse aynı tutma, zıt sonuç. Belirleyici olan **eğitim rejimi**, kapının tutma sıklığı değil. İçerideki hangi özelliğin bunu sağladığı **henüz ölçülmedi** → U2d tanısı bunu ölçtü (yukarıdaki bölüm): belirleyici olan sınıf çapalarının aralığı.

## Yöntem

- Mimari: `horizon/gated.py` — mühürlü Core'un üzerine kapı. Her adımda korunan durum ya adayla değişir (WRITE) ya da **aynen kalır** (HOLD). Kapı yalnız `[e_t, s_{t−1}]` görür; test anında oracle yok. Kollar: `soft` (sigmoid karışım, türevlenebilir, kesin değil), `hard_st` (sert karar + straight-through; HOLD'da pencere önbelleği de donar), `supervised` (hard_st + koşu-farkında k=2 hedefine BCE kaybı).
- Eğitim: U1 döngüsü (2048 adım, batch 32, L128, AdamW, kırpma 1,0) + A8 sonluluk koruması; seçim doğrulama kaybına göre. Değerlendirme U2b uç noktalarında: 500 dizi × nötr 128/512 × ön/orta/son, temiz 128/1024/4096, artı kapı tanıları (tutma oranı, yanlış-tutma/yanlış-yazma).
- Protokoller: `results/U2c-v2` (lr 0,003, 18/18 tamam), `results/U2c-lr0003` (lr 0,0003, aynı protokol değerleri, 18/18 tamam), `results/U2c-lr` (geliştirme probu, 8/8). Referans `none` = U1'in seçilmiş checkpoint'leri, aynı uç noktalarda (U2b'den).
- Durum etiketleri: analiz betiğinin kuralı tohum başına sıkı olduğu için altı hücrenin altısını da "MIXED" yazıyor; bu yüzden raporda etiket değil **sayılar** kullanılıyor (aynı karar U2/B1 raporlarında da alınmıştı).

## Olay kaydı: çökme, düzeltme, tekrar üretim

`hard_st` kolu 513. adımda düştü. Sebep bilim değil, **kayıt kodu**: straight-through gradyanı 9,15e16'ya çıkınca karelerin toplamı float32'de taşıp `inf` oldu ve `allow_nan=False` yazıcı hata verdi. Eğitim matematiği etkilenmemişti (inf norm zaten kırpmayla sıfır adıma dönüşür).

Düzeltme: gradyan normu float64'te hesaplanıyor, sonsuz norm sayılan bir atlama olarak işaretleniyor, hiçbir satır inf/NaN taşıyamıyor (`tests/test_guards.py` regresyon testleri). Düzeltme dondurulmuş kaynak kümesini değiştirdiği için döngü **baştan** koşuldu (`results/U2c-v2`); yarım kalan koşul `results/U2c/interrupted-20260922T2333/` altında korunuyor.

Yan ürün — **tekrar üretim kanıtı**: v1'in altı `soft` koşulu v2'de bit-bit yeniden üretildi (seçilen checkpoint SHA256'ları, seçilen adımlar, kırpma sayıları, tüm doğruluklar özdeş; tek fark kırpma öncesi normun 10. hanesi). CPU eğitimi günler, süreçler ve yama arası **tam tekrar üretilebilir** — B2'deki fp16 GPU bulgusunun (aynı korpusun yeniden kodlanması %10,5 listede fark veriyordu) tam tersi.

## Sınırlar

- 32 genişlikte oyuncak modeller, iki sentetik görev ailesi, üç tohum, tek eğitim bütçesi (2048 adım). Dil yok, ölçek yok.
- lr karşılaştırması **sabit adım bütçesiyle** yapıldı (havuz C17): "düşük lr daha kötü" ifadesi "aynı adım sayısında daha kötü" demektir.
- `hard_st`'in beş-durumdaki tohumlar arası dalgalanması (84–100%) kararsız rejimde ölçüldü; kararlı rejimde ortalama %79'a geriliyor ve o seviyede temiz 4096 genellemesi bozuluyor (−0,471). Yani "kararlı lr" bu kol için de bedava değil.
- Kapının neden işe yaradığına dair içsel mekanizma ölçülmedi; U2d tanısı (`horizon/gated_trajectory.py`) tam da bunu hedefliyor.

## Durum

`hard_st`: **her iki görev ailesinde ve her iki lr seviyesinde MEASURED_POSITIVE** (onarım %94/%79 ve %99/%92); tek zayıf noktası düşük lr'de beş-durumun temiz uzunluk genellemesi. `soft`: beş-durumda pozitif, yığında negatif. `supervised`: yığında pozitif, beş-durumda negatif. Denetim: `.superpowers/sdd/phase2/audit-u2c.md`; mühür: `audit/phase2-U2c-manifest.json`, sha256 96694038… (iki lr seviyesi, prob ve U2d tanısı dahil).
