# Z1-G — HOLD/WRITE kapısı için GRU kontrolü (2026-10-02)

**Hüküm (ön-kayıtlı kuralın koddan okunuşu): R2 — düz GRU kayıyor.** `python -B -m horizon.gru_control --verdict` → `R2 (five rows: 6 of 6)`. Ön-kayıt: `results/Z1-G/README.txt`, commit 00c6846 (00:50), koşulardan önce; kural kodu `horizon/gru_control.py::verdict`, 20 test.

## Soru

Kampanyanın ilk pozitif mimari sonucu, öğrenilmiş HOLD/WRITE kapısı (U2c), Core'un durumunu nötr (kimlik) belirteç koşularında tutuyor. Bir GRU'nun zaten güncelleme kapısı var. Aynı veride aynı bütçeyle eğitilen düz bir GRU de nötr koşuda durumu tutuyorsa kapı standart bir özelliğin yeniden icadı olurdu ve Core'un sürüklenmesi "öğrenilmiş yineleme" değil Core'un zayıf toplamalı geri beslemesinin kusuru olurdu.

## Protokol

U1 ile birebir (sayılar `horizon/sweep.py::protocol()`'dan okunur, kopyalanmaz; aynı kaynak tohumları, yani eğitim batch'leri ve test problemleri Core kollarınınkiyle aynı, tohuma göre eşli): 2.048 güncelleme, batch 32, eğitim uzunluğu 128, AdamW lr 0,003, wd 0,01, kırpma 1; doğrulama her 128 güncellemede 256 problem, seçim en düşük doğrulama son-CE (eşitlikte erken adım); tohumlar 117/143/191; görevler `five` ve `stack`; temiz uzunluklar 128…4096 (1.000 problem); nötr eklemeler 128 ve 512 yuva × ön/orta/son. Kollar: GRU genişlik 32 (6.597 parametre; Core kollarının genişliği) ve 64 (25.477; Core'un ≈ 24k'sına eşlenmiş). CPU, tek iş parçacığı. Ağırlıklar kayıtlı (`results/Z1-G/*.pt`, Git dışında).

## Sonuç 1 — `five`: nötr eklemede kayıp (puan; tohum başına 117/143/191; ham dosyalardan hesaplanır: `--compare --task five`)

| kol | 128 ön | 128 orta | 128 son | 512 ön | 512 orta | 512 son |
|---|---|---|---|---|---|---|
| Core, kapısız (U1) | 0,2 | **44,1** | **46,4** | 0,2 | **59,4** | **53,0** |
| Core + yumuşak kapı (U2c-v2) | 0,0 | 0,0 | 0,0 | 0,0 | 1,5 | 0,0 |
| Core + `hard_st` kapı | 3,5 | 3,9 | 0,0 | 22,5 (0/68/0) | 25,9 (8/54/15) | 3,6 |
| Core + `supervised` kapı | 1,3 | 2,1 | 6,1 | 1,9 | 24,3 (61/3/9) | 13,0 |
| **GRU w32** | 42,9 (72/56/0) | **52,3** (17/60/80) | **55,2** (20/66/80) | 54,2 (99/64/0) | **69,8** (67/62/80) | **63,3** (40/70/80) |
| **GRU w64** | 4,5 (0/0/14) | **57,1** (72/59/41) | **59,1** (67/51/59) | 33,3 (0/0/100) | **73,5** (81/80/59) | **73,0** (80/80/59) |

Ön-kayıtlı dört hücrede (128/512 × orta/son) altı GRU satırının altısı da ≥ 16,9 puan kaybetti (hepsi "kayıyor" ölçütünü sağladı: ≥ 2 hücrede ≥ 10, ≥ 2/3 tohum, w64'te 3/3). R1 (≤ 2 puan) hiçbir satırda sağlanmadı.

## Sonuç 2 — `stack` (betimleyici; hükme girmez): nötr eklemede kayıp (puan)

| kol | 128 ön | 128 orta | 128 son | 512 ön | 512 orta | 512 son |
|---|---|---|---|---|---|---|
| Core, kapısız (U1) | 4,5 | 4,4 | **56,5** | 10,9 | 6,6 | **58,6** |
| Core + yumuşak kapı | 1,9 | 3,7 | 48,2 | 6,1 | 6,4 | 61,1 |
| Core + `hard_st` kapı | 3,6 | 3,6 | **0,0** | 9,7 | 5,9 | **0,0** |
| Core + `supervised` kapı | 2,9 | 5,1 | −0,4 | 11,0 | 8,3 | −0,3 |
| GRU w32 | 0,0 | 1,4 | 15,2 (19/14/12) | 0,0 | 4,1 | 32,2 (52/26/19) |
| GRU w64 | 0,0 | 0,5 | 12,3 (10/21/6) | 0,0 | 1,6 | 30,7 (24/44/24) |

## Sonuç 3 — temiz uzunluk (son-konum doğruluğu; tohum başına 117/143/191)

`five`: GRU w32 ve w64 altı satırın altısında 128…4096'nın hepsinde **1,000**; Core (kapısız, U1): 4096'da 1,000 / 1,000 / 0,947 (üçüncü tohum 2048'de 0,987).
`stack`: eğitim uzunluğunda doğruluk iki yerden okunur. Nötr-bölünme tabanı (1.000 problem): GRU w32 0,951 / 0,966 / 0,970, w64 0,950 / 0,965 / 0,977; Core 0,946 / 0,940 / 0,938. Temiz 128 testi: GRU w32 0,949 / 0,961 / 0,957, w64 0,954 / 0,969 / 0,972; Core 0,936 / 0,936 / 0,954. Yani GRU 0,95–0,98 bandında, Core 0,94–0,95 bandında (U1'in "eğitim ufkunda bile ≥ 0,95 iki tohumda sağlanmıyor" bulgusu temiz-128 testindeki sayılardır). 4096'da GRU w32 0,907 / 0,916 / 0,885, w64 0,915 / 0,912 / 0,905; Core 0,801 / 0,813 / 0,715.

## Okuma

1. **Kapı standart bir GRU kapısının yeniden icadı değil (R2).** Aynı veri ve bütçeyle eğitilen düz GRU, kapısız Core kadar (orta/son eklerde 52–74 puan, Core 44–59) kayıyor; yumuşak kapılı Core ≈ 0. GRU'nun doğal güncelleme kapısı, eğitimde hiç görülmeyen uzun nötr koşuda durumu tutmayı kendiliğinden öğrenmiyor.
2. **Kayma biçimi.** `five`'da GRU ön eklemeye de duyarlı (w32'de 43/54 puan; Core ≈ 0). w64 tohum 191'in 512 ön ekinde doğruluk **tam 0,000** ve karışıklık matrisi köşegen sıfır, köşegen dışı ≈ düzgün (40–69/hücre): ön ek aynı 512 belirteçten oluştuğu için GRU'nun durumu bütün problemlerde aynı noktaya gidiyor; bu nokta otomatın geçerli ama yanlış bir başlangıç durumuysa her problemde doğru cevaptan farklı bir sınıf üretilir. Bu, temsil çöküşünden çok **başlangıç durumunun kayması**yla tutarlı. Bu bir yorum; test edilmedi (Core'un kayması ise U1 denetiminde tek bir çekici sınıfa çöküş olarak ölçülmüştü).
3. **GRU Core'dan zayıf değil, çoğu yerde güçlü.** Temiz uzunlukta `five`'da kusursuz (Core'un bir tohumu 4096'da düşüyor); `stack`'te 4× az parametreyle (w32) eğitim uzunluğunda Core'dan yüksek (0,95–0,98'e karşı 0,94–0,95) ve 4096'da 0,885–0,916 tutuyor (Core 0,715–0,813). `stack` son-ek kaybı GRU'da 12–32 puan, kapısız Core'da 56–59: GRU burada sağlam bir taban. Ancak `hard_st` ve `supervised` kapılı Core son eklerde 0,0 / −0,4, yani kapı GRU'yu da geçiyor.
4. **Kampanya başlığına etkisi:** Core'un çekirdeğinin kendisi, düz bir GRU'ya temiz uzunlukta üstünlük göstermiyor (`stack`'te geride). Üstünlük, kapı eklentisinde ve nötr-koşu dayanıklılığında. "RLT yinelemesi GRU'dan iyidir" cümlesi bu verilerle desteklenmiyor; "HOLD/WRITE kapısı, aynı eğitimle düz GRU'nun sahip olmadığı bir dayanıklılık veriyor" cümlesi destekleniyor.

## Sınırlar

Tek protokol (lr 0,003, 2.048 güncelleme), ayar yok (GRU için de, Core kolları için de); üç tohum; GRU dikkatsiz bir taban; LSTM kolu yok (ön-kayıt gereği yalnız R1 olsaydı eklenirdi). Kapı kolları U2c-v2'den (lr 0,003): `hard_st` tohum 143'te 512 ön/orta eklerde 68/54 puan kaybediyor, yani kapılı kollar da tek tip değil (U2c raporunun lr ve kol ayrımları geçerli). **Ölçülmeyen asıl soru:** GRU'ya eğitimde uzun nötr koşular gösterilseydi tutmayı öğrenir miydi (GRU + nötr-koşu müfredatı)? Bu sonuç "kapı gerekli" değil, "bu veriyle ve bu bütçeyle GRU kendiliğinden öğrenmiyor" der. Nötr belirteçler tam kimlik; `stack` hükme girmedi.

## Yeniden üretim

`PYTHONPATH=. python -B -m horizon.gru_control --out results/Z1-G --lane 0 --lanes 2` (ve `--lane 1`), `--verdict`, `--compare --task five|stack`. Ham sonuçlar: `results/Z1-G/z1g-gru{32,64}-{five,stack}-s{117,143,191}.json` (+ `.pt`), tablolar `comparison-five.json`, `comparison-stack.json`.
