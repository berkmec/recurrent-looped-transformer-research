# U2 — sürüklenmeyi teşhis etme ve düzeltme (2026-09-18)

## Bölüm 1 — Ölçüm katmanı (eğitimsiz; U1'in seçilmiş modelleri)

**Yöntem** (`horizon/trajectory.py`, testler `tests/test_trajectory.py` RED 3 → GREEN 3): decode döngüsüyle her konumdaki gizli durum z_t alınır (Core.forward ile 1e-5 içinde özdeş, test). Beş-durum: eğitim dağılımından 256 dizi × 128 konumda oracle sınıfına göre **çapa merkezleri** (5 × 32); her konumda en yakın merkez ↔ oracle karşılaştırılır. 500 taze taban dizisine (nötr ad alanı) 128/512 nötr ön/orta/son eklenir; koşu içi **yer değiştirme eğrisi**, koşu sonu **çekici histogramı**, koşudan sonraki 64 gerçek işlemde **onarım eğrisi**. Yığın: sonlu çapa yok; koşu öncesi duruma ve hizalı taban yörüngesine RMS-normalize uzaklık.

**Bulgular (beş durum, 3 model):**

| | s117 | s143 | s191 |
|---|---|---|---|
| Taban dizilerde çapa tutarlılığı | 1,000 | 1,000 | 1,000 |
| Merkezler arası min uzaklık / sınıf içi yayılım | 27,9 / 3,7–8,1 | 17,9 / 2,5–4,9 | 31,1 / 4,6–10,4 |
| Yer değiştirme, 128 nötr son: k=1,2 → 4 → 16 → 128 | 0,00 → 0,23 → 0,44 → 0,44 | 0,004 / 0,008 → 0,22 → 0,61 → 0,62 | 0,00 → 0,00 → 0,01 → 0,39 (k=32'de 0,29) |
| Koşu sonunda çekici (500 durum) | sınıf 2: 310; 0: 99; 4: 91; **1 ve 3: 0** | sınıf 0: 402; 2: 98; **1, 3, 4: ≈0** | 4: 215; 0: 184; 2: 100; **1: 0** |
| Onarım (orta ek, 64 gerçek işlem) | 0,40 → 0,40 | 0,60 → 0,60 | 0,30 → 0,30 |
| Ön ek (128/512) yer değiştirme | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |

Okuma: (1) Sürüklenme **hızlı bir geçiş**: ilk iki nötr adım zararsız, 4–16 adımda durum sınıf bölgesinden çıkıp doyuyor; 512'ye kadar çok az artıyor (s117 orta: 0,42 → 0,61). (2) Nötr haritanın sabit noktaları oracle sınıflarının bir **alt kümesi**: her modelde 1–2 sınıf çekici, bazıları (s117'de 1 ve 3) nötr koşu altında hiç kararlı değil. (3) Başlangıç durumu (sınıf 0, ön ek) 512 nötrde bile kararlı. (4) Beş-durumda **onarım sıfır**: yer değiştiren durumlar sonraki 64 işlemle geri gelmiyor (permütasyon otomatı; hata sabit bir kayma olarak taşınıyor). (5) Model çapaları o kadar net ki, "en yakın çapa" tabanda oracle ile %100 örtüşüyor — bu, çapaya çekme müdahalesinin yanlış çapa riskini tabanda sıfıra indiriyor.

**Bulgular (yığın, 3 model):** koşu içinde koşu-öncesi duruma uzaklık k=1'de 2,2–2,4 (RMS birimi) ile başlayıp 5,6–6,5'te doyuyor; koşudan sonra hizalı taban yörüngesine uzaklık 64 işlemde 4,1–5,0 → 1,0–1,9'a düşüyor (üzerine yazma onarımı); son ekte onarım imkânsız → doğruluk 0,29–0,43.

**Karar (ledger `Ruling`):** müdahale hedefi "geçiş başlamadan durumu sınıf bölgesinde tutmak". Bölüm 2'de test-zamanı üç kol: nötr adımda decoder durumunu **dondurma** (adımı geri al), **en yakın çapaya çekme** (beş durum), **yanlış çapa kontrolü**; yığında yalnız dondurma. Bunlar nötr token'ı oracle'dan bilir → mimari çözüm değil, **tavan ölçümü**; öğrenilmiş no-op algılama sonraki adımdır.

## Bölüm 2 — Test-zamanı müdahaleler (oracle nötr bilgisi; tavan ölçümü)

**Yöntem** (`horizon/interventions.py`, testler `tests/test_interventions.py` RED 3 → GREEN 3; protokol `results/U2-intervene/frozen.json`, hash `dbdd3395…`, U1 protokolü `56ef56a1…`): U1'in seçilmiş modelleri, eğitim yok. Kollar: `none` (Core.forward ile özdeş), `freeze` (her nötr token 2 konumunda decoder durumu s ve pencere önbelleği dk/dv bir önceki adıma geri alınır, logits tekrarlanır), `snap` (beş-durum: nötr konumda s en yakın sınıf merkezine çekilir; okuma çekilen durumdan), `wrong_snap` (kontrol: bir sonraki merkez). Uç noktalar: 500 taze dizi × nötr ek 128/512 × ön/orta/son + temiz 128/1024/4096 (500 problem). Nötr token oracle'dan bilinir → bu bir mimari çözüm değil, öğrenilmiş bir no-op algılayıcısının **tavanı**. Üç tohum, keşif aralıkları.

**Bulgular (beş durum, 3 tohum; "kazanç" = kolun `none`'a göre son-konum doğruluk farkı, ortalama [min, maks]):**

| Kol | 128 orta | 128 son | 512 orta | 512 son | Temiz 128 / 1024 / 4096 (fark) |
|---|---|---|---|---|---|
| none (taban) | 0,60 / 0,40 / 0,70 | 0,61 / 0,38 / 0,61 | 0,40 / 0,22 / 0,62 | 0,43 / 0,37 / 0,61 | 1,00 / 1,00 / 0,99 |
| freeze | **+33,3** [+27,6, +44,0] | **+41,5** [+34,8, +51,0] | **+45,5** [+35,4, +58,4] | **+47,8** [+38,8, +53,0] | **−4,5 / −25,3 / −46,7** (s117 −6,2/−26,0/−62,8; s143 −7,4/−48,8/−76,2; s191 0/−1,2/−1,0) |
| snap | +29,7 [+18,4, +40,2] | +31,5 [+15,2, +40,2] | +38,9 [+38,0, +40,6] | +37,6 [+33,4, +40,4] | −0,4 / −2,5 / −11,0 (s191: 0/0/+2,0) |
| wrong_snap | −34,9 | −33,1 | −21,3 | −26,7 | **−80,7 / −80,7 / −78,9** (şans ≈ 0,20) |

Taban değerler tohum sırasıyla s117 / s143 / s191. Ön ek koşularında (durum = başlangıç durumu) hiçbir kol bir şey değiştirmiyor; freeze ön ekte −5 puan (doğal token 2'lerde de donduğu için).

**Bulgular (yığın, 3 tohum, yalnız freeze):** son ek **+55,1** [+49,8, +62,2] (128) ve **+58,3** [+52,2, +64,4] (512) — tam onarım (0,29–0,43 → 0,93–0,94 = taban); orta/ön ek değişmiyor (±0,3); temiz 128/1024/4096 farkı −0,9 / +0,3 / +0,8 — **maliyetsiz**.

**Okuma:**
1. **Sürüklenme, nötr koşu kaybının tamamıdır.** Son ekte durumu dondurmak beş-durumda kaybın tamamını (kalan fark 0,0 puan), yığında da tamamını geri alıyor. Orta ekte beş-durumda +33/+46 puan geri geliyor ama tabana 5–8 puan uzak kalıyor: koşudan sonra gelen gerçek işlemler dondurulmuş önbellek/konum farkıyla yürüyor.
2. **Düz dondurma dağıtılabilir bir müdahale değil (beş-durum).** Token 2 doğal bir işlem (konumların %12,5'i); her doğal token 2'de de dondurunca temiz uzun dizilerde çöküyor: s117/s143'te 4096'da −63/−76 puan. Tohuma bağlı: yavaş sürüklenen s191 modeli hem freeze'i hem snap'i temizde neredeyse maliyetsiz tolere ediyor. Yığın modelleri ise doğal kimlik işlemlerinde dondurmayı hiç önemsemiyor.
3. **Çapaya çekme kısmi.** snap orta/son ekte +30/+39 puan geri alıyor ama 13,6–19,5 puan taban altında kalıyor; temizde 4096'da ortalama −11 puan (s117 −15, s143 −20). s191'de ise her yerde 1,000 ve temiz 4096'da +2 puan. `wrong_snap` kontrolü şansa (0,20) düşürüyor → çapalar sınıf bilgisini gerçekten taşıyor; "en yakın çapa" tabanda oracle ile %100 örtüştüğü hâlde snap'in kısmi kalması, sürüklenmiş durumun en yakın çapasının artık doğru sınıf olmadığını gösteriyor (çekici sınıflara geçiş, Bölüm 1).
4. **Durum etiketleri:** analiz betiğinin kural tabanlı etiketi tüm kolları "MIXED" yazıyor; `wrong_snap` için bu, tabanı zaten şansta olduğu için kuralın bir yapaylığıdır — kontrol kol tasarım gereği ve temiz kapısıyla **MEASURED_NEGATIVE**'dir. Bilimsel karar: `freeze` = **tanı olarak pozitif** (sürüklenme = kayıp), **müdahale olarak negatif** (temiz kapısı); `snap` = **negatif** (kısmi + temiz maliyeti); yığın `freeze` = **pozitif tavan** (son ek tam onarım, maliyetsiz).

**Bölüm 3'ün gerekçesi:** Bölüm 1'de k ≤ 2 nötrde yer değiştirme sıfırdı. O hâlde dondurma her token 2'de değil, ardışık (k+1)'inci nötrden itibaren yapılırsa doğal tekil kimlik işlemlerine dokunmadan uzun koşular durdurulmalı. Bu U2b'dir.

## Bölüm 3 — Koşu-farkında dondurma (U2b): kayıpsız tavan

**Yöntem** (`horizon/interventions_runaware.py`, testler `tests/test_interventions_runaware.py` RED 3 → GREEN 3; protokol `results/U2b-intervene/frozen.json`, hash `20b73983…`): Bölüm 2'deki dondurma yalnız **ardışık (k+1)'inci nötrden itibaren** uygulanır (`freeze_k1`, `freeze_k2`); k=0 düz dondurmaya bit-bit eşit (test). Aynı modeller, aynı uç noktalar, aynı 500 dizi; `none` satırları Bölüm 2 ile özdeş. Oracle nötr bilgisi → tavan.

**Bulgular (beş durum, 3 tohum; kazanç `none`'a göre, ortalama [min, maks]):**

| Kol | 128 orta | 128 son | 512 orta | 512 son | Kendi tabanına kayıp (en kötü hücre) | Temiz 128 / 1024 / 4096 |
|---|---|---|---|---|---|---|
| freeze_k1 | +42,1 [+30,2, +57,6] | +46,3 [+38,8, +61,0] | +55,7 [+37,4, +75,4] | +52,5 [+39,0, +61,6] | 5,4 puan (s117, 512 orta) | −0,1 / −0,1 / **−2,4** (s143 −7,6) |
| **freeze_k2** | **+43,5** [+30,4, +60,0] | **+46,6** [+38,8, +62,0] | **+58,6** [+38,0, +78,0] | **+52,9** [+39,0, +62,6] | **0,6 puan** (s191, 128 ön) | **0,0 / 0,0 / +0,1** |

`freeze_k2`: üç tohumda tüm nötr hücreler kendi tabanının ≤ 0,6 puan içinde (s117 ve s143'te bütün hücreler tam 1,000, sıfır çevirme), temiz üç uzunlukta fark 0,0 / 0,0 / +0,07 puan. Analiz etiketi: **MEASURED_POSITIVE_CEILING** (tüm kapılar, 3/3 tohum). `freeze_k1` hafif kayıplı: doğal iki-nötr koşular (%1,6/konum) da dondurulunca s143'te 4096'da −7,6 puan.

**Bulgular (yığın, 3 tohum):** son ek `freeze_k2` **+55,7** [+51,8, +61,4] (128) / **+58,9** [+52,8, +63,6] (512) — taban değerine tam onarım (0,29–0,43 → 0,93–0,95); temiz 128/1024/4096 farkı 0,0 / 0,0 / +0,1. Orta/ön ek hücrelerinde kazanç ≈ 0 (±0,4): yığın modelinin bu hücrelerdeki 3,6–10 puanlık kaybı **sürüklenmeden kaynaklanmıyor** (Bölüm 1: yığında durum sonraki işlemlerle üzerine yazılarak onarılıyor; kalan kayıp koşudan sonraki gerçek işlem zincirinde) ve dondurma bunu değiştirmiyor. Analiz kuralı bu hücreleri "kendi tabanına kayıp" sayıp kolu "MIXED" etiketliyor; bilimsel karar: yığında `freeze_k2` = **maliyetsiz, son ek için tam onarım, orta/ön için etkisiz**.

**Okuma:** Bölüm 1'in "ilk iki nötr adım zararsız, 4–16'da geçiş" gözlemi doğrudan bir müdahaleye çevrildi: durumu ikinci ardışık nötrden sonra sabitlemek, beş-durumda nötr koşu kaybını üç tohumda tamamen ve temiz dizilere hiçbir bedel ödemeden ortadan kaldırıyor. Bu, RLT çekirdeğinde "korunan kayıt / değişebilir hesap" ayrımının (dış paket, P1) ölçülmüş bir **tavanıdır**: hangi adımın nötr olduğu oracle'dan verilmiştir. Sonraki adım (U2c) aynı kararı modelin kendisinin [e_t, s_{t−1}]'den vermesini öğrenip öğrenemeyeceğidir; kod ve testler hazır, eğitim koşusu izne bağlı.

**Kayıtlar:** ham koşul dosyaları `results/U2-intervene/*.json`, `results/U2b-intervene/*.json`; analizler `analysis.json`; kapsamlı manifest `audit/phase2-U2-manifest.json` (mühürleme sonrası); bağımsız denetim `.superpowers/sdd/phase2/audit-u2.md`.

**Denetim notu (2026-09-18, `audit-u2.md`):** `results/U2-measure` çalıştırılırken ayrı bir dondurma kaydı alınmadı; koşul dosyaları U1 protokol hash'ini (56ef56a1…) ve seçilmiş checkpoint hash'lerini taşıyor, `trajectory.py`'nin koşu anındaki hash'i kayıtsız — dosya, 5 dakika sonra U2-intervene tarafından dondurulan hash ile aynı. Sonradan alınan kayıt: `results/U2-measure/frozen-posthoc.json`. Bu bölümün sayıları bu sınırla okunmalı.
