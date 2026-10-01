# U1 — t ekseni: sabit-durum ve büyüyen-bellek görevlerinde uzun ufuk (2026-09-18)

**Durum:** beş durum (sabit bellek) **MIXED**; yığın (büyüyen bellek) **MEASURED_NEGATIVE**. Kapsam: yalnız t (token/işlem) ekseni; k (başlık turu) ve d (bağımlılık derinliği) eksenleri başlık/graf mekanizması gerektirdiğinden U1b'ye ertelendi (ledger `Ruling`). Kök `__RLT_ROOT__/2026-09-18-phase2`, dal `codex/rlt-phase2`. Mühürlü kökler değişmedi.

## Soru

Eğitim ufku 128 iken, aynı küçük RLT çekirdeği (w32, 2 baş, ~24k parametre) (a) yeterli durumu sabit olan bir görevde ve (b) gereken belleğin problemle büyüdüğü bir görevde 128 → 4.096 uzunluğa ne kadar genelliyor; nötr (kimlik) koşularının konumu ve uzunluğu son cevabı ne kadar bozuyor; etkin ufuk (bütün önceki noktalarda ≥ %95) nerede?

## İki görev ailesi

- **five:** mevcut beş-durum otomatı (3 token, 5 sınıf); doğru durum sabit büyüklükte.
- **stack:** push a / push b / nötr / pop (4 token, 3 sınıf: üstteki sembol a/b/boş); pop-on-empty etkisizdir. Sembol yığını problemle büyür (4.096'da ortalama en büyük derinlik 73, 128'de ≈13); sonlu otomat çözemez. Core'a dokunulmadı; yalnız gömme tablosu tohumlu `nn.Embedding(4, w)` ile değiştirildi (girdi-sözlüğü genişletmesi, mimari değişiklik değil; makbuzlarda kayıtlı).

Protokol (`results/U1/frozen.json`, hash `56ef56a1…`): 2048 güncelleme, batch 32, uzunluk 128, AdamW lr 0,003, doğrulama her 128 adımda 256 örnek (son-durum CE minimumu seçilir), tohumlar 117/143/191, testler 128/256/512/1024/2048/4096 × **1.000 taze problem**, nötr ekleme 0/128/512 × ön/orta/son (aynı 128'lik tabana), eşik 0,95, yeni kaynak tohum uzayları. Ham diziler uint8 `.npz`.

## Sonuçlar — temiz uzunluk genellemesi (son-durum doğruluğu, 1.000 problem/hücre)

| Görev / tohum | 128 | 256 | 512 | 1024 | 2048 | 4096 | Etkin ufuk |
|---|---:|---:|---:|---:|---:|---:|---:|
| five s117 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 4096 |
| five s143 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 4096 |
| five s191 | 1,000 | 1,000 | 0,998 | 1,000 | 0,987 | 0,947 | 2048 |
| stack s117 | 0,936 | 0,900 | 0,864 | 0,834 | 0,836 | 0,801 | 0 |
| stack s143 | 0,936 | 0,888 | 0,854 | 0,826 | 0,798 | 0,813 | 0 |
| stack s191 | 0,954 | 0,868 | 0,835 | 0,805 | 0,759 | 0,715 | 128 |

Beş durumda üç tohumdan ikisi bütün noktalarda %100; üçüncüsü 2.048'de %98,7 ve 4.096'da %94,7 (eşiğin 0,3 puan altı). Yığında eğitim ufkunda bile ≥ %95 iki tohumda sağlanmıyor ve doğruluk uzunlukla tekdüze düşüyor (prefix doğruluğu da: 0,97 → 0,83); 4.096'da tahmin histogramı hâlâ üç sınıfa dağılmış (sabit-sınıf çöküşü yok). Şans düzeyi: five 0,20; stack ≈ 0,4 (a/b) — bir tohum 0,715'e yaklaşıyor.

## Sonuçlar — nötr koşular (kayıp = taban − ekli, puan; 3 tohum)

| Görev | 128 ön | 128 orta | 128 son | 512 ön | 512 orta | 512 son |
|---|---:|---:|---:|---:|---:|---:|
| five | +0,2 (0,0/0,0/0,5) | **+44,1** (40,8/60,9/30,6) | **+46,4** (39,2/60,8/39,2) | +0,2 | **+59,4** (59,5/79,9/38,7) | **+53,0** (58,8/61,1/39,2) |
| stack | +4,5 (5,6/5,0/2,8) | +4,4 (3,6/4,8/4,9) | **+56,5** (64,9/56,4/48,2) | +10,9 | +6,6 | **+58,6** (65,8/53,7/56,2) |

Doğru→yanlış dönüşler (1.000 problemde): five 128-son 392/608/392, 512-orta 595/799/387; stack 128-son 663/578/514 (yanlış→doğru yalnız 14/14/32), 512-orta 60/80/105. Hedef "nötr uzatmada ≤ 2 puan kayıp" hiçbir tohum/görevde sağlanmadı (yalnız beş-durumun ön ekinde sağlanır).

## Yorum (ölçüm değil, hipotez)

1. Sabit bellekli görevde uzunluk genellemesi büyük ölçüde başarılı; başarısızlığın kaynağı uzunluk değil, durum kurulduktan sonra gelen **boş koşu**. Bu, önceki üç kampanyanın (Faz 1, R2, sürüklenme mini deneyi) bulgusunu 4× daha uzun eğitim ufkunda ve 1.000 problemlik hücrelerde doğruluyor.
2. Nötr koşunun zararı **konum × görev yapısına** bağlı: beş-durumda orta ve son eşit derecede yıkıcı, yığında yalnız son. Yığında orta ekten sonra gelen 64 gerçek işlem bozulan üst sembolü **üzerine yazıyor** (onarım), beş-durum bir permütasyon otomatı olduğundan sürüklenmiş durum sonraki işlemlerle **onarılmıyor**. Yani sürüklenme = durumun yanlış bir noktaya kayması; sonraki işlemler bu noktayı ezebiliyorsa görünmez kalıyor.
3. Büyüyen bellek görevi bu çekirdeğin bu bütçedeki kapasitesini aşıyor (doğrulama kaybı hâlâ düşüyordu; bkz. denetim): 24k parametrelik 32-boyutlu durum, derinliği 13'ü aşan yığını eğitim uzunluğunda bile tam taşıyamıyor; uzunlukla derinlik büyüdükçe kayıp artıyor. Bu sonuç sabit-durum başarısının büyüyen-bellek görevlerine **atfedilemeyeceğini** planın istediği biçimde gösteriyor.

## Hesap bütçesi (gerçek)

Koşul başına eğitim 26–29 dk (CPU tek iş parçacığı, 3 paralel yuva; indirmeler ve B1 geliştirme eşzamanlı olduğundan host yükü yüksekti), değerlendirme 17,4 dk (4.096 uzunluğu 550 s); 6 koşul toplam ≈ 1,6 saat duvar süresi. GPU kullanılmadı. Ham çıktı 45 MB.

## Sınırlar

t ekseni; tek genişlik (32); üç tohum; 16.384/65.536 noktaları koşulmadı (kalibrasyonda 4.096'nın maliyeti 550 s/1.000 problem — 16.384 ≈ 40 dk/koşul, yapılabilir ama bu döngüde açılmadı); nötr ekler yalnız 128'lik tabana; "eşdeğer yeniden yazım" ve "kimlik dizisi" varyantları koşulmadı; sonuçlar doğal dile veya büyük modele taşınmaz. Keşif aralıkları çokluk düzeltmesiz.

## Bağımsız ham denetim

`.superpowers/sdd/phase2/u1-independent-audit.md` (+ `.json`, betik `u1_raw_audit.py`; taze denetçi ajan, CPU tek iş parçacığı, salt-okunur, 24 s): **PASS 328/0.** Dondurma/protokol/kaynak pin'leri, 6 makbuz ve mühür, 78/78 `.npz` sha256 ve `metadata_hash`; 2048 gerçek güncelleme (kayıp örn. s117 1,668 → 0,00014), bütün eğitim tohumları ve girdi hash'leri yeniden üretildi, seçilen adım = en erken minimum doğrulama kaybı (2048/2048/2048/2048/1792/1792), `initial.pt` `torch.manual_seed` ile, `selected.pt` 32×128 örnekte saklanan tahminlerle yeniden üretildi; denetçinin **kendi** five/stack oracle'ları 78 hedef dizisinin hepsiyle özdeş, test/nötr girdileri dondurulmuş tohumlardan yeniden üretildi, nötr dosyalar = taban + 0/64/128 konumuna 128/512 token-2 ve özdeş son hedef, eğitim/test satır örtüşmesi sıfır; 78 satırın tüm metrikleri ve `analysis.json` 1e-12 içinde.

Denetçinin gözlemleri (hata değil, rapora işlendi): (1) nötr-son dosyalarında tahminlerin %79–85'i sınıf 0'da toplanıyor — sürüklenme tek bir çekici duruma çöküş; (2) beş-durumda doğrulama kaybı 256/384/1152. adıma kadar şans düzeyinde kalıp aniden düşüyor; geç öğrenen s191, 4096'da zayıflayan tohum; (3) **yığın koşullarında doğrulama kaybı dondurulmuş 2048-güncellemelik bütçede hâlâ düşüyordu** — yığın sonucu "bu bütçede öğrenilemedi"dir, kapasite sınırı iddiası değil; daha uzun eğitim yeni protokol gerektirir; (4) beş-durumda kırpma öncesi gradyan sıçramaları 17.766'ya kadar (s117, adım 272), kırpıldı.

## U2/U3'e girdi

- U2'de müdahale hedefi net: **kurulmuş durumun boş koşularda sabit kalması**; ölçüt olarak yalnız son cevap değil, durum yörüngesinin taban yörüngeden uzaklığı ve "onarılabilirlik" (sonraki işlemlerin hatayı ezme oranı) eklenmeli. Sürüklenme mini deneyi (pencere/değişmezlik kaybı/reread) üç ucuz yolun işe yaramadığını göstermişti; U2 adayları: doğrulanmış sınırdan yeniden okuma, kaynak-kontrollü önceki doğru duruma bağlanma, sınırlı sönüm.
- U3'ün ilk alanı olarak beş-durum otomatı uygun: iki tohum bütün noktalarda %100 → öğrenilmiş geçişin durum bölgeleri için invariant adayı var; ancak nötr geçişin (token 2) daraltıcı olmadığı ölçülmüş bir gerçek — teorem hedefi buna göre kurulmalı (kimlik geçişi için ayrı koşul).

## Kanıt yolları

`results/U1/` (frozen.json, 6 koşul, analysis.json, workers/, processes/), testler `tests/test_horizon_axes.py` (RED 6/6 → GREEN 6/6: `.superpowers/sdd/phase2/u1-red.log`, `u1-green.log`), zaman pilotu `u1-pilot-timing.txt`, bağımsız denetim `u1-independent-audit.md/json` (PASS 328/0), mühür `audit/phase2-U1-manifest.json`.
