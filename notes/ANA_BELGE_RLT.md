# RLT araştırma kampanyası — ana belge

> **Son güncelleme:** 2026-10-01 22:57 (yerel saat). Son işlenen sonuçlar: D5 (bitti), T1p, T1w64 (bitti), D6 adım 250, 500 ve 750 ara kontrolleri, T1w256 aşama 1. Çalışan işler: D6 (GPU, adım ≈ 950/1000), T1w256 (CPU, aşama 2; aşama 1 bitti: kopyalama devresi 250–500. adım).
>
> **Kural:** Bu belgenin güncel tutulması proje kuralıdır. Her yeni sonuç aynı oturumda buraya işlenir, atlanmaz. Bir bulgu düzeltilirse eski metin silinmez; "DÜZELTME" notuyla yanında kalır. Sıra: önce ham sonuç dosyası, sonra döngü raporu (`notes/`), sonra bu belge. Burası bir özet değil bir haritadır: her satır bir rapora ve bir ham dosyaya bağlanır. Bu belge ve her şey Türkçedir; sayılar, etiketler (`MEASURED_NEGATIVE`, `MIXED`…) ve kod dilden bağımsızdır.

## 1. Bir bakışta

**Soru.** Küçük yinelemeli bir çekirdek (RLT: tek kodlayıcı katmanı, gizli durum geri beslemeli kod çözücü) ve onun etrafında kurulan "başlık belleği" uzun bir hesaplama boyunca doğru bilgiyi koruyup kullanabilir mi; bunlar doğal dile ve daha büyük bir bayt-düzeyi modele taşınır mı?

**Cevap, sekiz maddede (ayrıntı ve sınırlar ilgili bölümde):**

1. Sabit durumlu görevlerde yinelemeli çekirdek uzunluğa genelliyor: beş-durumda 3 tohumdan 2'si 4.096'ya kadar %100, parite 1.024'te %100; geri beslemesiz kontrol şansta (parite ≈ %48–50, beş-durum ≈ %19–21). (Faz 1, U1)
2. Çöküş uzunlukta değil, durum kurulduktan sonra gelen **uzun nötr koşuda**: durum 4–16 adımda 1–2 çekici sınıfa kayıyor ve beş-durumda onarılmıyor. Nötr koşu kaybının tamamı budur. (U1, U2)
3. Üç ucuz çare (geniş yerel pencere, değişmezlik kaybı, durum-geçmişine attention) bunu çözmedi; üçü de `MEASURED_NEGATIVE`. (sürüklenme mini deneyi)
4. Durumu nötr adımlarda **aynen tutmak** kaybı bitiriyor: nötr adımı bilen bir oracle ile tavan (U2b) ve modelin kendi öğrendiği HOLD/WRITE kapısıyla (U2c, `hard_st`) iki görev ailesinde ve iki öğrenme oranında. Kampanyanın ilk pozitif mimari sonucu. Sınır: 32 genişlikli oyuncak modeller, sentetik görevler.
5. Kapının neden işe yaradığına dair üç nedensel deneme (U4-sep, U4-sep2, U4-inject) olumsuz bitti ve üçü de kendi ön iddiasını düzeltti ya da reddetti.
6. Anlamsal erişim ve dil modeliyle ayrıştırma hedeflerin çok altında kaldı: B1/B2 (recall@10 ≈ 0,66; hedef 0,95), D1 (ilişki-F1 0,03; yanlış-kabul riski ≥ 0,97).
7. 30M bayt modeli (M-hattı): kapı-aktarım hükmü kapıyı gerçekte sınamadı (kapı hiç kapanmadı); uzak bilgi uçurumu bir **eğitim aralığı etkisi** (ince ayarla ≈ 6× kaydı); iğne görevi konuma kilitli çıktı; MQAR ince ayarı anahtar eşleştirmesini öğretti ama harfe özgü (rastgele anahtarlarda 0,95, görülmemiş harflerde 0,41); gerçek kelime/isim anahtarları eğitimde yokken şans düzeyinde (0,00–0,28), eğitime karıştırılınca görülmemiş kelimelerde 0,93 (D6, ara kontrol).
8. 70× küçük çekirdekte (0,44M) aynı örüntü yeniden üretiliyor: tek-olgu başlangıcı + MQAR eşleştirmeyi öğretiyor, eşleştirme harfe özgü (T1p: pangram dolgusu T–Z gömmelerini eğittiği hâlde T–Z'de yok), genişlik 64'te devre 4.000 güncellemede oluşmadı.

**Anahtar eşleştirmesi en güçlü model (şu an):** D5, `results/M1-finetune-distance/D5-mqar4-broad/selected.pt` (31.131.904 parametre; eğitim sembollerinde ilk rakam 0,95, tam eşleşme 0,90; görülmemiş T–Z 0,41; gerçek kelime/isim anahtarlarında şans düzeyi). D6 (gerçek anahtarlı eğitim) bitince aday olur; ara kontrolde gerçek anahtarlarda 0,93. *DÜZELTME notu:* 30M modeller sorulduğunda önce D2 söylenmişti (D2: eğitim harflerinde 0,91); D5 ve D6 sonuçlarıyla bu liste güncellendi. Ağırlıkların tamamı iyi/kötü ayrımı yapılmadan kayıtlıdır (§10).

## 2. Güncel durum (2026-10-01 akşam)

| Alan | Durum |
|---|---|
| Çalışan | **D6** (GPU, D5 + gerçek anahtarlı MQAR ince ayarı, 1.000 güncelleme, ≈ 23:00'te biter; ara kontroller her 250 adımda CPU'da). **T1w256** (CPU, genişlik 256, iki aşama: 1.500 + 4.000 güncelleme). |
| Biten bu oturumda | D5 (1.500), T1p, T1w64; sonuçlar `notes/M1_EK_KAPI_OLU_2026-10-01.md`. |
| Hazır, başlamadı | **L1**: sıfırdan 20.297.472 parametreli bayt modeli, 500M bayt, plato için durdurulmayan uzun koşu (`scripts/run_longrun.sh`). D6 tarifi kesinleştirdikten sonra elle başlatılacak. |
| Bekleyen karar | Kapının geçerli testi (kapanabilir kapı, dikkati sınırlı kol, manipülasyon kontrolü) tasarımı; B taraması (`feedback=False` + `gate_input='input'`, lr 1e-3, ≈ 5 saat CPU); kesintiye karşı Windows zamanlanmış görevi. Hepsi kullanıcı onayı bekliyor. |
| Yayın | Depo açık kaynak yapılacak (MIT) ve eğitilmiş bütün modeller paylaşılacak; hazırlık bitti, **gönderme adımı** otomatik izin sisteminde bekliyor (bkz. §10). |
| Güvenlik | VM erişim bilgisi proje klasörlerinden ve depo geçmişinin yayın kopyasından çıkarıldı; depo dışında şifreli bir klasörde tutulur. |

## 3. Harita

### 3.1 Kökler ve dallar

Dört kök **tek bir git deposunun** çalışma ağaçlarıdır (ortak `.git`: `2026-09-16-e2e/.git`); tarihçe doğrusal bir zincirdir: e2e → diagnostics → drift → phase2.

| Dizin (`E:\rlt-research\…`) | Ne | Dal · commit | Mühür |
|---|---|---|---|
| `2026-09-16-e2e` | Faz 1: 14 paket | `codex/rlt-e2e` · 3c118c7 | `audit/final-manifest.json`, sha 146eb693…, 9.073 dosya |
| `2026-09-16-phase2-plan` | Faz 2 planı (12 deney + G0) ve dış yorum değerlendirmesi | git yok | plan dosyaları `docs/phase2-plan-snapshot/` altında bayt bayt kopya |
| `2026-09-17-diagnostics` | R1–R6 tanı döngüsü | `codex/rlt-review-diagnostics` · f83b527 | `audit/review-diagnostics-manifest.json`, sha 30bf6c54…, 2.483 dosya, 1,29 GiB |
| `2026-09-18-drift` | sürüklenme mini deneyi | `codex/rlt-drift-mini` · c01639f | `audit/drift-mini-manifest.json`, sha 0ffb20ad… |
| `2026-09-18-phase2` | Faz 2 ve M-hattı (canlı kök) | `codex/rlt-phase2` | döngü başına `audit/phase2-*-manifest.json` |
| `2026-09-20-rlt2-15m-2b` | **ayrı proje**: RLT-2 15M, üç model × 2B token (bkz. §12) | `codex/rlt2-15m-2b` (ayrı depo) | ayrı |

Diğer klasörler: `datasets`, `hf-cache`, `models` (üçüncü taraf: indirilen veri/model; bu projenin ürünü değil), `mline-data` (cnn_dailymail bayt korpusu: eğitim 209.720.360 B, doğrulama ≈ 8,4 MB), `vm-mirror` (M1 koşularının bulut makinesinden indirilmiş kopyası: sonuçlar, günlükler, 8 kontrol noktası), `tmp`, `.venv-phase2` (Faz 2 ortamı: torch 2.14.0+cu130, transformers 5.17.0).

### 3.2 Canlı kökün düzeni (`2026-09-18-phase2`)

- `core/`, `learning/`, `memory/`, `integration/`, `reports/`, `briefs/`: Faz 1 (mühürlü; `core/model.py` = mühürlü Core, **düzenlenmez**). `archive/`: Faz 2 paket uygulamasının eklediği append-only arşiv.
- `review_cycle/`: R1–R6. `drift/`: sürüklenme mini.
- `horizon/`: U-döngüleri. `semantic_memory/`: B1/B2. `language/`: D1. `protocols/`: dondurulmuş protokoller ve pin'ler. `mline/`: M-hattı.
- `results/`: sonuç dosyaları (§9). `notes/`: döngü raporları. `audit/`: mühür manifestleri ve bağımsız denetim betikleri. `tests/`: 87 dosyalık pytest takımı. `scripts/`: kuyruk ve koşu betikleri. `docs/`: planlar ve dış tasarım paketi (`docs/external/rlt_gelistirme_v0_1/`). `.superpowers/sdd/`: ajan defterleri, denetim raporları, test günlükleri.
- Kural olarak yeni deney mühürlü köklerde çalıştırılmaz; protokol, veri açılmadan önce SHA-256 ile dondurulur.

## 4. Kronoloji

| Tarih | Olay |
|---|---|
| 2026-09-16 | Faz 1 kapandı (14 paket, 63 eğitim koşulu, 6.144 + 1.536 yürütme, 108 test); Faz 2 planı yazıldı; dış yorum değerlendirildi |
| 2026-09-17/18 | R1–R6 tanı döngüsü, bağımsız denetimlerden geçti (f83b527) |
| 2026-09-18 | Sürüklenme mini deneyi (c01639f). Faz 2 kökü; G0; U1 (4114d62); U2/U2b (a6a3e6c); dış tasarım paketinin 1. dalgası (9b7df88). 19:57'de elektrik kesintisi |
| 2026-09-19 | B1 mühürlendi (3c8e587); D1 finali başladı; 14:09'da ikinci kesinti ve GPU→RAM taşması |
| 2026-09-20 | D1 mühürlendi (aaf45f7); B2 başladı |
| 2026-09-22/23 | B2 (668d54e); U2c (fe65ba5, düzeltme 5385e6a, r2 7f47357); M-hattı planı (f7f5e26) |
| 2026-09-23/24 | U4-sep (43cf35e), U4-sep2 (7c20aa4), U4-inject (9e53d5c): üçü de olumsuz; 24 Eylül gecesi, bağımsız denetim düzeltmelerinden sonra (`-r2` / `-r3` manifestleri) mühürlendi |
| 2026-09-25 | M0 sondası: tam sıfır (neden: iğne satırlarının %83,7'si boş dolgu); paketlemeyle düzeltildi |
| 2026-09-27 | M1 hükmü `GATE_DOES_NOT_TRANSFER`; bulut makinesi kapatıldı; kanıt git'e girdi (8fe3feb) |
| 2026-10-01 | Mesafe tanısı, RoPE/kanal/akış deneyleri, ince ayar (ft384), yem-olgu probu, D1–D6, T1 serisi, L1 hazırlığı; 16:53'te elektrik kesintisi (18:41'de dönüş, işler kurtarıldı); açık kaynak hazırlığı |

## 5. Bulgular tablosu

Etiketler: `MEASURED_POSITIVE` / `MEASURED_NEGATIVE`: önceden yazılmış ölçüt geçildi / geçilmedi. `MIXED`: farklı koşullar farklı sonuç verdi. `IMPLEMENTED_TESTED`: uygulandı ve testlendi, bilimsel iddia değil. `UNMEASURED`: ölçülmedi. Aralıklar keşif amaçlıdır, çokluk düzeltmesizdir. Hiçbir sonuç doğal dile ya da büyük modele genellenmez.

| Döngü | Soru | Etiket | Ana sayı | Rapor |
|---|---|---|---|---|
| Faz 1 (14 paket) | RLT çekirdeği, başlık belleği, pencere, doğrulayıcı… | 2 `IMPLEMENTED_TESTED`, 2 `MEASURED_NEGATIVE`, 10 `MIXED` | bkz. §6.1 | `reports/RLT_SONUC_RAPORU.md` |
| R1–R6 | Faz 1 sonuçlarının tanısı | altı kontrol, bağımsız denetimli | segment yürütmesi L1024'te CUDA tepe bellek 0,53× | `notes/RLT_R1-R6_SONUC_RAPORU.md` |
| Sürüklenme mini | Pencere / değişmezlik kaybı / durum-geçmişi attention sürüklenmeyi azaltır mı? | `MEASURED_NEGATIVE` (3/3) | paritede pencere 16: −45…−49 puan | `notes/DRIFT_MINI_RAPORU.md` |
| G0 | Faz 2 ortak hazırlığı | tamam | modeller/veri sürüm sabitli | `notes/G0_KAYIT.md` |
| U1 | 128 → 4.096 uzunluk genellemesi | beş-durum `MIXED`, yığın `MEASURED_NEGATIVE` | beş-durum 2/3 tohum 4.096'da %100; nötr koşu −44…−59 puan | `notes/U1_RAPORU.md` |
| U2 | Sürüklenme nedir? | tanı: sürüklenme = kaybın tamamı | düz `freeze` temizde çöker; `snap` kısmi | `notes/U2_RAPORU.md` |
| U2b | Koşu-farkında dondurma | `MEASURED_POSITIVE` (tavan, oracle) | `freeze_k2`: nötr kayıp ≤ 0,6 puan, temiz maliyet ≈ 0 | `notes/U2_RAPORU.md` §3 |
| U2c | Öğrenilmiş HOLD/WRITE kapısı | `hard_st` `MEASURED_POSITIVE` | onarım %94/%79 (beş-durum), %99/%92 (yığın), iki lr | `notes/U2c_RAPORU.md` |
| U2d | Dayanıklılığı içeriden ne ayırır? | tanı (n = 12), sonradan düzeltildi | çapa aralığı / sürüklenme | `notes/U2c_RAPORU.md`, `notes/U4sep_RAPORU.md` |
| U4-sep | Ayrım kaybı nedensel mi? | deney kendi sorusunu sınamadı | düşük lr kolu `INTERVENTION_INEFFECTIVE` | `notes/U4sep_RAPORU.md` |
| U4-sep2 | Menteşeli aralık/sürüklenme müdahalesi | önceden kayıtlı iddia reddedildi (`CONFOUNDED_BY_TRAINING`) | oran 5/5 yükseldi, onarım 2'sinde arttı | `notes/U4sep2_RAPORU.md` |
| U4-inject | Toplamalı geri besleme yerine attention-injection | `MEASURED_NEGATIVE` (6/6) | enjeksiyon kolları şansta (≈ 0,2), kontrol 3/3 kusursuz | `notes/U4inject_RAPORU.md` |
| B1 | Sözcüksel vs anlamsal erişim (MuSiQue, 2.417 sorgu) | `MEASURED_NEGATIVE` (5 kol) | en iyi recall@10 0,665 / kapsama@10 0,325 (hedef 0,95 / 0,90) | `notes/B1_RAPORU.md` |
| B2 | Çok-atlamalı erişim | `MEASURED_NEGATIVE` | 0,663 / 0,336; birleşim tavanı 0,488 | `notes/B2_RAPORU.md` |
| D1 (dil) | 4B modelle metinden başlık/iddia/kaynak | `MEASURED_NEGATIVE`; `finite_parser` `MISSING_ASSET` | ilişki-F1 0,03; yanlış-kabul riski ≥ 0,97 | `notes/D1_RAPORU.md` |
| M0 | 30M iğne sondası | tam sıfır, neden bulundu | iğne satırlarının %83,7'si boş dolgu | `notes/M_HATTI_PLANI.md`, `results/M0-*` |
| M1 | 30M: kapı aktarılıyor mu? | `GATE_DOES_NOT_TRANSFER` (kapıyı sınamadı) | kapı hiç kapanmadı (hold ≈ 5e-6) | `notes/M1_EK_KAPI_OLU_2026-10-01.md` |
| M-mesafe | Uzak bilgi neden 0? | eğitim aralığı etkisi | uçurum 64 → 384 ile ≈ 6× kaydı | aynı not, §6.5 |
| MQAR (D2–D6) | Anahtar eşleştirmesi öğrenilir mi? | kısmen / harfe özgü / anahtar türüne bağlı | D5: eğitim sembolleri 0,95; D6: görülmemiş kelimeler 0,93 (ara) | aynı not |
| T1 | Küçük çekirdekte aynı soru | yeniden üretildi | T1p: T–Z 0,31–0,37; T1w64: oluşmadı | aynı not |

## 6. Ayrıntılı bölümler

### 6.1 Faz 1 — on dört paket (`2026-09-16-e2e`, mühürlü)

Kaynak: `reports/RLT_SONUC_RAPORU.md` (Faz 1 kök `README.md`, `SPEC.md`). Resmî RLT kaynağı sabitli: `cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39` (yazarların 26–29M modelleri yeniden üretilmedi; küçük bağımsız uygulamalar kullanıldı). 63 eğitim/devam eğitimi koşulu, bağlı sistemde 6.144 yürütme, gerçek durum penceresinde 1.536 yürütme, 108 test, önceki üç araştırmadan 424 kanıt dosyası değişmeden doğrulandı.

| No | Paket | Etiket | Kapsam özeti |
|---|---|---|---|
| 1 | RLT çekirdeği ve kontrol | `IMPLEMENTED_TESTED` | gizli durum, yerel KV, global encoder belleği; nedensellik ve akış/toplu eşlik |
| 2 | Uzun ufuk | `MIXED` | 32'de eğitim, 32–1.024 test; parite, beş durum, tohum ayrı |
| 3 | Ek hesaplamanın yararı | `MIXED` | t/k/d ayrıldı; kimlik son ekleri tekdüze kazancı desteklemiyor |
| 4 | Gecikmiş hata ve eğitim maliyeti | `MIXED` | prefix öğrenimi ≠ son-cevap gözetimi; son-cevap koşulları şansa yakın |
| 5 | Metinden başlık ve ilişki | `MIXED` | görülen kalıpta başarılı, yeni kalıpta aktarım zayıf |
| 6 | İleri/çift yönlü ağ | `MIXED` | eğitim benzeri görevde büyük yarar (%26,11 → %95,70); derinlik 8'de tek başına yetmez (%9,90) |
| 7 | Seçici revizyon | `MEASURED_NEGATIVE` | önceden belirlenen güçlü ölçüt iki ailede de geçmedi |
| 8 | Başlık penceresi | `IMPLEMENTED_TESTED` | gerçek z sınırı saklama/okuma; 70.080 iz, 8.760 sınır/okuma, aktif tepe geçmiş 8 → 4 |
| 9 | Sıkıştırma ve yeniden okuma | `MIXED` | sabit bağlamda eşdeğer; güncel kısıttan sinirsel yeniden başlatma derinde kayıp (%37,63 → %20,57) |
| 10 | Ham arşiv ve vektör erişimi | `MEASURED_NEGATIVE` | sözcük-karma recall@5 yalnız %4,17 |
| 11 | Yeterlilik ve durma | `MIXED` | %24,14 daha az gerçek tur, %38,93 sertifikalı kapsam; yeni derinlikte kapsam %0,13 |
| 12 | Gerçek RLT/başlık entegrasyonu | `MIXED` | metin→başlık→model→arşiv→denetim çalıştı; ayrı geri besleme kazancı küçük/belirsiz |
| 13 | Üretilmiş izleri ağırlıklara aktarma | `MIXED` | kolay görevde zarar (MSE 0,0026 → 0,0117), derinde iyileşme (0,698 → 0,558); aralıklar sıfırı içerir |
| 14 | Ölçek, aktarım ve maliyet | `MIXED` | ≈ 24 bin – 1,51 milyon parametre; büyük modelde 3 tohumdan 1'i öğrendi |

Küçük çekirdek (512 yeni örnek/hücre, 3 eğitim ortalaması): parite/32/RLT %100 (32, 128, 1.024), geri beslemesiz %48,24–%49,87; beş-durum/32/RLT %100 → %98,63, geri beslemesiz ≈ %19–21. Her adımda hedefli öğrenme ile yalnız son cevaptan öğrenme aynı değil: son-cevap altı koşul %47,46–%51,56 (şansa yakın; yalnız bu 2.000 güncellemelik protokol için, eşdeğerlik iddiası değil). Kimlik son eki (128 anlamsal-kimlik işlem): son ek yokken %100 olan modellerde etki model ve tohuma göre değişti (örn. parite/32/RLT/tam: %100 / %50,20 / %49,80). 1,51M modelde başarılı tohum 1.024'te %99,61, diğer ikisi %48,44; kimlik son eki başarılı modelde %99,41 → %50,59. Doğrulayıcı: ana bağlı sistemde doğru sayısal aday %98,18, kabul edilen doğru son cevap kapsamı %38,93, gerçek güncelleme sayısı %24,14 azaldı.

Açık sınırlar (rapordan): serbest doğal dil, öğrenilmiş genel konu bölme, güvenilir anlamsal vektör erişimi, geniş model/RL eğitimi, enerji, toplam RAM için sabit üst sınır ve sonsuz doğru akıl yürütme doğrulanmadı veya ölçülmedi. Model ağırlıkları, tahmin dizileri ve SQLite arşivleri Git dışındadır ama nihai manifestte listelidir.

### 6.2 Dış yorum değerlendirmesi ve R1–R6 tanı döngüsü

**Dış yorum (2026-09-16, `2026-09-16-phase2-plan/reviews/`).** Dokuz bilimsel eleştiri kayıtlara karşı sınandı; yürütme talimatı sayılmadı. Başlıca: "gözetim yoğunluğu mimariden belirleyici" bir sonraki deneyin hipotezi olabilir, mevcut sonuç değil; kimlik son eki bulgusu güçlü ama tek nedene bağlanamaz (eğitimde kimlik vardı, olasılığı 0,125; sınanan blok/uzunluk dağılımı); **"%48,44 sabit sınıf" iddiası ham tahminlerce yanlışlandı** (iki dosyada 248/512 doğru; sabit sınıf 261 ya da 251 verirdi); doğrulayıcının bağlı-bileşen sınırı geçerli ama gereğinden geniş olabilir (yol sınırı daha dar); bellek ölçümü CUDA değil süreç çalışma kümesiydi. Plana altı tanı kontrolü (R1–R6) önerildi.

**R1–R6 (2026-09-17/18; `notes/RLT_R1-R6_SONUC_RAPORU.md`; manifest 30bf6c54…; commit f83b527; bağımsız denetimler geçti).**
- **R1** Yol sertifikası: dallanmışta dar kazanç, 6 / 0 / 22 ek kabul.
- **R2** Bloklu nötr yerleşim: dayanıklılık vermedi.
- **R3** Etiket yoğunluğu: tek rastgele etiket yoğun etiketle aynı sonucu verdi; son-konum etiketi ve geri beslemesiz çekirdek şansta.
- **R4** TBPTT8 = tam BPTT (beş-durum).
- **R5** Faz 1'deki "3 tohumdan 1'i öğrendi" sonucu **öğrenme oranı duyarlılığıydı**: lr 0,003 başarısız, lr 0,0003 ile 3/3 taze tohum %100.
- **R6** Segment yürütmesi L1024'te CUDA tepe belleğini 0,53×'e indirdi; yalnız-detach ve checkpoint etkisiz.

### 6.3 Sürüklenme mini deneyi (`2026-09-18-drift`; `notes/DRIFT_MINI_RAPORU.md`)

Aynı girişler ve başlangıç ağırlıklarıyla üç müdahale: yerel KV penceresi 4 → 16, nötr girişte durum-değişmezlik kaybı (λ = 1), durum-geçmişine attention (reread). 22/24 koşul tamam (2 reread koşusu sonsuz gradyanla ıraksadı, kanıtı korunarak `INCOMPLETE`). Protokol 1496384366fc62f9…, bağımsız ham denetim 42/42 PASS, taban kolu mühürlü tabanı 78/78 son noktada özdeş yeniden üretti.

1. **Pencere 16 sürüklenmeyi artırdı**: beş-durum nötr-32 orta/son −30/−35, paritede (tabanın 3/3 dayanıklı olduğu görevde) nötr-32 ve nötr-128 −45…−49 puan, t95 aralıkları sıfırı dışlıyor.
2. **Değişmezlik kaybı yerel işe yarıyor, uzakta kırılıyor**: beş-durum nötr-32 orta/son +9/+7; nötr-128 ön ekte taban 1,00/0,99/1,00 iken 0,21/0,19/0,99 (−53 puan).
3. **Reread kararsız ve zararlı** (bu spesifikasyonla: anahtar/değerler normalize edilmemiş durumdan alındı; "durum-geçmişi attention" fikrinin genel özelliği değil): 6 koşuldan 2'si ıraksadı; gradyan sıçraması reread'e özgü değil (taban dahil her kol), reread'e özgü olan normun inf'e kaçması.
4. Denetçinin ham veri gözlemi: 128'lik nötr son ek her kolda durumu 1–2 çekici duruma çökertiyor (deterministik çoktan-bire ≥ 0,98).

Faz 2'ye girdi: koşunun konumu (kurulmadan önce/sonra) ve uzunluğu ayrı faktör; ölçüt sabit nokta kararlılığı; reread denenecekse anahtarlar normalize edilmeli.

### 6.4 Faz 2 (`2026-09-18-phase2`)

**G0 (ortak hazırlık; `notes/G0_KAYIT.md`).** Pin'li modeller: Qwen3-Embedding-0.6B `97b0c614…`, Qwen3-Reranker-0.6B `e61197ed…`, Qwen3-4B `1cfa9a72…`; veri: MuSiQue `c8f4f8c9…` (train+dev), LongMemEval-cleaned `98d7416c…` (yalnız S ve oracle), BigCodeBench `b74c0d0b…`. Ortam `.venv-phase2` (torch 2.14.0+cu130). Cevap modeli kalibrasyonu: Qwen3-4B 8-bit 2,0 tok/s, nf4 7,2 tok/s → **nf4**. Sınırlar: Türkçe dilimi yok, insan anotasyonu yok.

**U1 — t ekseni (`notes/U1_RAPORU.md`; protokol 56ef56a1…; denetim 328/0).** İki görev ailesi: beş-durum (sabit bellek) ve yığın (büyüyen bellek). Eğitim ufku 128, test 128–4.096 × 1.000 problem, nötr ekler 128/512 × ön/orta/son. Beş-durum: s117 ve s143 her noktada 1,000; s191 4.096'da 0,947. Yığın: eğitim ufkunda bile ≥ %95 yok, uzunlukla tekdüze düşüş (0,936 → 0,801 / 0,954 → 0,715), sabit-sınıf çöküşü yok; doğrulama kaybı 2.048 güncellemede hâlâ düşüyordu, yani "bu bütçede öğrenilemedi", kapasite sınırı iddiası değil. Nötr koşular (kayıp, puan): beş-durum 128-orta +44,1, 128-son +46,4, 512-orta +59,4, 512-son +53,0 (ön ek +0,2); yığında yalnız son ek (+56,5 / +58,6). Okuma: sürüklenme = durumun yanlış bir noktaya kayması; yığında sonraki işlemler bozulan üst sembolü üzerine yazıp onarıyor, beş-durum permütasyon otomatı olduğundan onarmıyor.

**U2 — sürüklenmeyi teşhis ve düzeltme (`notes/U2_RAPORU.md`).**
- *Ölçüm (eğitimsiz):* çapa tutarlılığı 1,000; yer değiştirme k = 1,2 nötr adımda ≈ 0; s117 ve s143'te 4–16 adımda 0,22–0,62'ye doyuyor (s191 daha yavaş: 128 adımda 0,39); koşu sonunda 1–2 çekici sınıf (diğer sınıflar ≈ 0); beş-durumda onarım yok (yer değiştirme sonraki 64 gerçek işlemde 0,30–0,60'ta kalıyor), yığında üzerine yazma onarımı.
- *Test-zamanı müdahaleler (oracle nötr bilgisi, tavan ölçümü):* düz `freeze` nötr kaybın tamamını geri alır ama doğal token 2'ler de donduğu için temiz uzun dizilerde çöker (4.096'da −46,7 ortalama; s117/s143 −63/−76); `snap` (en yakın çapa) kısmi (+30/+39) ve temizde −11; `wrong_snap` şansa düşürür (çapalar sınıf bilgisini taşıyor). Yığında `freeze` son ekte tam onarım, maliyetsiz.
- *U2b koşu-farkında dondurma:* yalnız ardışık (k+1)'inci nötrden itibaren dondur. `freeze_k2`: üç tohumda tüm nötr hücreler kendi tabanının ≤ 0,6 puan içinde, temizde 0,0 / 0,0 / +0,07; yığında son ek +55,7 / +58,9. Etiket `MEASURED_POSITIVE_CEILING`. Protokoller: U2-intervene dbdd3395…, U2b 20b73983…; mühür a6a3e6c, manifest 109bd679…. Sınır: `U2-measure` çalıştırılırken ayrı dondurma kaydı alınmamıştı, sonradan kayıt (`frozen-posthoc.json`).

**U2c — öğrenilmiş HOLD/WRITE kapısı (`notes/U2c_RAPORU.md`; `horizon/gated.py`).** Kapı yalnız `[e_t, s_{t−1}]` görür; HOLD'da korunan durum aynen kalır. Kollar: `soft` (sigmoid karışım), `hard_st` (sert karar + straight-through, HOLD'da pencere önbelleği de donar), `supervised` (hard_st + U2b'nin k=2 kuralına BCE). 36 koşul (3 kol × 2 görev × 3 tohum × 2 lr). 512 nötr adımlık son ek için onarım: `soft` beş-durumda +%100, yığında −%3; `hard_st` beş-durumda %94, yığında %99; `supervised` yığında %99, beş-durumda %71 ve temiz 4.096'da −0,385. Düşük lr'de (0,0003) `hard_st` %79 / %92 ile pozitif kalan tek kol; `soft` ve `supervised` negatife düşer. **Yöntem dersi (C16):** eğitim dağılımındaki doğrulama, ölçülen özelliğe (sürüklenme dayanıklılığı) kör: lr'yi doğrulamaya bakarak seçmek sonucu sessizce yok ederdi. **U2d tanısı (n = 12, korelasyonel):** dayanıklıyı kırılgandan ayıran tek nicelik sınıf çapalarının aralığı ve bunun sürüklenmeye oranı (dayanıklı ≥ 2,6, kırılgan ≤ 2,5); *bu bulgunun sayıları U4-sep'te düzeltildi (§7).* Olay: `hard_st` 513. adımda çöktü; sebep bilim değil kayıt kodu (kare toplamı float32'de taştı, `allow_nan=False` yazıcı hata verdi); düzeltme (norm float64, sonsuz norm atlama sayılır) kaynakları değiştirdiği için döngü baştan koşuldu (`results/U2c-v2`), v1'in altı `soft` koşusu v2'de bit-bit yeniden üretildi (CPU eğitimi tam tekrar üretilebilir). Mühür fe65ba5 + 5385e6a + r2 7f47357; manifestler 96694038… ve 06098fab….

**U4-sep (`notes/U4sep_RAPORU.md`).** U2d'nin nedensel testi için çapa ayrımını artıran yardımcı kayıp (`horizon/sep_loss.py`). **Deney kendi sorusunu test etmedi:** kayıp aralık/sınıf-içi-yayılım oranını hedefliyordu, bulgu aralık/sürüklenme idi (alan adı `separation_ratio` müdahaleyi bulgudan değil alan adından kurdurdu). Kapı kayba dejenere bir kaçış yolu verdi (kapı kapanır, durum donar, yayılım sıfırlanır, oran tırmanır, görev öğrenilmez). Düşük lr 2/2 şansa çöktü; yüksek lr çöküşün kaçılabilir tuzak olduğunu gösterdi (897. adımda kaçış, sonra oran 2.722). Ön-kayıtlı karar kuralı çıktısı: `INTERVENTION_INEFFECTIVE`. §7 düzeltmeleri raporun sonunda.

**U4-sep2 (`notes/U4sep2_RAPORU.md`; protokol 5e10c8d7…; commit 7c20aa4).** Hedef: aralık/sürüklenme, menteşeli (`horizon/geom_loss.py`), donmuş U2c kontrol noktalarında ince ayar, her biri için aynı bütçeli yalnız-görev-kaybı kontrolü. Oran 5/5 çiftte yükseldi; onarım 2'sinde arttı (soft/s117 +95,1; soft/s143 +24,9), 3'ünde düştü (supervised/s117 −106,0; s143 −2,9; s191 −51,8). Etiket `CONFOUNDED_BY_TRAINING`: "oranı dayanıklı aralığa taşı, dayanıklılık peşinden gelir" iddiası reddedildi. Tasarım kusuru: supervised kontrol kolları boş kontrol değil.

**U4-inject (`notes/U4inject_RAPORU.md`; protokol 523dba46…; commit 9e53d5c).** Toplamalı geri besleme yerine attention-injection (FLT, arXiv 2605.18797 uyarlaması). 9 koşul: `additive` (mühürlü Core) 3/3 doğrulama 1,000; `inject` ve `inject_mix` 0/3 (0,215–0,309; şans 0,200). **Bu sonuç makalenin yöntemi hakkında değil benim uyarlamam hakkında:** `horizon/inject.py`'de önceki durum yalnız sorgu olarak girer, içeriği hiçbir yoldan yeni duruma akmaz; `inject_mix` planlandığı gibi "geniş kanal" testi değildir. Gradyan iddiası geri çekildi (ilk 64 adımda iki kolun gradyanı aynı).

**B1 — sözcüksel vs anlamsal erişim (`notes/B1_RAPORU.md`; protokol makbuzlu; mühür 3c8e587, manifest 5cb28aa3…).** MuSiQue-Ans dev, 2.417 sorgu, 21.100 paragraf. En iyi `hybrid_rerank` recall@10 0,665, kapsama@10 0,325. dense 0,600 / 0,247; bm25 0,487 / 0,134; hybrid (RRF) 0,588 / 0,223 (hibrit < dense); `hash128` şansta (0,048). Reranker +6,6 puan recall ama sorgu başına 1,93 s. Sorun aday üretiminde: altın paragrafı ilk 10 dışında kalan 1.878 sorgunun 1.322'sinde en az bir altın ilk 50'nin de dışında. Atlama sayısıyla çöküş (4 altın: 0,012). Ölçek merdiveni: her 10× korpus ≈ −10…−14 puan. Reranker ilk final CUDA OOM (tam-vocab logits) → yama + parite testi; başarısız deneme `results/B1-attempt1-oom/` korunuyor.

**B2 — çok-atlamalı erişim (`notes/B2_RAPORU.md`; protokol 7c658de8…; mühür 668d54e).** `multihop_rerank` 0,663 / 0,336 (B1: 0,665 / 0,325, gürültü düzeyi). Birleşim tavanı 0,488; ikinci atlama son top-10 altınlarının yalnız %6'sını getirdi; ucuz köprü çıkarıcı genel terimler üretiyor ("American", "United", "States"). Geliştirme dilimi (1.707 paragraf) +10 puan vaat etmişti, final +1,1: küçük korpusta seçim riski ölçüldü. fp16 yeniden kodlama bit-aynı değil (254/2.417 listede fark): kollar aynı vektör matrisini paylaşmalı.

**D1 (dil) — metinden başlık/iddia/kaynak (`notes/D1_RAPORU.md`; protokol 38f8d005…; mühür aaf45f7).** Qwen3-4B nf4, 600 MuSiQue sorusu. `lm_schema` ilişki-F1 0,032, `lm_zero` 0,021 (hedef 0,95); kabul edilen 1.472 / 1.418 atlamanın 1.423 / 1.387'si yanlış (risk üst sınırı 0,97 / 0,98); 300 programatik bozmadan beklenen 400 atlamanın 330 / 247'sini sahte kabul. İlişki-F1 metriği yapısal olarak hedefe ulaşamazdı (≈ 1.000/1.570 atlamada "ilişki" serbest alt-soru; tavan ≈ 0,5; **protokol tasarım hatası**); ikincil görünümde zincirin tamamı %7–8. `finite_parser` kolu `MISSING_ASSET`. Kesintiler: bir elektrik kesintisi + altı GPU→RAM taşması yeniden başlatması, 0 satır kaybı.

**Paket uygulama programı (2026-09-18 gece).** Dış tasarım paketi (`docs/external/rlt_gelistirme_v0_1/`) okundu, kullanıcı onayıyla 1. dalga (9b7df88): `audit/receipt.py` (E12 makbuz), `audit/stats.py` (Holm/risk/tam-log/manifest), `archive/store.py` (append-only arşiv), `horizon/guards.py` (A8 sonluluk koruması), `horizon/remat.py` (A5), `tests/run_all.py` (E11), B1 zinciri makbuzlu v2. Uygulanmayanlar (protokol değişikliği gerektirir): ≥ 5 tohum, 16k/64k ufuk, hedef değişimi.

### 6.5 M-hattı — 30M bayt modeli ve uzak bilgi (`notes/M_HATTI_PLANI.md`, `notes/M1_EK_KAPI_OLU_2026-10-01.md`)

**Plan (commit f7f5e26, 2026-09-23).** Tek amaç model eğitmek; 16 ölçülmüş bulgu "inşa et / inşa etme" kararına bağlandı. 30M = mühürlü Core'un dil genişliğinde: `Core(width=1152, heads=8, classes=256)` + bayt gömme = **31.131.904 parametre**; kapı yalnız 2.305 parametre ekler. Korpus: cnn_dailymail bayt düzeyinde (eğitim 209,7 MB, doğrulama 8,4 MB). Hız ölçüldü ve iki tahmin de yanlış çıktı: 4,39–5,15 s/adım yerelde, batch 8 → 32 ile 2,70×, GPU kullanımı %5, CUDA grafiği 1,20×, `torch.compile` Windows'ta yok (Triton). `mline/faststep.py` mühürlü Core ile 1e-7 farkla aynı. İğne sondası yeniden tasarlandı: biçim 0–128 dolguda öğretilir, 512/2048'de ölçülür; anahtar alfabeleri A–M / N–Z ayrık.

**M0 (2026-09-25).** 5.000 adımlık sonda (81,9 MB): dil öğrenmesi iyi (bpb 2,267 → 1,655) ama 76 ölçümün hepsinde iğne tam eşleşmesi 0,000, eğitim dağılımında bile. Neden: iğne eğitim satırlarının %83,7'si boş dolguydu, cevap baytları toplam gradyanın binde birini alıyordu. Paketleme 5,5× düzeltti (`results/M0-packed`). `results/M0-padded/REASON.txt` "3358'de durduruldu" diyordu; koşu sonradan 5.000/5.000'e tamamlandı: başlık düzeltildi, eski metin dosyanın altında aynen korundu. Test koşucusunun 42 testi hiç çalıştırmadığı bulundu (unittest, fonksiyon-stili dosyalar) → pytest'e geçildi, takım 329/329 oldu.

**M1 (2026-09-27, bulut makinesi, tek tohum 117; `results/M1/`).** Hüküm `GATE_DOES_NOT_TRANSFER`. plain: doğrulama bpb 1,7049, iğne@0 0,387; hard_st: 1,7122, iğne@0 0,902; **iki kol da 128/512/2048 dolguda 0,000**. Ön-kayıtlı tahmin (a) tuttu (bpb bedeli +0,0073, eşik 0,02), (b) düştü, (c) ölçülmedi (`soft` kolu bütçe için düşürüldü, protokolde `dropped_arms`). Uzunluk-eşli kontrol ayırdı: sorun uzunluk değil mesafe (near kontrol, mesafe 0, aynı genişlik: hard_st tekrar dolguda 0,76 / 0,77 / 0,35 [128/512/2048], metin dolgusunda 0,13 / 0,09 / 0,004; plain tekrar 0,19 / 0,15 / 0; far hepsinde 0,000). Eğitim dolgusu [0, 64]: 128 = eğitilen aralığın 2 katı. U5-lr (A ekseni): girdi-yalnız kapının beş-durum/s143 çöküşü öğrenme oranı duyarlılığıydı, kırık kapı değil (lr 1e-3'te durum-okuyanı geçiyor). B taraması (`feedback=False` + `gate_input='input'`, paralel-tarama ikinci koşulu) koşmadı; altyapı `horizon/scan_sweep.py`'de hazır. Kanıt git'te (8fe3feb); `*.pt` Git dışında, her koşu dizininde yerini söyleyen `CHECKPOINTS.txt` var.

**DÜZELTME — kapı hiç kapanmadı (2026-10-01).** M1 `hard_st` koşusunda `updates.jsonl` hold oranı ortalama 4,8e-6 (4.000 adım; yalnız 40 adımda > 0, hepsi ilk ≈ 100 adımda); kayıtlı kontrol noktasında iğne girdilerinin her segmentinde HOLD 0,0000, kapı logitleri en az +5,4 (ortalama +16…+25, doymuş sigmoid); kanal probu: kapı-açık = tam, birebir. Yani `hard_st` = `plain` (+2.305 parametre): **M1 kapıyı sınamadı**; iki kol arasındaki iğne@0 farkı (0,90 vs 0,39) kapı etkisi değil, koşu-koşu değişkenlik. Önceki "kapı biçimi öğreniyor, hatta plain'in iki katı" yorumu **yanlıştı**; "M-hattı basamak birde durur" sonucu desteksizdi. Ders: manipülasyonun uygulandığını günlükten oku (`feedback_manipulation_check_first`).

**Mesafe tanısı (2026-10-01).**
- *DÜZELTME — mimari:* mühürlü `core/model.py` **tam nedensel dikkate** sahip: `encode` bütün önekte causal self-attention (RoPE, kafa boyutu 144, taban 10.000); `decode` bütün kodlayıcı belleğine cross-attention; yalnız kod çözücü self-attention'ı pencere 4; durum geri beslemesi zayıf toplamalı kanal (alpha 0,1). "Uzağı taşıyan tek şey durumdur" varsayımım (Engram raporu, araştırma brifleri) yanlıştı: iğne bir geri-getirme görevi, dikkatle çözülebilir.
- *Mesafe taraması (egitim yok, repeat dolgu, 128 örnek):* eğitim dolgusu [0, 64]. plain, eğitim anahtarları: 0–64 arası 0,99–1,00; 72'de 0,484; 80'de 0,023; 96'da 0,008; ≥ 128'de 0. hard_st: 72'de 0,586, 80'de 0,031, ≥ 96'da 0. Görülmeyen anahtarlar (EVAL_KEYS): plain ≈ 0,40–0,52 düz aralık, sonra 0. **Şekil: eğitilen azami mesafenin hemen ötesinde keskin uçurum, aralık içinde düz**: "kısa eğit, uzun test et" imzası; p^N yavaş sönme değil.
- *Sıfır-eğitim müdahaleler:* RoPE enterpolasyonu: PI (0,5 / 0,25 / 0,125) aralık içinde bile yıkar (≈ 0); NTK ×2 uçurumu ≈ 72 → ≈ 100'e taşır (0,90@72 ve @80, 0,72@96) ama dolgu 0'ı çökertir (0,07); NTK ×4 aralık içi doğruluğu 0,3–0,4'e düşürür; ≥ 8 gürültü. Sıfır-eğitim düzeltme yok. Geri beslemeyi kapatma (alpha = 0): plain'de dolgu ≤ 32'de büyük ölçüde korunur (0,67–0,83), ≥ 48'de 1,00 → 0,00: temiz cerrahi değil. Araçlar: `mline/rope_probe.py`, `channel_probe.py`, `state_swap.py`.
- *İnce ayar `ft384-plain-s217` (M1 plain'den 500 adım, dolgu U[0, 384], lr 3e-4, iğne payı 0,25, tohum 217):* görülen anahtarlarda dolgu 0: 0,98; 32–256: 1,00; 320: 0,98; 384: 1,00; **448: 0,008**; 512: 0,18; 768: 0,04; ≥ 1.024: 0,00. Görülmeyen anahtarlarda @128 0,00 → 0,65, @512 0,00 → 0,19; doğal metin dolgusu @512 hâlâ 0,00 (eğitimde yalnız tekrar dolgusu); bpb 1,705 → 1,648. **Uçurum eğitim aralığıyla birlikte ≈ 6× taşındı (64 → ≈ 384–448): mesafe çöküşü bir eğitim aralığı etkisi.** Sınırlar: tek tohum, 128 örnek, yalnız tekrar dolgusu, karşılaştırma koşusu yok (iğne payı, lr, optimizer sıfırlama aynı anda değişti).
- *Mekanizma testleri:* tam-akış nakilleri: geri besleme akışı bütünüyle donör/ortalamayla değişse de ilk rakam alıcıdan geliyor (donör 0,00–0,02). alpha = 0 ince ayarlı modelde görülen anahtar 32/64/128/256/384: 0,99 / 0,97 / 0,95 / 0,73 / 0,008: kanal her iki modelde kendi eğitim maksimumuna yakın mesafelerde gerekli hâle geliyor, içeriği önemsiz. Soru-hizalı akış aşılama (yakın satırın akışı uzak satıra) hiçbir şey değiştirmedi, uzak akış yakın satırı bozmadı → uçurum **dikkat yolunda**, "tekrarlayan durum çöküşü" hipotezi doğrudan çürütüldü. Zamanda kaydırma: kendi akışını 8 adım bayat vermek 1,00 → 0,35–0,41; geri besleme olguyu değil şablondaki yeri/fazı taşıyor.
- *Yem-olgu (decoy) probu:* iki model (ve bağımsız hard_st %94–100) sorulan anahtardan bağımsız **en yakın** olgunun değerini söylüyor; ince ayarlı model ilk olguyu tercih ediyor: **iğne görevi konuma kilitli, anahtar-adresli belleği ölçmüyor.** (Ders: probu eğitimden önce yem/karşı-örnekle sına.)
- Literatür araştırması (6 araştırmacı, notlar Masaüstünde) ve rapor "Yinelemeli modellerde uzak bilgi taşıma" (Ek A, Ek B; Ek C yazılacak) repoda değil, Masaüstünde.

**Yem-olgu ince ayarları ve MQAR (`mline/decoy_train.py`, `mqar_eval.py`, `attn_probe.py`; sonuçlar `results/M1-finetune-distance/`).**
- **D1 / D1b** (3 olgu, rastgele sorulan; 500 + 1.000 güncelleme, ikinci koşuda %50 iğne): model "İLK olguyu söyle" kestirmesine kilitlendi (yem sorulunca yemin değeri %0–5, ilk olgunun %95–100); daha çok adım kestirmeyi sertleştirdi. Kendi şekliyle skor: ilk rakam 0,39, tam 0,09.
- **D2** (MQAR: 4 olgu / 4 soru, her olgu bir kez; orijinal plain'den 1.000 güncelleme, dolgu 0–64): gerçek anahtar eşleştirmesi, **yalnız eğitim harflerinde**. Eğitim-dağılımı skoru ilk rakam 0,91 / 0,88 / 0,94 / 0,94 (tam 0,88); görülmemiş T–Z'de soru 1 = 0,25 (şans). Dikkat probu (anahtar-koşulluluk): orijinal M1'de 8 kafanın hepsi 0,00 ± 0,02; D2'de iki kafa (1 ve 7) dört yerleşimde anahtara duyarlı (en iyi +0,43…+0,58).
- **D3** (D2 + anahtar uzunluğu 1–4 karışık): 4 harflik anahtar, eğitim harfleri 0,85 / 0,87 / 0,88 / 0,91 (tam 0,73–0,77); 1 harflik daha zayıf (0,46 / 0,62 / 0,70 / 0,79); görülmemiş T–Z'de yok. Geçiş 500. (şans) ile 750. (0,51) adım arasında; aynı iki kafa (1 ve 7). Tek-olgu sondası TRAIN@32 0,58 (D2'de ≈ 1,0): MQAR eğitimi tek-olgu biçimini bir miktar bozuyor.
- **Eğitim-dağılımı skoru (`mline/mqar_eval.py`) önceki yem-probu okumalarını düzeltir:** yem probu (iki olgu, tek soru) eğitimde görülmeyen bir yerleşimdi ve öğrenmeyi olduğundan kötü gösteriyordu.

| model | eğitim harfleri (soru 1/2/3/4, ilk rakam) | tam eşleşme | görülmemiş T–Z |
|---|---|---|---|
| orijinal M1 plain (30M) | 0,34 / 0,12 / 0,06 / 0,09 | 0,05 | 0,34 / 0,06 / 0,19 / 0,09 |
| D2 (30M, 1.000 güncelleme) | 0,91 / 0,88 / 0,94 / 0,94 | 0,88 | 0,25 / 0,34 / 0,44 / 0,41 |
| T1a (küçük Core, sıfırdan MQAR) | 0,39 / 0,33 / 0,47 / 0,80 | 0,18 | 0,38 / 0,38 / 0,52 / 0,81 |
| T1b (küçük Core, tek-olgu başlangıcı + MQAR) | 0,42 / 0,36 / 0,48 / 1,00 | 0,47 | 0,39 / 0,41 / 0,45 / 1,00 |

- **DÜZELTME — eleme kısayolu:** MQAR üretecinde her olgu en fazla bir kez soruluyordu; son soru "henüz sorulmamış olgu" diye bulunabiliyordu (küçük modeller tam bunu öğrendi: T1b son soruda 1,00, görülmemiş harflerde de). İlk soruya bakmak şart; düzeltme `--with-replacement` (her soruda olgu bağımsız çekilir). D1/D1b'nin `mqar-eval.json` dosyaları 4 olgu/4 soru şekliyle skorlandığı için dağılım dışı; doğrusu `mqar-eval-3x1.json`.
- **Gerçek anahtarlar (kelime ve isimler; doğrulama metninden 300 anahtar, eğitim listesinde yok, 4–24 bayt):** **D2 ve D3 tam 0,00** (140 cevap): cevap konumunda rakam yerine HARF üretiyorlar ("The code for Heathrow is " → 'a…'): doğal anahtarda "geri getirme kipi" hiç tetiklenmiyor. D5@500: 0,14.
- **D5** (D2 tarifi + 53 sembollü alfabe [T–Z hiç anahtar değil] + yerine koyarak soru, 1.500 güncelleme): adım 500 eğitim şekilli skor 0,62 / 0,62 / 0,69 / 0,62 (D2/D3 aynı adımda 0,28–0,38; tek-olgu sondası bozulmadı: TRAIN@32 0,94, @128 1,00). Adım 1000: eğitim 0,80 (tam 0,67), T–Z 0,38, gerçek anahtarlar 0,21. **Adım 1500 (son):** eğitim sembolleri 0,97 / 0,95 / 0,95 / 0,94 (tam 0,94 / 0,91 / 0,88 / 0,88; toplam 0,95 / 0,90); görülmemiş T–Z 0,45 / 0,46 / 0,41 / 0,33 (toplam 0,41, tam 0,29); gerçek anahtarlar 0,28 (tam 0,04). Dikkatte anahtara duyarlı kafa sayısı 2 → 4 (kafa 1: +0,62, 3: +0,38…+0,44, 7: +0,44…+0,49, 6: +0,26…+0,30), T–Z'de zayıf iz (en iyi +0,07…+0,08). Okuma: 53 sembol harfe özgü yönlendirmeyi ortadan kaldırmadı; gerçek anahtarlarda eşleştirme yok.
- **D6** (D5'in son ağırlıkları + 1.000 güncelleme, örneklerin yarısı eğitim metninden çekilmiş 2.000 gerçek kelime/isim anahtarı, dolgu 0–128, yerine koyarak soru; devam ediyor): **adım 250 (ara kontrol, soru başına 32 örnek):** gerçek anahtarlarda (görülmemiş 300 kelime/isim) ilk rakam **0,93, tam 0,80** (D5 son: 0,28 / 0,04); eğitim sembolleri 0,96 / 0,91; görülmemiş T–Z rastgele dizgileri 0,38 / 0,23 (değişim yok); tek-olgu sondası @128 1,00, @512 0,02. Okuma: gerçek anahtarlar eğitimde görülünce eşleştirme görülmemiş gerçek kelime/isimlere aktarılıyor; D2/D3/D5'teki 0,00–0,28 mimari sınır değil eğitimdeki anahtar türü gibi görünüyor (tek tohum, tek kolla). Sınırlar: ara kontrol, dolgu 0–128, uzak mesafe ölçülmedi, gerçek "başlık + açıklama" senaryosu değil. **Adım 500 (ara kontrol):** gerçek anahtarlarda 0,94 / tam 0,86 (kazanç korundu, tam eşleşme yükseldi); eğitim sembolleri 0,94 / 0,91; T–Z 0,33 / 0,20 (değişim yok); dikkatte T–Z'ye duyarlılık yavaşça artıyor (en iyi kafa +0,06…+0,08 → +0,14…+0,16); tek-olgu @512 0,02 → 0,19 (eğitim dolgusu 0–128'in 4 katı; doğal metin dolgusu @512 hâlâ 0). **Adım 750 (ara kontrol):** gerçek anahtarlarda **0,99 / tam 0,91** (250 → 500 → 750: 0,93 → 0,94 → 0,99; tam 0,80 → 0,86 → 0,91); eğitim sembolleri 0,96 / 0,90; T–Z 0,35 / 0,18 (üç ölçümde değişim yok); dikkatte T–Z'ye duyarlılık hâlâ zayıf (en iyi kafa +0,09…+0,11); tek-olgu @512 0,25, doğal metin dolgusu @512 0. *Son (1.000. adım) skor bekleniyor.*

**T1 — küçük Core, sıfırdan, CPU (`mline/mqar_scratch.py`, 0,44M parametre; `scripts/run_t1*.sh`).** Ucuz test yatağı (≈ 75 dk/koşu).
- T1s (yalnız tek-olgu iğne) 750. adımda görevi buldu ve **konuma kilitli çıktı** (ilk olgu sorulunca sorulan değer %0, yem sorulunca %96): 30M plain'in küçük sadık benzeri. T1a (sıfırdan MQAR) 3.000 adımda 1. soruda 0,39: öğrenmedi.
- T1b (tek-olgu başlangıcı + MQAR, 3.000 adım): 4 harflik anahtar, eğitim harfleri 0,84 / 0,92 / 0,97 / 1,00 (tam 0,81 / 0,84 / 0,89 / 1,00): gerçek anahtar eşleştirmesi 0,44M parametrede (ama eleme kısayoluyla karışık). Tek-olgu başlangıcı eşleştirmeyi engellemedi, **hızlandırdı** (kopyalama devreleri hazır).
- T1r (yerine koyarak, eleme yok): T1r-b eğitim harflerinde 0,97–0,98 (4 harf; tam 0,91–0,97; 1 harf 0,92–0,94), geçiş 2.000–2.500 arasında keskin (2.000'de 0,39, 2.500'de 0,96). T1r-a (sıfırdan): 0,30–0,45, öğrenmedi. Görülmemiş T–Z: 0,27–0,39.
- T1d (indüksiyon ön eğitimi: rastgele dizgi + tekrarı) 2.000 adımda oluşmadı (kopyalama doğruluğu 0,013, şans 0,016).
- T1g (53 sembol, yerine koyarak, tek-olgu başlangıcı): adım 2.500'de eğitim harfleri 0,73, T–Z 0,36. T1h (T1r-b devresi + 53 sembol): eşleştirme yeni sembollere (küçük harf, rakam) 250 adımda aktarıldı (0,97).
- **DÜZELTME — küçük modelde T–Z karışıklığı:** küçük modelin eğitim verisinde U–Z harfleri hiç geçmiyordu, gömmeleri rastgele kalıyordu; bu yüzden T1b/T1g/T1h/T1r'deki "T–Z'ye genelleme yok" **genelleme kanıtı sayılmazdı**. Adil test T1p.
- **T1p** (T1s2 ağırlıkları + MQAR, yerine koyarak, 53 sembol, dolgu = pangram; 4.000 güncelleme): ön-kayıtlı okuma ÖĞRENDİ ⇔ train_k4_first ≥ 0,8 (evet: 0,93@3.000, 0,99@4.000, tam 0,98), GENELLER ⇔ held_k4_first ≥ 0,6 (hayır: 0,31–0,37, 4.000'de 0,305). Pangram T–Z gömmelerini eğitti, yine de eşleştirme aktarılmadı: **küçük modelde de T–Z başarısızlığı yalnız yapısal değil, devrenin harfe özgü olması.** D2/D3 bulgusu 70× küçük modelde, gömme karışıklığı giderilerek yeniden üretildi.
- **T1w64** (genişlik 64, 2 kafa): aşama 1 (tek-olgu, 1.500 güncelleme) 1,00; aşama 2 (T1r-b ile aynı bayraklar) 4.000 güncellemede eğitim harflerinde 0,35–0,46'da kaldı, ön-kayıtlı 0,8 ölçütü **ulaşılmadı** (genişlik 128'de 2.000–2.500). Kapasite mi, yavaş oluşum mu, ayrılamıyor (iki genişlik, tek tohum). **T1w256** (genişlik 256, 8 kafa; 21:47'de başladı): **aşama 1 bitti (22:52)**: tek-olgu kopyalama devresi 250–500. adımlar arasında oluştu (genişlik 128: ≈ 750, genişlik 64: 1.000–1.250); yani kopyalama devresinin oluşum adımı genişlikle tekdüze erkenleşiyor (üç genişlik, tek tohum). Aşama 2 (anahtar eşleştirme, 4.000 güncelleme) çalışıyor.

**L1 — sıfırdan 20M parametre, 500M bayt, durmayan uzun koşu (hazır, başlamadı; `mline/longrun.py`, `scripts/run_longrun.sh`).** Fikir kullanıcıdan: parametreyi 20M'ye indirip token'ı 500M'ye çekmek, plato olsa bile sonuna gitmek, çünkü sonda önemli bir şey değişiyor olabilir ve kaçırılmamalı. Model: genişlik 928, 8 kafa, **20.297.472 parametre**, bayt sözlüğü. Eğitim: AdamW lr 6e-4, wd 0,01, kırpma 1; batch 32 × 512 = 16.384 bayt/adım ⇒ 500M bayt ≈ 30.518 güncelleme; WSD programı (ısınma 500, kararlı, son %10'da taban 0,1'e sönüm); satırların ≈ %25'i sentetik (tek-olgu uzak, MQAR 4 rastgele, MQAR 4 gerçek anahtar; `m2-draft` tarifi üç aşamalı); her 1.000 adımda anlık görüntü (≈ 81 MB), her 250'de kontrol noktası, her 1.000'de ölçüm pili (doğrulama bpb; MQAR eğitim harfleri/T–Z/gerçek anahtar; tek-olgu mesafe 0/64/128/256/384/448; dikkat koşulluluğu); güncelleme başına `answer_loss` ve `first_digit_accuracy` günlüğü; aynı koşu olarak sürdürme; **yalnız art arda sonsuz kayıpta durur** (`DIVERGED.txt`). Önceden yazılmış eşikler: ÖĞRENDİ = train_k4_first ≥ 0,8; GENELLER = held_k4_first ≥ 0,6 VE natural_first ≥ 0,6; ERİŞİM = single_fact_exact@384 ≥ 0,9; geçiş = first_digit_accuracy'nin 500 güncellemelik hareketli ortalamasının ilk kez > 0,5 olduğu güncelleme. Tahmini hız 5,5–6 s/adım (≈ 2 gün), gerçek GPU'da ölçülmedi. D6 tarifi kesinleştirince elle başlatılacak; kontrol kuyruğu (B2/B6/B5) L1'in arkasında bekliyor.

### 6.6 Kampanya örüntüleri

1. **Güvenli görünen seçenek daha kötü ölçüldü** (dört kez): düşük lr, denetimli kapı, yumuşak karışım, enjeksiyon. Ortak payda: durumun etkisini sınırlayan mekanizmalar, durumun taşıyabildiğini de sınırlıyor.
2. **Öğrenme oranı çözüm ailesini belirledi** (dört kez): Faz 1 w256 başarısızlığı, U2c, U4-sep, U5-lr.
3. **Eğitim dağılımındaki doğrulama ölçülen özelliğe kördü** (U2c C16, M1 kapı): lr/ayar doğrulamaya bakarak seçilmez.
4. **Manipülasyon kontrolü önce:** bir kol farkı yorumlanmadan manipülasyonun uygulandığı günlükten okunur (M1'de kapı hiç kapanmamıştı).
5. **Probu eğitimden önce kestirmeyle sına** (iğne en yakın olguyla çözülüyordu; MQAR'da eleme).
6. **Uçurum eğitim aralığı etkisi; anahtar eşleştirmesi anahtar dağılımı etkisi:** ikisi de "mimari sınır" gibi görünüp eğitim verisiyle oynatılabildi.
7. **Ders aynı kayıttan ölçü almak:** önceki bir hat aynı duvara çarptıysa onun ne değiştirdiğine bak (karşılaştırma bir sonraki deneyi kendiliğinden üretti).

## 7. Düzeltmeler defteri

Eski metin hiçbir yerde silinmedi; düzeltme ilgili raporda yanında durur. Bu liste bir dizindir.

| Tarih | Ne yanlıştı | Nasıl bulundu / düzeltildi | Yer |
|---|---|---|---|
| 2026-09-16 | "%48,44 sabit sınıf" iddiası | ham tahminler yeniden sayıldı (248/512 ≠ 261 ya da 251) | `2026-09-16-phase2-plan/reviews/` |
| 2026-09-18 | U2-measure'ın dondurma kaydı yoktu | sonradan kayıt (`frozen-posthoc.json`), rapora sınır yazıldı | `notes/U2_RAPORU.md` |
| 2026-09-18 | Sürüklenme: gradyan sıçraması reread'e özgü sanıldı | bağımsız denetim: her beş-durum kolunda var | `notes/DRIFT_MINI_RAPORU.md` |
| 2026-09-20 | D1 ilişki-F1 metriği tavanı ≈ 0,5; `subject_swap` geri dönüşü | denetim; sonraki sürümde düzeltilecek | `notes/D1_RAPORU.md` |
| 2026-09-22 | B2 geliştirme dilimi (1.707 paragraf) iyimserdi (+10 puan vaadi, final +1,1) | final sonucu | `notes/B2_RAPORU.md` |
| 2026-09-23 | U2d "dayanıklı/kırılgan örtüşmesiz" iddiası | lr'ye göre gruplama artefaktı; sonuca göre gruplayınca örtüşüyor, yalnız aralık/sürüklenme ayırıyor | `notes/U4sep_RAPORU.md` §1 |
| 2026-09-23 | U4-sep yanlış büyüklüğü (yayılım) hedefliyordu; ölçek-değişmezlik testi çökme-değişmezliğini kapsamıyordu; 24 adımlık ölçüm yanlış okundu | düşük lr 2/2 şansa çöktü | `notes/U4sep_RAPORU.md` §5 |
| 2026-09-24 | U4-sep §6 "çökme kaçışı yapısal olarak kapalı" | 28 dk sonra 64 adımlık pilot çürüttü; `drift_floor=0,5` | `notes/U4sep_RAPORU.md` §7 D1 |
| 2026-09-24 | U4-sep sayım "2'si hiç başlamadı" | denetim: s143 lr 0,003 18 adım koştu | aynı rapor |
| 2026-09-24 | U4-inject: `inject_mix` "geniş kanal" testi sanıldı; gradyan iddiası | koda bakıldı: durum yalnız sorgu; gradyan ilk 64 adımda aynı | `notes/U4inject_RAPORU.md` §2 |
| 2026-09-25 | M0-padded `REASON.txt` "3358'de durduruldu" | koşu 5.000/5.000 tamamlandı; başlık düzeltildi | `results/M0-padded/REASON.txt` |
| 2026-09-25 | Test koşucusu 42 testi çalıştırmıyordu | pytest'e geçildi | `tests/run_all.py` |
| 2026-09-27 | M1 "kapı biçimi öğreniyor" | kapı hiç kapanmamıştı (hold ≈ 5e-6) | §6.5 |
| 2026-10-01 | "Uzağı yalnız durum taşır" (Core'un dikkati varmış) | mühürlü `core/model.py` okundu | §6.5 |
| 2026-10-01 | "near kontrol ≈ 0,9" notu | `analysis.json`'dan okundu: 0,76 / 0,77 / 0,35 | §6.5 |
| 2026-10-01 | Yem probu öğrenmeyi olduğundan kötü gösterdi (dağılım dışı yerleşim) | eğitim-dağılımı skoru | §6.5 |
| 2026-10-01 | MQAR eleme kısayolu | ilk soruya bakıldı; `--with-replacement` | §6.5 |
| 2026-10-01 | Küçük modelde "T–Z genellemiyor" yapısal sanıldı (gömme karışıklığı) | T1p adil testi: yine genellemiyor | §6.5 |
| 2026-10-01 | D2 "en iyi model" cevabı | D5/D6 geldikten sonra güncellendi | §1 |

## 8. Araçlar ve betikler

- **`horizon/`:** kapı: `gated.py`, `gated_sweep.py`, `gated_trajectory.py`; ölçüm ve müdahale: `trajectory.py`, `interventions.py`, `interventions_runaware.py`; koruma ve bellek: `guards.py` (A8), `remat.py` (A5); U4: `inject.py`, `inject_sweep.py`, `sep_loss.py`, `sep_sweep.py`, `sep_finetune.py`, `geom_loss.py`, `u4sep_compare.py`, `u4sep2_analyze.py`, `u4sep2_start_geometry.py`; B taraması altyapısı: `scan_sweep.py`, `scanable.py`; ortak/yardımcı: `sweep.py`, `metrics.py`, `task_families.py`, `lr_probe.py`, `u2d_compare.py`. Mühürlü kaynaklar düzenlenmez (`gated.py`, `gated_sweep.py`); yeniden uygulamalar bit-eşdeğerliği kanıtlanarak yazılır.
- **`mline/`:** veri ve model: `data.py`, `lm.py`, `needle.py`, `fastcore.py`, `faststep.py`; eğitim: `train.py`, `emergence.py` (M0/M1 sondası; `--init-from/--lr/--train-filler`), `decoy_train.py` (yem-olgu / MQAR ince ayarı), `mqar_scratch.py` (küçük Core sıfırdan), `induction.py`, `longrun.py` (L1); ölçüm: `mqar_eval.py` (eğitim-dağılımı skoru, gerçek anahtarlar), `decoy_probe.py`, `attn_probe.py` (anahtar-koşulluluk), `rope_probe.py`, `channel_probe.py`, `state_swap.py`; ortak: `analyze.py`, `checkpoint.py`, `diagnose.py`.
- **`scripts/`:** `run_decoy_experiments.sh` (GPU kuyruğu v10, boşta ise sırayla D1…D6; yeniden başlatma güvenli), `decoy_midcheck.sh` (CPU ara kontrol; 250 adımda bir; `HATA` yazar), `run_t1*.sh` (T1 serisi), `run_t1_width.sh` (genişlik denemeleri), `run_longrun.sh` (L1, sürdürmeyi korur), `run_distance_controls.sh` (B2/B6/B5 kontrol kuyruğu), `_gpu_busy.ps1`, `calibrate.py`, `vm_run.sh`, `vm_setup.sh`, `vm_watchdog.py`, `pull_from_vm.py` (bulut makinesi araçları; erişim bilgisi betikte durmaz).
- **`semantic_memory/`:** `embedding.py`, `retrieval.py`, `rerank.py`, `multihop.py`, `lm_bridge.py`, `run_b2.py`, `run_b3.py`. **`language/`:** `dataset.py`, `extract.py`, `generator.py`, `run_d1.py`, `run_d1_final.py`, `score_semantics.py`. **`protocols/`:** `g0_pins.py`, `lm_calibration.py`, `batch_calibration.py`, `G0-pins.json`. **`audit/`:** `receipt.py`, `stats.py`, `phase2_manifest.py`, `delivery_integrity.py`, `audit_u4sep*.py`, `audit_u4inject.py`.
- **Tuzaklar (kendi hatalarımızdan):** Bash aracından yazılan kodda `\n` gerçek satır sonuna dönüyor (ters eğik çizgili içerik için Edit/Write); arka plan Bash işleri 10 dk'da sonlanıyor (bekleme için izleyici); çalışan bir betik düzenlenmez (öldür, düzenle, yeniden başlat; betikler idempotent); bir eylemin "bitti" demesi sonucun varlığını kanıtlamaz (çıktı dosyasını kontrol et).

## 9. Sonuç dizinleri haritası (`results/`)

| Dizin | İçerik |
|---|---|
| `G0`, `B1-development`, `B1`, `B1-attempt1-oom`, `B2`, `D1`, `D1-final` | Faz 2 erişim ve dil döngüleri; `*-attempt1-oom` başarısız ilk deneme (korunur) |
| `U1`, `U2-measure`, `U2-intervene`, `U2b-intervene`, `U2c`, `U2c-v2`, `U2c-lr`, `U2c-lr0003`, `U2d` | sürüklenme döngüleri; `U2c` yarım kalan v1 `interrupted-…` altında korunur |
| `U4sep-lr0003`, `U4sep-lr003`, `U4sep2`, `U4inj` | U4 döngüleri (yarıda kesilen koşular karantinada, `REASON.txt` ile) |
| `U5lr`, `U5scan` | M1'den sonra lr ekseni ve B taraması altyapısı |
| `M0-emergence*`, `M0-packed*`, `M0-padded`, `M1` | M0/M1 koşuları ve analizleri (`M1/analysis.json`, `frozen.json`) |
| `M1-diagnosis` | mesafe taramaları, RoPE, kanal, durum takası, yem probu |
| `M1-finetune-distance` | ft384, B2 kontrol, D1…D6, T1 serisi, `queue.log` (kuyruk olayları) |
| `R1`, `review-diagnostics-r6`, `training`, `drift` | R1–R6 ve sürüklenme mini döngülerinden (atalardan) gelen sonuçlar |

Her koşu dizininde (çoğunda): `README.txt` (ön-kayıtlı okuma ve tarif), `train.log`, `probe.jsonl` (güncelleme başına), `decoy.json`, `mqar-eval*.json`, `sweep-*.json`, `attn-*.json`, `selected.pt`/`latest.pt` (Git dışında).

## 10. Model ağırlıkları ve yayın durumu

- **Envanter (2026-10-01, 21:55):** 960 `.pt` dosyası, 12,60 GB, **732 farklı içerik** (83 hash birden çok yolda), 956'sı zip arşivi, okuma hatası yok, 271'i optimizer durumu taşıyor. Kökler: Faz 1 e2e 239 dosya (251 MB), diagnostics 319 (1,12 GB), drift 68 (8 MB), phase2 326 (9,23 GB), `vm-mirror` 8 (1,99 GB). Gömülü metadata taraması temiz (yalnız 4 dosyada yerel Windows yolu).
- **Karar (kullanıcı, 2026-10-01):** eğitilmiş bütün modeller, iyi/kötü ayrımı yapılmadan paylaşılacak; çöken ve yarım kalan koşular dahil. `selected.pt` raporlarda değerlendirilen ağırlıklar; `latest.pt`/`previous.pt`/`resume.pt`/`stepN.pt` eğitim durumu.
- **Plan:** GitHub Release (her dosya < 2 GiB; en büyüğü 374 MB), özgün baytlar korunur (SHA-256 mühür manifestleriyle örtüşür), ad = yolun `/` yerine `__` konmuş hâli, aynı içerik bir kez yüklenir; manifest `docs/model-manifest.json` olarak depoya girer. D6 ve L1 bitince eklenir. Yükleme betiği ve hazırlık dosyaları depo dışında hazır.
- **Durum:** depo yayını (MIT lisanslı, özel depo olarak) ve model yüklemesi hazır; **gönderme adımı otomatik izin sisteminde bekliyor** (kullanıcı onayı sohbetle verildi, ama sistem bu sınıf eylemi izin kuralı ya da kullanıcının kendi çalıştırması olmadan geçirmiyor). Özelden açığa çevirme kullanıcının kendi komutuyla yapılacak.
- **Güvenlik:** gönderim öncesi denetimde tam tarama (token, özel anahtar, IP, kullanıcı@IP, e-posta) sıfır; izlenen ağırlık/dizi/anahtar dosyası yok; VM erişim dizgileri hiçbir commit'te yok; depodaki dosyalar mühür manifestlerindeki SHA-256'larla bayt bayt örtüşüyor (yalnız belgelenmiş eski sürüm manifestlerde 18 fark). Hâlâ açık: commit'lerde yazar adı ve e-posta herkese açık olacak; 112 commit'in 106'sında oturum bağlantısı var; notlardaki serbest metinde kişisel bilgi desen taramasıyla görülemez.

## 11. Açık sorular ve sıradaki işler

1. **D6 bitişi** (≈ 23:00): son skor (gerçek anahtarlar, T–Z, tek-olgu mesafe, dikkat); sonra L1 tarifini kesinleştir.
2. **L1'i başlat** (elle; ilk 100 adımda gerçek adım süresini ölç, tahminleri güncelle).
3. **T1w256** sonucu: genişlik 64'te oluşmayan devre 256'da erken mi oluşur (üç nokta: 64, 128, 256).
4. **Kapının geçerli testi** (kullanıcı onayı bekliyor): kapanabilir kapı (HOLD önyargısı, normlanmış girdi, kapıya özel lr), dikkati sınırlı bir görev (yoksa iğne görevi durumu hiç gerektirmez), manipülasyon kontrolü ve `MANIPULATION_NOT_APPLIED` kuralı; ayrı döngü, ayrı önkayıt.
5. **B taraması** (`feedback=False` + `gate_input='input'`, lr 1e-3, ≈ 5 saat CPU): kullanıcının sözü bekleniyor.
6. **Gerçek anahtarlarla uzak mesafe:** D6'nın dolgusu 0–128; ≥ 256'da gerçek anahtarlı eşleştirme ve doğal metin dolgusu ölçülmedi.
7. **Kontrol kuyruğu** (B2 / B6 doz-yanıt / B5 tekrarlar) L1'in arkasında bekliyor; ft384'ün karşılaştırma kolu eksik (aynı tarif, dolgu 0–64).
8. **Düşük öncelik:** B3 (LM köprü çıkarıcı; önce birleşim tavanı), D2/D5 üretici batching (sessiz makine ister), Windows zamanlanmış görevi (L1 ve kuyruk kesintiden sonra kendiliğinden devam etsin; kullanıcı onayı bekliyor).
9. **Masaüstü raporuna "Ek C"** (D1b–D6 ve T1 sonuçları) yazılacak.
10. **Yayın:** gönderme izni; model yüklemesi; yüklemeden sonra manifest + README güncellemesi; açığa çevirme.

## 12. Kaynaklar ve ilgili ayrı proje

- RLT fikri ve resmî kodu: <https://github.com/yifanzhang-pro/recurrent-looped-tranformer> (Faz 1'de sabitli commit `cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39`). Bu depo onun yeniden üretimi değildir.
- U4-inject: FLT, arXiv 2605.18797 (uyarlama; bkz. §6.4 kusur notu). Mesafe araştırması: Engram (DeepSeek; hash tabanlı N-gram koşullu bellek) M1 mesafe sorununu çözmez, kapı normalizasyonu ve mesafe eğitimi ipuçları verdi. Mamba-3 ve DeepSeek-V2 MLA: RLT-2 projesinin kaynakları.
- **RLT-2 15M (`2026-09-20-rlt2-15m-2b`, ayrı proje ve ayrı git deposu):** üç modelin her biri 2 milyar eğitim token'ı: RLT-2 (14.989.056 parametre), RLT-2 + Mamba-3 (14.980.296), RLT-2 + Mamba-3 + MLA (14.977.352); toplam hedef 6 milyar token, 512 token bağlam, 16.384 hedef token'lık etkili güncelleme, FineWeb-Edu sample/10BT (sürüm `87f09149…`). Veri hazırlığı **tamamlandı** (`data/production/progress.json`: 1.646.688 belge görüldü, 10.179 tekrar, eğitim 2.000.000.000 token / 1.634.842 belge, doğrulama 2.000.000 token / 1.667 belge, 5.649 s). Eğitim sonuçları bu belgeye girmemiştir: eğitim durumu bu oturumda incelenmedi; README "bir dosyanın veya çalıştırıcının bulunması eğitimin tamamlandığı anlamına gelmez" der. Depoda makale PDF'i (`references/rlt/`) izleniyor; yayın kararı ayrı (üçüncü taraf telif).

