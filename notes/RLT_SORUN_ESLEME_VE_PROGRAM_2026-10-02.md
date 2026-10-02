# RLT iyileştirme programı: makale sorunları ↔ 64'lük havuz ↔ kanıtımız ↔ matematik ↔ deney kuyruğu

2026-10-02 06:50 (yerel). Kullanıcı kararı (06:00): *çözülemeyen ve kısmen çözülen problemleri araştırma ve matematikle çöz; RLT makalesinin sorunlarıyla karşılaştır; amaç RLT'yi mantığını ve tekniğini bozmadan tamamen iyileştirip GERÇEK bir model eğitmek, laf olsun diye iş yapılmıyor.*

**Bu not yeni bir sonuç taşımaz.** Okunan şeyleri (makale PDF'i, 64'lük havuz, ana belge) eşler, türetmeleri ve ÖNGÖRÜLERİ sonuçlardan ÖNCE yazar (bu dosyanın commit zamanı ön-kayıttır), deney kuyruğunu sıralar. Kanıt etiketleri: **[V]** bu oturumda ham kaynaktan (yerel makale metni, ana belge, sonuç dosyası) okundu; **[S]** ikincil (bir özetleyici aracılığıyla okundu, doğrulanmadı); **[R]** hafızadan literatür, açılmadı; **[T]** kendi türetmem, belirtilen varsayımlar altında; **[H]** öngörü/hipotez.

Kaynaklar: makale `__RLT_ROOT__\papers\Recurrent_Looped_Transformer_Zhang-Feng-Qin_2026-09-17.pdf` (+ `.txt`; Zhang, Feng, Qin; 12 Eylül 2026, **17 Eylül revizyonu**; satır numaraları `.txt`'e göre); problem havuzu `docs/external/rlt_gelistirme_v0_1/source/problem_havuzu.md` (18 Eylül; sonraki M-hattı, Z1 ve L1 bulguları havuzda YOK, aşağıda N1–N10 olarak eklendi); dış çözüm tasarımı aynı klasörde `RLT_64_MADDE_COZUM_TASARIMI.md` (kendi beyanı: asıl projeye uygulanmadı, kazanımlar ölçülmedi; burada önerilerinden biri olarak okunur).

## 1. Kısa cevap

1. **Makalenin kendi sorunları** (§2'de tablo): sıralı kod çözücü maliyeti (eğitimde ve prefill'de), toplamada eğitilen genişliğin ötesinde çöküş, düz mod-5'in ilklemeye duyarlılığı, standart S5'in öğrenilememesi, derinlik bölünmesinin göreve bağlılığı, eğitim kararsızlığı, karışık karşılaştırma ve yalnız sentetik görevler. **Bağımsız değerlendirme** ("Does Recurrence Pay?", 15 Eylül; **[S]**) dil kaybında kazanç bulmadı (50M: Δ = 0,0004 nat; 140M: RLT 0,0464 nat kötü) ve ≈ 20× GPU saati gördü.
2. **Bizim kanıtımızla örtüşme:** makalenin 10 sorunundan 7'si havuzdaki A/D/F maddelerinde zaten vardı; 3'ü (toplama, derinlik bölünmesi, S5-swaps) bizde hiç ölçülmedi.
3. **Çözülmüş sayılabilecekler** (sınırlarıyla): sabit durumlu görevlerde uzunluk genellemesi (parite, S₃, S₄: Core 1.024'e kadar hatasız, tek eğitim tohumu); boş koşuda sürüklenmenin açık HOLD/WRITE kapısıyla giderilmesi (oyuncak, genişlik 32); plato/tohum sorununun uzunluk müfredatıyla aşılması (S₄ ve S₅'te; makalenin "standart S5 zor" bulgusu bizim kurulumda öğrenilebildi, bkz. M4).
4. **Çözülmeyenler ve kısmen çözülenler, öncelik sırasıyla** (neden bu sıra: ucuzluk × gerçek modelin tasarımını belirleme gücü): (a) S₅'te 4–8× kopuşunun nedeni; (b) durum kalıcılığı ve adım-başı hata tabanı; (c) derinlik (sayaç) dışdeğerlemesi, yani RLT'nin "sonsuz derinlik" iddiasının asıl testi; (d) zaman ekseninde paralellik/hız (gerçek model eğitilebilirliği); (e) mesafe uçurumu ve içerik-adresli erişim; (f) kapının LM ölçeğinde geçerli testi; (g) gerçek model (eşleşmiş transformer tabanıyla).
5. **Dürüst konumlandırma:** RLT'nin yapısal değeri durum-izleme görevlerinde aranmalı (makale de orada kazanç buluyor), dil kaybında değil (bağımsız değerlendirme bulamadı; kendi L1'imiz bu soruyu sormuyor). Gerçek model, RLT'nin yapısal olarak ayrıldığı görev ailesini (S_N bileşimi, aritmetik, uzun nötr koşulu durum izleme) eşleşmiş bir transformer tabanıyla karşılaştırmalı.
6. **Fundamental bir ödünleşim** (§4.5, **[T]+[R]**): tam RLT geri beslemesi (önceki tokenin SON kod çözücü durumu) + zaman ekseninde paralel eğitim + keyfi uzunlukta tam S₅ izleme **birlikte olamaz** (TC⁰ ≠ NC¹ varsayımı altında). Seçenekler: sıralı RLT-1 (derinlik T), parça-düzeyi RLT-2 (T/B), birleşme-uyumlu durum şeridi (log T). Bu, hız çabasının RLT'nin "mantığını" ne kadar koruyacağına dair açık bir karar noktasıdır.

## 2. Makalenin kendi sorunları

| # | Makalenin ifadesi (kaynak) | Sayılar | Bizim havuz / kanıt | Durum |
|---|---|---|---|---|
| M1 | Sıralı kod çözücü güncellemesi eğitimde ve prompt prefill'de zorunlu (özet l.14; §2.6 l.261; B.2 l.944-946: "T sequential decoder transitions… can reduce hardware utilization") **[V]** | RLT-2 (parça-düzeyi geri besleme) 4+4: chunk4 2,27×, chunk8 3,12-3,37×, RLT-0 4,17-4,30× daha hızlı (CPU, mod-5, adım 1.001-2.000; J.9 l.2260) **[V]**; kalite bedeli: düz mod-5 uzunluk 63, RLT-1 7+1 %99,61, chunk4 %18,95, chunk8 %43,75, RLT-0 %18,85, Transformer %34,96; chunk4 4+4 %92,68 (l.2253-2256) **[V]**; RLT-2 yalnız J.9'da, tek tohum, yalnız mod-5 | D1 (YAPISAL), D2, D3; L1: 6,25-7,9 s/adım, 16.384 bayt/adım ≈ 2,1-2,6K bayt/s | AÇIK → E-H |
| M2 | Toplama: eğitilen operand genişliğinin ötesinde doğruluk düşüyor (l.451-455) | 9 basamak: RLT-1 7+1 %68,05±3,64 vs Transformer %58,61±2,44; 32 basamak: hepsi %14,89-16,84 **[V]** | bizde aritmetik görevi yok | AÇIK (L2 üreteçleri) |
| M3 | Düz mod-5 ilklemeye duyarlı (l.419-423) | 4+4: %17,58 / %19,53 / %98,96 (tohum 42/43/44) **[V]** | A7 (tohum piyangosu), A10 (plato); S₄ 4.000 güncelleme sabit uzunluk ✗, müfredat + 15.000 ✓ | KISMİ: müfredat platoyu aşıyor, tohum tekrarı sürüyor (kuyruk 5) |
| M4 | Standart S5 zor (l.438-440, 472-473) | 32 permütasyon, 2.000 adım, tepe öğrenme oranı 1e-4 (l.295, 314); eğitim aralığında bile son-durum %0,52-2,47 (şans %0,83); 512'de %0,72-1,43 **[V]** | Bizim Core (410k parametre, müfredat 1→16, 30.000 güncelleme, lr 3e-3) S₅ standardını eğitim aralığında öğrendi (1,00), 128'e kadar ≥ 0,95, 512'de 0,54; LSTM w208 0,56, GRU w256 0,32 **[V]** | KISMİ-ÇÖZÜLDÜ: makalenin başarısızlığı aralık-içi, yani bütçe/lr/müfredat etkisi (hangisi ayrılamaz: genişlik, lr, müfredat birlikte farklı). RLT'ye özgü üstünlük S₅'te yok (LSTM ≈ Core) |
| M5 | Tercih edilen kodlayıcı/kod çözücü bölünmesi göreve bağlı (l.473-475) | parite 5+3 ve 7+1; S5-swaps 4+4 | bizde tek bölünme (1+1) | AÇIK |
| M6 | Eğitim kararsızlığı: parite 5+3, tohum 42, %100 → %48,05 → %100 (l.503-506) | | A8 (gradyan sıçramaları; sayaç GRU P=32'de kırpma adımların %98'inde) | KISMİ |
| M7 | Karşılaştırma geri besleme, dikkat yapısı, parametre ve hesabı birlikte değiştiriyor; RL ölçülmedi (l.508-512) | RLT-1 26,10-28,73M vs Transformer 25,31M parametre | bizde `core_nofb`, GRU, LSTM, NoPE-transformer; parametre-eşit S₄/S₅; hesap eşit değil (Core 5.213 s, LSTM 1.313 s) | KISMİ |
| M8 | Uzunluk dışdeğerleme başarısızlıkları: mod-5 255'te hepsi ≈ şans; S5-swaps 512 işlemde yalnız 4+4 %55,70±25,78 (l.461-475) | Transformer %0,85±0,30 | N1 (S₅ kopuşu), N2 (sayaç) | AÇIK → E-A, E-B, E-C |
| M9 | Eksik/devam eden koşular; 2.000 adım; derinlik-16 ve mod-5 varyantları tek tohum (J.x) | | bizim pilotlar da tek tohum | KURAL: dondurulmuş protokolde ≥ 3 tohum |
| M10 | Kapsam: yalnız sentetik algoritmik görevler; dil sonucu yok; RL ölçülmedi | | M-hattı/L1 bayt-dil | AÇIK → gerçek model |
| X1 | Bağımsız değerlendirme **[S]**: kayıp 50M 3,8845 vs 3,8841; 140M 3,6884 vs 3,6420; LAMBADA 0,147 vs 0,163; 8,4K vs 124K token/s; ≈ 22 sa vs 1,14 sa (500M token); bağlam-uzunluğu dışdeğerlemesinde kazanç yok; sınırlar: ≤ 140M, 500M token, tek veri karışımı, "açık kalıcı durum isteyen görevler" ele alınmadı | | L1'imiz eşleşmiş transformer tabanı olmadan koşuyor | AÇIK → L2'de eşleşmiş taban şart |

Not: makale sayfasının başka bir yerinde (özetleyici **[S]**) 79K parametreli bağımsız bir topluluk denemesi var: 128 işlemde parite %60,8, beş-durum %20,7 (şans %50 / %20): zayıf kanıt, tek kaynak.

**Makale ile bizim farklarımız (karşılaştırmanın şartları):** makale 8 katman, genişlik 512, 26-29M parametre, 2.000 adım, küresel batch 512, tepe lr 1e-4, AdamW (0,9; 0,95), wd 0,1 (l.287-298); bizim Core 1+1 katman, genişlik 32-1.152, tek tohum, lr 3e-3 (GRU/LSTM S₅: 1e-3), batch 64, uzunluk müfredatı. Görev tanımları S5'te örtüşüyor (standart: 120 permütasyon; bizde `perm5` aynı); S5-swaps (özdeşlik + 10 transpozisyon) bizde yok.

## 3. 64'lük havuz: kanıt durumu (A1-G7) ve eklenenler (N1-N10)

Durum sözlüğü: **ÇÖZÜLDÜ** (ölçülmüş çözüm, sınırlarıyla), **KISMİ**, **ÇÖZÜLMEDİ** (denendi, hedef karşılanmadı), **AÇIK** (denenmedi/ölçülmedi), **KURAL** (süreç dersi, kural olarak kapatıldı), **KARAR** (kullanıcı kararı bekliyor), **KISIT** (donanım/yapısal kısıt; yönetilir, çözülmez). Kanıt kaynağı: havuz metni + ana belge (bu oturumda okundu); "→" sonraki adım.

### A. Çekirdek dinamik (RLT'nin kendisi)
| ID | Konu | Durum | Kanıt | → |
|---|---|---|---|---|
| A1 | Boş koşuda sürüklenme | KISMİ | U2c `hard_st`: onarım %94/%79 (beş-durum), %99/%92 (yığın), iki lr, genişlik 32; GRU da kayıyor (Z1-G, R2: 52-74 puan kayıp); LM'de kapı SINANMADI (M1: hold ≈ 5e-6) | E-G |
| A2 | Üç ucuz çare | KURAL (kapandı, negatif) | pencere 16 / değişmezlik kaybı / durum-geçmişi dikkati: 3/3 `MEASURED_NEGATIVE` | |
| A3 | Zarar görünürlüğü göreve bağlı | KURAL | her müdahale iki cebirde (onarılabilir/onarılamaz) sınanır | |
| A4 | Başlangıç kararlı, bilgi taşıyan durum değil | AÇIK | Jacobian/Lyapunov/gürültü marjı ölçülmedi | E-B |
| A5 | Uzak kredi ataması | ÇÖZÜLMEDİ | tbptt8-final'de başlangıç parametresi güncellenmedi; M-mesafe: uçurum eğitim aralığıyla ≈ 6× kaydı (64 → ≈ 448) | E-A, mesafe-çeşitli eğitim |
| A6 | Büyüyen bellek (yığın/sayaç) | KISMİ | yığın eğitim ufkunda %94, 4.096'da %72-81; sayaç: Core/GRU/LSTM eğitilen tepeye doyan sayaç (P = 8, 16) | E-C |
| A7 | Tohum piyangosu | KISMİ | pilotlar tek tohum; S₄ tohum 2-3 kuyruk 5'te | kuyruk 5 |
| A8 | Gradyan sıçramaları | KISMİ | kırpma var, kararlılık kapısı yok; sayaç GRU P=32'de kırpma 11.813/12.000 adımda | kararlılık günlüğü |
| A9 | Öğrenme oranı duyarlılığı | AÇIK | kol başına lr taranmadı; makale lr 1e-4, bizde 3e-3 / 1e-3 | E-F |
| A10 | Geç öğrenme (plato) | KISMİ-ÇÖZÜLDÜ | uzunluk müfredatı S₄ (15.000) ve S₅ (30.000) platoyu aştı; sayaç P=32 GRU 12.000 güncellemede hâlâ aralıkta öğrenmedi | E-C (tepe müfredatı) |

### B. Doğrulama, başlık ve bellek katmanı (RLT çevresindeki sistemler; gerçek model yolunda ikincil)
| ID | Konu | Durum | Kanıt |
|---|---|---|---|
| B1 | Doğrulayıcı kapsamı | ÇÖZÜLMEDİ | Faz 1: IID kabul %39, yeni derinlikte ≈ 0 |
| B2 | Anlamsal erişim | ÇÖZÜLMEDİ | recall@10 0,665 (hedef 0,95) |
| B3 | Öğrenilmiş ayrıştırma yeni kalıba aktarılmıyor | ÇÖZÜLMEDİ | D1 ilişki-F1 0,03, yanlış-kabul ≥ 0,97 |
| B4 | Seçici revizyon | ÇÖZÜLMEDİ | ≥ %10 kazanç kapısı geçilmedi (Faz 1) |
| B5 | Sözcük-karma erişimi | ÇÖZÜLMEDİ | recall@5 %4 |
| B6 | Sıkıştırma | KISMİ | son z saklanıyor; anlamsal özet 37 → 21 puan |
| B7 | Başlık grafiği yeni derinlikte | ÇÖZÜLMEDİ | IID %96, derinlik 8 %10 (ara gözetimle %33) |
| B8 | Symbolica/Graphica | KARAR | park (G7) |

### C. Ölçüm ve yöntem
| ID | Konu | Durum |
|---|---|---|
| C1 | Test kümesi yeniden kullanımı | KURAL (Z1: dev/test ad alanı ayrı; test açılmadı) |
| C2 | Üç tohum, aralıklar | KISMİ (pilotlar tek tohum) |
| C3 | Türkçe dilimi yok | AÇIK / KARAR (G3) |
| C4 | Model-yazımı veri riski | KURAL (L2: programatik üreteç; etiketle) |
| C5 | Tek model hakemi | AÇIK (D2 yok) |
| C6 | Nicemleme sadakati | AÇIK (RLT hattından bağımsız) |
| C7 | Kısmi okuma dersi | KURAL |
| C8 | Test paketi ham dosya ister | KURAL (bilinçli) |
| C9 | Çoklu uç nokta | KURAL (eşikler ön-kayıt: L1, Z1) |
| C10 | Kampanya birleştirme | KURAL |

### D. Donanım, performans, ölçek
| ID | Konu | Durum | Kanıt | → |
|---|---|---|---|---|
| D1 | Zaman ekseni paralelleşmez | KISIT (yapısal; makalenin de 1 numaralı sınırı) | bağımsız değerlendirme ≈ 20× **[S]**; L1 20M: 6,25-7,9 s/adım | E-H, §4.5 |
| D2 | Kernel-başlatma darboğazı | KISIT | CUDA grafiği 1,20×, batch 8 → 32 ile 2,70× (M-hattı) | |
| D3 | Detach-yalnız TBPTT tasarruf etmiyor | KISIT | segment yürütmesi L1024'te 0,53× | |
| D4 | 8 GB VRAM, 4 çekirdek | KISIT | L1: 500M bayt ≈ 53 saat | |
| D5 | Değerlendirme maliyeti | KISIT | | |
| D6 | Bant genişliği/disk | KISIT | C: 3,1 GB boş, E: 31 GB | |
| D7 | CPU çekişmesi | KURAL | ≤ 2 deney süreci | |
| D8 | 16k/64k uzunlukları açılmadı | AÇIK | | |
| D9 | Reranker OOM | ÇÖZÜLDÜ | taban model + yalnız son konum lm_head | |

### E. Süreç ve altyapı
E1-E8, E12: KURAL (bire bir ders olarak kapatıldı). E9 (git remote yok): ÇÖZÜLDÜ (GitHub 2026-10-01). E10 (kod tekrarı): AÇIK (bilinçli). E11 (dağınık test paketleri): KISMİ (`tests/run_all.py` var; bu oturumda koşulmadı).

### F. Faz 2 planı ve iddialar
| ID | Konu | Durum | Kanıt |
|---|---|---|---|
| F1 | B1 hedefleri | ÇÖZÜLMEDİ | en iyi recall@10 0,665 / kapsama 0,325 |
| F2 | D1 hedefleri | ÇÖZÜLMEDİ | ilişki-F1 0,03 |
| F3 | D2 kapsamı | KARAR | G2 |
| F4 | Gerçek RLT dil eğitimi yerelde fizibilite | KISMİ | L1 ölçtü: 20M, 500M bayt ≈ 53 saat; 30M → 300M yerelde değil |
| F5 | U3 biçimsel güvence | AÇIK | nötr geçiş daraltıcı değil → kimlik/çekici koşulu (bkz. §4.1); E-B ölçümü gerekir |
| F6 | U1 k ve d eksenleri | KISMİ | d (derinlik) = Z1; k (başlık turu) açık |
| F7 | "Genel doğal dil" iddiası için kanıt yok | AÇIK | tüm sonuçlar sentetik/kontrollü |
| F8 | LongMemEval-V2, Huginn karşılaştırması, enerji | AÇIK | yapılmadı |

### G. Açık kararlar
G1 (sürüklenme için mimari hamle): **kısmen yanıtlandı**: mimari kapı oyuncakta pozitif, GRU'dan ayrışıyor; LM'de AÇIK. G2, G3, G5, G6, G7: KARAR (kullanıcı). G4 (yedekleme): ÇÖZÜLDÜ (GitHub).

### N. Havuza eklenenler (M-hattı, Z1, L1'den; 18 Eylül sonrası)
| ID | Problem | Durum | Kanıt |
|---|---|---|---|
| N1 | S₅'te Core'un 4-8× kopuşu: nedeni bilinmiyor (dikkat? durum?) | AÇIK | pencere başına ilk-hata ×1e-4: 0·0·0·3,5·23,5·26,2·17,7; GRU/LSTM baştan sabit taban; S₄'te aynı mimari 1.024'te sıfır **[V]** |
| N2 | Sayaç: derinlik dışdeğerlemesi yok (doyan sayaç) | ÇÖZÜLMEDİ | P = 8: Core/GRU/LSTM ilk-iniş 0,064/0,058/0,054/0,058/0,061 aynı; P = 16 GRU/LSTM 0,029-0,032; yalnız gerçek derinlik 0'da doğru; P = 32 GRU 12.000 güncellemede aralıkta öğrenmedi (prefix 0,873; ilk-iniş 0,015) **[V]** |
| N3 | Mesafe uçurumu = eğitim-aralığı etkisi; iğne görevi konuma kilitli | KISMİ | uçurum 64 → ≈ 448 (6×) ince ayarla; yem-olgu probu: en yakın olgu **[V]** |
| N4 | Anahtar eşleştirme harfe/anahtar türüne bağlı | KISMİ | D5/D6: görülmemiş gerçek anahtarlar 0,28 → 0,957 yalnız eğitime karıştırılınca; T–Z rastgele 0,38 **[V]** |
| N5 | Kapı LM ölçeğinde geçerli sınanmadı | AÇIK | M1: hold 4,8e-6, kapı hiç kapanmadı **[V]** |
| N6 | Transformer tabanı (NoPE, 4 katman, w64) S₄/S₅'i öğrenemedi | KISMİ | S₄ 60.000 güncelleme müfredatla bile uzunluk 8'de takıldı; taban zayıflığı mı, tabanın kötü ayarı mı (konum kodlamalı/geniş denenmedi) | 
| N7 | RLT eğitim hızı | KISIT | = D1; L1 ölçüsü |
| N8 | L1 eşikleri | DEVAM | 2.000. adım pili 0/3; tek-olgu getirme dolgu 0-64'te çalışıyor, ≥ 128'de 0 **[V]** |
| N9 | Core'un S₄ avantajı tek eğitim tohumuna dayanıyor | KISMİ | kuyruk 5 (tohum 2-3) |
| N10 | GRU/LSTM sabit hata tabanı ≠ Core kopuşu: mekanizma farkı | AÇIK | hazard tablosu `results/Z1-pilot/hazard-by-window-s4-s5.txt` |

**Sayımlar (tablolardan hesaplanır, aşağıdaki "Sayım" bölümünde).**

## 4. Matematik

Tüm türetmeler Core (kod çözücü katman sayısı L_D = 1) için; makalenin RLT-1'i aynı yapıdır (α = 0,1, merge: u_t = e_t + α·σ(W_g[e_t; r_{t-1}] + b_g) ⊙ W_s r_{t-1}, r = RMSNorm(s)) [V].

### 4.1 Durum haritası, bilgi taşıma koşulu ve neden çekici yapısı gerekli (A1, A4, F5, N5)
s_t = Φ(u_t; M_t, C_{t-1}) ve u_t = e_t + α g_t ⊙ (W_s r_{t-1}). Önbellek yolları hariç Jacobian
J_t = ∂s_t/∂s_{t-1} = (∂Φ/∂u) · α [diag(g_t) W_s + diag(W_s r) diag(g_t(1−g_t)) W_{g,r}] · (diag(w)/ρ)(I − ŝŝᵀ/d), ρ = rms(s_{t-1}), ŝ = s/ρ.
Normdan ‖J_t‖ ≲ α ‖∂Φ/∂u‖ ‖W_s‖ ‖w‖∞ / ρ **[T]**. α sabit (0,1) ve artık akış normu ρ eğitimde büyür; kalıcılık için ‖W_s‖/ρ ≈ 10 mertebesinde olmalıdır: öğrenilebilir ama **eğitim baskısı ister**. Nötr koşuda baskı yoksa daraltıcı çözüm seçilir.
*Bilgi taşıma koşulu:* k adım sonra s_{t+k}'nin s_t'ye bağımlılığı ∏ J. Bilgi ancak şu iki yapıdan biriyle taşınır **[T]**: (i) *sürekli (çizgi) çekici*: bilgi taşıyan altuzayda tekil değerler ≈ 1, dik yönlerde < 1; ince ayar ister, gürültüyle kayar (Seung 1996 **[R]**); (ii) *kesikli izole çekiciler*: her yerde büzülen J, bilgi "hangi havzada" olduğunda; gürültüye dayanıklı. Gözlem "nötr koşuda durum 4-16 adımda 1-2 çekiciye çöküyor, tahminlerin %79-85'i tek sınıfta" = havzaların birleşmesi **[H]** (eğitim nötr koşuyu göstermediğinden havza sayısı yetmiyor). U2c kapısı bu yapıyı mimariyle zorlar (z = hold·prev + (1−hold)·yeni: kimlik yolu). M1'de kapı kapanmadığı için (hold ≈ 5e-6) **kapı LM ölçeğinde sınanmadı**; kapıyı kapanmaya zorlayan şey eğitim görevinde tutma baskısı olmasıdır (N5).
*Öngörü [H]:* S₄'te Core'un 1.024'te sıfır hatası (203.200 dizi-adım) kesikli çekici çözümüdür; yani nötr/sapma gürültüsüne karşı yeterli havza marjı vardır. Bunu ölçmek: durum gürültüsü marj eğrisi (E-B).

### 4.2 Hata tabanı ve marj: neden küçük marj artışı büyük güvenilirlik getirir (N10, A10)
Her adımda durum göreli σ gürültüsü alıyorsa (yaklaşım/öğrenilmiş haritanın pürüzü) ve havza yarı genişliği r ise (m = r/σ), havzadan çıkma olasılığı Gauss varsayımıyla p(m) = erfc(m/√2) **[T; varsayımlar: gürültü Gauss, havza küresel, adımlar bağımsız]**; uzunluk L için hatasız kalma olasılığı ≈ exp(−p L), hazard λ = p(m). Hesaplandı (erfc'in tersi): λ = 2,4e-3 → m = 3,04; 1e-3 → 3,29; 1e-4 → 3,89; 1e-5 → 4,42; 1e-6 → 4,89; 1e-9 → 6,11. Sonuç: **hata tabanını 1e-3'ten 1e-6'ya düşürmek marjı yalnız ×1,49 artırmak demektir**; L = 4.096'da hatasız kalma olasılığı λ = 1e-3'te %1,7, λ = 1e-6'da %99,6. GRU/LSTM'in sabit tabanı (S₄: 1,15e-4…7,6e-4; S₅: 5e-4…3e-3) ve Core'un S₅'te ≈ 2,4e-3'e çıkan tabanı bu marj ölçeğinde m ≈ 3,0-3,9'a denk gelir. *Mühendislik çıkarımı [H]:* marjı büyüten her şey (durum gürültüsü enjeksiyonuyla eğitim, ayrıklaştırma/VQ, "snap" katmanı) hazard'ı kuvvetli biçimde düşürür; bu, durum kalıcılığı için **en ucuz** müdahaledir (yalnız eğitim değişir). Test: E-D.

### 4.3 Dikkat seyrelmesi ve log-n ölçeği (N1, N3, A5)
Doğru anahtarın logit farkı g (rakip ortalamasına göre), n anahtar varken ağırlık a_n = e^g/(e^g + (n−1)). Çözülebilirlik için a_n ≥ θ ise n ≤ n* = 1 + e^g(1−θ)/θ (θ = 0,5: n* = 1 + e^g). Eğitim n_tr = 16 ise gereken g ≥ ln 15 = 2,71; model marjı gereğin ln k nat üstündeyse **kopuş çarpanı ≈ k** (4-8× kopuş = 1,39-2,08 nat fazla marj) **[T]**. *Log-n ölçeği (Chiang & Cholak 2022 **[R]**):* logitler κ(n) = max(1, ln n / ln n_tr) ile çarpılırsa g' = κ g; a_n ≥ θ ⇔ g ln n / ln n_tr ≥ ln(n−1) + ln(θ/(1−θ)); büyük n'de bu g ≥ ln n_tr'ye iner: **eğitim aralığında çözüm marjı yeterliyse (g ≥ ln n_tr ≈ 2,77) seyrelme kopuşu n'den bağımsız ortadan kalkar [T]**. κ(256) = 2,0; κ(1.024) = 2,5. Bu, RLT çekirdeğini değiştirmeyen (yalnız logit sıcaklığı) ve ÜCRETSİZ (eğitimsiz) bir dışdeğerleme müdahalesidir.
*RoPE ayrı bir etkidir:* eğitilen göreli uzaklık aralığının ötesindeki kayma açıları dikkat desenini bozar (M-hattında PI aralık içini bile yıktı, NTK ×2 uçurumu 72 → ≈ 100 kaydırdı ama uzaklık 0'ı bozdu). Seyrelme yalnız n'ye, RoPE ise uzaklığa bağlı; **pencere kırpma her ikisini de eğitim aralığında tutar, log-n yalnız birincisini düzeltir**: E-A bu ikisini ayırır (w16 / temp karşılaştırması).
*Öngörü [H]:* 4-8× kopuş ln k ≈ 1,4-2,1 nat marjla uyumlu, CE kaybının sürdürdüğü büyük marjla (∼ ln(1/kayıp)) değil; bu yüzden dikkat seyrelmesi tek başına ≈ 1.000× kopuş vermezdi → salt seyrelme açıklaması şüpheli, RoPE/OOD ya da durum hassasiyeti de olası. E-A bunu karar noktasına getirir.

### 4.4 Sayaç: kod kapanışı, tanh doyması, dönel sayaç (N2, A6)
*Kod kapanışı:* eğitimde derinlik P'yi hiç aşmıyor (tepede yansıtıcı), yani "P'de it" geçişi eğitimde YOK. Model {0..P} kodunu öğrenir; kod sonlu olduğundan P'de it → P'de kalma (doyma) kodun doğal kapanışıdır, "en basit çözüm" argümanından bağımsız **[T]**. Sonlu bir eğitim dağılımı sınırsız sayaç ile (P+K+1) durumlu otomatı ayırt etmez; dışdeğerleme çarpanını **kodun öğrenilmiş yapısı** (kafes mi, uzatılabilir çizgi mi) belirler. Çizgi çözümünü seçtirmenin yolu *ölçek-çeşitli* eğitimdir (tepe/aşma derinliği log-tekdüze çekilir: büyük bir kafes pahalı, çizgi ucuz) **[H]** (E-C).
*tanh doyması ve LSTM ≈ 2,5× (R4 uyarısı 06:47: bu BİR HİPOTEZDİR; Weiss vd. sapmanın nedenini kapı eşitsizliği birikimi olarak verir ve iki mekanizma ayrılmadı; 2,56 ile 2,67 tesadüf olabilir):* LSTM'de h = o ⊙ tanh(c) ve hücre c sınırsız toplanabilir, ama okuma tanh'tan geçer: 1 − tanh(c) < 1e-3 için c > 3,8, < 1e-7 için c > 8,4 (float32'de tanh(8) = 1 − 2,3e-7). Eğitim aralığı P'nin c ≈ 3'e haritalandığı bir ölçekte (δ = 3/P adım başına) doygunluk c ≈ 8'de başlar: **dışdeğerleme çarpanı ≈ 8/3 ≈ 2,7 [T/H]**; literatürdeki "LSTM ≈ 2-2,5×" (a^n b^n n ≤ 100 → 256; **[R]**, ajan notu, doğrulanmadı) ile uyumlu: aritmetik değil okuma doygunluğu sınırı. GRU durumu [−1, 1]'de sınırlı: sınırsız sayaç yok. **Core:** geri besleme RMSNorm'dan geçer (ölçeğe duyarsız): sayım yalnız YÖN olarak taşınabilir. Dönel sayaç: itme +ω_i, çekme −ω_i açısı (RoPE'ye benzer çoklu frekans bankası; tek frekans 2π/ω'da örtüşür, ω_i'ler artan frekans oranlarıyla çarpımsal olarak aralığı büyütür); norm korunur, itme-çekme tam iptal olur (R(−ω)R(ω) = I), kayıp yok **[T]**. Tabanda ise zemin (0'da çekme sabit) dönel haritayı bozar: "derinlik 0" testi tüm fazların ≈ 0 olması olarak okunmalı (MLP). Bu bir *tasarım adayı*dır (E-I), önce E-C ve E-D bakılır.
*Not:* P = 32 GRU'nun aralıkta öğrenmemesi (12.000 güncelleme) bir ekstrapolasyon bulgusu DEĞİL, bütçe/plato: sayaç ancak ilerleyici tepe müfredatıyla (8 → 16 → 32 → 64) anlamlı sınanır.

### 4.5 Zaman paralelliği: bağımlılık grafiği derinliği (D1, M1)
*Sabit-nokta (Picard/Jacobi) paralelleştirme:* s^{(m+1)}_t = F_t(s^{(m)}_{t−1}) tüm t'ler için eşzamanlı. Hata e^{(m)}_t = (J̃_t … J̃_{t−m+1}) e^{(0)}_{t−m} → ≤ ρ^m eğer ‖J‖ ≤ ρ < 1; m süpürmeden sonra ilk m konum, ρ'dan bağımsız, tam doğrudur **[T]**. Yani m ≪ T süpürmede yakınsama yalnız **büzülme** (kısa bellek) varsa hızlıdır; *gerçek uzun-menzil durum bağımlılığı* (S₅ çarpımı) için m = Θ(T) süpürme gerekir. Bu yüzden: RLT'nin zayıf geri beslemesi (α = 0,1) ve kısa belleği olan dil gibi işlerde ucuz bir hızlanma olabilir (çıkarım ile eğitim uyumsuzluğu riski var), uzun-menzil durum izlemede olmaz.
*İfade gücü:* S₅ kelime problemi NC¹-tamdır (Barrington 1989 **[R]**); sabit derinlikli transformerler TC⁰'dadır (Merrill & Sabharwal 2023 **[R]**); dolayısıyla TC⁰ ≠ NC¹ varsayımı altında keyfi T'de tam S₅ izleme derinliği T ile büyüyen hesap gerektirir: (a) sıralı RLT-1: T; (b) RLT-2 parça B: T/B (B içindeki B permütasyonun çarpımı tek kod çözücü katmanında yapılmalı: B ≥ 2'de zorlaşır; makale RLT-2'yi yalnız mod-5'te, tek tohumla ölçtü ve chunk4 7+1'de %99,61 → %18,95 düştü); (c) birleşme-uyumlu durum şeridi (grup değerli geçişlerin ilişkisel taraması, derinlik log T; Grazzi vd. 2411.12537 ve Householder çarpımı yaklaşımı **[R]**: n−1 yansıma S_n'i temsil eder). (c) RLT'nin "tam kod çözücü durumunun geri beslenmesi" özelliğini bırakır; (a)-(b) korur. NoPE-transformerin S₄/S₅'i öğrenememesi (N6) bu çerçeveyle tutarlı ama bütçe/ayar belirsizliği nedeniyle kanıt değil.
*Karar:* hız çabası RLT'nin mantığını ne kadar koruyacağına dair açık karar (§7, soru 2).

### 4.6 "RLT mantığını bozmama" ölçütü
Her iyileştirme (K1) önceki tokenin SON kod çözücü çıktısını kapılı birleştirmeyle sonraki girdiye geri besleme özelliğini korur, (K2) nedensel kodlayıcı belleğini ve pencereli kod çözücüyü korur, (K3) **kapatılabilir**dir: kapalıyken aynı ağırlıklarla RLT-1'e birebir iner (testlenir; ablasyon temizliği), (K4) adım başına sıralı maliyeti O(1)'in üstüne çıkarmaz. Mühürlü `core/model.py` değiştirilmez; varyantlar alt sınıf olarak yazılır (E-A'daki `IntervenedCore` örneği gibi).

## 5. Deney kuyruğu (ön-kayıtlı öngörüler)

Kısıt: aynı anda ≤ 2 deney süreci (L1/T1 hariç), BelowNormal; test bölünmesi açılmaz (yalnız dev). Her satırın ilk iki sütunu sonuçtan ÖNCE yazılmıştır.

| ID | Soru / yöntem | Öngörü [H] ve çürütme | Maliyet | Karar bağı |
|---|---|---|---|---|
| **E-A** | S₅ Core kopuşu: dikkat mi durum mu? Kayıtlı `perm5-core-w128-s1-cur.pt` ile eğitimsiz müdahale: w16, w32, w64 (kodlayıcı + bellek), w16-yalnız-kodlayıcı, w16-yalnız-bellek, log-n sıcaklığı (hepsi / yalnız kodlayıcı / yalnız bellek); dev bölünmesi, tohum 1-3 havuzlanır; pencere başına ilk-hata oranı (`horizon/attention_intervention.py`) | **Okuma kuralı (ön-kayıt):** H_taban = [128,1024) havuzlanmış hazard. *DİKKAT (H1)* ⇔ en iyi {w16, w32, temp} için H_i ≤ 0,2·H_taban VE [0,64) hazard'ı ≤ 1e-4. *DURUM (H2)* ⇔ tüm müdahaleler için H_i ≥ 0,7·H_taban. Aksi hâlde KARMA, hangisinin etkili olduğu raporlanır. Yerelleştirme: yalnız-kodlayıcı ya da yalnız-bellek tek başına ≤ 0,3·H_taban ise o yol sorumlu. *Geçerlilik koşulu:* taban satırı eğitimde kayıtlı sayıları birebir üretmeli (fark ≤ 1e-6), yoksa sonuç okunmaz. Dikkat seyrelmesi 4-8×'i tek başına açıklamaz öngörüsü (§4.3): p(H1 tek başına) ≈ 0,35 [H] | CPU, ≈ 15-25 dk | H1 → gerçek modelde log-n/pencere standartlaşır; H2 → E-B/E-D |
| **E-B** | Durum gürültüsü marj eğrisi: her adımda geri besleme durumuna göreli σ ∈ {0; 0,01; 0,02; 0,05; 0,1; 0,2} Gauss gürültüsü; ilk-hata oranı λ(σ); S₄ Core, S₅ Core, GRU, LSTM, kayıtlı ağırlıklar. Ek: Lyapunov üstleri (QR) ve basit Jacobian tekil değerleri | m (§4.2) kolay okunur: S₄ Core için λ(σ=0,1) ≈ 0 ve S₅ Core'un σ-eşiği (λ > 1e-3) S₄'ünkinin en az yarısı (marj sırası hazard sırasıyla uyumlu). Çürütme: sıralama ters ya da λ(σ) erfc biçimine uymuyorsa Gauss-havza modeli reddedilir | CPU, ≈ 30 dk | m büyütme müdahalesi (E-D) değer taşır mı |
| **E-C** | Sayaç: ilerleyici tepe müfredatı (P = 8 → 16 → 32 → 64) ve ölçek-çeşitli aşma eğitimi (GRU, LSTM; sonra Core). İlk-iniş ölçüsü | Aşma görmeyen (şimdiki) tabanda ilk-iniş ≈ şans (0,03-0,06). Ölçek-çeşitli eğitimde D = 2 × (en büyük eğitilen aşma)'da ilk-iniş ≥ 0,5; Core'un RMSNorm'lu geri beslemesinde GRU/LSTM'den kötü. Çürütme: ilk-iniş < 0,2 → "sayaç yalnız mimari değişiklikle" | CPU, GRU/LSTM ≈ 15-60 dk/aşama, Core ≈ 3 sa/aşama | RLT sonsuz derinlik iddiası; E-I gerekli mi |
| **E-D** | Durum-gürültüsü enjekte eden eğitim (σ_train ∈ {0,05; 0,1; 0,2}) S₄/S₅ GRU, LSTM, Core | σ_train = 0,1'de sabit hazard tabanı ≥ 5× düşer; aralık-içi doğruluk ≥ 0,99 kalır (§4.2: marj ×1,5 → hazard ≥ 1.000× düşebilir). Çürütme: < 2× düşüş ya da aralık-içi doğruluk < 0,95 | CPU, ≈ 3-5 dk (GRU) … 90 dk (Core S₅) | Durum kalıcılığının en ucuz tarifi gerçek modele girer mi |
| **E-F** | S5-swaps (özdeşlik + 10 transpozisyon) ve eğitim uzunluğu 32 ile makaleyle birebir görev: Core, GRU, LSTM, NoPE-transformer; 512 işlemde son-durum. Makalenin lr 1e-4 ve bizim lr ile | Core 512'de ≥ %55 (makalenin en iyisi 4+4: %55,70±25,78; Transformer %0,85); transformer tabanı aralık-içi öğrenir (makale: %99,09). Çürütme: Core < %20 → bizim uygulamamız makalenin mekanizmasından zayıf | CPU, birkaç saat | Uygulama sadakati; M5 |
| **E-G** | Kapının geçerli testi: çok-zaman-ölçekli (chrono) HOLD kapısı + nötr-aralıklı S₄/S₅ + manipülasyon kontrolü (kapı gerçekten kapandı mı: hold oranı) | Chrono ilkleme (unutma yanlılığı ≈ ln(U[1, T_max] − 1); Tallec & Ollivier 2018 **[R]**) hold'un ortalama ≥ 0,5 olmasını sağlar; nötr kayıp ≤ 5 puan; çürütme: hold < 0,1 (manipülasyon uygulanmadı) ya da kayıp > 20 puan | Yeni kod + CPU | N5, A1 |
| **E-H** | RLT-2 (parça-düzeyi geri besleme) kolu: B ∈ {2, 4, 8}; B = 1'de RLT-1'e birebir (test); hız ve S₄/S₅ doğruluğu | B = 4'te adım hızı ≥ 2×; S₄ hatasızlığı korunur (B ≤ 2), S₅ bozulur (B ≥ 4: B permütasyonun çarpımı tek katmanda yapılmalı, §4.5). Çürütme: S₅ B = 4'te de bozulmuyorsa makalenin yöntemi bizim görevde bedelsiz | Yeni kod + CPU | D1/M1 |
| **E-I** | Yapılandırılmış (Householder/dönel) birleştirme kolu (§4.4-4.5; K3 geçerli) | E-C ve E-D bakılır; yalnız onların sonuçları yetersizse | Yeni kod | |
| **L2** | Gerçek model: akıl yürütme izli karışım + eşleşmiş transformer tabanı (§6) | | GPU, günler | |

### 5.1 E-B okuma kuralı (koşudan ÖNCE yazıldı; kod `horizon/state_noise.py`, 9 test)
Modeller (kayıtlı ağırlıklar, hepsi dev bölünmesi, tohum 1, n = 100, uzunluklar 16/64/256/1.024): S₄ Core w64, S₄ GRU w64 ve w128, S₄ LSTM w64 ve w112, S₅ Core w128, S₅ GRU w256, S₅ LSTM w208. Gürültü: taşınan duruma (Core: önceki son kod çözücü çıktısı; GRU: h; LSTM: h ve c) her adımda, ikinci adımdan itibaren, göreli Gauss σ ∈ {0; 0,01; 0,02; 0,05; 0,1; 0,2} (σ × satır rms'i). Ölçü: ilk 64 konumdaki (pencereler ≤ 64) ve tüm pencerelerdeki havuzlanmış ilk-hata oranı λ(σ). **σ_tol** = λ_erken(σ) ≥ 1e-3 olan en küçük σ (ızgarada yoksa "> 0,2").
- **Ö1:** S₄ Core için λ_erken(σ = 0,1) < 1e-3 (σ_tol ≥ 0,2).
- **Ö2:** σ_tol(S₅ Core) < σ_tol(S₄ Core) (120 durum, daha küçük havza).
- **Ö3 (Gauss-havza modeli):** (σ, λ) çiftlerinden 1e-4 < λ < 0,3 olanların ≥ 3'ü olan her model için örtük yarıçap r_i = σ_i · m(λ_i) değişim katsayısı ≤ 0,35 (tek bir r). Sağlanmazsa o model için Gauss-havza modeli REDDEDİLİR (eğriler yine raporlanır; §4.2'nin nicel iddiası desteksiz kalır).
- **Ö4:** S₄'ün dört GRU/LSTM modelinde gürültüsüz taban sırası (GRU w64 7,6e-4 > LSTM w64 5,3e-4 > LSTM w112 4,5e-4 > GRU w128 1,15e-4) ile σ_tol sırası ters orantılı (Spearman ρ ≤ −0,6).
- Geçerlilik: σ = 0 satırı, kayıtlı sonuç dosyasındaki aynı noktaların erken-pencere ilk-hata sayılarını ve pozlanmayı üretmeli (Core: birebir; GRU/LSTM σ = 0'da elle açılmış hücrelerle koşar, kütüphane katmanıyla fark ≤ 1e-5: ≤ 2 ilk hata ve pozlanmanın %1'i kadar fark kabul; 06:33'te, koşudan önce eklendi). Üretmeyen modelin satırları okunmaz. Çürütme: Ö1 tutmazsa S₄ Core'un hatasızlığı marj kaynaklı değildir.

### 5.2 E-D okuma kuralı (koşudan ÖNCE yazıldı; kod: `depth_run.train(state_noise=σ)` + `state_noise.NoisyRecurrent/NoisyCore`, 4 test; betik `scripts/run_ed.sh`)
Koşular: S₄ GRU w64 ve LSTM w64, tohum 1, 15.000 güncelleme, müfredat 1,2,3,4,6,8,12,16, lr 3×10⁻³ (taban koşularla aynı), **eğitim-zamanı** durum gürültüsü σ_train ∈ {0,05; 0,1; 0,2} (6 koşu; sıra: 0,1 önce). Değerlendirme TEMİZ (σ = 0), Z1'in standart noktaları (16 … 1.024). Ölçüler: uzunluk 16'da son-konum doğruluğu (aralık-içi) ve tüm pencerelerde havuzlanmış ilk-hata oranı (`horizon/hazard_summary.py` ile). Tabanlar: `perm4-gru-w64-s1-cur` (7,6×10⁻⁴) ve `perm4-lstm-w64-s1-cur` (5,3×10⁻⁴); tohum 2-3 tabanları kuyruk 5'ten gelince **en iyi taban** (üç tohumun en düşüğü) karşılaştırma değeri olur (talih payını elemek için); gelene kadar yalnız tohum 1 tabanıyla "geçici" okunur.
- **DESTEKLENİR** (kol başına): en az bir σ_train için aralık-içi doğruluk ≥ 0,99 VE havuzlanmış hazard ≤ (en iyi taban)/5.
- **ÇÜRÜTÜLÜR** (kol başına): tüm σ_train için hazard ≥ (en iyi taban)/2 ya da aralık-içi doğruluk < 0,95.
- **KARMA:** arası. Çok büyük σ_train'de öğrenmenin bozulması (aralık-içi doğruluk düşer) çürütme değil, sınır bulgusudur.
- *Ek (mekanizma):* yeni modellerin E-B marj eğrisi: Gauss-havza modeli ve marj yorumu doğruysa örtük yarıçap σ_train ile artar (r(σ_train = 0,1) ≥ 1,5 × r(taban)); artmıyorsa hazard düşse bile "marj büyüdü" yorumu desteksizdir.

### 5.3 E-C okuma kuralı (koşudan ÖNCE yazıldı; görev `count8_32` / `count8_64` (`horizon/depth_tasks.py`, 5 test), betik `scripts/run_ec.sh`)
Görev: etiket P = 8'e doyar (9 sınıf; eski sayaçla aynı çıktı sözlüğü), ama EĞİTİM her dizide log-tekdüze çekilen bir derinliğe (8 … R_max; `count8_32`: R_max = 32, uzunluk 256; `count8_64`: R_max = 64, uzunluk 512) tırmanır, iner ve dibe yakın dolaşır (değerlendirme biçimi); yani 8'in üstüne çıkan her aşma sayılmak zorundadır. Test: D ∈ {8, 16, 32, 64, 128, 256, 512} (R_max'ın 1×, 2×, 4× … katları). Kollar: GRU w64 ve LSTM w64, tohum 1, 12.000 güncelleme, lr 3×10⁻³, müfredatsız (Core CPU'da bu uzunlukta 9 saat/koşu: yalnız GRU/LSTM sonucu bakılırsa). Ölçü: ilk-iniş doğruluğu `reentry` (D başına), aralık-içi doğruluk (D = 8, `prefix`).
- **Geçerlilik:** aralık-içi (D = 8) `prefix` ≥ 0,95. Değilse sonuç bütçe/plato kaydıdır (P = 32 GRU'da olduğu gibi), dışdeğerleme okunmaz.
- **Ö1 (tarif çalışıyor mu):** eğitim aralığında (D ≤ R_max) ilk-iniş ≥ 0,9. < 0,5 ise "log-tekdüze aşma bu bütçede sayma öğretmiyor" (müfredat gerekir).
- **Ö2 (LSTM, tanh doyması ≈ 2,7×; §4.4 `[T/H]`):** D = 2·R_max'ta ilk-iniş ≥ 0,5, D = 4·R_max'ta < 0,5. **Çürütme:** D = 4·R_max'ta ≥ 0,95 ise sınır tanh doyması değildir (kesinlikle > 4×).
- **Ö3:** GRU'nun dışdeğerleme çarpanı LSTM'inkinden küçük ya da eşit (GRU durumu [−1, 1]'de sınırlı).
- **Çarpan** ρ = (ilk-iniş ≥ 0,95 olan en büyük D) / R_max (D ≤ R_max noktalarından başlayıp kesintisiz). Literatür emsali LSTM ≈ 2-2,5× (**[R]**, ajan notu).
- *Sonraki adım koşulu:* ρ ≥ 4 çıkarsa (tarif gerçekten sayaç öğretiyor) Core ve dönel şerit (E-I) aynı görevde sınanır; ρ ≈ 1-2 ise sayaç dışdeğerlemesi tarif değil mimari sorunudur ve E-I öne çıkar.

### 5.4 E-B2 okuma kuralı (kod `horizon/state_growth.py`, 6 test; betik `scripts/run_eb2.sh`)
**Yazılma anı ve bilgi durumu:** 06:42; E-A'nın 9 satırından yalnız ilk üçü görüldükten sonra (taban, w16, w32: hazard'lar tabanla pratikte aynı), E-A'nın geri kalanı, E-B ve E-B2 koşmadan ÖNCE. Yani E-A'nın ara satırları bu kuralın yazımını etkilemiş olabilir (kopuşun dikkat dışı, durumda olduğu izlenimi).
Ölçü: kayıtlı ağırlıklar (S₄ Core w64, S₅ Core w128, S₄ GRU w64, S₄ LSTM w64, S₅ GRU w256, S₅ LSTM w208), dev, tohum 1, 50 dizi × 256 konum; t0 = 8'de taşınan duruma göreli ε ∈ {0,001; 0,01} büyüklüğünde **tek bir tekme** (rastgele yön); float64; d_t = |Δdurum| / |durum|; tekmeden sonra [1,25), [25,121), [121,…) pencerelerinin her birinde ortalama log d_t'nin eğimi (adım başına büyüme oranı) ve tahmin-çevrilme oranı (tekmeli ve temiz koşuda argmax farklı).
- **G1:** S₄ Core: tüm pencerelerde büyüme ≤ 0 (genişlemeyen) ve ilk iki pencerede çevrilme oranı < 0,001.
- **G2:** S₅ Core: son pencerede büyüme > +0,005/adım YA DA çevrilme oranı S₄ Core'unkinin ≥ 5 katı. Bu, kopuşun "kararsız mod" yorumunu destekler. **Çürütme:** S₅ Core'da tüm pencerelerde büyüme ≤ 0 ve çevrilme oranı ≈ S₄'ünkine yakın → kopuş üstel kararsızlık değildir; açıklama başka bir mekanizmada (veri-yönelimli, nadir tetiklenen) aranır.
- **G3:** GRU/LSTM (S₄, S₅): tüm pencerelerde büyüme < 0 (büzülen): hata tabanı genişlemeden değil gürültü marjından (E-B) gelir.
- *Ek kaydedilen ayrıntı (okuma kuralının parçası DEĞİL; 06:43, koşudan önce eklendi):* tek-adım Jacobian'ın en büyük tekil değeri (taşınan durum üzerinden; Core'un üç önbellek anahtar/değeri temiz değerlerinde sabit) t ∈ {4, 16, 32, 64, 128, 192, 255} konumlarında, 8 dizi, float64: §4.1'deki ‖J‖ ≲ α‖W_s‖/ρ tahminini ve R3 ajanının "ρ_t medyanı < 1" öngörüsünü ölçer; yorum yalnız betimleyici.
- *Sınır:* tek tekme tüm Lyapunov spektrumunu ölçmez; rastgele yön en hızlı genişleyen yöne yakınsar (üst Lyapunov üssü). "Büzülüyor" sonucu "hiç genişleyen yön yok" değil, "tipik tekme genişlemiyor" demektir.

### 5.5 R3 ajanından gelen yeni deney adayları (okuma kuralları uygulanmadan ÖNCE yazılacak; şimdilik yalnız kayıt)
| ID | Soru (ajan: `R3_…md` §5) | Ajanın öngörüsü `[H-ajan]` | Durum |
|---|---|---|---|
| E-J | Doğrusal RNN + Householder çarpımları (DeltaProduct n_h ∈ {1, 2, 4}) Z1 protokolünde (S₄, S₅, sayaç): tarama-paralel tavan ve RLT'siz kontrol | S₄: n_h = 1 ≤ 0,3, n_h = 2 ≥ 0,95 @256; S₅: n_h = 4 ≥ 0,9 @128, n_h = 2 ≤ 0,5; sayaç P = 8: aralıkta ≥ 0,99, ilk-iniş ≤ 0,15 | kod yok; E-D/E-C sonrası |
| E-K | Sayaç için (max,+) tarama şeridi (x ↦ max(a, x + b), bileşke altında kapalı): "sayaç için geri besleme gerekmez" ve etiket doyması sorusu | ilk-iniş ≥ 0,9 (olasılık ≈ 0,6); ≤ 0,15 ise sorun mimari değil eğitim dağılımı | kod yok |
| E-L | Hız bütçesi mikro-ölçümü (GPU, L1 şekli): tam sıralı adım, geri beslemesiz tavan, CUDA-graph, stateful TBPTT, parça B, Jacobi K | tavan/sıralı 5-9×; CUDA-graph 3-5×; stateful-TBPTT 3-6× | GPU L1 ile paylaşılır: **kullanıcı onayı gerekir** (üçüncü süreç) |
Ertelenen: R3-E3 (Jacobi-K ile eğitim): eğitim/çıkarım uyuşmazlığı riski ve RLT'nin "exact replay" ilkesine aykırılık. E-H (RLT-2, R3-E2 ile aynı) ve E-B2 (R3-E1'in çıkarım-yalnız kısmı) zaten kuyrukta.

### 5.6 R4 ajanından gelen eklemeler: E-C0 (tohum dağılımı) ve E-M (`core_nofb` sayaç kontrolü); okuma kuralları koşudan ÖNCE yazıldı (06:50)
**R4 raporu geldi (06:47; 51 kaynak, ajan hepsini okudu; ben yalnız okudum, bağımsız doğrulamadım).** Kilit bulgular: (a) sonlu eğitim derinliği sonlu sayaç verir, "doyan sayaç" bulgumuz literatürün (Chang & Bisk 2025) beklenen sonucu; (b) gradyanla eğitilen sayaçlar tipik 2-3× dışdeğerler, en iyi tohum 5-12× (33× yalnız ikincil kaynaktan), >20× yalnız gradyansız arama/MDL/kesin-artışlı ünite/sınırlı derinlikte; gradyanla 32× (8 → 256) emsal bulunamadı (sınırlı tarama); (c) başarılı sayaç-LSTM'lerin hepsi MSE+sigmoid ya da skaler regresyon kullandı, bizde softmax-CE var (etiket kapanışı: "P'de kal"); (d) ölçek-çeşitli/log-tekdüze tepe için doğrudan emsal YOK (en yakınlar: uzunluk dağılımı çeşitliliği); (e) bizim ilk-iniş ölçütümüz literatürün tanıma ölçütlerinden daha katıdır, literatür çarpanları ÜST sınırdır; (f) **§4.4'teki "LSTM ≈ 2,5× = tanh doyması (8/3)" türetmem Weiss'in nedeniyle (kapı eşitsizliği birikimi) ayrılmadı: yalnız bir hipotezdir, 2,56 ile 2,67 tesadüf olabilir.**
- **E-C0 (R4-E1; kod: `horizon/readings.py::ec0`, betik `scripts/run_ec0.sh`):** mevcut `count` tasarımı (tepe 8, uzunluk 64, 4.000 güncelleme, lr 3×10⁻³, genişlik 64), GRU ve LSTM, **tohum 1-8** (tohum 1 mevcut; 14 yeni koşu ≈ 36 dk CPU). Soru: "tüm yinelemeli kollar doyan sayaç öğrendi" bulgusu tek tohuma mı dayanıyor (tohum piyangosu: en iyi tohum literatürde tipik × 4-10)? Geçerli koşu: aralık-içi (D = 8) `prefix` ≥ 0,95. **EVRENSEL DOYMA** ⇔ geçerli koşuların ≥ %90'ında ilk-iniş(D = 16) ≤ 0,10. **PİYANGO** ⇔ geçerli koşulardan ≥ 2'sinde ilk-iniş(D = 16) ≥ 0,5 (gerçekten 2×'te sayan bir tohum). Aksi hâlde KARMA. (Öngörü [H-ajan]: ≥ 15/16 koşuda ≤ 0,10; ≥ 1/16'da ≥ 0,5 olasılığı ≈ 0,15.)
- **E-M (`core_nofb` sayaç kontrolü):** geri beslemesiz Core (paralel yol: dakikalar), `count`, tohum 1, 4.000 güncelleme. Betimleyici (eşiksiz): Core'un küresel KV dikkati NoPE-benzeri bir oran yoluyla (ara derinlik ≈ itme oranı × konum) saymayı tek başına sağlıyor mu? Geri beslemeli Core'un sayaç sonucunu (yalnız derinlik 0'da doğru) bu kontrolle karşılaştırırım.
- **E-C2 (R4-E2, ŞARTLI; kod yok):** ucu açık skaler regresyon (Huber) okuması: etiket kapanışı ("P'de kal") mı doymanın kaynağı? Yalnız E-C öğrenilebilirlik gösterir ama dışdeğerleme sınırlı çıkarsa yazılır (E-C kuralı).
- **E-C'nin kalan açık noktası:** test uzunluğu ≥ 4D uzunluk sürüklenmesini derinliğe karıştırır (R4/El-Naggar uyarısı: ilk hata ≈ 20× uzunlukta); `length_control` (eğitim dağılımı, 1×/4×/16× uzunluk) JSON'da kayıtlı, E-C okurken ilk bakılacak ek ölçüdür.

### 5.7 E-N okuma kuralı: kimlik-dolgu (nötr koşular), eğitimsiz (R1-E5-ii'den; kod `horizon/neutral_runs.py`, 11 test; betik `scripts/run_en.sh`; koşudan ÖNCE yazıldı 06:55)
Soru: eğitilmiş S_N kolunun adım-başı ilk-hata oranı **zamana** mı (hiçbir şey olmazken durum kayıyor) yoksa **işe** mi (her gerçek güncelleme küçük hata ekliyor) bağlı? Her belirteç p_id olasılıkla özdeşlik permütasyonu (belirteç 0), aksi hâlde düzgün rastgele grup elemanı; etiket çalışan çarpım, yani özdeşlik belirteci durumu tam korumalı. p_id ∈ {0; 0,5; 0,9; 0,99}, uzunluk 512, n = 100, dev, kayıtlı ağırlıklar (S₄ Core w64, S₅ Core w128, S₄ GRU w64, S₄ LSTM w64, S₅ GRU w256, S₅ LSTM w208). Ölçü: konum ≥ 128 pencerelerinde ilk-hata oranı hem **adım başına** hem **gerçek güncelleme (özdeşlik olmayan belirteç) başına**.
- **İŞ (WORK):** p_id ∈ {0,5; 0,9} için güncelleme-başı oran, p_id = 0 değerinin [0,6; 1,67] katı içinde ve (log'da) adım-başı orandan ona daha yakın.
- **ZAMAN (TIME):** aynı koşul adım-başı oran için.
- **NÖTR ÇÖKÜŞ:** adım-başı oran p_id = 0 değerinin 3 katından büyük (özdeşlik koşuları modeli bozuyor: A1 sürüklenmesinin S_N'deki yeniden üretimi).
- **DİĞER / KARMA / ÇOK AZ OLAY (< 10 ilk hata):** raporlanır, yorum yok. p_id = 0,99 yalnız betimleyicidir (geriye birkaç gerçek güncelleme kalır). p_id = 0'da ilk-hata olmayan modelde (S₄ Core) atfedilecek bir şey yoktur.
- *Öngörü [H-ajan R1]:* güncelleme gürültüsüyse adım-başı oran ≈ (1 − p_id) × taban (p_id = 0,9'da ≈ 2,4×10⁻⁴ S₅ Core'da); uzunluk/durum-tutma ise adım-başı oran taban ± %30. Benim öngörüm: GRU/LSTM için İŞ (sabit hata tabanı marjdan, §4.2), S₅ Core için ZAMAN ya da NÖTR ÇÖKÜŞ olasılığı daha yüksek (kopuş ≥ 128 konumda başlıyor: zamana bağlı bir etken); ikisi de ölçülmedi.

## 6. Gerçek model yolu

1. **Ne iddia edilir:** "küçük, bayt-düzeyi bir RLT, durum-izleme/çok-adımlı görevlerde eşit-parametre ve eşit-token bir transformer'a yapısal bir üstünlük gösteriyor mu". LM kaybında üstünlük iddia edilmez (bağımsız değerlendirme bulamadı).
2. **Gerekenler:** (a) eşleşmiş transformer tabanı (parametre, token, veri sırası, optimizer aynı); (b) RLT-0/`core_nofb` ablasyonu; (c) görev ailesi: S_N bileşimi (metin biçiminde, derinlik düğmeli, iz yazan), aritmetik (toplama, çok basamaklı), uzun nötr koşulu durum izleme; (d) uzunluk/derinlik dışdeğerleme ölçüsü ilk-hata hazard'ı; (e) E-A/E-B/E-D sonuçlarına göre log-n/pencere, durum gürültüsü ve HOLD ilkleme tarifi.
3. **Veri:** L2 anketi (`notes/L2_VERI_ANKETI_2026-10-02.md`): programatik üreteçler lisans sorunu getirmez; CNN/DailyMail yayın riski (§6.8, §10).
4. **Başlatma kapısı:** E-A, E-B, E-D bitmeden L2 başlatılmaz; L1 (GPU) bitmeden ya da kullanıcı onayı olmadan GPU paylaşılmaz.

### 6.1 R3 ajanından (06:45) gerçek model yoluna gelen not
Bağımsız değerlendirmenin özetleyici-aktarımlı sayısı: 140M'de RLT 3,688, geri beslemesiz (A0) 3,666, Transformer 3,642 nat (`[S]`; ajan α'nın 0,1 mi 1 mi olduğunu belirsiz buluyor). Doğruysa **geri besleme bu ölçekte dil kaybını kötüleştiriyor** ve ≈ 7,85× yavaşlatıyor (A0 66,0K vs RLT 8,4K token/s). Bu, gerçek model için kritik bir tasarım sorusu doğurur: geri besleme dil yetkinliği için değil durum izleme için tutulmalı (seyrek sıralı katman ya da yalnız belirli konumlarda geri besleme gibi, hepsi K3 ölçütünü sağlayan kapatılabilir eklentiler) ve eşleşmiş bir A0/transformer tabanı olmadan LM iddiası yapılmamalı. L1'in (düz RLT-1) bu soruya cevabı yok.

## 7. Kullanıcıya açık sorular
1. L2 için GPU L1 ile paylaşılsın mı, L1'in ardından mı (L1 ≈ 53 saat, 01.10 23:39'da başladı)?
2. Hız: RLT-2 / birleşme-uyumlu şerit gibi RLT'nin tam-durum-geri-besleme özelliğini kısmen bırakan seçenekler, "mantığını bozmadan" ölçütüne girer mi? (Önerim: yalnız K3'ü sağlayan (kapatılabilir) eklentiler; RLT-2 B = 1'de RLT-1'e iner, girer.)
3. L2 metin korpusu ve yayın/lisans kararı (ana belge §6.8, §10).

## 8. R1, R2, R5 ajan raporları (06:50-06:55; R3 ve R4 §5.5-5.6'da): ne alındı, kuyruğa ne işlendi
Hepsi `notes/literatur_2026-10-02/` altında; `[V]` ajanın kendi okumasıdır, benim doğrulamam değildir (R3'te iki arXiv başlığı hariç).
**R1 (dikkat/uzunluk, 143 satır, 34 kaynak).** (a) Kopuş çarpanı hedef-dışı softmax kütlesinden hesaplanabilir (D(n) = Σ exp(z_j − z_hedef); çarpan ≈ D_c/D₀) ve log n ölçeği marj ≥ ln n₀ ise seyrelmeyi siler: §4.3 ile aynı biçim (Chiang & Cholak; SSMax). (b) **İğne uçurumu seyrelme değil konum/logit OOD'u:** n 64 → 80 (1,25×) iken doğruluk 0,99 → 0,02; saf seyrelme için (n/n_c)^−k, k ≈ 35-40 gerekir (ajanın hesabı); literatür aynı yönde (Vasylenko 2026: 2-10M MQAR, RoPE 2×'te çöküyor; Anson 2025: 162M iğne, LogN+RoPE ve p-RoPE tek başına yetmiyor, birlikte 0,96/0,94). (c) S₅ Core için ajanın hesabı (d_kafa 32, n₀ = 16): 16 RoPE çiftinden yalnız 2'si eğitimde periyodunu tamamlıyor; 7-9. çiftlerin 1-rad mesafesi 56-178 ≈ gözlenen ilk-hata aralığı (64-256) `[H hesap; tesadüf adayı]`. **E-A'nın ara okuması (9 satırdan 7'si; yalnız bilgi):** göreli uzaklığı eğitilen aralıkta tutan pencere satırları (w16, w32, w64, w16_enc, w16_mem) kopuşu değiştirmiyor (hazard'lar tabanla pratikte aynı), yani kodlayıcı ve bellek dikkatinde RoPE-faz-OOD açıklaması bu iki yol için desteksiz; son hüküm 9 satır bitince `attention_intervention.verdict()` ile. (d) Eğitimsiz müdahalelerin ≤ 30M'de sistematik karşılaştırması literatürde yok (E-A bu boşluğu dolduruyor). (e) Önerilen: E1 S₅ eval-only karar paneli (E-A'yı kapsar; ek P4 düşük-frekans dondurma, P5 oracle-durum, P6 dikkat kapatma), E2 iğne modeli eval-only (GPU, L1 ile paylaşım: bkz. aşağı), E3 iğne PoSE ince ayarı, E4 S₅ sıfırdan kısmi RoPE/SSMax/tohum (≈ 4 saat), E5 eşit-içerik ve **kimlik-dolgu** (= E-N), E6 S₄ alt-alfabe uzun-dizi karışımı (≈ 3 saat). Eşit-içerik hesabı (C4): S₅'te n ≈ |G| = 120'de tekrarlı belirteç bollaşıyor (tesadüf adayı).
**R2 (durum kalıcılığı/hassasiyet, 161 satır, 85 kaynak).** (a) **Kalıcılık = zaman ekseninde kimlik (skip) yolu + o durum dağılımının eğitimde görülmesi; biri tek başına yetmiyor** (Block-Recurrent'in yok-sayma optimumu; Stuffed Mamba; suRNN Selective Copy T = 4096: GRU %16,3 → seçici güncelleme %97,2): bizim A1/N5 gözlemleriyle tutarlı; M1'de hold ≈ 5×10⁻⁶ kapıyı çürütmez. (b) Hata kontrolü: afin/ortogonal taşıyıcı hatayı düzeltmez, durum-bağımlı doğrusal-olmayan harita düzeltebilir; GRU/LSTM sabit tabanı z = marj/σ ≈ 3,0-3,4 ve hazard = 2Q(z) (ajanın `[H]` okuması, §4.2 ile aynı fikir); kesikli yakalama (Zeng 1993, Das-Mozer 1993, Lamb 2019) tabanı küçültür sıfırlamaz (Maass-Sontag). (c) RMSNorm radyal yönü siler (kazanç 1/rms); α = 0,1 ile ajanın öngörüsü κ ≈ 0,1-0,65, FTLE ≈ −2,3 … −0,4 nat/adım; RLT-1'de zaman ekseninde skip yok. (d) Ortogonal/dönel taşıyıcı RMSNorm ile uyumlu ve uzun kopyayı çözer ama unutamaz ve düzeltmez. (e) Çekirdeğe eklenebilir dört tasarım (hepsi sıfırlanınca RLT-1): **D1** ikili hold-kaydı + chrono ön yanlılığı, **D2** yön-taşıyan kayıt, **D3** FSQ yakalama, **D4** jitter + zoneout; deneyler E1 (eğitimsiz FTLE/pertürbasyon/marj = E-B, E-B2), E2 (maruz bırakma × kapı faktöriyeli, oyuncak), E3 (D1 + GRU-chrono kontrolü), E4 (D3 + D4: GRU hazard tabanı), E5 (D2), E6 (sayısal hassasiyet). (f) Ajanın M3 notu: Core S₅'te [0,64) hazard < 8×10⁻⁵ (z > 3,95) → 128 ötesi 2,4×10⁻³ (z = 3,04): "ani marj kaybı ≥ %23" ve bu bir durum-dağılımı kayması olabilir: **E-B2 artık konum pencerelerinde durum rms'ini de kaydediyor** (R2 E1-iii; okuma kuralına dahil değil).
**R5 (RLT ekosistemi/gerçek model, 163 satır).** (a) Resmî "depo" yalnız proje sayfası (eğitim kodu, RLT-2 yapılandırması, kontrol noktası yok; makale özel `nanogptpro-dev`'e atıf yapıyor ve o depo 404). (b) İki bağımsız LM testi geri beslemeyi yararsız/zararlı buldu (Lehmann & Nakamura; RLTX 16M bayt enwik8: 1,8556 vs α = 0 1,8196), **ikisi de α = 1 ve çok kısa (≈ 1.900 / 800 adım)**; α = 0,1 ve 30.518 güncelleme rejimini hiçbir bağımsız çalışma sınamamış: L1'in rejimi doğrulanmamış, çürütülmemiş. (c) Durum izlemede yineleme gerçek (RLTX S₅ 4×: RLT %100, α = 0 %0,8; makale swaps-S5 512'de %55,7 vs %0,85) ama GRU'yu geçmiyor: resmî README'nin topluluk sonucunda GRU %100/%99,97 vs RLT %60,8/%20,7, GRU satırı 13 Eylül'de tablolardan silinmiş (git geçmişi), şekil duruyor; bizim Z1-G ile uyumlu. (d) Slayt (28 Eylül): swaps 64 işlemde RLT-1 %100, RLT-2 B = 4 %19,6 ± 6,5, B = 8 %6,8 ± 1,8: **parça-düzeyi geri besleme durum izlemeyi bozuyor** (E-H öngörüsüyle aynı yönde). (e) Komşu çapraz-token mimariler (T²MLR, LRT, Full-bandwidth, Recurrent Transformer) küçük ama tutarlı LM kazancı bildiriyor, RLT'den farklı (orta katman, sıfır-başlangıç, paralel yaklaşık eğitim) ve 1,3-4× maliyetle; T²MLR duvar-saati eşitlenince tabanın kazandığını yazıyor. (f) **L1 yargısı (ajan, `[H]`): büyük ölçüde HAYIR** (iğne/MQAR pili dikkat belleğini ölçer, yinelemenin işi olan durum izlemeyi ne eğitir ne ölçer; α = 0 kontrol kolu yok). **Benim değerlendirmem:** ajanın gerekçesine katılıyorum (N5, N8 ve §1 madde 5 ile aynı yönde); ama L1 bilgisiz değil: bayt-LM'de eğitilebilirlik, kayıp eğrisi ve eğitim-aralığı-içi getirme ölçülüyor. Eksik olan **A0 ikizi** (aynı veri, aynı sıra, geri besleme kapalı) ve durum-izleme probu. (g) Önerilen: E1 L1 anlık görüntülerinde "geri besleme kullanılıyor mu" (eğitimsiz; α_eval ∈ {0; 0,05; 0,1; 0,2}, durum ikamesi, konum dilimleri; CPU, ≈ 0,5-2 sa/anlık görüntü), E2 S₄-Retrieval (durum-anahtarlı getirme; ≈ 40-80 CPU-saat), E3 bayt-LM A0-eşli α merdiveni (≈ 20 CPU-saat), E4 geri besleme-geç program, E5 hız mühendisliği, E6 L1 son ağırlığıyla "kanal öğretilebilir mi" ince ayarı. **GPU gözlemi (06:50, `nvidia-smi`):** L1 GPU'da %97 kullanımda ve 7.487 / 8.192 MiB VRAM'de: eşzamanlı ikinci bir GPU işi (A0 ikizi, R1-E2, R3-E6) **sığmaz**; bunlar L1 durunca ya da bitince, ya da CPU'da küçük ölçekte koşabilir.
**Kuyruğa işlenenler:** E-N (§5.7), E-C0 ve E-M (§5.6), E-B2'ye durum rms'i; adaylar (kod yok): R5-E1 (L1 anlık görüntülerinde kanal kullanımı), E-J/E-K/E-L (§5.5), R2-E2/E3/E4 (kapı faktöriyeli, D1 chrono kaydı, D3+D4), R1-E4/E6. **Kullanıcı kararı gerektirenler:** A0 ikizi için GPU (L1'i durdurmadan olmaz; VRAM), E-L (GPU).

## Sayım (§3 tablolarından `awk` ile sayıldı, 06:24; A-G toplam 64 madde)
| Durum | Sayı | Maddeler |
|---|---|---|
| ÇÖZÜLDÜ | 3 | D9, E9, G4 |
| KISMİ (KISMİ-ÇÖZÜLDÜ dahil) | 11 | A1, A6, A7, A8, A10, B6, C2, E11, F4, F6, G1 |
| ÇÖZÜLMEDİ | 9 | A5, B1, B2, B3, B4, B5, B7, F1, F2 |
| AÇIK | 10 | A4, A9, C3, C5, C6, D8, E10, F5, F7, F8 |
| KURAL (süreç dersi, kapandı) | 18 | A2, A3, C1, C4, C7-C10, D7, E1-E8, E12 |
| KARAR (kullanıcı) | 7 | B8, F3, G2, G3, G5, G6, G7 |
| KISIT (yönetilir, çözülmez) | 6 | D1-D6 |

Toplam 64. KURAL/KARAR/KISIT hariç **teknik çekirdek 33 madde**: 3 çözüldü, 11 kısmi, 9 çözülmedi, 10 açık. Bu 9 "çözülmedi"nin 6'sı B bölümünde (doğrulayıcı, anlamsal erişim, ayrıştırma, başlık grafiği: RLT'nin çevresindeki sistemler; gerçek model yolunda ikincil); RLT çekirdeği ve eğitimiyle ilgili açık/kısmi olanlar A1, A4-A10, D1-D2 (hız), F4-F6, G1 ve N1-N10. Eklenen N bölümü (10 madde): AÇIK 3 (N1, N5, N10), ÇÖZÜLMEDİ 1 (N2), KISMİ 4 (N3, N4, N6, N9), KISIT 1 (N7), DEVAM 1 (N8).

**Not:** sohbette daha önce verdiğim "8 problemden 2 çözüldü (oyuncak), 2 kısmi, 2 denendi-çözülmedi, 2 açık" sayımı 8 birimlik farklı bir gruplamaydı; bu tablo havuzun 64 maddesinin ham kanıta karşı sayımıdır. İki sayım aynı birimde değildir, karşılaştırılmaz; geçerli olan bu dosyadaki maddedir.
