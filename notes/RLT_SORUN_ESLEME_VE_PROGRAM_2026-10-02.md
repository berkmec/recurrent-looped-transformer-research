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
*tanh doyması ve LSTM ≈ 2,5×:* LSTM'de h = o ⊙ tanh(c) ve hücre c sınırsız toplanabilir, ama okuma tanh'tan geçer: 1 − tanh(c) < 1e-3 için c > 3,8, < 1e-7 için c > 8,4 (float32'de tanh(8) = 1 − 2,3e-7). Eğitim aralığı P'nin c ≈ 3'e haritalandığı bir ölçekte (δ = 3/P adım başına) doygunluk c ≈ 8'de başlar: **dışdeğerleme çarpanı ≈ 8/3 ≈ 2,7 [T/H]**; literatürdeki "LSTM ≈ 2-2,5×" (a^n b^n n ≤ 100 → 256; **[R]**, ajan notu, doğrulanmadı) ile uyumlu: aritmetik değil okuma doygunluğu sınırı. GRU durumu [−1, 1]'de sınırlı: sınırsız sayaç yok. **Core:** geri besleme RMSNorm'dan geçer (ölçeğe duyarsız): sayım yalnız YÖN olarak taşınabilir. Dönel sayaç: itme +ω_i, çekme −ω_i açısı (RoPE'ye benzer çoklu frekans bankası; tek frekans 2π/ω'da örtüşür, ω_i'ler artan frekans oranlarıyla çarpımsal olarak aralığı büyütür); norm korunur, itme-çekme tam iptal olur (R(−ω)R(ω) = I), kayıp yok **[T]**. Tabanda ise zemin (0'da çekme sabit) dönel haritayı bozar: "derinlik 0" testi tüm fazların ≈ 0 olması olarak okunmalı (MLP). Bu bir *tasarım adayı*dır (E-I), önce E-C ve E-D bakılır.
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

## 6. Gerçek model yolu

1. **Ne iddia edilir:** "küçük, bayt-düzeyi bir RLT, durum-izleme/çok-adımlı görevlerde eşit-parametre ve eşit-token bir transformer'a yapısal bir üstünlük gösteriyor mu". LM kaybında üstünlük iddia edilmez (bağımsız değerlendirme bulamadı).
2. **Gerekenler:** (a) eşleşmiş transformer tabanı (parametre, token, veri sırası, optimizer aynı); (b) RLT-0/`core_nofb` ablasyonu; (c) görev ailesi: S_N bileşimi (metin biçiminde, derinlik düğmeli, iz yazan), aritmetik (toplama, çok basamaklı), uzun nötr koşulu durum izleme; (d) uzunluk/derinlik dışdeğerleme ölçüsü ilk-hata hazard'ı; (e) E-A/E-B/E-D sonuçlarına göre log-n/pencere, durum gürültüsü ve HOLD ilkleme tarifi.
3. **Veri:** L2 anketi (`notes/L2_VERI_ANKETI_2026-10-02.md`): programatik üreteçler lisans sorunu getirmez; CNN/DailyMail yayın riski (§6.8, §10).
4. **Başlatma kapısı:** E-A, E-B, E-D bitmeden L2 başlatılmaz; L1 (GPU) bitmeden ya da kullanıcı onayı olmadan GPU paylaşılmaz.

## 7. Kullanıcıya açık sorular
1. L2 için GPU L1 ile paylaşılsın mı, L1'in ardından mı (L1 ≈ 53 saat, 01.10 23:39'da başladı)?
2. Hız: RLT-2 / birleşme-uyumlu şerit gibi RLT'nin tam-durum-geri-besleme özelliğini kısmen bırakan seçenekler, "mantığını bozmadan" ölçütüne girer mi? (Önerim: yalnız K3'ü sağlayan (kapatılabilir) eklentiler; RLT-2 B = 1'de RLT-1'e iner, girer.)
3. L2 metin korpusu ve yayın/lisans kararı (ana belge §6.8, §10).

## Sayım (bu notun tablolarından, `grep` ile hesaplandı)
(Aşağıdaki sayılar yazıldıktan sonra tablolardan sayılarak doldurulur.)
