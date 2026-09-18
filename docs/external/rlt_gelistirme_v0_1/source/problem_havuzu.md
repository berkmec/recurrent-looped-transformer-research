# RLT araştırması — problem havuzu (2026-09-18)

Amaç: tek tek düşünüp çözmek için bütün açık ve olası problemlerin listesi. Her madde: **durum** (MEVCUT = ölçüldü/yaşandı; OLASI = henüz karşılaşılmadı ama yolda), kanıt, neden önemli, aday yön. Kaynaklar: Faz 1 (`2026-09-16-e2e`), R1–R6 (`2026-09-17-diagnostics`), sürüklenme mini deneyi (`2026-09-18-drift`), Faz 2 G0/U1/B1/D1 (`2026-09-18-phase2`). Sıra önem sırası değildir.

---

## A. Çekirdek dinamik (yinelemeli durum)

**A1. Kurulmuş durum boş koşuda kayıyor (sürüklenme) — MEVCUT, en büyük.**
Kanıt: 4 kampanyada tutarlı. U1 beş-durum: nötr 128/512 orta/son −31…−80 puan; ön ek ≈ 0. Denetçi: koşu sonunda tahminlerin %79–85'i tek sınıfa yığılıyor (tek çekiciye çöküş).
Neden: "sonsuz derinlik" iddiasının kalbi; ek hesaplama turu bu durumu daha çok bozar.
Aday yön: U2 ölçüm katmanı (çapa merkezleri, yörünge uzaklığı, onarım eğrisi) → çapaya çekme / son gerçek işlemden yeniden okuma; kimlik geçişi için ayrı invariant (U3).

**A2. Üç ucuz çare işe yaramadı — MEVCUT.**
Pencere 4→16 sürüklenmeyi artırdı (parite −45…−49 puan); nötr-değişmezlik kaybı 32'de +7/+9 ama 128 ön ekte −53; reread 2/6 ıraksadı ve kalanlarda daha kötü.
Aday yön: müdahaleyi eğitim-dağılımı hilesi değil, sabit-nokta yapısına yönelik kurmak.

**A3. Zararın görünürlüğü göreve bağlı (permütasyon vs üzerine yazma) — MEVCUT.**
Yığında orta ek zararsız (sonraki push'lar üst sembolü ezer), beş-durumda orta ek son kadar yıkıcı. Risk: yanlış görev seçimiyle sorun "çözülmüş" görünür.
Aday yön: her müdahale iki cebirde (onarılabilir / onarılamaz) test edilmeli.

**A4. Başlangıç durumu kararlı, bilgi taşıyan durum değil — MEVCUT (yorum).**
512 nötr ön ek zararsız. Öğrenilen dinamiğin başlangıçta bir kararlı nokta, bilgi taşıyan bölgelerde ise çekici var.
Aday yön: durum uzayındaki sabit noktaları ve Jacobian spektrumunu doğrudan ölçmek (U2/U3).

**A5. Uzak kredi ataması yok — MEVCUT.**
Yalnız son etiketle öğrenme şansta (R3, Faz 1). Tek rastgele konumlu etiket = yoğun etiket; sorun etiket sayısı değil, mesafe (tbptt8-final'de başlangıç parametresi hiç güncellenmedi).
Neden: "üretilmiş izleri sonraki eğitime aktarma" ve gecikmiş ödülle öğrenme fikirlerinin ön koşulu.
Aday yön: ara hedef üretimi (doğrulanmış aday adımlardan), yardımcı kayıplar; RL değil.

**A6. Büyüyen bellek görevi öğrenilmiyor — MEVCUT (bütçeye koşullu).**
Yığın: eğitim ufkunda %94, 4096'da %72–81; doğrulama kaybı bütçede hâlâ düşüyordu.
Aday yön: daha uzun eğitim (yeni dondurma), genişlik/durum boyutu taraması; sabit-durum başarısı buraya atfedilmez.

**A7. Tohum piyangosu — MEVCUT.**
Aynı protokolde bir tohum 4096'da %94,7, diğerleri %100; Faz 1'de nötr son ek 1/3, tanı kampanyasında 3/3 dayandı; büyük modelde LR ile 0/3→3/3.
Aday yön: tohum sayısı ≥5 ve tohum başına rapor; "n tohumdan k'si" dili.

**A8. Gradyan sıçramaları — MEVCUT.**
Kırpma öncesi norm beş-durumda 17.766'ya, reread'de inf'e. Kırpma çoğunu tutuyor; kararlılık kapısı yok.
Aday yön: gradyan/norm günlüğü + kararlılık kapısı her protokolde zorunlu.

**A9. Öğrenme oranı duyarlılığı ölçeğe göre değişiyor — MEVCUT.**
w256'da 0,003 3/3 başarısız, 0,0003/0,001 3/3 başarılı. Her yeni ölçekte LR taraması şart.

**A10. Geç öğrenme (plato) — MEVCUT.**
Beş-durumda doğrulama kaybı 256/384/1152. adıma kadar şansta kalıp aniden düşüyor; sabit bütçe bazı tohumları yarım bırakıyor.

## B. Doğrulama, başlık ve bellek katmanı

**B1. Doğrulayıcı kapsamı düşük — MEVCUT.**
IID'de kabul %39, yeni derinlikte ≈ 0; yol sınırı yalnız dallanmış topolojide 6/0/22 ek kabul. Depth-8 abstention çözülmedi.
Aday yön: kapsam problemi aday kalitesinde (bkz. B3); sertifika mekanizması değil.

**B2. Anlamsal erişim hedefi çok uzak — MEVCUT.**
MuSiQue geliştirme: dense kapsama@10 0,445, hibrit 0,36 (daha kötü!), +reranker 0,515; hedef %90/95. Çok atlamalı sorularda ikinci atlamanın paragrafı sorguyla bağ taşımıyor.
Aday yön: iteratif (atlama-atlama) erişim; sorgu yeniden yazımı; hibrit füzyonu yeniden tasarla (RRF zarar veriyor).

**B3. Öğrenilmiş ayrıştırma yeni kalıba aktarılmıyor — MEVCUT (Faz 1).**
Yeni cümle düzenlerinde tam abstention. D1 bunu gerçek dil modeliyle ölçecek (sonuç bekleniyor).

**B4. Seçici revizyon negatif — MEVCUT (Faz 1).** Etkilenen cevapta ≥%10 kazanç kapısı geçilmedi.

**B5. Sözcük-karma erişimi negatif — MEVCUT.** recall@5 %4; B1'de taban kol olarak duruyor.

**B6. Sıkıştırma: son z'yi saklamak çalışıyor, anlamsal özet değil — MEVCUT.** Kısıt özetinden yeniden başlatma derinlik 8'de 37→21 puan.

**B7. Başlık grafiği tek başına yeni derinlikte yetersiz — MEVCUT.** Çift yön IID'de %96, derinlik 8'de %10; ara gözetimle %33.

**B8. Symbolica/Graphica park — KARAR.** Ayrıştırma/aday kalitesi çözülmeden motor eklemek anlamsız; olası yeri başlık-grafiği izomorfizmi (B2/B3).

## C. Ölçüm ve yöntem

**C1. Test kümesi kampanyalar arası yeniden kullanıldı — MEVCUT (açıklandı).**
Sürüklenme deneyi ve tanı kampanyası aynı test tohum uzayını kullandı; seçim için kullanılmadı ama "taze" değil. Faz 2'de yeni uzaylar açıldı.

**C2. Üç tohum + keşif aralıkları — MEVCUT.** t95 aralıkları çokluk düzeltmesiz; iki-tohumlu satırlar df=1.

**C3. Türkçe dilimi yok — MEVCUT.** MuSiQue/BigCodeBench İngilizce; çeviri = model yazımı, ayrı etiket gerekir; insan anotasyonu yok.

**C4. Model-yazımı veri riski — OLASI.** D1 kontrolleri programatik pertürbasyon (insan verisinin dönüşümü); LLM ile üretilecek her veri ayrı etiketlenmeli, altın yerine geçmemeli.

**C5. Tek model hakemi — OLASI.** D2 açık uçlu cevaplarda hakem kalibrasyonu insan olmadan sertifika değil.

**C6. Nicemleme sadakati ölçülemedi — MEVCUT.** 4B fp16 8 GB'a sığmıyor; nf4 vs fp16 karşılaştırması yok; D1/D2 sonuçları nicemlemeye koşullu.

**C7. "Sabit sınıf çöküşü" gibi anomalilerin kısmi okunması — YAŞANDI.** Denetim JSON'unun ilk 16 satırından genelleme yaptım; denetçi düzeltti. Kural: liste tamamı sayılmadan sonuç yazılmaz.

**C8. Faz 1 test paketi worktree'de 7 testte ham dosya ister — MEVCUT.** `memory/checkpoints/extractor-17.pt` git dışı; mühürlü kökte 108/108. Yeni köklerde bilinçli olarak "beklenen başarısızlık".

**C9. Çoklu son nokta / seçici raporlama — OLASI.** 71 karşılaştırmadan olumlu seçmek tehlikeli; birincil son noktalar önceden dondurulmuş, ama okur bunu bilmeli.

**C10. İki kampanyayı birleştirme cazibesi — OLASI.** Farklı tohum/protokol pin'leri; tohumlar bağımsız örnek sayılmaz.

## D. Donanım, performans, ölçek

**D1. Zaman ekseni paralelleşmez — YAPISAL.** RLT'de s_t→s_{t+1} doğrusal olmayan; dizi-paralel eğitim yok.

**D2. Kernel-başlatma darboğazı — MEVCUT.** w32'de GPU CPU'dan yavaş (0,25 vs 0,14 s/güncelleme); GPU boş dönüyor.
Aday yön: batch büyütme, CUDA Graphs / torch.compile(reduce-overhead), encoder/decode boru hattı.

**D3. Bellek: detach-only TBPTT hiç tasarruf etmiyor — MEVCUT.** Segment yürütmesi L1024'te tepeyi 0,53×; checkpoint 1,9× yavaş ve tasarrufsuz.

**D4. 8 GB VRAM / 4 çekirdek / 16 GB RAM — MEVCUT.** 4B fp16 sığmaz; 8-bit 2,0 tok/s, nf4 7,2 tok/s. D4'ün 30M→300M dil eğitimi yerelde ölçülmeden söz verilemez; kiralık GPU yok (karar).

**D5. Değerlendirme maliyeti — MEVCUT.** L4096'da 1.000 problem 550 s; reranker sorgu başına 2,9 s (B1 final ≈ 3 saat); D1 örnek başına ~15–30 s.

**D6. Bant genişliği/disk — MEVCUT.** 4B indirme 23 dk (~5 MB/s); E: 57 GB boş; LongMemEval-M (2,6 GB) indirilmedi.

**D7. CPU çekişmesi ölçümü kirletiyor — YAŞANDI.** R6 L512 satırları host yükü 0,50/0,57 ile güvenilmez işaretlendi; eşzamanlı işler süre ölçümlerini bozuyor.

**D8. 16.384 / 65.536 uzunlukları açılmadı — MEVCUT.** Kalibrasyona göre yapılabilir (~40 dk/koşul) ama protokolde yok.

**D9. Reranker ileri geçişi bütün konumlar için sözlük-boyutlu logits üretiyor — YAŞANDI.** B1 finali 58. dakikada CUDA OOM (4,63 GiB tek tahsis: batch×seq×151.936×2 bayt). Düzeltme: taban model + yalnız son konuma lm_head. Genel ders: 8 GB'lık kartta her `CausalLM.forward` çağrısı logits şeklini kontrol etmeli.

## E. Süreç, altyapı, tekrar üretilebilirlik

**E1. Windows venv kurulumu — YAŞANDI.** WinError 32 (geçici dosya kilidi) cu130 torch'u düşürdü, sonraki adım CPU torch kurdu; E:'de TEMP ile düzeldi. Kural: kurulum sonrası CUDA doğrulaması zorunlu.

**E2. venv `python.exe` yönlendirici — MEVCUT.** Popen.pid ≠ işçi pid; launcher pid'leri bağlayıcı değil, işçi tarafı pid zinciri esas.

**E3. Satır sonu / kodlama olayları — YAŞANDI.** write_text CRLF'ye çevirdi; konsol UTF-8 değil (Türkçe bozuk görünür); dosyalar UTF-8. Kural: bayt-düzeyi yazım, hash öncesi kontrol.

**E4. API kota kesintisi — YAŞANDI.** Denetçi ajan 429 ile kesildi; JSON tamamdı, md sabah üretildi. Her ajan çıktısını önce makine-okunur yazmalı.

**E5. Kontrolör devir teslimi — YAŞANDI.** Codex→Claude geçişi ledger + `events.jsonl` ile doğrulandı; ledger'ın güncel tutulmaması riski (01:09'da kalmıştı).

**E6. Mühürleme ve sonradan ekleme — MEVCUT.** Kök-geneli manifest, yeni dosyalarla geçersizleşir; Faz 2'de döngü-kapsamlı manifestlere geçildi; kapanışta kök-geneli tek manifest gerekir.

**E7. Runner'lar kesintide devam etmiyor (fail-closed) — MEVCUT/KARAR.** Belirsiz kesilmiş segment yeniden yazılmaz; çöken koşul INCOMPLETE. Uzun koşularda maliyet: yeniden koşu gerekir.

**E8. Heredoc/tırnak hataları — YAŞANDI.** Kabuk üzerinden Python yaması birkaç kez bozuldu (bir kez ledger'a REPL başlığı sızdı, düzeltildi). Kural: yamalar dosya olarak.

**E9. Git remote yok — MEVCUT.** Commit'ler yerel; disk kaybı = kayıp. Yedekleme kararı yok.

**E10. Kod tekrarı — MEVCUT.** drift/run.py, horizon/sweep.py, semantic_memory/retrieval.py, language/run_d1.py aynı freeze/receipt kalıbını kopyalıyor; ortak çekirdek yok (kasıtlı: dondurulmuş kaynakları ayrı tutmak için, ama bakım maliyeti).

**E11. Test paketleri dağınık — MEVCUT.** `tests/` (Faz 1 + yeni), `review_cycle/test_*.py`, `drift/test_drift.py`; tek komutla tümü koşmuyor.

**E12. Zincir loglarında yanlış çıkış kodu — YAŞANDI.** `echo "$(date) exit=$?"` deseninde komut ikamesi `$?`'yi sıfırlıyor; B1 çökmesi "exit=0" göründü. Kural: `rc=$?` önce yakalanır; karar dosya varlığından verilir.

## F. Faz 2 planı ve iddialar

**F1. B1 hedefleri (recall@10 ≥ %95, kapsama ≥ %90) bu yöntemlerle ulaşılmaz görünüyor — MEVCUT.** Final koşuyor; olumsuz çıkarsa "anlamsal bellek" iddiası sıfır noktasında.

**F2. D1 hedefleri (F1 ≥ 0,95, risk ≤ %1, kapsam ≥ %80) — OLASI zor.** 4B nf4 ile serbest çıkarım; sonuç henüz yok.

**F3. D2 3.000 problem × kollar × bütçeler = 12.000+ bölüm — OLASI.** Yerelde 7 tok/s ile haftalar; kapsam daraltılmalı veya donanım kararı.

**F4. D4 (gerçek RLT dil eğitimi) yerelde fizibilite belirsiz — OLASI.** 30M pilotunun 100 güncellemelik maliyeti ölçülmeden plan yapılamaz; D2 ile karıştırılmamalı (INCONCLUSIVE_CAPABILITY_FLOOR riski).

**F5. U3 (biçimsel güvence) — OLASI.** Öğrenilmiş geçişin ağırlık kimliğine bağlı invariant: nötr geçişin daraltıcı olmadığı ölçüldü → teorem hedefi kimlik geçişi için ayrı koşul ister; Lean kurulumu yok.

**F6. U1 k ve d eksenleri ertelendi — MEVCUT.** Başlık turu ve bağımlılık derinliği eksenleri koşulmadı; U1 kısmi.

**F7. "Genel doğal dil / sınırsız doğru düşünme" iddiaları için kanıt yok — MEVCUT.** Bütün sonuçlar sentetik/kontrollü ve İngilizce MuSiQue.

**F8. LongMemEval-V2, Huginn karşılaştırması, enerji ölçümü — YAPILMADI.**

## G. Açık kararlar (senin vereceğin)

**G1.** Sürüklenme için mimari hamle: çapalama (dış müdahale) mi, eğitim hedefi mi, mimari değişiklik mi? (U2 ölçümü sonrası.)
**G2.** D2 kapsamı: 3.000 problem yerelde mi, daraltılmış mı, kiralık GPU mu?
**G3.** Türkçe: model çevirisi (etiketli) mi, insan verisi mi, UNMEASURED mı kalacak?
**G4.** Yedekleme: git remote / dış disk.
**G5.** Kod ortaklaştırma vs dondurulmuş kaynak ayrımı (E10).
**G6.** 16k/64k uzunlukları ve daha uzun yığın eğitimi için bütçe.
**G7.** Symbolica/Graphica: hangi koşulda yeniden açılır?
