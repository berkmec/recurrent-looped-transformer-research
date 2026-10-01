# RLT ve başlık belleği — açık araştırma kaydı

**English summary.** An open, evidence-first research record on the *Recurrent Looped Transformer* (RLT) and a "heading memory" built around it. It covers a fourteen-package first phase, diagnostic cycles (R1–R6), a drift mini-experiment, and a second phase: long-horizon state drift and a learned HOLD/WRITE gate, semantic retrieval, language parsing, and a 30M byte-level model with its long-distance problem. Protocols are frozen with SHA-256 before final data is opened, each cycle ends with a raw-data audit and a sealed manifest, seeds are reported one by one, and **negative results are kept and reported as results**. The prose is Turkish; code, numbers and verdict labels (`MEASURED_NEGATIVE`, `MEASURED_POSITIVE`, `MIXED`, …) are language-neutral. Model weights, raw prediction arrays and SQLite archives are *not* in this repository — see [Veri erişilebilirliği](#veri-erişilebilirliği). Licence: MIT.

## Bu depo ne?

Küçük, kontrollü modellerle şu soruların kaydı: yinelemeli bir çekirdek uzun bir hesaplama boyunca doğru durumu koruyabilir mi; yapılandırılmış bir "başlık belleği" doğru bilgiyi bulup kullanabilir mi; ve bunlar doğal dile ve daha büyük bir bayt-düzeyi modele taşınır mı? Her döngü bir soru, veri gelmeden yazılmış bir karar kuralı, dondurulmuş bir protokol, sonuç ve çoğunda bağımsız bir ham veri denetimi içerir. Başarısızlıklar da korunur. Birkaç döngü kendi ön iddiasını kendi verisiyle düzeltmiştir; bu düzeltmeler ilgili raporlarda silinmeden, eski metnin yanında durur.

RLT fikri ve resmî kodu: <https://github.com/yifanzhang-pro/recurrent-looped-tranformer>. Bu depo onun yeniden üretimi değildir; küçük bağımsız uygulamalar kullanır.

## Başlıca bulgular

Sınırlar ve ayrıntı her satırdaki raporda. Sonuçlar sentetik görevlerde ve küçük modellerde ölçülmüştür; doğal dile ya da büyük modele genellenmez.

| Döngü | Soru | Sonuç | Rapor |
|---|---|---|---|
| Faz 1 | On dört paket: RLT çekirdeği, başlık belleği, pencere, doğrulayıcı… | Karışık ve olumsuz sonuçlar korunmuştur | [RLT_SONUC_RAPORU](/reports/RLT_SONUC_RAPORU.md) |
| R1–R6 | Faz 1 sonuçlarının tanısı | Segment yürütmesi L1024'te CUDA tepe belleğini 0,53×'e indirir; yoğun etiket ile tek rastgele etiket aynı sonucu verir (son-konum etiketi ve geri beslemesiz çekirdek şansta kalır); Faz 1'deki "3 tohumdan 1'i öğrendi" sonucu öğrenme oranı duyarlılığıydı | [R1–R6 raporu](/notes/RLT_R1-R6_SONUC_RAPORU.md) |
| Sürüklenme mini | Durum kurulduktan sonraki uzun nötr koşuda doğruluk çöküşü üç ucuz araçla azalır mı? | Hayır: geniş yerel pencere, değişmezlik kaybı ve durum-geçmişine attention'ın üçü de `MEASURED_NEGATIVE` | [DRIFT_MINI_RAPORU](/notes/DRIFT_MINI_RAPORU.md) |
| U1 | Eğitim ufku 128 iken 4096'ya uzunluk genellemesi | Sabit bellekli görev genelliyor (beş-durum: 3 tohumdan 2'si 4096'ya kadar %100); büyüyen bellekli görev (yığın) bu bütçede öğrenilemedi; nötr koşular cevabı yıkıyor (yaklaşık 44–59 puan) | [U1_RAPORU](/notes/U1_RAPORU.md) |
| U2 / U2b | Sürüklenme nedir, onarımın tavanı ne? | Sürüklenme nötr koşu kaybının tamamıdır; nötr adımı bilen (oracle) koşu-farkında dondurma kaybı sıfırlar (tavan ölçümü) | [U2_RAPORU](/notes/U2_RAPORU.md) |
| U2c / U2d | Modelin kendi öğrendiği HOLD/WRITE kapısı oracle'ın yerini tutar mı? | Evet, 32 genişlikli oyuncak modellerde: tam-kopya kapı (`hard_st`) iki görevde ve iki öğrenme oranında pozitif kalan tek kol; kampanyanın ilk pozitif mimari sonucu. Sınır: iki sentetik görev, üç tohum | [U2c_RAPORU](/notes/U2c_RAPORU.md) |
| U4 | Dikkat enjeksiyonu; çapa ayrımı yardımcı kaybı | Üçü de olumsuz: enjeksiyon kolları 6/6 çöktü; U4-sep kendi sorusunu sınamadı; U4-sep2'nin ön kayıtlı nedensel iddiası reddedildi | [inject](/notes/U4inject_RAPORU.md), [sep](/notes/U4sep_RAPORU.md), [sep2](/notes/U4sep2_RAPORU.md) |
| B1 / B2 | Anlamsal bellek erişimi (MuSiQue, 2.417 sorgu) | `MEASURED_NEGATIVE`: en iyi kol recall@10 0,665 / kapsama@10 0,325 (hedef 0,95 / 0,90); köprü terimli çok-atlamalı erişim 0,663 / 0,336 | [B1](/notes/B1_RAPORU.md), [B2](/notes/B2_RAPORU.md) |
| D1 (dil) | 4B modelin serbest metinden başlık/iddia/kaynak ayrıştırması | `MEASURED_NEGATIVE`: ilişki-F1 0,03 (hedef 0,95); yanlış-kabul riski üst sınırı 0,97; bozulmuş kontrollerin %62–83'ünü yine de kabul ediyor | [D1_RAPORU](/notes/D1_RAPORU.md) |
| M-hattı | 30M bayt modelinde uzak bilgi ve kapı | Kapı-aktarım hükmü (`GATE_DOES_NOT_TRANSFER`) kapıyı gerçekte sınamadı, kapı hiç kapanmadı; mesafe uçurumu eğitim aralığı etkisi (ince ayarla ≈6× kaydı); iğne görevi konuma kilitli çıktı; yem-olgu/MQAR ince ayarları anahtar kullanımını kısmen ve harfe özgü başlattı (gerçek kelime/isim anahtarlarında 0,00) | [M1 ek notu](/notes/M1_EK_KAPI_OLU_2026-10-01.md), [M-hattı planı](/notes/M_HATTI_PLANI.md) |

Not: "D1" adı iki yerde geçer. Faz 2'nin D1'i dil ayrıştırmasıdır (`results/D1-final`); M-hattının D1–D6'sı yem-olgu/MQAR ince ayarlarıdır (`results/M1-finetune-distance/D*`).

## Güncel durum (2026-10-01)

M-hattı sürüyor: D5/D6 (53 sembollü anahtar alfabesi ve gerçek anahtarlarla MQAR ince ayarı), T1 serisi (küçük çekirdekte sıfırdan anahtar eşleştirme) ve L1 (sıfırdan 20M parametreli bayt modeli, 500M bayt, plato yüzünden durdurulmayan uzun koşu; hazır, henüz başlamadı). Sonuçlar geldikçe notlara ve bu depoya eklenir.

## Depo düzeni

- `core/`, `learning/`, `memory/`, `integration/`, `reports/`: Faz 1 (mühürlü teslimat; ayrıntı için kök `README.md`).
- `review_cycle/`: R1–R6 tanı döngüsü. `drift/`: sürüklenme mini deneyi.
- `horizon/`: Faz 2 U-döngüleri (uzun ufuk, HOLD/WRITE kapısı, müdahaleler). `semantic_memory/`: B1/B2. `language/`: D1 (dil).
- `mline/`: M-hattı (bayt düzeyi dil modeli, iğne/MQAR probları, dikkat probları, uzun koşu eğiticisi).
- `protocols/`: dondurulmuş protokoller ve veri/model sürüm sabitlemeleri. `results/`: sonuç dosyaları ve günlükler. `notes/`: döngü raporları.
- `audit/`: mühür manifestleri (boyut + SHA-256), bağımsız ham veri denetimi betikleri. `.superpowers/sdd/`: ajan defterleri, denetim raporları, test günlükleri ve ortam dökümü (`venv-freeze.txt`).
- `tests/`: pytest takımı (PyTorch gerekir; bazı testler Git dışındaki varlıkları, örneğin `memory/checkpoints/extractor-17.pt` dosyasını ister). `scripts/`: kuyruk ve koşu betikleri.

Dallar: `main` Faz 2 zincirinin ucudur; `codex/rlt-e2e` (Faz 1), `codex/rlt-review-diagnostics` (R1–R6) ve `codex/rlt-drift-mini` tarihsel mühür noktalarıdır ve `main`'in atalarıdır.

## Veri erişilebilirliği

Git'e girmeyenler (`.gitignore`): model ağırlıkları ve kontrol noktaları (`*.pt`; yalnız Faz 2 kökünde yerelde ≈8,7 GB), ham tahmin dizileri (`*.npz`) ve Faz 1'in SQLite arşivleri (`*.sqlite`; 7 binden fazla dosya). Bunlar `audit/*manifest*.json` içinde boyut ve SHA-256 ile listelidir; bu yüzden manifest girdilerinin çoğu bir klonda bulunmaz. Depodaki dosyalar ise ilgili döngünün **son** manifestindeki karmayla bayt bayt örtüşür (ilk sürümlerde görülen farklar, belgelenmiş düzeltmelerden sonra yazılan `-r2` / `-r3` manifestlerinde yoktur); `.gitattributes` bu baytların satır sonu dönüşümüyle bozulmasını önler. Ham dizilerin ve ağırlıkların ayrı bir arşiv olarak yayımlanıp yayımlanmayacağı henüz karara bağlanmadı.

## Nasıl üretildi

Çalışma, bir araştırmacının yönlendirmesinde yapay zekâ ajanlarıyla (OpenAI Codex ve Anthropic Claude Code) yürütüldü. Her döngü ayrı bir ham veri denetiminden geçirilip mühürlendi; bunlar hakem denetimi değildir ve bazı denetimlerin bağımsızlık sınırı (örneğin aynı oturumda yapılmış olması) ilgili raporda yazılıdır. Hatalar ve düzeltmeler kasıtlı olarak silinmemiştir.

## Üçüncü taraf veri ve modeller

Bu depo üçüncü taraf veri kümesi ya da model ağırlığı dağıtmaz. Kullanılan veya sabitlenenler: MuSiQue, LongMemEval, BigCodeBench, CNN/DailyMail (bayt düzeyi dil modeli korpusu) ve Qwen3 modelleri. Sürüm sabitlemeleri `protocols/G0-pins.json` içindedir; her birinin kendi lisansı ve atıf koşulları geçerlidir. Sonuç dosyaları bu verilerden türetilmiş kimlikler, kısa alıntılar veya model çıktıları içerebilir.

## Lisans

MIT — bkz. [LICENSE](/LICENSE).
