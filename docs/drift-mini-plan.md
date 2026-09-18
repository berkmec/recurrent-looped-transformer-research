# Sürüklenme mini deneyi — ön-kayıtlı plan (2026-09-18)

**Soru:** R2/R5'te ölçülen durum sürüklenmesi (durum kurulduktan sonra gelen uzun nötr koşularda son-durum doğruluğunun çökmesi) eğitim verisi değiştirilmeden, aynı girişler/başlangıçlar altında, yalnız (a) daha geniş yerel pencere, (b) nötr girişte durum-değişmezlik kaybı veya (c) durum-geçmişine attention ile azalır mı?

**Kök:** `E:/rlt-research/2026-09-18-drift`, dal `codex/rlt-drift-mini`, taban `f83b527`. `core/model.py` ve `review_cycle/*` salt-okunur içe aktarılır; yeni kod `drift/` altında. Mühürlü kökler değişmez.

## Kollar (hepsi rastgele yerleşim, yoğun gözetim, w32/2 baş, lr 0,003, AdamW wd 0,01 clip 1, batch 32, uzunluk 32, 2048 güncelleme, tohumlar 117/143/191)

| Kol | Değişen tek şey | Kaynak sadakati |
|---|---|---|
| `base` | yok (Core pencere 4, decode döngüsüyle eğitim) | RLT çekirdeği |
| `win16` | Core pencere 16 | RLT çekirdeği, hiperparametre |
| `idinv` | ek kayıp λ·E_nötr[‖z_t − z_{t−1}‖² / (E‖z_{t−1}‖² + ε)], λ = 1, yalnız x_t = 2 konumlarında | RLT çekirdeği + eğitim kaybı |
| `reread` | her adımda z_t ← z_t + W_o·Attn(q = W_q·RMS(z_t), K/V = W_k/W_v·[z_k ; e_k], k < t), tek baş, nedensel, W_o sıfır-başlangıç; okuma ve sonraki adım z_t' ile | **sapma:** durum-geçmişine attention, RLT değil |

Aynı görev/tohumda dört kolun Core başlangıç ağırlıkları ve 2048 eğitim minibatch'i özdeş (hash ile denetlenir). `reread` ek parametreleri ayrı tohumla başlatılır; W_o = 0 olduğundan ilk ileri geçiş `base` ile aynıdır (test).

## Görevler ve son noktalar

Birincil görev **beş durum** (R2'de en büyük sürüklenme). İkincil parite. Her seçilmiş model (doğrulama son-durum CE minimumu, her 128 adımda 256 örnek) taze test kümelerinde: uzunluk 32/128/1024 (512'şer) ve aynı 32'lik tabana 0/8/32/128 nötr ön/orta/son ekleme; kaynak tohum uzayı tanı kampanyasıyla aynı (`training_data.source_seed`, test/nötr ad alanları), dolayısıyla test girişleri mühürlü tanı kampanyasındakilerle özdeştir; bu deneyde seçim için kullanılmaz.

**Birincil son nokta:** beş durum, nötr-128 **orta** ve **son**; kol − base son-durum doğruluğu, tohum başına eşli, 3 tohum ortalaması ve t95 (df = 2, 4,3027·s/√3). **İkincil:** nötr-32 orta/son, L1024 temiz doğruluk (bedel), parite aynı son noktalar.

## Karar kuralı (ölçümden önce)

- Bir kol için `MEASURED_POSITIVE`: birincil iki son noktada da ortalama ≥ +20 puan ve t95 sıfırı dışlıyor **ve** L1024 temiz doğruluk kaybı ≤ 5 puan.
- `MIXED`: yön tutarsız veya aralık sıfırı kapsıyor.
- `MEASURED_NEGATIVE`: ortalama ≤ 0 ya da L1024 bedeli > 5 puan.
- `win16` beklenti: etki yok (pencere kontrolü). Hiçbir sonuç doğal dile veya büyük modele genellenmez.

## Bütçe ve sağlama

24 koşul, 3 CPU yuvası (tek iş parçacığı), tahmin 1–1,5 saat eğitim + ~30 dk değerlendirme. `base` kolunun test sonuçları mühürlü tanı kampanyasının `small-{task}-w32-rlt-random-dense` sonuçlarıyla karşılaştırılır (aynı tohum, aynı girişler; farklı yürütme yolu: decode döngüsü vs `forward`) — sapma varsa rapor edilir. Kırmızı/yeşil test kanıtı `drift/test_drift.py`. Protokol ve kaynak hash'leri final test açılmadan `results/drift/frozen.json` ile dondurulur; olumsuz sonuç geçerli sonuçtur.
