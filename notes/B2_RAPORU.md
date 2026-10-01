# B2 — Çok-atlamalı erişim (köprü terimi → ikinci sorgu), MuSiQue (2026-09-20)

## Bulgular (önce)

1. **MEASURED_NEGATIVE; B1'in en iyi koluna göre kazanç ölçüm gürültüsü düzeyinde.** Aynı 2.417 sorgu / 21.100 paragraf: `multihop_rerank` recall@10 **0,663** [0,653, 0,674], kapsama@10 **0,336** [0,318, 0,353] — B1 `hybrid_rerank` 0,665 / 0,325. Hedefler (0,95 / 0,90) uzak. `multihop` (yeniden sıralamasız birleşim) dense'e göre +0,4 puan recall, +1,8 puan kapsama.
2. **Sebep aday üretiminde değil, köprü teriminde.** Birleşim tavanı (tüm altınlar 31 belgelik birleşim içinde) **0,488** — yani mükemmel bir reranker bile bu adaylarla kapsamayı 0,49'un üstüne çıkaramazdı. Ucuz köprü çıkarıcı (soruda geçmeyen büyük harfli terimler) genel sözcükler üretiyor: en sık köprüler "American" (207 sorgu), "United" (122), "States" (79), "British", "John", "List". İkinci atlama bu yüzden yeni altın getirmiyor: son top-10'daki 4.060 altın isabetin yalnız **247'si** (%6) ikinci atlamadan geldi.
3. **Atlama sayısına göre kapsama (multihop_rerank / B1 hybrid_rerank):** 2 altın 0,502 / 0,470; 3 altın 0,233 / 0,254; 4 altın 0,015 / 0,012. İki atlamalıda +3 puan, üç atlamalıda −2 puan: köprü mekanizması tam da hedeflediği uzun zincirlerde çalışmıyor.
4. **Geliştirme (100 train sorgusu, 1.707 paragraf) iyimser görünmüştü:** kapsama@10 0,56 → 0,66, tavan 0,86 — küçük korpusta rastgele köprü bile isabet ediyor. Final bunu doğrulamadı; geliştirme diliminin ölçek geçerliliği yok (B1'de ölçülen merdiven etkisiyle tutarlı).
5. **Maliyet:** sorgu başına 2,2 s (dense 3 ms, B1 reranker 1,9 s); toplam 1,5 saat; reranker 76.043 çift.

## Yöntem

- `semantic_memory/multihop.py` (testler `tests/test_multihop.py` 3/3, `tests/test_run_b2.py` 1/1 GREEN, zincir içinde tekrar koşuldu): soru → dense ilk atlama (k1 = 20) → ilk 3 paragraftan köprü terimleri (`capitalised_terms_v1`: soruda geçmeyen, durak-sözcük olmayan büyük harfli terimler, en çok 3) → "soru + köprü" ikinci sorguları (k2 = 20) → RRF birleşimi (k = 60, kaynak izi hop 1/2) → reranker (Qwen3-Reranker-0.6B, derinlik 50) → **tam 10 belge**. Kollar: `dense` (ilk atlama), `multihop` (birleşim), `multihop_rerank`. Metrikler son listede, k'de kesilmiş; birleşim tavanı ayrı.
- Kodlayıcı B1'in geliştirme seçimi (Qwen3-Embedding-0.6B, L256, talimat açık); B1 pinleri; sorgu vektörleri önbellekli.
- Protokol `results/B2/frozen.json` (hash `7c658de8…`), kaynak hash'leri; makbuz `results/B2/receipt.json` COMPLETE (run_id `b2-20260920T101051Z`, testler geçti, 2/2 artifact). Ham sıralamalar ve köprüler `final-per-query.json`.

## Yorum

- Tasarım paketinin B2 önerisi ("ilk kanıttan ikinci sorguyu türet") ilkesel olarak doğru yerde — B1 boşluğun aday üretiminde olduğunu göstermişti — ama **köprü çıkarımı** çözülmeden işe yaramıyor. Bu koşu, çıkarıcının en ucuz sürümünün yetersizliğini ölçtü; sonraki kol köprü terimini LM ile (ilk paragraftan varlık/ilişki çıkarımı, IRCoT tarzı) üretmeli ve önce **tavanı** (birleşimde tüm altınlar) izlemeli: tavan 0,49'dan 0,9'un üstüne çıkmadan reranker'a bakmanın anlamı yok.
- D1 sonucu burada bir uyarı: aynı 4B modelin ayrıştırma çıktısı %7–8 doğruydu; LM köprüsü de kalibrasyon sorununu taşıyabilir. Köprü kalitesi ayrı ölçülmeli (köprü terimi altın ikinci paragrafta geçiyor mu).
- Türkçe dilimi UNMEASURED. Deterministik erişim, tohum yok; belirsizlik sorgu-grubu bootstrap'ından.

## Yeniden üretilebilirlik notu (denetim bulgusu)

B2'nin `dense` kolu B1'in `dense` koluyla aynı kodlayıcı ayarı, aynı korpus ve aynı sorgulara sahip; buna rağmen **2.417 sorgunun 254'ünde (%10,5) top-10 listesi en az bir konumda farklı** çıktı (recall@10 0,599538 vs 0,598986; Δ = −0,055 puan). Sebep fp16 GPU kodlamasının koşular arası bit-aynı olmaması. Sonuçları etkilemiyor, ama karşılaştırmalı kollar bundan sonra **aynı vektör matrisini paylaşmalı**; tekrar üretilebilirlik iddiası "bit-aynı" değil "metrik olarak ≤ 0,1 puan" düzeyindedir.

## Durum

`dense`, `multihop`, `multihop_rerank`: **MEASURED_NEGATIVE**. Denetim: `.superpowers/sdd/phase2/audit-b2.md` — **14 PASS / 0 FAIL / 0 UNVERIFIED**, rapor sayıları 16/16 doğrulandı; denetimin bağımsızlık sınırı orada belirtilmiştir (ayrı ajan haftalık API kotasına takıldı, denetim aynı oturumda ayrı bir betikle yapıldı). Mühür: `audit/phase2-B2-manifest.json`.
