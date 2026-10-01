# B2 bağımsız denetim — 2026-09-22

**Sonuç: 14 PASS / 0 FAIL / 0 UNVERIFIED.** Rapordaki 16 sayının 16'sı saklanan değerlerle birebir örtüşüyor; hiçbir düzeltme gerekmedi.

**Denetimin bağımsızlık sınırı (önce söylenmeli):** bu denetim ayrı bir ajanla başlatıldı ama ajan haftalık API kotasına takılıp düştü (`rate_limit`, 20 Eylül); denetim bu yüzden **aynı oturumda, ayrı bir betikle** (`C:/Users/pc/.claude/jobs/4797b78c/tmp/audit_b2.py`) yeniden yapıldı. Metrikler (recall/coverage/ndcg/mrr, grup-bootstrap) deney kodundan bağımsız olarak yeniden yazıldı ve ham `final-per-query.json` üzerinden hesaplandı; ancak "yazan ile denetleyen aynı model" sınırı bu döngü için geçerlidir. U2/B1/D1 döngülerinin denetimleri ayrı ajanlarla yapılmıştı.

## Kontroller

| # | Kontrol | Sonuç |
|---|---|---|
| 1a | `frozen.sha256` = sha256(frozen.json); `protocol_hash` = kanonik JSON özeti (`7c658de8…`) | PASS |
| 1b | 5 kaynak dosya hash'i çalışma ağacıyla aynı | PASS |
| 1c | Protokol değerleri k1 20 / branches 3 / k2 20 / RRF 60 / rerank 50 / top_k 10 / `capitalised_terms_v1` / L256; veri ve model pinleri B1 ile birebir aynı | PASS |
| 2 | Makbuz COMPLETE, rc 0, 2/2 artifact, hash'ler güncel; `b2-tests.log` 4 test OK | PASS |
| 3 | 2.417 sorgu / 21.100 paragraf; altın dağılımı 1252/760/405 ham dosyadan yeniden sayıldı; `per_query_sha256` ve veri kümesi hash'i doğru | PASS |
| 4a | Üç kolun tüm metrikleri (k=1,5,10) ham sıralamalardan yeniden hesaplandı: **maksimum fark 0,0** | PASS |
| 4b | recall@10 ve kapsama@10 için grup-bootstrap %95 aralıkları (n=1000, tohum 2026091802, grup = sorgu kimliği) birebir | PASS |
| 4c | Birleşim tavanı 0,487795; ortalama birleşim 31,50 belge; köprü/sorgu 3,0; köprüsüz sorgu 0 | PASS |
| 5a | Her kolun her listesi tam 10 belge; `multihop_rerank` çıktısının tamamı birleşim kökenli (0 ihlal) | PASS |
| 5b | Son top-10'daki 4.060 altın isabetin 247'si (%6,1) yalnız ikinci atlamadan; en sık köprüler American 207 / United 122 / States 79 | PASS |
| 5c | Kapsama@10 altın sayısına göre: dense 0,388/0,142/0,000 · multihop 0,419/0,147/0,002 · multihop_rerank 0,502/0,233/0,015 | PASS |
| 6 | Sorgu kimlikleri ve altın kümeleri B1 ile birebir aynı | PASS |
| 7 | Rapor sayıları 16/16 | PASS |
| 8 | Ledger'da B2 dondurma, başlatma, geliştirme ve final satırları var | PASS |

## Bulgu (yeni, rapora eklendi)

**Yeniden kodlama bit-aynı değil.** B2'nin `dense` kolu ile B1'in `dense` kolu aynı kodlayıcı ayarları, aynı korpus ve aynı sorgularla çalıştığı hâlde **2.417 sorgunun 254'ünde (%10,5) top-10 listesi en az bir konumda farklı**; toplu etki küçük: recall@10 0,599538 (B1) vs 0,598986 (B2), Δ = −0,055 puan. Sebep fp16 GPU kodlamasının çalıştırmalar arası tam yinelenebilir olmaması (B1'in kendi ölçümü: toplu/tekil kodlama farkı |Δkosinüs| ≤ 4,8e-6). Sonuçlar için zararsız ama **tekrar üretilebilirlik iddiası "bit-aynı" değil, "metrik olarak ≤ 0,1 puan" düzeyinde** olmalı; ileride karşılaştırmalı kollar aynı vektör matrisini paylaşmalı.

## Yorum

- Negatif sonuç sağlam: kazanç (kapsama +1,1 puan) hem güven aralıklarıyla hem de mekanizma ölçümüyle (247/4.060 = %6 ikinci-atlama katkısı, tavan 0,49) tutarlı.
- Tavan (0,488) metrikten bağımsız bir üst sınır: bu aday kümesiyle mükemmel bir yeniden sıralayıcı bile hedefi (0,90) geçemezdi. Bu, "reranker'ı iyileştir" yönünü kapatıp "köprü çıkarımını iyileştir" yönünü açan asıl kanıttır.
- Geliştirme dilimi (100 sorgu, 1.707 paragraf) finali yanlış yönde tahmin etti (+10 puan kapsama vaadi, gerçekleşen +1,1): küçük korpusta geliştirme seçimi yapmanın riski burada ölçüldü.

## Doğrulanmayanlar

Model kimliği/çıktı kalitesi (yalnız dosya içi tutarlılık), köprü çıkarıcının kendi doğruluğu (köprü terimi altın ikinci paragrafta geçiyor mu — ölçülmedi, sonraki kol için önerildi), Türkçe dilimi (UNMEASURED), GPU ölçümlerinin tekrar üretimi.
