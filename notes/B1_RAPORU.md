# B1 — Sözcüksel vs anlamsal erişim, MuSiQue (2026-09-18/19)

## Bulgular (önce)

1. **Beş kolun hiçbiri dondurulmuş hedeflere (recall@10 ≥ 0,95, kapsama@10 ≥ 0,90) yaklaşmıyor → MEASURED_NEGATIVE.** En iyi kol `hybrid_rerank`: recall@10 **0,665** [0,655, 0,676], kapsama@10 **0,325** [0,308, 0,342] (2.417 sorgu, 21.100 paragraf, grup-bootstrap %95).
2. **Yoğun (dense) tek başına sözcükselden iyi; hibrit (RRF) yoğundan kötü.** dense recall@10 0,600 / kapsama 0,247; bm25 0,487 / 0,134; hybrid 0,588 / 0,223. Geliştirme aşamasındaki "füzyon zarar verir" bulgusu finalde doğrulandı (paket B5'in uyarısıyla tutarlı).
3. **Reranker +6,6 puan recall, +7,9 puan kapsama** kazandırıyor (0,5995→0,6655; 0,2466→0,3252) ama sorgu başına **1,93 s** (p50) — dense 3 ms. Yeniden sıralama penceresi (50) kaçırılanların çoğuna ulaşamıyor: hibrit listesinde altın parçası ilk 10 dışında kalan 1.878 sorgunun **1.322'sinde en az bir altın ilk 50'nin de dışında** → hiçbir reranker düzeltemez. Sorun **aday üretiminde**, sıralamada değil.
4. **Kapsama atlama sayısıyla çöküyor** (hybrid_rerank): 2 altın → 0,470; 3 altın → 0,254; 4 altın → **0,012** (dense: 0,390 / 0,142 / **0,000**). Çok atlamalı sorularda köprü paragrafı soru metniyle anlamsal olarak ilişkili değil; tek sorguyla bulunamıyor. Bu, B2 çok-atlamalı erişim kolunun (ilk atlamadan köprü terimi → ikinci sorgu) hedeflediği boşluk.
5. **Ölçek merdiveni** (250 sorgu, 1k/10k/100k/tam 21k): hybrid_rerank recall@10 0,833 → 0,692 → 0,592 (100k, 78.900'ü dış havuzdan) → 0,651 (tam). Her 10× korpus ≈ −10..−14 puan; 100k'da reranker'lı en iyi kol bile 0,59.
6. **hash128 (Faz 1 sözcük-karma) şansta:** recall@10 0,048, kapsama 0,002 — Faz 1'in negatif sonucu daha büyük korpusta da geçerli (kontrol olarak tabloda kalır).

## Yöntem

- Veri: MuSiQue-Ans dev (`musique_ans_v1.0_dev.jsonl`, pin G0), 2.417 sorgu; korpus = yüklenen satırların tüm paragrafları (21.100 tekil). Altın = `is_supporting` paragraflar (2/3/4 altın: 1.252/760/405 sorgu). Merdiven ek paragrafları train dosyasından (havuz 80.862; final sorgularının paragraflarıyla kesişmeyen).
- Modeller (G0 pin): Qwen3-Embedding-0.6B (son-token havuzlama, sorgu talimatı açık, maks 256 token — geliştirmede seçildi: `results/B1-development/development.json`, kapsama@10 0,445 dense / 0,515 rerank), Qwen3-Reranker-0.6B (yes/no logit, derinlik 50, fp16, CUDA). Kollar: `hash128` (Faz 1 `memory.api.embed`), `bm25` (k1 1,5, b 0,75), `dense` (kosinüs, tam arama), `hybrid` (RRF k=60, füzyon derinliği 100), `hybrid_rerank`.
- Protokol `results/B1/frozen.json` (geliştirme seçimi override'larıyla donduruldu), kaynak hash'leri doğrulandı; ham sıralamalar `results/B1/final-*.json`, `ladder*-*.json`; makbuz `results/B1/receipt.json` (E12: run_id, exit 0, 5/5 kol, sözleşme testleri geçti, artifact hash'leri) — **COMPLETE**.
- Toplu/tekil kodlama farkı: maks |Δkosinüs| 4,8e-6 (64 belge) — batching sonucu değiştirmiyor.
- Süre: kodlama 102.545 metin 1.615 s (GPU); reranker 170.850 çift 7.172 s; toplam ~2,6 saat.

## Yeniden koşu ve yama (kayıt)

İlk final denemesi (18 Eylül, 58. dakika) CUDA OOM ile düştü: reranker bütün konumlar için sözlük boyutlu logits üretiyordu (4,63 GiB tek tahsis). `semantic_memory/rerank.py` yamalandı: taban model + `lm_head` yalnız son gizli vektör üzerinde. Başarısız deneme `results/B1-attempt1-oom/` altında korunuyor. **Parite testi** (`tests/test_rerank_parity.py`, 19 Eylül 02:24, CPU, 2/2 GREEN): yamalı skor = eski tam-logits skoru (çift başına 1e-4), toplu = tekil (sol dolgu). Dolayısıyla yama sonucu değiştirmiyor; yalnız belleği.

## Yorum ve sınırlar

- Hedefler bu erişim ailesiyle (tek sorgu → top-10, yeniden sıralama) ulaşılabilir görünmüyor; **negatif sonuç geçerli sonuçtur**. Boşluğun yeri ölçüldü: köprü paragraflar aday listesine hiç girmiyor (1.322/2.417 sorguda ilk 50 dışında).
- Sonraki kol: **B2** (`semantic_memory/multihop.py`, `run_b2.py`, testler GREEN; koşu GPU sırasında). Birleşim tavanı (tüm altınlar birleşimde) ayrıca raporlanacak.
- Türkçe dilimi UNMEASURED (MuSiQue İngilizce). Üç tohum yok: erişim deterministik; belirsizlik sorgu-grubu bootstrap'ından.
- Reranker gecikmesi (1,9 s/sorgu, 8 GB kartta fp16) üretim ölçeğinde D5 (batching/önbellek) gerektirir; sayılar kirli değil (koşu sırasında başka GPU işi yoktu; CPU'da 22:20–22:35 arasında U2 bağımsız denetimi ve mühürleme/commit çakıştı (U2b 22:19'da bitmişti) — yalnız kodlama süresini etkileyebilir, doğruluğu değil).

## Durum

`hybrid_rerank`, `dense`, `hybrid`, `bm25`, `hash128`: **MEASURED_NEGATIVE** (her ikisi de hedefin altında; kol başına `target_recall` / `target_coverage` alanları false, `status` "MEASURED"; bkz. `results/B1/final.json`). Denetim: `.superpowers/sdd/phase2/audit-b1.md` — 13 PASS / 1 FAIL (bu belgedeki alan adı; düzeltildi) / 1 UNVERIFIED (MRR: ham dosyalar ilk 50 sırayı saklar, metrik ilk 100 ile hesaplandı; fark ≤ 7,4e-4). Mühür: `audit/phase2-B1-manifest.json`.
