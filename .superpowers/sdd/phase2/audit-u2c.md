# U2c bağımsız denetim — 2026-09-23

**Sonuç: 10 PASS / 0 FAIL / 0 UNVERIFIED.** Raporun sayısal iddialarının tamamı ham dosyalardan yeniden hesaplandı ve örtüştü; düzeltme gerekmedi.

**Bağımsızlık sınırı (önce söylenmeli):** ayrı bir denetim ajanı haftalık API kotasına takıldığı için (20 Eylül, `rate_limit`, 24 Eylül'de sıfırlanır) bu denetim de B2'deki gibi **aynı oturumda, ayrı bir betikle** yapıldı (`C:/Users/pc/.claude/jobs/4797b78c/tmp/audit_u2c.py`). Metrikler deney kodundan bağımsız yeniden yazıldı ama "yazan ile denetleyen aynı model" sınırı bu döngü için geçerlidir. U2/B1/D1 döngüleri ayrı ajanlarla denetlenmişti.

## Kontroller

| # | Kontrol | Sonuç |
|---|---|---|
| 1 | `frozen.sha256` = sha256(frozen.json); `protocol_hash` = kanonik özet (`25d9d376…`); 8 kaynak dosya hash'i çalışma ağacıyla aynı | PASS |
| 2 | 18 koşulun 18'i `complete.json` ile kapanmış | PASS |
| 3 | Her koşulda `train.json`, `selected.pt` ve `updates.jsonl` hash'leri kayıtla uyuşuyor; her birinde tam 2048 adım satırı | PASS |
| 4 | `analysis.json`'daki her özet hücresi değerlendirme dosyalarından yeniden hesaplandı (ortalama kayıp **ve** df=2 t95 aralığı) — fark yok | PASS |
| 5 | 18 × 7 nötr hücresinde tutma ve yanlış-yazma oranları [0,1] aralığında | PASS |
| 6 | Rapordaki onarım oranları U2b referansından bağımsız yeniden hesaplandı: soft/five 1,000 · soft/stack −0,031 · hard_st/five 0,941 · hard_st/stack 0,989 · supervised/five 0,710 · supervised/stack 0,989 | PASS |
| 7 | v1 ↔ v2 tekrar üretimi: ortak 6 koşulda checkpoint SHA'ları ve tüm doğruluklar **bit-bit aynı** | PASS |
| 8 | lr probu kendi dondurmasını doğruluyor; protokolünde test/nötr uç noktası alanı **yok**; kayıp sıçraması yalnız `hard_st` × lr 0,003 hücresinde (6), diğer yedi koşulda 0 | PASS |
| 9 | İki kesinti de kanıt olarak korunmuş: `interrupted-20260918T1955` (2 koşul) ve `interrupted-20260922T2333` (1 koşul) | PASS |
| 10 | Rapordaki anahtar sayılar (0,149 / 0,152 / 0,153 / 0,1515 / 9,15e16 / 1,6e20 / 0,284) belgede mevcut | PASS |

## Yorum notları

- **Durum etiketleri yanıltıcı, sayılar değil.** `analysis.json` altı hücrenin altısını da "MIXED" yazıyor, çünkü kural tohum başına sıkı (her hücre kendi tabanının 2 puan içinde olmalı). Örneğin `soft/five` ortalama kaybı 1,5 puan ama bir tohumda 4,6 puan olduğu için MIXED. Rapor doğru olanı yapmış: etiket yerine onarım oranlarını vermiş.
- **Tekrar üretim kanıtı güçlü.** Aynı koşullar dört gün arayla, farklı süreçte, düzeltilmiş kayıt koduyla bit-bit aynı ağırlıkları üretti. Kampanyanın determinizm iddiası CPU eğitimi için doğrulanmıştır (GPU çıkarımı için değil — B2'de tersi ölçülmüştü).
- **Denetlenmeyen:** modelin içsel mekanizması (U2d tanısı ayrı koşuyor), lr0003 seviyesi (henüz tamamlanmadı), ve sonuçların oyuncak ölçek dışına aktarılabilirliği.
