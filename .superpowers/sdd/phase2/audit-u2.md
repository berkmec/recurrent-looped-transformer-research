# U2 döngüsü — bağımsız denetim (2026-09-18, salt-okunur)

**Karar: 26 PASS / 2 FAIL / 0 UNVERIFIED** (28 kontrol; makine-okunur ayrıntı `audit-u2.json`). Hiçbir deney kodu çalıştırılmadı, model yüklenmedi; yalnız dosyalar okundu ve istatistikler yeniden hesaplandı. İki FAIL'in ikisi de **sayısal sonuçları değiştirmiyor**: biri eksik bir mühür dosyası, diğeri raporun düzyazı/parantez içi sayılarında beş küçük tutarsızlık. Ham koşul dosyaları, analiz ve tablo değerleri birbiriyle tutarlı.

## 1. Bulgular (önce hatalar)

### FAIL-1 — `results/U2-measure` mühürsüz
- `frozen.json` ve `frozen.sha256` yok. Ledger L18 "Repair arms follow only after the measurement is **frozen** and reported" diyor; mühür kaydı yok.
- Hafifletici kanıt: 6 koşul dosyası + `summary.json` U1 protokol hash'ini (`56ef56a1…`) taşıyor; `selected_sha256` değerleri U2/U2b dosyalarıyla birebir aynı (aynı modeller); `summary.json` üçlüleri `[displaced_at_run_end, repaired_within_horizon, final_accuracy]` dosyalardan aynen türüyor.
- Doğrulanamayan: `horizon/trajectory.py`'nin koşu anındaki (17:48–17:55) kaynak hash'i. Şimdiki dosya, U2-intervene'in 18:00'de mühürlediği `b463a0a1…` ile eşleşiyor; aradaki 5 dakikada değişmediği kayıtla gösterilemiyor.

### FAIL-2 — Rapor sayıları (`notes/U2_RAPORU.md`): 212 eşlenen sayıdan 202 tutuyor, 10 tutmuyor (5 madde)
| # | Bölüm / ifade | Raporda | Depoda (JSON) |
|---|---|---|---|
| a | B2 snap: "temizde 4096'da −11 puan **(s117 −20)**" | s117 −20 | s117 **−15,0** (0,850 vs 1,000); −20,0 olan **s143** (0,800); s191 +2,0; ortalama −11,0 ✓ |
| b | B2 snap: "**13–20** puan taban altında kalıyor" | 13–20 | hücre ortalamaları 13,6 / 14,9 / 19,5 / 15,0 (128 orta/son, 512 orta/son) → **13,6–19,5**; tohum bazında s117/s143 19,4–37,0, s191 0,0 |
| c | B2 ve B3 yığın: "**0,31**–0,43 → 0,93–0,94(5)" | 0,31 | none son ek 128/512: s117 0,314/**0,292**, s143 0,404/0,416, s191 0,432/0,348 → min **0,29** (Bölüm 1'de doğru: "0,29–0,43"); 0,31 yalnız 128 son için doğru |
| d | B1 tablo s143: "k=1,2 → **0,00**" | 0,00 | k=1 0,004, k=2 **0,008** (→ 0,01) |
| e | B1 yığın düzyazısı: "2,2 → 6,0 … ~5,0 → 1,2–1,6" | 3 model için | k=1: 2,17–2,43; doyum 5,59–6,54; koşu sonrası k=1: 4,14–5,00; k=64: **1,00–1,95** (128 orta 1,51/1,19/1,00; 512 orta 1,95/1,59/1,63). Değerler s117'ye yakın, "3 model" için aralık daha geniş. Ledger L20 aynı "5.0->1.2-1.6"yı tekrarlıyor |

Küçük ifade tutarsızlığı (sayı hatası değil): B3 metni "temiz üç uzunlukta fark 0" derken tablo "+0,1" yazıyor; depoda 4096 için +0,07 pp (s191 +0,2, diğerleri 0).

Tabloların **tüm** kazanç/kayıp/temiz değerleri (B2 freeze/snap/wrong_snap, B2 yığın freeze, B3 freeze_k1/k2, B3 yığın freeze_k2; ortalama, min, maks, tohum bazlı temiz değerler, "5,4 puan (s117, 512 orta)", "0,6 puan (s191, 128 ön)", "s143 4096 −7,6", "s117/s143 −63/−76", B1 çapa histogramları, onarım eğrileri, yer değiştirme eğrileri s117/s191, merkez uzaklıkları/yayılımlar) depodaki değerlerle yarım-birim yuvarlama içinde **tutuyor** (liste: `audit-u2.json` → `report_claim_table`).

## 2. Geçen kontroller (kanıt özeti)
1. **Mühür bütünlüğü** — U2-intervene (`5c012e60…`), U2b-intervene (`b4c36dd7…`), U1 (`e98d7ef3…`): `frozen.sha256` = sha256(`frozen.json`); `protocol_hash` = kanonik JSON sha256'sı (`dbdd3395…`, `20b73983…`, `56ef56a1…`); tüm `sources` (6 / 7 / 5 dosya) şimdiki dosyalarla birebir; iki intervene mührünün `u1_protocol_hash`'i U1'inkine eşit.
2. **Tamlık** — U2/U2b: 6/6 koşul dosyası, hepsi mühür hash'ini taşıyor; kollar protokole uygun (five: none/freeze/snap/wrong_snap; stack: none/freeze — U2b: none/freeze_k1/freeze_k2 ×2); nötr 7 anahtar, temiz 128/1024/4096, n=500; `analysis.json` içindeki gömülü koşul kopyaları dosyalarla özdeş. U2-measure: `summary.json` + 6 koşul dosyası, her birinde 6 koşu.
3. **Döngüler arası tutarlılık** — 6 koşulda `none` kolu U2 ↔ U2b: son doğruluk, çevirme sayıları, `prediction_changes`, `prefix_accuracy`, `source_seed` bit-bit özdeş; `selected_sha256` aynı.
4. **Çevirme aritmetiği** — 216 hücre: (doğruluk − taban)×n = doğru→yanlış farkı, 1e-6·n içinde; `prediction_changes` ≥ çevirme toplamı.
5. **Analiz yeniden üretimi** — 2×36 özet hücresi: `gain_vs_none_pp`, `loss_vs_own_base_pp`, `diff_vs_none_pp` ortalama/değer/min/maks 1e-9 içinde; 2×60 t95 aralığı = ortalama ± 4,302652729·s/√3 (1e-6 içinde; örn. freeze 128 orta 33,33 ± 22,97 = [10,36, 56,30]); `within_target` (≤ 2 pp), `no_regression` (≥ −1 pp) ve tüm `status` etiketleri belirtilen kuralla yeniden üretildi.
6. **Test kanıtı** — RED loglar "FAILED (failures=3)", GREEN loglar "OK" ile bitiyor (u2-measure-red / green-v2, u2-intervene-red / green, u2b-red / green); üç test dosyası mevcut. Test paketi çalıştırılmadı.
7. **Ledger** — L18 U2 ölçüm kararı; L21 "Ruling (U2 part 2)"; L26/L27 freeze önbellek ve snap okuma kararları; L32 "Ruling: U2b = run-aware freeze"; **L68 "19:55 USER INTERVENTION: '4 python süreci … OOM olursa pc çöker … başka başlatma'** … U2c workers STOPPED"; **L70 "19:57–20:03 POWER OUTAGE (Event 6008 … Event 41 reboot 20:03:31): all jobs died — U2 5/6 done, U2b 4/6 done"**; L71 "tek tek al başla" sıralı yeniden başlatma; L75 wrong_snap etiket kararı; L77 U2b analiz; L79 denetim görevlendirmesi.

## 3. Yorum notları
- **`wrong_snap` = MIXED etiketi kural yapaylığı.** Temiz fark ortalamaları −80,7 / −80,7 / −78,9 pp, `no_regression` 9/9 false; ama "kendi tabanına kayıp" ≈ 0 (−0,7 … +0,9 pp) çünkü kolun tabanı zaten şansta (0,196 / 0,228 / 0,202). Kural "herhangi bir hücre hedef içinde → MIXED" dediği için kontrol kol MEASURED_NEGATIVE yerine MIXED çıkıyor. Raporun ve ledger L75'in bu durumu açıkça beyan etmesi doğru; etiket değil sayı esas alınmalı.
- **`stack/freeze_k2` = MIXED** aynı yapaylığın ikinci yüzü: orta/ön hücrelerde "kendi tabanına kayıp" 3,6 / 4,3 / 6,4 / 9,9 pp (ortalama) modelin müdahale öncesi kendi kaybı (none'a göre kazanç 0,0–0,1 pp); son ek +55,7 / +58,9 pp, temiz 0,0 / 0,0 / +0,07. Raporun "maliyetsiz, son ek için tam onarım" okuması sayılarla uyumlu; s117'de k2 son ek 0,928 vs taban 0,944 (−1,6 pp) tek istisna.
- **`five/freeze_k2` = MEASURED_POSITIVE_CEILING** 3/3 tohumda doğrulandı: nötr kayıp ortalamaları 0,0–0,2 pp (en kötü tek hücre 0,6 pp, s191 128 ön), temiz fark 0,00 / 0,00 / +0,07. Oracle nötr bilgisi → tavan; mimari iddia değil.
- df=2 t-aralıkları çok geniş (freeze 128 orta [10,4, 56,3]); "keşif aralığı" kaydı yerinde, üç tohumla çıkarımsal iddia yapılmamalı.

## 4. Doğrulanmayanlar
- U2-measure için koşu-anı kaynak hash'i (mühür yok; bkz. FAIL-1).
- Test paketleri çalıştırılmadı; RED/GREEN loglar kayıt olarak alındı.
- Hiçbir koşul yeniden koşulmadı: değerler iç tutarlılık (aritmetik, `none` özdeşliği, analiz yeniden üretimi) açısından doğrulandı, hesabın kendisinin doğruluğu değil.
- Bölüm 1 düzyazı aralıkları ("0,2 at k=4", "0,4–0,6 at k=16") yalnız tablodaki model bazlı değerler üzerinden kontrol edildi.
