# M1 ek notu — kapı hiç kapanmadı (2026-10-01)

## Bulgu

M1'de `hard_st` kapısı 4000 adımın hiçbirinde anlamlı ölçüde kapanmadı. **`hard_st` = `plain` + 2 305 parametre. M1 kapıyı sınamadı.**

- Eğitim günlüğü (`results/M1/m1-hard_st-w1152-s117/updates.jsonl`): `hold_rate` ortalaması **4,8e-6**. Yalnızca 40 adımda sıfırdan büyük, hepsi ilk ~100 adımda. Sonrası 0,0000.
- Kayıtlı kontrol noktasında (`selected.pt`), needle girdilerinin her segmentinde HOLD **0,0000** (olgu, dolgu, soru; dolgu 0/32/64/128). Kapı logitleri en düşükte **+5,4**, ortalama **+16…+25**: sigmoid doymuş, straight-through gradyanı ≈ 0. Kapı başlığının bias'ı 1,99, yani başlangıçtaki +2'den değişmemiş; logit büyüklüğü girdi normundan geliyor.
- `gate_open` (kapı zorla WRITE) ile `full` çıktıları **16 hücrede birebir aynı** (`results/M1-diagnosis/channel-hard_st.json`).
- `tests/test_channel_probe.py`: açık kapılı `GatedCore`, aynı ağırlıklı düz Core ile sayısal olarak aynı.

## Devir notunda neyi geçersiz kılıyor

- **"Kapı yakın erişimi 2,3 kat iyileştiriyor, bedelsiz" (`DEVIR_2026-09-27.md`) geçersiz.** İğne@0 farkı (0,902 ↔ 0,387) kapıdan gelmiyor; aynı fonksiyonun iki eğitim koşusu arasındaki değişkenlik. Eğitilen aralıkta bile görülmemiş anahtar harfleriyle ≈0,4 ↔ ≈0,9 fark var: anahtar genellemesinin koşu-koşu değişkenliği (tek çift, n=1).
- **Hüküm etiketi `GATE_DOES_NOT_TRANSFER` kural gereği doğru, ama "kapı 30M'e transfer olmuyor, M-hattı birinci basamakta durur" çıkarımı desteksiz.** Sınanan bir kapı yoktu.
- **Hüküm kuralında manipülasyon kontrolü yoktu.** `mline/lm.py` bunu önceden uyarmıştı ("kapı hiç kapanmazsa null okunamaz") ve günlüğe `hold_rate` eklenmişti, ama hüküm onu okumuyordu. Eklenmeli: `hold_rate` eşiğin altındaysa hüküm `MANIPULATION_NOT_APPLIED`.
- Uzunluk-eşli kontrol hâlâ geçerli (toplam uzunluk sınır değil). Ama metin dolgusunda yakın kontrol de zayıf (hard_st 0,13/0,09/0,004); temiz "mesafe" okuması yalnızca tekrar dolgusunda.

## Mesafe uçurumu (kapıdan bağımsız, iki kolda da aynı)

`results/M1-diagnosis/distance-sweep-{plain,hard_st}.json` (128 öğe, tekrar dolgusu, eğitim dolgusu [0,64]):

| | 0–64 | 72 | 80 | ≥128 |
|---|---|---|---|---|
| plain, görülen anahtar | 0,99–1,00 | 0,48 | 0,02 | 0,00 |
| hard_st, görülen anahtar | 0,94–1,00 | 0,59 | 0,03 | 0,00 |

Aralık içinde düz, eğitilen azami mesafenin hemen ötesinde keskin çöküş: "kısa eğit, uzun test et" imzası; tutma güvenilirliğinin yavaş sönmesi değil.

## Çekirdek mimari

`core/model.py` **tam nedensel dikkat** içeriyor (kodlayıcı tüm önekte self-attention, kod çözücü tüm kodlayıcı belleğine cross-attention, RoPE, baş boyutu 144, taban 10000). Yalnızca kod çözücü self-attention'ı window=4. Durum geri beslemesi zayıf toplamalı bir kanal (alpha 0,1). "Uzağı taşıyan tek şey durum" varsayımı yanlıştı; iğne görevi dikkatle çözülebilir.

## Sıfır-eğitim müdahaleleri (mevcut kontrol noktaları, hepsi testli)

- **RoPE enterpolasyonu** (`mline/rope_probe.py`, `rope-rescale-*.json`): konum enterpolasyonu (0,5/0,25/0,125) aralık içinde bile ≈0; NTK×2 uçurumu ≈72'den ≈100'e kaydırıyor (0,90@72/80, 0,72@96) ama dolgu 0 çöküyor; NTK×4+ aralık içi doğruluğu yiyor. Devre yerel ofsetlere çok hassas; sıfır-eğitim düzeltme yok.
- **Geri besleme kapatma** (`mline/channel_probe.py`, `channel-*.json`): `alpha=0` düz kolda dolgu ≥48'de 1,00 → 0,00 (≤32'de büyük ölçüde duruyor). Temiz bir cerrahi değil (modelin eğitildiği fonksiyonu bozuyor).
- **Durum takası** (`mline/state_swap.py`, `state-swap-plain.json`): tek adımlık takas (olgudan sonra / sorudan önce) cevabı **hiç değiştirmiyor** (donör-izleme 0,00, alıcı-izleme baz çizgiyle aynı). İlk rakam durumdan okunmuyor; dikkatten geliyor. Tam akış nakli (`donor-*`, `mean-*`; `state-transplant-plain.json`): geri besleme akışının **tamamı** donörle ya da parti ortalamasıyla değiştirilince de ilk rakam alıcıyı izliyor (donör 0,00–0,02); görülen anahtarlarda hiçbir şey değişmiyor (1,00). Yalnızca görülmeyen anahtarların sonraki rakamları akışın istatistiğine duyarlı (dolgu 64 tam eşleşme 0,42 → 0,20 donör, 0,03 ortalama). Kanal olguyu taşımıyor; "genel sinyal" gibi davranıyor (hipotez).

## Açık

- **İnce-ayar sonucu** (`results/M1-finetune-distance/ft384-plain-s217`; M1 düz kontrol noktasından 500 adım, dolgu [0,384], lr 3e-4, tek tohum, repeat dolgu): uçurum eğitim aralığına bağlı. Görülen anahtarlarda 0–384 arası 0,98–1,00, **448'de 0,008**, 512'de 0,18 (açıklanamayan çıkıntı), 768'de 0,04, 1024 ve 2048'de 0,00. Görülmeyen anahtarlarda 0–384 arası ≈0,40–0,55 (ince-ayar anahtar genellemesini artırmadı), 448'de 0,00. Uçurum 64 → 384 ile birlikte ~72 → ~400–450'ye (~6×) taşındı. Sınırlar: tek tohum, 128 örnek, yalnız tekrar dolgusu, **karşılaştırma koşusu yok** (iğne payı, lr, optimizer sıfırlama aynı anda değişti). Doğal metin dolgusu eğitilmedi (@512: 0,00). Ayrıntılı rapor: `Desktop\reports\Yinelemeli modellerde uzak bilgi taşıma.md`.
- Kapıyı gerçekten sınamak için gereken: kapanma eğilimi (HOLD ön yargısı, normlanmış girdi, kapıya özel lr) ve dikkatin sınırlandığı bir görev (yoksa iğne görevi durumu hiç gerektirmez). Bu ayrı bir döngü, ayrı bir önkayıt.
- Araştırma raporu: `C:\Users\pc\Desktop\reports\Yinelemeli modellerde uzak bilgi taşıma.md` (yazılınca).
