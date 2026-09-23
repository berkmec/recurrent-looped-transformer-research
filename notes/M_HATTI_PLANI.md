# M-hattı — ölçülmüş her sonucu model eğitmeye bağlayan araştırma hattı (Faz 3)

**Tek amaç: model eğitmek.** Bu belge, kampanyanın bugüne kadar ürettiği bütün sonuçları — pozitif, negatif, geri
çekilmiş, hepsi — o amaca göre yeniden okur. Negatif sonuçlar burada "başarısızlık" değil **kısıt**: neyi inşa
etmeyeceğimizi söylerler ve arama uzayını daraltırlar, ki bu pozitif sonuç kadar değerlidir.

İlke: *karşılaştırma sayısı arttıkça gelişme artar.* Bu yüzden hat, tek bir "doğru model" aramaz; her basamakta
**eşli karşılaştırma** üretir ve modelleri saklar.

---

## 1. Kanıt envanteri → mimari karar

| # | Ölçülen sonuç | Modele etkisi |
|---|---|---|
| 1 | **U2c hard_st**: öğrenilmiş HOLD/WRITE kapısı, tam bit-düzeyi koruma. 3 tohum × 2 görev × 2 lr'de pozitif kalan **tek** kol (beş %94/%79, yığın %99/%92) | **İnşa et.** Hattın 1. basamağı bu. Tek tekrarlanmış mimari bulgu. |
| 2 | **U2b**: koşu-farkında dondurma test zamanında kayıpsız onarım tavanı veriyor | **Çıkarım zamanı seçeneği olarak ekle.** Yeniden eğitim gerektirmiyor, bedava. |
| 3 | **U2c soft**: yumuşak karışım beş-durumda %100, yığında −%3 | **Yumuşak kapıyı kullanma.** Dil modelinin durumu *içerik taşır* (yığın gibi, beş-durum gibi değil) → yumuşak karışımın dilde de sızdırması beklenir. Bu, M1 için önceden kayıtlı tahmindir. |
| 4 | **U2c supervised**: kapıyı elle tasarlanmış k=2 kuralına denetlemek **zarar verdi** (beş: %71, temiz-4096 −0,385) | **Kuralı elle dayatma.** Kapı kendi kuralını bulsun. |
| 5 | **U2c çapraz-lr + U4-sep faz geçişi + R5 (w256 lr 3e-4 ile 3/3 kurtuldu)** | **lr birinci sınıf deney değişkeni.** Üç ayrı döngüde çözüm ailesini lr belirledi, hız değil. M2 bunu tarar. |
| 6 | **C16**: 6 koşulun 5'inde doğrulama 1,000 iken dayanıklılık uç noktası çöküyordu | **Perplexity tek başına kör.** Dayanıklılık değerlendirmesi 1. günden itibaren olmalı. |
| 7 | **U2d → U4-sep → U4-sep2**: mekanizma bilinmiyor; oran hipotezi reddedildi, mutlak sürüklenme 4/5 | **Geometri düzenleyicisi KOYMA.** Sürüklenmeyi tanı olarak ölç, kayba sokma. İki döngü bunu denedi ve ikisi de hedefi ıskaladı. |
| 8 | **B1 0,325 · B2 0,336 (hedef 0,90) · D1 F1 0,03**; darboğaz aday üretiminde | **Başlık belleği yığınını ilk modele koyma.** Ayrı hat olarak beklesin; doğrulanmış bir uygulaması yok. |
| 9 | **Sürüklenme mini**: pencere 16 sürüklenmeyi artırıyor; değişmezlik kaybı çöküyor; reread kararsız | **Üç şeyi tekrar deneme.** Arama uzayından çıktılar. |
| 10 | **B2**: fp16 yeniden kodlama bit-aynı değil (254/2417 listede fark) | **Karşılaştırmalı kollar aynı ağırlık/matrisi paylaşmalı.** Yöntem kuralı. |
| 11 | **R6**: segment yürütme CUDA tepesini L1024'te 0,53×'e indiriyor | **8 GB'da uzun bağlam için kullan.** |
| 12 | **R4**: beş-durumda TBPTT8 = tam BPTT | **Kesilmiş BPTT uygulanabilir** → bellek kazancı. |
| 13 | **A5/remat**: tam BPTT + yeniden hesaplama, öğrenilmiş `initial` durumuna kredi veriyor | **Eğitim döngüsünde kullan.** |
| 14 | **U4-sep2'nin asıl kazancı**: eşli kontrol kolu | **Her kol eşli koşsun**, tek terim farkla. Kontrolsüz tasarımda +95 puan "başarı" diye raporlanırdı. |
| 15 | **U4-sep**: yardımcı kayıp yeterli lr'de kodu bedelsiz aşırı ayrık geometriye taşıyabiliyor; çöküş kaçılabilir tuzak | **Uyarı:** erken eğitimde kapının tamamen kapanması geçici olabilir; erken durdurmadan önce bekle. |
| 16 | **U4-sep2 seçim kusuru**: taban eğitimi ile ince ayar yalnızca hedeflenen terimde ayrışmalı | **Yöntem kuralı**, her karşılaştırmada. |

### Buradan çıkan mimari

Mühürlü `core/model.py` zaten bir dil modeli olabiliyor: `Core(width, heads, classes=vocab)` + kelime dağarcığı
boyutunda gömme. Yani **yeni mimari icat etmiyoruz** — ölçülmüş mimariyi büyütüyoruz ve üstüne tek kanıtlanmış
eklentiyi (`GatedCore`, hard_st) koyuyoruz. Bu, hattın en güçlü tarafı: 30M'de test edilen şey, oyuncak ölçekte
3×2×2'de tekrarlanmış olan şeyin *aynısı*.

---

## 2. Dilde "nötr koşu" karşılığı

Oyuncak deneylerin tamamı şu şemaya dayanıyordu: *bilgi önce gelir, araya bilgi taşımayan uzun bir koşu girer, soru
sonra sorulur.* Dildeki birebir karşılığı **dolgu içinden iğne çekme**: bir olgu erken konur, N dolgu belirteci
araya girer, sonra o olgu sorulur.

Bu yüzden M-hattının dayanıklılık ölçümü, perplexity'nin yanında:
- **iğne@N**: N ∈ {0, 128, 512, 2048} dolgu belirteci için olgu geri çağırma doğruluğu,
- **uzunluk genellemesi**: 512'de eğit, 1024/2048/4096'da ölç,
- **temiz perplexity**: müdahalenin dil yeteneğine maliyeti.

C16'nın dersi burada uygulanıyor: perplexity iyi görünürken iğne@2048 çökebilir, ve asıl merak ettiğimiz ikincisi.

---

## 3. Basamaklar

**M0 — altyapı (hesap gücü düşük).** Korpus E: sürücüsüne (C:'de 0,6 GB kaldı), tokenizer, 30M `Core` tabanlı LM,
A8 korumalı + remat'lı eğitim döngüsü, değerlendirme takımı (perplexity + iğne@N + uzunluk). Dondurulmuş protokol,
her zamanki gibi.

**M1 — kapı, eşli (ana deney).** Üç kol, aynı veri, aynı bütçe, aynı tohum: `plain` (mühürlü Core, kontrol) ·
`hard_st` (tam koruma) · `soft` (yumuşak karışım).

**Tohum sayısı — plan ile kod arasındaki farkın gerekçesi:** yerelde **tek tohum** (117) koşuyoruz, protokolde de
öyle yazıyor. Ölçülen hız (kol başına ~8,5 saat) üç tohumu yerelde 25 saat yerine 75 saate çıkarırdı. Tek tohum
**bir ilk bakıştır, tekrarlanmış sonuç değildir** ve karar kuralı bunu kendisi damgalıyor (`provisional=True`,
`mline/analyze.py`). Replikasyon VM'e gidiyor — kampanyanın kendi tarihi bunu gerektiriyor: lr üç ayrı döngüde
çözüm ailesini belirledi ve C16 tek koşulun doğrulama doğruluğunun asıl özelliğe kör olduğunu ölçtü.

**İğne mesafe merdiveni:** eğitim dolgusu **0–64 bayt**; değerlendirme 128 / 512 / 2048, yani eğitimin gördüğü en
uzun mesafenin **2×, 8× ve 32× katı**. Sıfır dolgu kontrol noktasıdır. Böylece genelleşmesi gereken şey biçim
değil **mesafe** olur — U1'in merdiveninin (128'de eğit, 4096'da ölç) dildeki karşılığı.
*Önceden kayıtlı tahmin:* hard_st perplexity'yi bozmadan iğne@512 ve iğne@2048'i `plain`in üstüne çıkarır; `soft`
hard_st'in altında kalır, çünkü dil durumu içerik taşır (U2c yığın sonucu).
*Yanlışlama:* hard_st iğne@N'de `plain`i geçmezse, kapı oyuncak ölçeğe özgüdür ve hat 1. basamakta biter.

**M2 — öğrenme oranı taraması.** En az üç seviye. Üç ayrı döngüde çözüm ailesini lr belirlediği için bu bir ayar
işi değil, deneyin kendisi.

**M3 — uzunluk genellemesi.** 512'de eğit, 4096'ya kadar ölç. U1'in dildeki karşılığı.

**M4 — çıkarım zamanı dondurma.** U2b'nin koşu-farkında dondurmasını eğitilmiş LM'e uygula. Yeniden eğitim yok,
maliyet yok; işe yararsa bedava kazanç.

**M5 — (sonra) bellek hattı.** B-hattı aday üretimi sorununa çözüm bulana kadar bekler. Bu modele girmez.

---

## 4. Yöntem kuralları (hepsi ölçülmüş bir hatadan geliyor)

1. Her kol **eşli** koşar; kollar tek terimde ayrışır. *(U4-sep2)*
2. Taban eğitimi ile ince ayar **yalnızca hedeflenen terimde** ayrışmalı. *(U4-sep2 seçim kusuru)*
3. Karşılaştırmalı kollar aynı ağırlık matrisini paylaşır; fp16 yeniden kodlama bit-aynı değildir. *(B2)*
4. Tahmin ve karar kuralı **koşmadan önce** koda yazılır, eşikleriyle. *(U2c'den beri)*
5. Perplexity'ye bakıp dayanıklılık hakkında hüküm verilmez. *(C16)*
6. Başlatmadan önce ölç: adım maliyeti, RAM, VRAM. *(dört-süreç olayı ve U4-sep'in 24 adımlık yanlış okunan ölçümü)*
7. Mühürlü kaynaklara dokunulmaz; yeni modül yazılır, bit-aynılık testle kanıtlanır. *(U2c mühürleri)*

---

## 5. Donanım gerçeği

RTX 2060 SUPER 8 GB · 16 GB RAM · E: 48,7 GB boş, **C: 0,6 GB boş**.

- 30M: **ölçüldü, rahat değil.** batch 8'de 4,39-5,15 s/adım (kol başına), üç kol 3000 adımda **12,1 saat**; batch 32'de
  2148 bayt/s, bir epoch **27 saat/kol**. GPU kullanımı tepede %5 — iş yükü aritmetik değil ardışıklık sınırlı.
  Kaldıraçlar ölçüldü: batch 8→32 **2,70×** (batch 48 host belleğine taşıyor, 10 kat yavaşlama), CUDA grafiği **1,20×**.
  `torch.compile` bu makinede yok (Triton kurulu değil); Linux'ta olacağı için `mline/faststep.py` ön şart olarak yazıldı.
- 300M: yerel olarak gerçekçi değil; yukarıdaki ölçümlerden doğrusal olarak çok daha ağır.
- **İş bölümü (2026-09-24 kararı):** 30M **bu makinede** eğitilir; daha büyük model **VM'de** olur. Yerel donanım
  30M'i taşıyor, ötesini taşımıyor.
- **Bulut bütçesi 40 USD ve ne satın alacağı.** Yerel M1'in tek tohumluk bir ilk bakış olduğu göz önüne alınınca,
  bu para M1'i tekrarlamak yerine **replikasyona** gitmeli: 3 tohum × 3 kol × 3 lr seviyesi. Gerekçe kampanyanın
  kendi ölçümleri: lr üç ayrı döngüde çözüm ailesini belirledi (U2c çapraz-lr, U4-sep faz geçişi, R5), ve C16 tek
  koşulun doğrulama doğruluğunun asıl merak edilen özelliğe kör olduğunu gösterdi. Ölçülen maliyet: fırlatma-sınırlı
  iş yükünde pahalı kart getirisi düşük, bu yüzden kart seçimi VRAM (dolayısıyla batch) üzerinden yapılmalı,
  FLOPS üzerinden değil. Spot uygun: koşucu her koşulu ayrı mühürlüyor, kesinti en fazla bir koşul kaybettirir.
- Bütün önbellek ve korpus **E:'ye** yazılır; C: dolu.
- Eğitim, boş RAM eşiğin altındayken **başlatılmaz**.
