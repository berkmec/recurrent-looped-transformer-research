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

## 4b. Ön-kayıt: hangi sayı nerede duruyor, hangisi durmuyor

Bu bölüm bir denetim bulgusuna cevap: rapor, bir eşiğin **mühürlü** mü yoksa **serbest** mi olduğunu okuyucuya
söylemeden "önceden kayıtlı" diyemez. U4-sep2'de eşikler gerçekten koşudan önce sabitlenmişti, ama yalnızca
`.superpowers/sdd/phase2/progress.md` içinde — gitignore'lu, sürümsüz, benim her an yeniden yazabileceğim bir
dosyada. Bu bir söz, kayıt değil.

M1'de üç şey değişti:

1. **Ön-kayıt protokolün içinde.** `mline/train.protocol()` artık bir `preregistration` bloğu taşıyor.
   Blok `protocol_hash`'in içinde, o da `frozen.sha256`'nın içinde; dondurmadan sonra tek bir eşiği değiştirmek
   `verify_frozen`'ı reddettirir. Tahmin de, eşik de, olası sonuç listesi de oradadır.
2. **Puanlayan modül kanıtla birlikte özetleniyor.** `mline/analyze.py` artık `SOURCES` listesinde, yani mühürlü
   manifest yalnızca sayıları değil **kuralın kendisini** sabitliyor. "Kural koşudan önce yazılmıştı" iddiası
   artık benim sözüme değil, dosya özetine dayanıyor.
3. **Tek kaynak.** Protokole yazılan eşikler `analyze.THRESHOLDS`'ın kendisinden kopyalanıyor ve bir test ikisinin
   aynı olduğunu zorluyor. U4-sep2'de kuralın okuduğu ölçü (64 yuva) ile hedefin kalibre edildiği ölçü (512 yuva)
   sessizce ayrıştı; aynı sessiz ayrışma burada teste takılır.

**Dondurulmuş eşikler** (hepsi `preregistration.thresholds`, hepsi `protocol_hash` kapsamında):

| eşik | değer | ne yapar |
|---|---|---|
| `bpb_regression` | 0,02 | hard_st'in dil maliyeti bu kadarı aşarsa tahmin (a) düşer |
| `needle_gain` | 0,05 | hard_st'in `plain` üstündeki iğne kazancı **iki uzun dolguda birden** bunu aşmalı |
| `long_filler` | [512, 2048] | hükmün okunduğu iki nokta |
| `measurable_floor` | 0,02 | bunun altındaki skorlar "ölçülemedi" sayılır, "çalışmadı" değil |

**Dondurulmuş OLMAYAN, dolayısıyla kanıt diye sunulamayacak sayılar:** adım sayısı (4000) ve batch (32) ölçüme
göre seçildi ama bir tahmini sınamıyor, ayar parametresidir; sonda (`mline/emergence.py`) bütçe boyutlandırma
amaçlı bir geliştirme koşusudur ve kendi başlığında kapı hakkında kanıt olmadığını yazar. Sondanın hiçbir
ölçümü yukarıdaki eşiklerin türetilmesinde kullanılmadı.

**Olası sonuçlar, tümü önceden adlandırılmış** (`preregistration.statuses`): `GATE_TRANSFERS`,
`GATE_AT_A_COST`, `GATE_DOES_NOT_TRANSFER`, `NOT_MEASURABLE_AT_THIS_LENGTH`, `NOT_MEASURABLE_AT_THIS_BUDGET`,
`INCOMPLETE`. Bir test, kuralın döndürebildiği her durumun bu listede olmasını zorluyor — listede olmayan bir
sonuç, kimsenin öngörmediği bir sonuçtur.

### Uzunluk kontrolü: neden eklendi

Denetim, hükmün dayandığı iki noktanın da **modelin eğitildiği uzunluğun üstünde** olduğunu gösterdi. Ölçüldü:
eğitim uzunluğu 512 bayt, değerlendirme dizisi dolgu 512'de 565 bayt (1,10×) ve dolgu 2048'de 2101 bayt (4,10×).
Bu haliyle sıfır bir skorun iki ayrı açıklaması var ve ayırt edilemiyorlar:

- model bilgiyi **mesafe boyunca taşıyamıyor** — deneyin sorduğu şey;
- model **bu uzunlukta diziyi hiç işleyemiyor**, mesafe ne olursa olsun — konumsal dışa-yayılım sorunu, kapıyla
  ilgisi yok.

Ayıran kontrol: `placement='near'`. Aynı genişlikte dizi kurar, ama dolguyu başa alır ve olguyu sorunun hemen
önüne koyar — uzunluk sabit, mesafe sıfır. `far`da düşüp `near`da geçen kol bir **bellek** sonucudur; ikisinde de
düşen kolun `far` skoru kapı hakkında hiçbir şey söylemez ve hüküm bunu `NOT_MEASURABLE_AT_THIS_LENGTH` olarak
raporlar. Çaresi de farklıdır: daha uzun koşu değil, daha uzun eğitim uzunluğu.

Bu, U2b'nin dondurma kontrollerinde yapılan hamlenin aynısı: bir düşüşün istediğin anlama geldiğine inanmadan
önce, o düşüşün daha sıkıcı bir sebebi olduğunu gösterebilecek koşulu kur.

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
- **Bulut bütçesi ~24 USD (1200 TL) ve ne satın aldığı.** Bütçe 40 USD olarak planlanmıştı ve 2026-09-25'te
  yaklaşık 24 USD'ye indi. 40 USD'lik plan tam replikasyondu (3 tohum × 3 kol × 3 lr = 27 koşu); bu artık
  sığmıyor, o yüzden sıra değişti, kapsam değil:
  **M1 yerelde** (3 kol × lr 6e-4 × tek tohum, bedava, ölçülen hızda 22,3 saat) · **M2 bulutta** (3 kol × 2 ek lr
  seviyesi, 6 koşu) · **M3 bulutta** (3 kol × 2 ek tohum, 6 koşu). Spot fiyatlarıyla 6+6 koşu yaklaşık 20 USD,
  gerisi pay. Önce lr, sonra tohum: kampanyanın kendi ölçümleri lr'yi üç ayrı döngüde çözüm ailesini belirleyen
  değişken olarak gösterdi (U2c çapraz-lr, U4-sep faz geçişi, R5), yani M1 olumsuz dönerse ilk sorulacak soru
  "başka lr'de de mi" olur. Ölçülen maliyet: fırlatma-sınırlı iş yükünde pahalı kart getirisi düşük, bu yüzden
  kart seçimi VRAM (dolayısıyla batch) üzerinden yapılmalı, FLOPS üzerinden değil. **Ucuz kartın yerelden hızlı
  olacağı varsayılmayacak** — GPU kullanımı tepede %5, darboğaz ardışıklık; ilk bir saat ölçüme ayrılır, kalan
  plan ölçülen s/adıma göre kurulur. Spot uygun: koşucu her koşulu ayrı mühürlüyor, kesinti en fazla bir koşul
  kaybettirir.
- Bütün önbellek ve korpus **E:'ye** yazılır; C: dolu.
- Eğitim, boş RAM eşiğin altındayken **başlatılmaz**.
