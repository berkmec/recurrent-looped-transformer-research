# U4-sep — çapa ayrımını artıran yardımcı kayıp (U2d bulgusunun nedensel testi)

**Durum: DENEY KENDİ SORUSUNU TEST ETMEDİ.** 6 koşuldan 3'ü tamamlandı, **2'si başlayıp yarıda kesildi**
(21 ve 18 adım, ikisi de karantinada), 1'i hiç başlamadı. *(Sayım 2026-09-24'te düzeltildi: önce "2'si hiç
başlamadı" yazılmıştı; bağımsız denetim, lr 0,003 kolunun s143 koşulunun 18 adım koştuğunu ve karantinaya
alınmadığını buldu.)* Süpürme
bilerek yarıda durduruldu — sebebi sonucu beğenmemek değil, **aletin yanlış büyüklüğü optimize ettiğini keşfetmek**.

Tarih: 2026-09-23 · Dondurulmuş protokoller `results/U4sep-lr0003` (`627993ae…`) ve `results/U4sep-lr003` (`cc97165a…`)
· Kod commit `66ade43`, düzeltme ve devam tasarımı `a5feb74`

---

## 1. Ana bulgular

**B1. Yardımcı kayıp yanlış büyüklüğü hedefliyordu.** Kayıp, `aralık / sınıf-içi yayılım` oranını maksimize ediyor.
U2d'nin bulgusu ise `aralık / sürüklenme` oranıydı. Yayılım ve sürüklenme farklı şeyler: yayılım aynı sınıftaki
durumların eğitim dağılımındaki dağınıklığı, sürüklenme ise nötr koşu boyunca durumun kat ettiği yol. `compare.json`
içindeki alanın adı `separation_ratio` olduğu için müdahale **bulgudan değil alan adından** kuruldu.

12 U2d koşulu, öğrenme oranına göre değil **ölçülen onarıma** göre yeniden gruplandığında:

| büyüklük | kırılgan | dayanıklı | ayırıyor mu |
|---|---|---|---|
| aralık / yayılım (`separation_ratio`) | 0,23 · 3,80 · **4,77** | **4,70** · 4,71 · **5,21** · 6,24 · 7,28 · 8,48 · 8,99 · 11,91 · 12,10 | **hayır** |
| aralık / sürüklenme | 0,06 · 0,49 · **0,57** | **0,92** · 1,25 · 2,49 · 2,64 · 2,81 · 2,89 · 3,74 | **evet, temiz** |
| aralık (tek başına) | 0,29 · 2,86 · **3,55** | **3,43** · 4,90 · 5,16 · 13,64 · … | **hayır** |

Kırılgan bir modelin `separation_ratio` değeri (4,77) iki dayanıklı modelinkinin (4,70 ve 5,21) arasında duruyor.

**B2. U2d'nin kendi "hiç örtüşme yok" iddiası da hatalıydı.** O iddia koşulları *öğrenme oranına göre* gruplayarak
üretilmişti (lr 0,003 = dayanıklı, lr 0,0003 = kırılgan). Bu gruplama `hard_st`'in düşük lr'deki üç koşulunu — onarım
+67 %, +72 %, +97 % — "kırılgan" kutusuna koyuyor. Sonuca göre yeniden gruplandığında U2d'nin yayımladığı aralıklar
tutmuyor: `min_center_distance` için "dayanıklı 13,64–33,78 / kırılgan 0,29–5,16" örtüşmeye dönüşüyor, eşik
2,49 | 2,64'ten 0,57 | 0,92'ye kayıyor, ve "mutlak sürüklenme iki grupta aynı (4,47 vs 4,47)" cümlesi de aynı
artefakt — sonuca göre ortalamalar 5,71'e karşı 4,06.

**Ayakta kalan:** `aralık / sürüklenme` hâlâ tek temiz ayırıcı ve U2d'nin nitel cümlesi — *"başarısızlık sürüklenme
değil, sınıf ayrımına göre sürüklenme"* — duruyor. Ona iliştirilen sayılar ve "sürüklenme eşit" iddiası durmuyor.

**B3. Kapı, kayba dejenere bir kaçış yolu veriyor.** Düşük lr'de iki tohumda da kapı 257. adımda %92–97 kapalıya
geçti. Kapı kapanınca durum donuyor, donmuş durumun yayılımı sıfıra gidiyor, ve paydası yayılım olan bir oranı
yükseltmenin en ucuz yolu tam olarak bu. Sonuç: oran tırmanıyor ama model görevi hiç öğrenmiyor.

**B4. Çöküş kalıcı değil, kaçılabilir bir tuzak.** Yüksek lr hücresi 897. adıma kadar düşük lr'dekiyle birebir aynı
tabloyu çiziyor, sonra kapı açılıyor (hold 0,017) ve görev bir anda çözülüyor. Öğrenme oranı, kaçışın adım bütçesine
sığıp sığmadığını belirliyor.

---

## 2. Ölçülen koşullar

| koşul | doğrulama | ayrım oranı | 512-suffix | temiz 4096 | kırpılan adım |
|---|---|---|---|---|---|
| lr 3e-4 · s117 | **0,207** | 0,58 | 0,198 | 0,184 | 796/2048 |
| lr 3e-4 · s143 | **0,215** | 0,40 | 0,216 | 0,196 | — |
| lr 3e-3 · s117 | **1,000** | **2722,55** | **1,000** | **1,000** | 1158/2048 |

Beş sınıflı görevde şans seviyesi **0,200**. Düşük lr'deki iki koşul da her uç noktada şansta; aynı tohum aynı lr'de
bu kayıp olmadan (U2c) doğrulama 1,000'e ulaşıyordu.

Yüksek lr koşulu her uç noktada 1,000 — ama bu **iyileştirme kanıtı değil**: U2c tabanı da aynı uç noktalarda %100
onarıyordu, yani gösterecek yer yoktu. Önceden yazılan (b) tahmini tam olarak bunu söylüyordu.

Bu koşulun gerçekten gösterdiği şey geometri: çapa aralığı 16,33 → **103,57**, yayılım 1,93 → **0,038**, oran
8,48 → **2722** (320 kat), göreve **sıfır maliyetle**. Yardımcı kayıp, yeterli öğrenme oranında kodu istediği yere
götürebiliyor.

Not: beş-durum otomatının ideal kodu zaten sınıf başına tek noktadır, dolayısıyla yayılımın sıfıra gitmesi bu görevde
dejenere değil **doğru**. İçerik taşıyan bir durum (yığın) için aynı şey varsayılamaz; ölçülmedi.

---

## 3. Koruma ve süreç

A8 koruması yüksek lr rejiminde 2048 adımın 1158'ini kırptı, kırpma öncesi en büyük norm 33 213, **atlanan adım 0**.
Koruma ve kırpma, aksi halde ıraksayacak bir rejimi ayakta tuttu.

Karantina: `results/U4sep-lr0003/interrupted-20260923T120928/` — s191 koşulu 21. adımda durduruldu, silinmedi,
gerekçesi `REASON.txt` içinde. Koşucu `complete.json` olmayan bir dizini kendiliğinden yeniden kullanmayı reddediyor.

---

## 4. Önceden kayıtlı karar kuralının verdiği sonuç

Karar kuralı (`horizon/u4sep_compare.py`, 7 test) veri üretilmeden önce yazılmıştı. Düşük lr kolu için verdiği
kategori **`INTERVENTION_INEFFECTIVE`** ve o kategorinin önceden yazılmış ifadesi aynen şu: *"yardımcı kayıp oranı
genişletmedi, dolayısıyla nedensel soru test edilmedi."* Oran kendi başlangıcına göre yükseldi ama U2c tabanının
oranına göre yükselmedi. Yeniden yorumlamaya gerek kalmadı — kural işini yaptı.

---

## 5. Kabul edilen tasarım hataları

1. **Test yanlış şeyi koruyordu.** `tests/test_sep_loss.py` ölçek-değişmezliği teste bağlamıştı (durumları 7 ile
   çarpınca oran değişmiyor, yani norm şişirerek doyurulamaz). Simetrik başarısızlığı — *durumu çökerterek*
   doyurulamaması gerektiğini — hiçbir şey kontrol etmiyordu. Ölçek-değişmezlik çökme-değişmezliği değil.

2. **Ölçüm sinyali içeriyordu, yanlış okundu.** Başlatma öncesi 24 adımlık ölçümde görev kaybı 1,60–1,66'da çakılıydı
   (ln 5 = 1,609 = şans) ve kapı 24. adımda 0,951'e çıkmıştı. İkisi de kaydedildi, "erken geçici durum" denip
   başlatıldı. 24 adım öğrenmeye karar vermek için gerçekten azdır, ama "görev kaybı tam şansta dururken kapı
   çarparak kapanıyor" altı koşul taahhüt etmeden önce daha uzun bir sondaj hak ediyordu.

3. **Hedef büyüklük bulgudan değil alan adından seçildi** (B1).

---

## 6. Devam: U4-sep2

Sorunun kendisi hâlâ açık ve `U4-sep2` onu soruyor (`horizon/geom_loss.py`, `horizon/sep_finetune.py`, commit
`a5feb74`; tahmin ve yanlışlama koşulları koşmadan önce ledger'a yazıldı):

- **Hedef düzeltildi:** `aralık / sürüklenme`, menteşeli — `max(0, hedef − oran)` — yani hedefe varınca itmeyi bırakır.
- **Çökme kaçışı yapısal olarak kapalı:** donmuş durumda aralık 0 ve sürüklenme 0 → oran 0 → menteşe maksimum ceza
  verir. U4-sep'i öldüren hamle artık modelin yapabileceği en kötü şey (`tests/test_geom_loss.py` bunu doğrudan test
  ediyor).
- **Sıfırdan eğitim yok:** görevi zaten çözen (doğrulama ≥ 0,95) ama kırılgan (onarım < %50) beş donmuş U2c kontrol
  noktası ince ayarlanıyor; görev kaybı modeli ayakta tutuyor.
- **Kontrol kolu zorunlu:** her kontrol noktası bir de **aynı bütçeyle, sadece görev kaybıyla** ince ayarlanıyor.
  Bu olmadan herhangi bir iyileşme "biraz daha eğitim"den ayırt edilemez.

**Sınır, dürüstçe:** sonuca göre gruplandığında kırılgan grup yalnızca 3 koşul ve üçü de aynı kol, aynı lr
(soft @ lr 0,0003). Yani "aralık/sürüklenme ayırıyor" bulgusu "kol × lr" ile karışık. Bu karışıklığı çözmenin tek
yolu müdahaledir — U4-sep2'yi değerli kılan da bu.


---

## 7. DÜZELTMELER (2026-09-24, bağımsız denetimden sonra)

Mühürlü metne dokunulmadı; düzeltmeler buraya eklendi ve ikinci bir manifest çıkarıldı.

**D1 — §6'daki çökme-güvenliği iddiası, bu rapor yazıldıktan 28 dakika sonra çürütüldü ve burada düzeltilmedi.**
§6 şöyle diyor: *"Çökme kaçışı yapısal olarak kapalı: donmuş durumda aralık 0 ve sürüklenme 0 → oran 0 → menteşe
maksimum ceza verir."* Bu **yanlış**. 64 adımlık bir pilot, gerçek çökmenin farklı işlediğini gösterdi: model durumu
donduruyor, sürüklenme tam 0'a gidiyor, ama sınıflar arasında küçük bir aralık kalıyor (0,05) ve **ham oran
patlıyor** — lr 1e-3'te 18, 3e-3'te 5298 — doğrulama 0,219 ve 0,266'ya düşerken. Yani çökme yine ödüllendiriliyordu,
başka bir yoldan. Düzeltme `drift_floor=0,5` ile yapıldı ve test, ölçülen çökme biçimine göre yeniden yazıldı
(`test_the_collapse_that_actually_happens_is_penalised`). O sırada commit mesajına yazılmıştı; **mühürlü rapora
yazılmamıştı**, ve bir denetçinin bakma ihtimali en düşük yer tam olarak orasıdır.

**D2 — §1'deki B1 tablosunun `aralık/sürüklenme` satırı 12 koşulun 10'unu listeliyor, eksiltme işareti olmadan.**
Eksik olan ikisi ve neden eksik oldukları:

| koşul | sürüklenme | aralık/sürüklenme |
|---|---|---|
| `u2c-soft-five-w32-s143@lr0.003` | 0,0116 | **1623,3** |
| `u2c-hard_st-five-w32-s117@lr0.003` | **0,0000** | **tanımsız (÷0)** |

İkisi de dayanıklı grupta ve kırılgan aralığın çok üstünde, yani sonucu değiştirmiyorlar. Ama satırın başlığındaki
**"evet, temiz"** nitelemesi, paydası bir koşulda **tam sıfır olan** bir metrik için hak edilmemiş bir kesinlik
taşıyor. Üstündeki satır 9 değerin dokuzunu da listeliyor, altındaki satır açık "…" kullanıyor; bu satır ikisini de
yapmadığı için okuyucu tam liste sanıyor. `audit/audit_u4sep.py` de `None` değerini sessizce eliyor ve kanıt
cümlesinde bundan söz etmiyor.

**D3 — §2 tablosunda `lr 3e-4 · s143` satırının "kırpılan adım" sütunu "—" gösteriyor.** Ham kayıtta
`guard.clipped = 963/2048` — iki düşük-lr koşusunun en yükseği. Em dash "hiç kırpılmadı" diye okunuyor.

**D4 — §1 B3'teki "iki tohumda da 257. adımda" ifadesi bir tohum için doğru.** s117'de `hold@257 = 0,919`; s143'te
`hold@257 = 0,066` ve 0,90'ı **287. adımda** aşıyor. Örüntü iki tohumda da var, ama belirtilen adım sayısı yalnız
birinde geçerli.
