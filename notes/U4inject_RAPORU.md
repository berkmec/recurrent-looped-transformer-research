# U4-inject — toplamalı geri besleme yerine attention-injection (FLT / arXiv 2605.18797 uyarlaması)

**Durum: MEASURED_NEGATIVE, 6/6 koşulda tekrarlandı.** Enjeksiyon kollarının hiçbiri görevi öğrenemedi; mühürlü
Core kontrolü üç tohumda da kusursuz. Ayrıca deneyin dayandığı ön kabulün yarısı **tabanda hiç görünmedi**.

Tarih: 2026-09-23/24 · Dondurulmuş protokol `results/U4inj` (`523dba46…`) · 9 koşul, tamamı tamamlandı

---

## 1. Ana tablo

| kol | tohum | doğrulama | norm büyümesi | ayrım oranı | grad tepe | kırpılan | 512-suffix | temiz 4096 |
|---|---|---|---|---|---|---|---|---|
| additive | 117 | **1,000** | 0,97 | 5,77 | 1,78e4 | 67 | 0,430 | 1,000 |
| additive | 143 | **1,000** | 1,20 | 4,99 | 57,2 | 83 | 0,374 | 1,000 |
| additive | 191 | **1,000** | 1,03 | 4,36 | 1,23e3 | 99 | 0,610 | 0,958 |
| inject | 117 | 0,234 | 0,31 | 0,10 | 9,6 | 39 | 0,220 | 0,204 |
| inject | 143 | 0,277 | 0,48 | 0,15 | 17,1 | 17 | 0,198 | 0,206 |
| inject | 191 | 0,246 | 0,47 | 0,11 | 20,8 | 31 | 0,196 | 0,226 |
| inject_mix | 117 | 0,215 | 0,40 | 0,07 | 6,61 | 66 | 0,198 | 0,186 |
| inject_mix | 143 | 0,309 | 0,27 | 0,07 | 2,35 | 22 | 0,216 | 0,204 |
| inject_mix | 191 | 0,215 | 0,27 | 0,09 | 2,93 | 22 | 0,194 | 0,178 |

Beş sınıflı görevde şans **0,200**. Öğrenen kol sayısı: additive **3/3**, inject **0/3**, inject_mix **0/3**.

---

## 2. Deneyin gerçekte ölçtüğü şey

Bu sonuç, **FLT makalesinin yöntemi hakkında değil, benim uyarlamam hakkında** bir ifadedir. Koda bakıldığında
(`horizon/inject.py`) her iki enjeksiyon kolunda da önceki durum **yalnızca sorgu** olarak giriyor:

```
a = io(attend(iq(pn(prev)), k, v))     # k, v = core.dn(e), yani mevcut girdiden
inject      →  z = e + a
inject_mix  →  z = w0*e + w1*a
```

Durumun **içeriği** hiçbir yoldan yeni duruma akmıyor; yalnızca penceredeki hangi girdiye bakılacağının ağırlıklarını
belirliyor. Dolayısıyla:

- İki kol arasındaki fark, `a`'nın `e` ile nasıl birleştirildiğidir — dışbükey bileşim **girdi ile enjekte edilen
  vektör** arasındadır, durum ile aday arasında değil.
- Bu yüzden `inject_mix`, planlandığı gibi bir "geniş kanal" testi **değildir**; o hipotezi test eden kol kurulmadı.

**Ölçülen tek ifade, 6/6 koşulda:** özyinelemeli durumu adıma *yalnızca attention sorgusu olarak* vermek, bu
mimaride öğrenmeyi engelliyor. FLT'nin "önceki döngü çıktısı" bir *dizi* olduğu için, onların ortamında aynı
darboğaz bulunmayabilir.

---

## 3. Ön kabulün yarısı tabanda yok

Makale toplamalı geri beslemeyi **artık patlamasıyla** suçluyor. Bu Core'da, bu ölçekte, patlama yok:
durum RMS'i 512 adımlık nötr koşu boyunca **×0,97 / ×1,20 / ×1,03** — üç tohumda da büyümüyor.

Dolayısıyla `inject_mix`'in bütün gerekçesi (dışbükey bileşim artık biriktiremez) **var olmayan bir arızayı**
hedefliyordu. Bu, kolun ilk sonucu gelmeden önce deftere yazıldı.

Enjeksiyon kollarında norm tersine **düşüyor** (×0,27–0,48), yani durum koşu boyunca sönüyor.

---

## 4. Gradyan sorusu: bu deney cevaplamıyor

Enjeksiyon kollarının kırpma öncesi tepe gradyanı 2,35–20,8; kontrolünki 57–17.766. İlk bakışta "enjeksiyon
gradyan patlamasını çözdü" gibi duruyor ve **ben de bir süre öyle raporladım**. Yanlış: öğrenmeyen bir modelin
gradyanları zaten küçüktür, yani iki açıklama aynı veriye uyuyor.

Ham kayıtlardan ayrıştırma (s117, ikisi de henüz hiçbir şey öğrenmemişken):

| adım aralığı | additive kayıp | inject kayıp | additive grad medyan | inject grad medyan |
|---|---|---|---|---|
| 1–64 | 1,6066 | 1,6085 | **0,145** | **0,146** |
| 257–512 | **0,1993** | 1,5487 | 0,041 | 0,236 |
| 1025–2048 | 0,0201 | 1,5320 | 0,002 | 0,213 |

İlk 64 adımda gradyanlar **birebir aynı**. Fark ancak kontrol kolu öğrenmeye başladıktan sonra çıkıyor. Ayrıca
kontrolün 1,78e4'lük tepesi **nadir bir sivrilik**; medyanı yakınsarken 0,002'ye iniyor, enjeksiyonunkinin (0,213)
altına. Tepeyi tepeyle karşılaştırmak bunu gizliyordu.

**Sonuç: U4-inject gradyan patlaması hakkında hiçbir şey kanıtlamıyor.** Elindeki tek karşılaştırma öğrenen bir
modelle öğrenmeyen bir model arasında. Bu soru, **ikisi de öğrenen** kollar gerektirir.

---

## 5. Kampanya örüntüsü

Bu, daha güvenli görünen seçeneğin daha kötü ölçüldüğü **dördüncü** vaka: düşük öğrenme oranı (kararlı, kırılgan),
kapıyı elle tasarlanmış kurala denetlemek (kararlı, zayıf), yumuşak karışım (türevlenebilir, içerik taşıyan durumda
sızdıran), ve şimdi attention-injection (kararlı, öğrenemeyen). Ortak payda: **durumun etkisini sınırlayan
mekanizmalar, durumun taşıyabildiğini de sınırlıyor.** Ölçek merdiveninde geri besleme biçimi seçilirken bu bir
önsel olarak kullanılmalı.

---

## 6. Sıradaki

- **Gerçek genişlik testi** (havuza): duruma pencerede kendi anahtar/değer girdisini ver, böylece *içeriği* yeni
  duruma çekilebilsin. Mevcut iki kol bunu test etmiyor.
- **Gradyan sorusu** ancak iki öğrenen kol karşılaştırılırsa cevaplanabilir; mevcut tasarım buna uygun değil.
- Ölçek merdiveni için taşınabilir sonuç yine U2c'nin kapısıdır; bu döngü bir yol **kapattı**, yeni yol açmadı —
  ki arama uzayını daraltmak da kazançtır.

---

## 7. DÜZELTME (2026-09-25) — kollar **tek terimde** ayrışmıyor, ikide ayrışıyor

§2 durumun içeriğinin yeni duruma akmadığını zaten söylüyor ve §6 eksik kolu adlandırıyor. Eksik olan şu:
aynı farkın **kontrol koluna göre** ne anlama geldiği. Kampanyanın kendi yöntem kuralı (U4-sep2'den öğrenilen,
`notes/M_HATTI_PLANI.md` §4 madde 1) şöyle: *eşli karşılaştırmanın iki kolu tam olarak tek terimde ayrışmalı,
yoksa sonuçtaki fark o terime atfedilemez.* U4-inject bu kuralı çiğniyor.

İki çözücü yan yana:

```
mühürlü Core.decode      z = e + alpha*sigmoid(gate([e,r]))*merge(r)   <- geri besleme ÖNCE
                         n = dn(z); k = dk(n); v = dv(n)               <- pencere K/V, geri besleme SONRASI z'den

InjectCore.decode        n = dn(e); k = dk(n); v = dv(n)               <- pencere K/V, enjeksiyon ÖNCESİ e'den
                         z = e + io(attend(iq(pn(prev)), K, V))        <- enjeksiyon SONRA
```

Yani enjeksiyon kolları **iki şeyi birden** değiştiriyor:

1. durumun nasıl girdiği (toplamalı `merge` yerine attention enjeksiyonu);
2. durumun kayan pencere önbelleğinde (`dk`/`dv`) **bulunup bulunmadığı**.

Mühürlü modelde önbelleğe yazılan anahtar/değerler geri besleme sonrası `z`'den türediği için durum geçmişi
pencerede de taşınıyor — **ikinci bir özyineleme kanalı**. Enjeksiyon kollarında bu kanal yok; önbellek yalnızca
kodlayıcı çıktısını taşıyor.

**Ölçüldü, iddia edilmedi** (`tests/test_inject_two_terms.py`): aynı girdi, yalnızca `s` değiştirilmiş iki önceki
durumla,

- mühürlü Core'da sonraki `dk` **değişiyor** (durum önbelleğe giriyor),
- `inject` ve `inject_mix`'te `dk` **bitwise aynı kalıyor** (durum önbelleğe girmiyor),
- ama `inject`'in çıktısı `z` yine **değişiyor** — yani kol özyinelemeliliğini tamamen kaybetmiş değil, sorgu
  yolundan hâlâ bağlı. (Bu üçüncü kontrol kasten var: ikinci maddeyi "enjeksiyon durumu hiç kullanmıyor" diye
  fazla okumayı engelliyor.)

### Sonucun sınırı, net olarak

- **Ayakta kalan:** "bu altı koşulda enjeksiyon kolları öğrenmedi, kontrol üç tohumda da öğrendi." Bu bir olgu ve
  `results/U4inj` içinde duruyor.
- **Ayakta kalmayan:** "attention enjeksiyonu başarısızdır." İkinci bir özyineleme kanalını kaybetmiş bir kolun
  daha kötü ölçülmesi, enjeksiyon mekanizması ne yaparsa yapsın beklenir. Sonuç, mekanizmayı **kanal kaybından
  ayıramıyor**.
- **Ayıracak kol kurulmadı:** enjeksiyon **artı** pencere K/V'sinin enjeksiyon sonrası `z`'den türetilmesi. §6'daki
  "gerçek genişlik testi" ile aynı kol değil — o, duruma pencerede kendi anahtar/değer girdisini vermekle ilgili;
  bu ise kontrolün zaten sahip olduğu yolu enjeksiyon koluna geri vermekle ilgili. İkisi ayrı deney.

**M-hattına etkisi yok.** M1 bu döngüden hiçbir şey taşımıyor; taşınan sonuç U2c'nin kapısı. §5'teki örüntü
gözlemi ("durumun etkisini sınırlayan mekanizmalar taşıyabildiğini de sınırlıyor") ise bu düzeltmeden sonra
**daha zayıf** durmalı: dört vakadan biri artık karışık tasarımlı, yani örüntü üç sağlam vakaya dayanıyor.

**Yöntem dersi:** kuralı U4-sep2'de öğrenip bir sonraki döngüde uygulamayı atladım. Kuralın kendisi rapora
yazılmıştı; kontrol listesi olarak kullanılmamıştı. Bundan sonra her kol tasarımında iki çözücü yan yana konup
farkların sayısı yazılacak.
