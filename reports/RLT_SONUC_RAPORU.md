# RLT ve başlık belleği — uçtan uca sonuç raporu

**On dört paketin tanımlanmış yerel deney döngüsü tamamlandı. Çalışan bir araştırma prototipi ve denetlenmiş olumlu, olumsuz, karışık sonuçlar var; genel ve sınırsız doğru düşünme iddiası doğrulanmadı.** Bu kapanış, yirmi fikrin her alanda başarılı olduğu veya otuz riskin ortadan kaldırıldığı anlamına gelmez.

Ana sonuçlara dahil edilen 63 eğitim/devam eğitimi koşulu, bağlı sistemde 6.144 yürütme ve gerçek durum penceresinde 1.536 ek yürütme tamamlandı. Ana bağlı sistem 512, pencere deneyi 256 ayrı kaynak problemi kullanır; koşul ve model tekrarları bağımsız yeni problem sayılmaz. Üç öğretmenden türetilen altı öğrencinin ağırlık ve çıktı kanıtları korunmuştur. 108 yazılım testi geçti. Önceki üç araştırmadaki 424 kanıt dosyası değişmeden doğrulandı.

## RLT'nin sonsuz derinlik iddiası

Kaynak mimaride t token sonrasında yinelenen yol t×L_D decoder bloğundan geçer; token başına kullanılan blok sayısı sabittir. Bu, hesaplama yolunun uzatılabilmesini ifade eder. Her ek adımda doğruluğun artması veya sınırsız bilginin korunması sonucu çıkmaz. [Sabitlenmiş resmî kaynak](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md).

Biz t = 32…1.024 tüketilen işlem, ayrı k = 1…8 başlık güncellemesi ve d = 8'e kadar yeni bağımlılık derinliği ölçtük. Parite/beş durum görevleri küçük, sonlu yeterli durum taşır. Bu görevlerde uzun ufuk başarısı, geniş kavramsal akıl yürütme veya sınırsız bilgi kapasitesi değildir. Kaynak yazarlarının 26–29M parametreli modellerinin yeniden üretimi yapılmadı.

## Araştırma hattındaki 14 paket

MIXED, ölçümün eksik olduğu anlamına gelmez; farklı koşulların farklı sonuç verdiğini belirtir. MEASURED_NEGATIVE, tanımlanan başarı ölçütünün karşılanmadığı geçerli bir deney sonucudur.

| No | Paket | Sonuç | Kanıtın kapsamı |
|---|---|---|---|
| 1 | RLT çekirdeği ve kontrol | IMPLEMENTED_TESTED | Gizli durum, yerel KV ve global encoder belleği; nedensellik ve akış/toplu eşlik. Bağımsız küçük uygulama. |
| 2 | Uzun ufuk | MIXED | 32 işlem uzunluğunda eğitimden sonra 32–1.024 testi; parite, beş durum ve eğitim tohumu ayrı. Sonlu test üst sınırı. |
| 3 | Ek hesaplamanın yararı | MIXED | t/k/d ayrıldı. Bazı ek turlar yararlı; kimlik son ekleri ve tur eğrileri tekdüze kazancı desteklemiyor. |
| 4 | Gecikmiş hata ve eğitim maliyeti | MIXED | Prefix öğrenimi ile son-cevap gözetimi ayrıldı; son-cevap koşulları şans düzeyine yakın. Gerçek backward/update ve süreç belleği ölçüldü. |
| 5 | Metinden başlık ve ilişki | MIXED | Öğrenilmiş çıkarıcı görülen kalıpta başarılı; yeni kalıpta aktarım zayıf. Başlık örgütleme sonlu varlıklar için deterministik. |
| 6 | İleri/çift yönlü ağ | MIXED | Aynı tur sayısında eğitim benzeri görevde büyük yarar; görülmemiş derinlikte çift yön tek başına yeterli değil. |
| 7 | Seçici revizyon | MEASURED_NEGATIVE | Etkilenen cevapta en az %10 kazanç ve etkilenmeyeni koruma ölçütü iki test ailesinin ikisinde de geçmedi. |
| 8 | Başlık penceresi | IMPLEMENTED_TESTED | Kaynak ve gerçek üretilmiş durum pencereleri ayrı uygulandı; üretilmiş aktif geçmişin tepe büyüklüğü 4/8 kayıt. |
| 9 | Sıkıştırma ve yeniden okuma | MIXED | Sabit bağlamda son z sınırı eşdeğer; güncel kısıtlardan sinirsel yeniden başlatma derin görevde kayıp verdi. |
| 10 | Ham arşiv ve vektör erişimi | MEASURED_NEGATIVE | Ham/sürüm/özet korundu; sözcük-vektörü recall@5 yalnız %4,17. Anahtar erişimi bunun başarısı sayılmadı. |
| 11 | Yeterlilik ve durma | MIXED | Ana IID testte %24,14 daha az gerçek tur ve %38,93 doğru sertifikalı kapsam; yeni derinlikte kapsam çok düşük. |
| 12 | Gerçek RLT/başlık entegrasyonu | MIXED | Metin→başlık→model→arşiv→denetim uçtan uca çalıştı. Ayrı geri besleme kazancı küçük/belirsiz; yeni cümlelerde cevap yok. |
| 13 | Üretilmiş izleri ağırlıklara aktarma | MIXED | Gerçek öğretmen izlerinden altı öğrenci eğitildi, arşiv kapalı çıktı eşliği geçti. Kolay görevde zarar, derinde koşullu kazanç. |
| 14 | Ölçek, aktarım ve maliyet | MIXED | Yaklaşık 24 bin ile 1,51 milyon parametre arasında duyarlılık; büyük modelde 1/3 tohum kısa problemi öğrendi. Enerji/RL/büyük dil modeli ölçülmedi. |

## Küçük çekirdek: uzunluk gerçekten neyi değiştirdi?

Aşağıdaki değerler üç eğitimin ortalama son durum doğruluğu. Her model/uzunluk hücresinde 512 yeni örnek; en sağda üç ayrı eğitimin 1.024 sonucunun aralığı var. Aralık güven aralığı değildir. Kontrol, RLT'nin global ve yerel dikkati korunup gizli durum geri beslemesi kapatılan sürümüdür.

| Görev / genişlik / model |32 |128 |1.024 |1.024 tohum aralığı |
|---|---:|---:|---:|---:|
| Parite / 32 / RLT / tam | %100,00 | %100,00 | %100,00 | %100,00–%100,00 |
| Parite / 32 / Geri beslemesiz | %48,24 | %47,14 | %49,87 | %48,05–%52,93 |
| Parite / 32 / RLT / kesilmiş, 8 adım | %100,00 | %100,00 | %100,00 | %100,00–%100,00 |
| Beş durum / 32 / RLT / tam | %100,00 | %99,87 | %98,63 | %96,09–%100,00 |
| Beş durum / 32 / Geri beslemesiz | %19,27 | %19,60 | %20,77 | %18,36–%22,66 |
| Parite / 48 / RLT / tam | %100,00 | %100,00 | %100,00 | %100,00–%100,00 |
| Parite / 48 / Geri beslemesiz | %49,35 | %47,46 | %51,11 | %49,02–%52,15 |

Her adımda hedef verilen öğrenme ile yalnız son cevaptan hedef verilen öğrenme aynı sonuç değildi. Ayrı son-cevap deneyinde tam ve kesilmiş gradyanla eğitilen altı koşulun 32 ve 128 uzunluktaki doğrulukları %47,46–%51,56 arasında, iki sınıflı görevin şans düzeyine yakın kaldı. Bu yalnız sabit 2.000 güncellemelik protokol için olumsuz bulgudur; şansla istatistiksel eşdeğerlik iddiası değildir. Prefix deneyinde 32 kat fazla gözetim hedefi vardır; iki rejim eşit etiket bütçesi karşılaştırması değildir.

Kesilmiş gradyan gizli durum ve yerel decoder KV yollarını sekiz adımda keser; encoder/global bellek yollarını kesmez. Küçük veya sıfır yerel türev tek başına unutma, sıfır olmayan türev de öğrenme kanıtı değildir. Gerçek checkpoint yeniden hesaplamasında backward/AdamW eşliği kontrol edildi. Kısa süreç benchmark'ında koşul başına üç tekrarın ortalama toplam tepe çalışma belleği 592,1–599,4 MiB arasında kaldı. Bu ölçüm belirgin toplam bellek tasarrufu göstermedi; aktivasyon belleği ayrı ölçülmedi.

![Çekirdeğin uzun ufuk sonuçları](E:/rlt-research/2026-09-16-e2e/reports/figures/core-long-horizon.png)

Ayrı kimlik-son-eki deneyinde aynı 32 işlemin sonuna cevabı değiştirmeyen 128 işlem eklendi. Aşağıdaki modellerin son ek yokken doğruluğu üç tohumda da %100'dü. Tablo son ekten sonraki doğrulukları gösteriyor. Rastgele uzun dizilerde başarı, farklı işlem düzeninde gereksiz adımlara dayanıklılık sağlamadı; etki model ve tohuma göre değişti.

| Görev / genişlik / model | Tohum 17 | Tohum 43 | Tohum 91 |
|---|---:|---:|---:|
| Parite / 32 / RLT / tam | %100,00 | %50,20 | %49,80 |
| Parite / 32 / RLT / kesilmiş, 8 adım | %100,00 | %55,47 | %100,00 |
| Beş durum / 32 / RLT / tam | %23,24 | %34,77 | %42,77 |
| Parite / 48 / RLT / tam | %100,00 | %100,00 | %50,00 |

## Senin başlık/ağ fikrinde işe yarayan ve yetersiz kalan parçalar

Sabit yerel tur sayısında çift yönlü ağ, eğitim benzeri yeni görevlerde son cevap doğruluğunu ileri yönün %26,11'inden %95,70'e çıkardı. Ancak gerçek yeni bağımlılık derinliği 8'de tek başına çift yönlü ağ %9,90'da kaldı; ileri yön %11,98 idi. Doğru ara değerlere ek gözetim vermek aynı derinlikte %33,46'ya çıkardı. Ek hedefler gerçek bir eğitim avantajıdır; sonuç yalnız ağ biçimine veya eşit hesaplama maliyetine yüklenemez.

Gerçek model geri beslemesi sonraki token durumuna bağlandı; ayrı ek yararı küçük ve belirsiz kaldı. Seçici revizyonun önceden belirlenen güçlü ölçütü geçmedi. Sözcük karma vektörüyle erişim@5 yalnız %4,17 idi; kalıcı arşivin doğru çalışması bunu başarılı anlamsal erişim yapmaz.

![Başlık ağı ve ara gözetim](E:/rlt-research/2026-09-16-e2e/reports/figures/learning-generalization.png)

Ana bağlı sistemde eğitim benzeri yeni problemlerde doğru sayısal aday %98,18, doğru ve doğrulayıcıdan geçen son cevap kapsamı %38,93 çıktı. Sabit 8 tura göre gerçek güncelleme sayısı %24,14 azaldı. Yeni derinlikte aday doğruluğu %37,63'e, kabul kapsamı %0,13'e indi. Yeni cümle düzenlerinde ayrıştırma reddi sonrası son cevapların tamamı eksik kaldı.

Güncel kısıt özetinden modeli yeniden başlatmak, derinlik 8 aday doğruluğunu %37,63'ten %20,57'ye düşürdü. Buna karşılık sabit bağlamda **gerçek son z durumunu saklayıp yeniden yükleyen** ayrı pencere deneyi bütün eşlenik izleri ve çıktıları korudu: 70.080 gerçek iz, 8.760 sınır/okuma, aktif tepe geçmiş 8→4. Bu ikinci mekanizma genel anlamsal özet değildir. Serileştirilmiş aktif yük yaklaşık yarıya inerken ek sınır kayıtları diski yaklaşık %21 büyüttü; toplam RAM tasarrufu gösterilmedi.

![Bağlı sistemde aday ve kabul kapsamı](E:/rlt-research/2026-09-16-e2e/reports/figures/connected-quality-coverage.png)

## Geçmişteki gerçek düşünme yolunu sonraki eğitime aktarma

İlk ara-gözetim deneyi doğru ara hedefler kullanıyordu. Ayrı deneyde modelin gerçekten ürettiği latent/sayısal yollar saklandı ve altı öğrenci devam eğitiminde kullanıldı. Bütün bölümün sertifikalanabildiği örnek bulunmadı; önceden belirlenen sorgulanan-bileşen seçeneği üç tohumda 313 / 418 / 333 bölüm sağladı. Bu seçim sığ örneklere kayıyor. Başarılı son aday, her ara adımın doğruluğunu kanıtlamıyor.

MSE ortalama karesel hatadır; daha düşük değer daha iyi sonucu gösterir.

| Yeni test | Yalnız öğretmen son cevabı: MSE | Gerçek ara izler de kullanıldığında: MSE |
|---|---:|---:|
| Eğitim benzeri |0,002615 |0,011706 |
| Gerçek derinlik 8 |0,697890 |0,557985 |

Kolay görevde üç tohumda da zarar, derin görevde cevap-hedefli kola göre üçünde de MSE iyileşmesi var. Özgün eşli karşılaştırmanın tohum-t %95 aralıkları iki bölümde de sıfırı içeriyor. Ayrı, sonradan eklenen başlangıç-modeli referansı da kolayda kötüleşme/derinde iyileşme yönünü gösterdi; bu özgün önceden kayıtlı üçüncü kol değildir. Yöntemin genel üstünlüğü sonucu çıkarılmadı.

Altı öğrencinin 512'şer örnekte arşiv ve üretici modülleri bulunmayan çıkarım dışa aktarımı kayıtlı çıktılarla tam eşleşti. Bu, çıkarımın saklanan öğretmen izlerini okuyarak cevap vermediğini gösteren dar bir paket kontrolüdür; işletim sistemi erişim izolasyonu değildir.

## Büyütünce çözüldü mü?

1,51M parametreli üç RLT eğitiminden yalnız biri bu ayarlarla kısa görevi öğrendi. Başarılı tohumun 1.024 adımdaki doğruluğu %99,61 oldu; diğer iki tohum 1.024'te %48,44'tü. Başarılı modelde, ayrı aynı 32 işlemin sonuna cevabı değiştirmeyen 128 kimlik işlemi eklemek doğruluğu %99,41'den %50,59'a düşürdü. Ek, girdinin işlem düzenini değiştirir; saf başlık-turu deneyi değildir.

Genişlikle dikkat başı sayısı ve küçük çekirdeğe göre cihaz değişti. Öğrenme oranı ölçek başına eniyilenmedi. Bu, belirtilen protokol altında duyarlılık ölçümüdür; ölçek yasası veya büyük modelin evrensel kapasite sınırı değildir.

![Ölçek, tohum ve kimlik işlemleri](E:/rlt-research/2026-09-16-e2e/reports/figures/scale-seeds-and-neutral.png)

## Teslimat ve açık geçerlilik sınırları

Çalışma ayrı E: kökünde yürütüldü. Önceki 424 kanıt değişmedi. Ham gözlemler, gerçek model izleri, başarısız ilk denemeler, model ağırlıkları, tahminler, protokoller ve bağımsız denetimler korundu. Ana bağlı sistem ve pencere deneyinin toplam 242.162 gerçek güncelleme izi ayrı SQLite arşivlerindedir; öğretmen eğitim arşivi ayrıca saklanır.

Kapanış öncesinde mevcut kaynak, protokol, ağırlık, tahmin ve arşiv dosyaları kaydedilmiş karmalarına karşı tekrar okundu. Bağımsız bellek deneyinin bazı eski dosyalarında önceden kaydedilmiş bütün-dosya karması yok; bu eksikler kapanış doğrulamasında ayrı listeleniyor. Veri tabanı içindeki kayıt karmaları bunun yerine bütün dosyanın geçmişteki durumunu kanıtlıyor sayılmadı. Son teslimat manifesti mevcut dosyaların bütünlüğünü korur; haricî imza veya eksik eski kayıtların yerine geçen kanıt değildir.

Serbest doğal dil, öğrenilmiş genel konu bölme, güvenilir anlamsal vektör erişimi, geniş model/RL eğitimi, enerji, toplam RAM için sabit üst sınır ve sonsuz doğru akıl yürütme **doğrulanmadı veya ölçülmedi**. Bu açık sınırlar, yerel deneylerin bitmediği anlamına gelmez. Yeni başarı iddiaları yeni deney gerektirir.

Kanıt yolları:

- [Çekirdek ayrıntıları](E:/rlt-research/2026-09-16-e2e/reports/core-report.md) ve [bağımsız çekirdek denetimi](E:/rlt-research/2026-09-16-e2e/reports/core-independent-audit.md).
- [Ağ, revizyon ve ara gözetim](E:/rlt-research/2026-09-16-e2e/reports/learning-report.md) ve [bağımsız öğrenme denetimi](E:/rlt-research/2026-09-16-e2e/reports/learning-independent-audit.md).
- [Metinden sonuca bağlı sistem](E:/rlt-research/2026-09-16-e2e/reports/connected-report.md) ve [6.144 arşivin denetimi](E:/rlt-research/2026-09-16-e2e/reports/connected-independent-audit.md).
- [Üretilmiş durum penceresi](E:/rlt-research/2026-09-16-e2e/reports/reasoning-window-report.md) ve [1.536 arşivin denetimi](E:/rlt-research/2026-09-16-e2e/reports/reasoning-window-independent-audit.md).
- [Gerçek izlerden sonraki eğitim](E:/rlt-research/2026-09-16-e2e/learning/distillation/report.md) ve [bağımsız aktarım ve başlangıç referansı](E:/rlt-research/2026-09-16-e2e/reports/distillation-independent-audit.md).
- [Büyük yerel modelin sonuçları](E:/rlt-research/2026-09-16-e2e/reports/scale-report.md), [yirmi fikrin ayrı karşılığı](E:/rlt-research/2026-09-16-e2e/reports/user-requirement-map.md), [matematik ve otuz risk kapsamı](E:/rlt-research/2026-09-16-e2e/reports/mathematical-interpretation.md).
- [Kapanış kaydı](E:/rlt-research/2026-09-16-e2e/audit/campaign-closure.json), [kapanış öncesi güncel karma kontrolü ve kapsam sınırları](E:/rlt-research/2026-09-16-e2e/audit/closure-pins.json), [bütün dosyaların SHA256 manifesti](E:/rlt-research/2026-09-16-e2e/audit/final-manifest.json), [doğrulama yönergesi](E:/rlt-research/2026-09-16-e2e/README.md).
