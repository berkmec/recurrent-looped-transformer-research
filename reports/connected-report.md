# Metinden sonuca bağlı sistemin ölçümü

**Durum: deney tamamlandı, sonuç MIXED.** Öğrenilmiş metin çıkarıcısı, sürümlü başlıklar, bağlı sinirsel RLT modeli, ham arşiv, pencere denetleyicisi ve aday doğrulayıcısı aynı akışta çalıştırıldı. 512 yeni kaynak problemi, üç ortak model/çıkarıcı tohumu ve dört koşul ile6.144 bölüm yürütüldü. Bunlar6.144 bağımsız problem değildir. Toplam yerel çalıştırma süresi2.220,09s oldu.

Bağımsız denetim24 koşulu,6.144 SQLite arşivini,253.746 ham kaydı,86.016 kaynak gözlemini ve172.082 gerçek sinirsel güncelleme izini yeniden kontrol etti. Kaynak etiketleri değerlendiricideki en küçük kareler çözümü yerine ayrı bir işaretli grafik geçişiyle hesaplandı; en büyük fark7,11e-15. Tahmin, kaynak, sürüm, başlık, arşiv ve özet karmaları eşleşti. Detay: `reports/connected-independent-audit.md`, `audit/connected-independent-20260916T175138.json`.

Protokol: `0b61590618f799ecece783cac7d43c08521f7d2ae901228f293a1df1c5a4e2ac`. Veriler ve koşullar açılmadan sabitlendi; sonuçlardan sonra ayar seçilmedi. Son model her tohumda önceden belirlenen combined600 kontrol noktasıdır. Her bölüm14 kaynak cümlesi, üç amaçlanan soru ve bir kısıt değişikliği içerir. Etiketler kaynakta yazılan sekiz ondalıklı sayılardan çıkarılır; doğrulayıcı cevabı modelin yerine yazmaz.

## Cevap adayının doğruluğu ve kabul edilen cevabın kapsamı

Hata eşiği0,1. “Doğru aday”, doğrulayıcı kabul etmese bile modelin sayısal tahmininin hedefe yakın olmasıdır. “Doğru ve kabul edilen”, hem kaynağa göre doğru hem de ayrıştırılmış kısıtların denetiminden geçmiş son cevapların bütün amaçlanan son sorulara oranıdır. Aşağıdaki sayılar üç modelin ortalamasıdır; her modelde her test ailesi256 aynı kaynak problemi kullanır.

| Test ailesi | Bellek / durma | Doğru aday | Doğru ve kabul edilen | Aday MSE |
|---|---|---:|---:|---:|
| Eğitim benzeri | Tam geçmiş / uyarlanır | %98,18 | %38,93 | 0,001551 |
| Eğitim benzeri | Tam geçmiş / sabit8tur | %98,18 | %25,13 | 0,001486 |
| Eğitim benzeri | Güncel kısıt özeti / uyarlanır | %97,66 | %41,28 | 0,002277 |
| Eğitim benzeri | Yeni cümle düzeni / tam geçmiş | %0 | %0 | Ölçülemez: aday yok |
| Bağımlılık derinliği8 | Tam geçmiş / uyarlanır | %37,63 | %0,13 | 0,301349 |
| Bağımlılık derinliği8 | Tam geçmiş / sabit8tur | %37,63 | %0 | 0,301349 |
| Bağımlılık derinliği8 | Güncel kısıt özeti / uyarlanır | %20,57 | %0 | 0,339339 |
| Bağımlılık derinliği8 | Yeni cümle düzeni / tam geçmiş | %0 | %0 | Ölçülemez: aday yok |

Sayılan kabul edilmiş çıktılarda bu örneklem içinde kaynak hatası gözlenmedi. Bu, gerçek hata olasılığının sıfır olduğuna dair kanıt değildir. Kabul sayısı0 ise koşullu risk bilinmiyor/null'dır. Derinlik8 uyarlanır koşulundaki%0,13 kapsam, üç modelin256'şar yürütmesinde yalnız bir kabulden gelir; kullanışlı genel çözme başarısı değildir. Her modelin kabul sayısı ve ayrı Wilson aralığı ham sonuçlarda korunur; aynı problemlerin tekrarlarını bağımsız sayarak dar bir birleşik aralık üretilmez.

## Durma işe yaradı mı?

Eğitim benzeri ailede uyarlanır durma, tam geçmişi kullanan sabit8tura göre son aday doğruluğunu korudu ve kabul edilen doğru cevapların kapsamını13,80 yüzde puanı artırdı. Üç tohumda fark sırasıyla13,28/13,28/14,84 puandı. Ortak örnek/tohum eşli betimleyici bootstrap aralığı10,29–17,32 puan; yalnız üç bağımsız eğitim vardır.

Tekrar oynatmalar dahil gerçekten yürütülen ortalama sinirsel güncelleme48'den36,41'e indi: %24,14 daha az güncelleme. Gözlenen bölüm süresi0,517s'den0,450s'ye indi. Süreler eşzamanlı diğer yerel işlerin olduğu ortamdan gelir; yalıtılmış performans veya enerji kazancı değildir. Sürekli MSE biraz arttığı için “bütün kalite ölçüleri aynen korundu” denemez.

Doğrulayıcı bazı ara adayları kabul ediyor; sekizinci turdaki aday aynı denetimi geçmeyebiliyor. Bu yüzden daha fazla tur her kalite/kapsam ölçüsünü tekdüze iyileştirmiyor. Yeni bağımlılık derinliğinde durma neredeyse hiç iş tasarrufu sağlamadı.

## Özet ve ham geçmiş aynı işlevi gördü mü?

Güncel kısıt özetinden yeniden başlatmak, eğitim benzeri ailede bütün ham geçmişi yeniden oynatmaya yakın aday doğruluğu verdi. Ancak derinlik8'de%37,63'ten%20,57'ye düşürdü: −17,06 yüzde puan; her üç tohumda kötüleşme var. Betimleyici eşli aralık −31,25 ile−5,07 puan. MSE de arttı.

Bu politika aktif kısıtları korur; sinirsel geçmiş durumunun eşdeğerliğini garanti etmez. Olay sırası, yeniden oynatma ve ilk görünüşe göre değişken yuvalaması farklılaşabilir. Sonuçtan yalnız bilgi kaybının tek sebep olduğu çıkarılamaz. Özet koşulunda güncelleme işinin düşmesi, daha az geçmiş tekrarının da sonucudur.

Buradaki kapasite4, kaynaktan gelen aktif kayıt kimliklerinin ortak sınırıdır. Gerçek başlık kimliği ham geçmişin ve özet sınırının hangi başlığa ait olduğunu belirler. Bütün model belleği, encoder önbelleği ve geçici yeniden kurulum alanı bu sınırın dışında kalır. Ham kayıtlar ve üretilen izler silinmez. Sinirsel güncelleme geçmişinin son durumdan gerçekten okunarak devam etmesi ayrıca `integration/reasoning-companion` deneyinde ölçülür; bu iki sıkıştırma aynı iddia değildir.

## Dil aktarımı ve doğrulama sınırı

Görülmüş cümle kalıplarında kaynak olayları tam çıkarıldı. Ayrılmış yeni cümle düzenleri ise bölüm içinde ayrıştırma reddi üretti; sonraki sorular karantinaya alındı ve son adayların tamamı eksik kaldı. Eksik cevaplar doğruluk paydasında tutuldu, MSE'ye sıfır yazılmadı. Bu sonuç serbest doğal dil başarısı değildir.

`VERIFIED_PARSED_CONSTRAINTS` yalnız çıkartılmış denklemleri denetler. Kaynak metne sadakati kendi başına kanıtlamaz. Bu deneyde kaynak karşılaştırması değerlendirme tarafında ayrıca yapıldı. Ana sistemin geçmiş erişimi görev/başlık/sürüm anahtarı kullanır; başarısız sözcük-vektörü erişimi sanki başarılı olmuş gibi sunulmaz.

## Ek tur eğrisi

Sabit tam geçmiş koşulunun gerçek izlerinden son sorudaki k=1..8 adayları yeniden ölçüldü. Önceki iki sorunun turları8,8 olarak sabittir; bu bütün sistemin farklı k bütçelerinde yeniden çalıştırılması değildir.

| Son soruda tur | Eğitim benzeri aday doğruluğu | Derinlik8 aday doğruluğu |
|---|---:|---:|
|1 | %97,66 | %29,56 |
|2 | %97,79 | %29,82 |
|3 | %97,66 | %38,28 |
|4 | %97,92 | %35,29 |
|5 | %98,18 | %33,59 |
|6 | %98,31 | %34,77 |
|7 | %98,31 | %36,46 |
|8 | %98,18 | %37,63 |

Bu eğri daha çok hesaplamanın bazı noktalarda faydalı, bazılarında zararlı olabildiğini gösterir. Üçüncü turu test sonucuna bakarak yeni bir çalışma ayarı olarak seçmedik.

Makinece okunabilir tam tablo, eşlenik karşılaştırmalar ve maliyetler: `integration/connected/summary.json`. İz eğrileri: `audit/connected-independent-20260916T175138-depth-curves.json`.
