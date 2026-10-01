# R1: sorguya bağlı artık sertifikası — matematiksel sözleşme

Bu sözleşme yalnızca son sürümü belirli, sayısal afin kısıtlar içindir. Kaynak
metnin gerçek dünyayı doğru tarif ettiği veya serbest doğal dilin doğru
ayrıştırıldığı burada kanıtlanmaz. Modelin cevabı hiçbir aşamada çözücünün
cevabıyla değiştirilmez.

Her etkin kenar `x_child - x_parent = d`, her çapa `x_a = b` biçimindedir.
Olaylar kronolojiktir; aynı mutlak/kenar başlık yüzeyinin son yazımı etkindir.
Bütün etkin bileşenlerin çevrim ve çapa tutarlılığı tam aritmetikle denetlenir.
Sorgu bileşeninde en az bir çapa bulunmalıdır. Bağlantısız, tutarlı fakat
çapasız bir başka bileşen sorguyu belirsiz yapmaz. Geçersiz türler, desteklenmeyen
olaylar, sonlu olmayan sayılar ve çelişkili kaynaklar kabul edilmez.

Model düğüm adayları `y_v` olsun. Bir çapanın artığı `r_a = |y_a - b|`, bir
kenarın artığı `r_uv = |y_v - y_u - d_uv|` olsun. Her çapa-sorgu yolu için:

`|y_q - x_q| <= r_a + sum_path r_uv = B(path)`.

Bu eşitsizlik, yol boyunca farkların teleskopik toplamı ve üçgen eşitsizliğidir.
Artıklar negatif olmadığından çok kaynaklı Dijkstra, kullanılabilir aday
düğümleri üzerindeki en küçük **yol toplamını** bulur. Bu, bütün olası
doğrulama yöntemleri arasında en sıkı sınır iddiası değildir. Bir yolda eksik
aday varsa o yol kullanılamaz; başka geçerli yol seçilebilir. Kullanılmayan
bir dalda aday eksik olması kabulü engellemez. Sağlanan her aday sayı sonlu
olmalıdır; bilinmeyen düğüm adları reddedilir.

Sayısal sözleşme: Python int değerleri tam tamsayı, float değerleri temsil
ettikleri tam IEEE rasyoneli olarak yorumlanır. Girdi değerleri ve artıklar
`Fraction` ile işlenir. Eşik, kullanıcıya görünen ondalık yazımının rasyonelidir;
örneğin `.1`, tam `1/10` eşiğidir. Kabul karşılaştırması rasyonellerle yapılır.
Rapor için kayan noktalı sınır ayrıca verilir; kabul kararında kullanılmaz.
Kayan noktalı gösterim taşarsa bu alan `null`, taşma işareti `true` olur;
tam rasyonel sertifika ve karar korunur.
Bu nedenle ek, kanıtlanmamış bir kayan nokta güvenlik payına gerek yoktur.

Sertifika çapa, düğüm yolu, kullanılan etkin yüzeyler, her tam artık ve toplam
sınırı içerir. Adayın sorgu değeri aynen döner. Ayrı sertifika denetimi, bu
yolu ve toplamı yeniden hesaplar; kısa yol algoritmasına güvenmez.

Sürüm/kapsam sözleşmesi: aday zarfı beklenen görev kapsamını, sorguyu ve mevcut
kaynak olaylarının SHA256 özetini taşır. Zarf bu üçüyle eşleşmeden sayısal
sertifika kullanılmaz. Bu bir kimlik doğrulama veya güvenilmeyen yazarın
doğruluğunu kanıtlama mekanizması değildir; yanlış görevden veya eski olay
sürümünden gelen kaydın tekrar kullanımını engelleyen yerel bağlam denetimidir.

Deneyde aynı adaylar iki doğrulayıcıya verilir. Sonlu testler bu matematiksel
sözleşmenin uygulamasını sınar; genel dilde doğruluk veya sınırsız doğru düşünme
sonucuna dönüştürülmez. Doğruluk eşiği sabit `.1`, kabul eşikleri önceden
belirlenmiş ve en fazla `.099999` olacaktır.
