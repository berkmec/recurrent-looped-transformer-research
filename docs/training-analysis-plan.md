# R2–R5 analiz sözleşmesi — ayrılmış sonuçlar açılmadan önce

Bu plan 48 eğitim koşulunun final testleri açılmadan önce kayda alınır. Kaynak
ve eğitim protokolü ayrıca dondurulur. İstatistiksel aralıklar keşif ve etki
büyüklüğü belirsizliği içindir; çok sayıda son nokta arasından olumlu sonuç
seçilerek doğrulanmış genel üstünlük ilan edilmez.

Her eğitim tohumunda karşılaştırılan kollar aynı giriş ve hedef örneklerini
kullanmalıdır. Karşılaştırma öncesi ham giriş/kimlik/hedef eşitliği denetlenir.
Koşulun seçilmiş ağırlıkları yalnız doğrulama verisiyle belirlenir. Son testte
değerlendirilen varsayılan ağırlık seçilmiş ağırlıktır; step64 eşit etiket
karşılaştırması ayrı bir sabit ara noktadır.

* **R2 birincil:** rastgele ve bloklu nötr sıralamayla eğitilmiş full RLT'nin
  128 nötr son-ek sonrasında son-durum doğruluk farkı; parity ve five ayrı.
  32/128/1024 uzunlukları, 0/8/32/128 nötr ekleri, ön/orta/son yerleşimleri
  ikincil ayrıştırmadır. Nötr ek öncesi doğru olanın yanlışa dönüşmesi ve ters
  dönüşüm ayrıca sayılır. Bloklu eğitim nötr sayısını artırmadığından uzun
  128-ek koşuluna maruz kalma iddiası kurulmaz.
* **R3 birincil:** parity L128'de aynı güncelleme/girdi bütçesinde dense/sparse/
  final denetimlerinin full ve nofeedback kolları; her denetimde full-minus-
  nofeedback farkı. Sparse ve final birer hedef taşır; yoğunluk ile hedefin
  konumu ayrıştırılır. Dense64 ile sparse2048/final2048 karşılaştırması eşit
  65.536 etiket taşır, fakat güncelleme/girdi/hesap eşit değildir. Bu eşit
  etiket karşılaştırması nedensel bir 'yalnız yoğunluk' sonucu sayılmaz.
* **R4 birincil:** five L128'de full-minus-TBPTT8 farkı, dense ve final ayrı.
  Küresel encoder ve mk/mv yolları iki kolda da açık kalır. Sonuç bütün gradyan
  yollarını sekiz adımda kesen bir model veya bütün görevler için genellenmez.
* **R5 birincil:** yalnız geliştirme doğrulamasıyla seçilmiş öğrenme oranında
  taze üç büyük modelin L32 ve L1024 doğruluğu, sınıf histogramları ve nötr ek
  kırılmaları. Aynı yüzde veya kötü doğruluk tek başına sabit sınıf çökmesi
  sayılmaz. Küçük modellerin farklı tohumlarıyla eşlenmiş genişlik üstünlüğü
  testi yapılmış gibi raporlanmaz.

R2–R4 için her tohumun doğruluk farkı yüzde puanında raporlanır. Üç tohumun
ortalaması ve yaklaşık %95 t aralığı (serbestlik2; katsayı4.302652729) verilir.
Kaynak problemleri aynı tohum/kollar içinde eşlenmiş bootstrap ile ayrıca
incelenebilir. Tohumlar veya aynı problemin nötr varyasyonları bağımsız örnek
gibi birleştirilmez. Aralık sıfırı içerirse yön belirsizliği açık yazılır;
önceden güç/MDE hesabı olmadan UNDERPOWERED etiketi kullanılmaz.

Kayıp/nonfinite/OOM/kesinti koşulları başarısız veya eksik paydadan çıkarılıp
başarılı örneklerle tamlık ilan edilemez. Gerçek duvar süresi süreç/çağrı
ölçümlerinden alınır; yalnız adım içi zamanlar günlük yazma ve başlatma
maliyetlerini kapsamaz. Eğitim adımı, etiket sayısı, aktif hesap ve model
parametre sayısı ayrı büyüklüklerdir.
