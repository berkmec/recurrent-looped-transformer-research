# RLT araştırma ajanına devir — uygulanacak işlemler

## Girdi ve sınır

Bu dizin bağımsız referans pakettir. Kullanıcının `drift/run.py`, `horizon/sweep.py`, `semantic_memory/retrieval.py`, `language/run_d1.py` dosyaları ve checkpointleri bu teslimde yok. Buradaki arayüzleri görmeden bu dosyalara otomatik string yaması uygulama. Önce yerel kaynakları gerçekten oku. Kaynak hash’lerini ve mevcut 64-maddelik durumu koru; eski kampanyayı değiştirme.

## 1. Tam durum envanteri

Mevcut RLT’nin z, tüm decoder SWA KV, encoder global memory, mask/position, sampler RNG ve varsa learned initial-state bileşenlerini çıkar. Her biri için: kim sahip, kim güncelliyor, gradyan gerekiyor mu, hangi checkpointte saklanıyor, ağırlık değişince yeniden kuruluyor mu?

Başlık düzeltmesinin aynı tokenı iki kez cache’e yazmadığını açık assertion ile sınama kapsamına al. z’nin korunmasını tam durumun korunması sanma. Referans pakette bu hatayı gösteren negatif test vardır.

## 2. A5 önce hesaplama grafiği düzeyinde

Aynı küçük model ve aynı terminal loss ile dört durum üret: full BPTT, mevcut TBPTT8, saf segment remat, yardımcı-etiketli kol. Başlangıç state parametresinin ve ilk input/encoder temsillerinin gradyanını kaydet. Sayısal parite toleransını dtype için önceden dondur. Shared parametreye son segmentten gradyan gelmesi, ilk state’e kredi geldiğini göstermez.

Orijinal step’i `recurrent_scan` benzeri saf fonksiyona uyarla; mutable cache ve eksik encoder bağımlılığı bulunuyorsa önce düzelt. Kayıp ve gradyan eşliği geçmeden uzun koşuya geçme. Bu test eşiği görevi daraltmaz, yanlış yürütmeyi önler.

## 3. A1 için üç farklı karar kanalı

A: eski RLT; B: nötr işlem etiketi verilen mekanik HOLD referansı; C: HOLD/WRITE kararını modelin verdiği yeni çekirdek. B’nin başarısını C’nin öğrenmesi olarak yazma. Girişteki gated merge yeterli değildir: decoder sonrasında korunan kayda yazma sözleşmesini denetle.

Token saati, semantik olay sayacı ve başlık düzeltme turu ayrı tutulabilir. Bu değişiklik yeni mimari koludur. Nötr ekten sonra gelecek gerçek işlem başarısı mutlaka test edilir; yalnız kayıt normu veya sabit final sınıf yetmez. Kalıcı kayıt, çalışma state’i, KV ve çıktı yoluna ayrı müdahale ederek drift’in hangi yoldan geri geldiğini bul.

Öğrenilmiş kapının yanlış HOLD (gerekli güncellemeyi kaçırma) ve yanlış WRITE (korunması gerekeni bozma) oranları ayrı ölçülür. Sert kapının eğitimi için oracle etiketli/supervised, yumuşak kapılı ve straight-through benzeri yaklaşık eğitim kolları ayrı tutulur; yaklaşık estimator tam gradyan diye raporlanmaz.

## 4. Canlı başlık devamı

Kompakt metin canlı recurrent state’in yerine otomatik konmasın. Raw sayfa önce kalıcılaşıp hash doğrulandıktan sonra aktif metni sıkıştır. Aynı başlık altında yeni sayfa aç. Model ağırlıkları değişince eski latent snapshot’ın geçerli olmadığını bildir ve güncel modelle ham izden yeniden oluştur.

Graf üzerinde her cevabın okuduğu kaynak sürümlerini kaydet. Değişiklikte etkilenen kapanımı bul. Seçici yeniden hesaplamayı full recompute ile eşleştir; daha hızlı ama farklı cevap veriyorsa optimizasyon geçmez. SQLite test kodu basit referanstır; tam model scheduler’ı değildir.

## 5. Retrieval/IR geliştirmesi

Aynı MuSiQue geliştirme pininde query→kanıt→köprü-varlık→yeni-query döngüsü kur. Alternatif dalları silme; aktif sırayı değiştir. Final recall@10 gerçekten son on kayıt üzerinden hesaplansın. Dense/hybrid/rerank negatif kollar tabloda kalsın. Kaynak sınırı, birim, olumsuzluk ve koşul alanlı IR üret; geçerli JSON ile anlam doğruluğunu karıştırma.

Graphica/Symbolica’yı yalnız IR ve varsayım-sürüm sözleşmesi olan alt işlemlere bağla. Çevrilemeyen genel düşünceyi reddetme; o parça doğal/latent başlık yolunda kalır. Motorsuz/basit motor/özel motor ve oracle/öğrenilmiş IR kolları ayrı.

## 6. Sistem optimizasyonu

Mevcut decoder’ı aynen paralelleştirdim iddiası kullanma. Aynı-semantik batch/fusion/cache optimizasyonları, RLT-2 chunk kolu ve yardımcı affine scan yeni mimari kolu ayrı dursun. GPU yoksa CUDA sonucu UNMEASURED kalır. CPU’da 65k HOLD testi başarısını 65k eğitilmiş RLT genellemesi diye sunma.

Reranker’da yalnız gerekli konum logits’i; padding ve çok-token skor tanımı için parite testleri. Her optimizasyonun warmup/compile, indirme/indeks hazırlama, transfer ve toplam tekrar kullanma maliyeti dahil olsun.

## 7. Tam kapsamlı kampanya

3000 problem, bütün kollar/bütçeler/tohumlar, 16k/64k, k/d ve uzun yığın eğitimini manifestten silme. İşleri yeniden başlatılabilir transactionlara böl. Fizibilite için 30M gerçek RLT pilotunda 100 güncelleme ölç; 300M hedefini aynı kalemlerle projekte et. Kiralık donanım varsayımı ekleme.

Test ailelerini ve seed uzaylarını yeni pinlerle dondur; yeni düzeltmeleri eski test performansına göre seçme. Eksik fixture, MISSING_ASSET; öldürülmüş koşul, INCOMPLETE. Başarısızlığı atanıp yeni tohum seçerek gizleme.

## 8. Çıkış sözleşmesi

Her bulgu: kaynak commit, config, veri hash, ağırlık hash, ortam, bütün örnek sayısı, bütün tohumlar, negatif sonuçlar, maliyet ve karar. Model tarafından yazılan metin makine makbuzunun yerine geçmez. Aynı başarıda süre düşmediyse "hızlandı" deme. Arşivde kayıt var diye "model öğrendi" deme. Model/motor kapalı aktarım ayrı testtir.

Bu paket için doğrulanan durum: 46 referans testi, dört mekanizma probu ve 64/64 çözüm eşleştirmesi. Asıl projenin açık maddelerini ancak yukarıdaki gerçek entegrasyon testlerinden sonra güncelle.
