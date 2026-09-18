# RLT — kapsamı koruyan çözüm ve geliştirme dosyası
## 64 madde / A1–G7 / 18 Eylül 2026

**Teslim türü:** kaynak-temelli mimari geliştirme + bağımsız çalıştırılabilir mekanizmalar + test raporu.
**Asıl projeye uygulama:** yapılmadı; özel RLT kaynak kodu, checkpointleri ve ham kampanya günlükleri bu oturumda verilmedi.
**Bilimsel durum:** asıl RLT’de kalite, hız, bellek ve eğitim kazanımı henüz ölçülmedi. Aşağıdaki mekanizma testlerini bu kazanımların yerine koymuyoruz.

### Temel sözleşme

3000 problem, mevcut kollar/bütçeler, 16k/64k uzunlukları, başlık derinliği ve düzeltme turları, Türkçe ve genel dil hedefi, 30M→300M dil eğitimi, kalıcı ham kayıt ve sonraki eğitimde düşünme yöntemini kullanma hedefleri korunur. Kiralık GPU varsayımı eklenmez. Başlık penceresi bağlam penceresiyle birleştirilmez. Erken compaction yok; kullanılabilir başlık sayfası dolduğunda ham veri önce arşivlenir. Sonraki sayfa aynı başlık altında açılabilir. İş planı parçalara bölünür; görev kapsamı veya kanıt eşiği küçültülmez.

Mekanik denetimler düşünmeyi kısaltmak için değil; aynı uzunlukta düşüncenin eski doğru kaydı bozmamasını, tekrar kullanılabilir işi yeniden üretmemesini ve eğitim sinyalinin yolda kesilmemesini sağlamak içindir. Çalışma dilimi dolduğunda bir işi beklemeye almak, çözülmüş saymak değildir. Çok uzun çalışma talebi hiçbir zaman "kısa örneklemde tamamlandı" diye sunulmaz.

### Kaynak ayrımı

`source/problem_havuzu.md` kullanıcı dosyasının değişmemiş kopyasıdır. "MEVCUT" bulgular bu dosyanın raporlarıdır; ham deneylerden burada yeniden hesaplanmış sonuçlar değildir. Dosyadaki sorun/karar numaraları değişmeden korunur. Yeni öneriler aşağıda açıkça öneridir. Dış teknik kaynaklar [S1]–[S12] işaretlidir. Kendi sayısal mekanizma deneylerimiz `results/mechanisms.json` ve testler `results/tests.json` içinde bulunur.

## 1. En önemli tasarım düzeltmesi: koruma ve düşünme farklı işlemler

Mevcut bulguya göre boş koşu bile kurulmuş durumu tek sınıfa sürükleyebiliyor. Bu nedenle sadece "daha iyi normalize et" veya "her şeyi bir çapaya yaklaştır" yeterli bir hedef değil. İki farklı doğru kayıt düşün: birinde durum A, diğerinde durum B olsun. İkisini de aynı noktaya yaklaştıran bir kural sonunda aralarındaki farkı siler. Bellek korumak için bütün uzayın tek çekiciye daralmasını istememeliyiz.

Önerilen çekirdekte üç katman vardır:

**Korunan kayıt:** kabul edilmiş ara sonuçlar, açık varsayımlar ve bunların kaynak/sürüm bağları. Uygun değişiklik gelmeyince aynen taşınır.

**Değişebilir çalışma:** aday düşünceler, karşılıklı başlık mesajları, hata arama ve alternatifler. Yeni token üretmeden de iç tur yapabilir. Korunan kaydı otomatik olarak her tur yeniden yazmaz.

**Token yürütme durumu:** RLT encoder belleği, katman KV’leri, konumlar ve recurrent akış. Token her defa tam bir kez tüketilir. Başlık düzeltmesi ikinci kez token-cache ekleme işlemi değildir.

Bu bir "düşünme yasağı" değil. Model alternatif üretebilir ve önceki kararı düzeltebilir; yalnız geçici aday ile geçerli kayıt aynı alan değildir. Yanlış olduğu ortaya çıkan eski sonuç silinmez: yeni sürüm oluşur, eski sürüme bağlı sonuçlar STALE olur. Doğrulayıcı yalnız kaynak koşullarına göre geçerli bir sonuç kurabiliyorsa VERIFIED; genel görevlerde doğrulama yoksa CANDIDATE/CHALLENGED kalır.

Tam kopya dalında hiçbir sayısal güncelleme yapılmadığında o kayıt değerinin korunması ağırlıkların eğitilmesine bağlı değildir. Ancak hangi olayın nötr olduğunu öğrenmek ayrı bir problemdir. Deneyde nötr maskeyi oracle’dan vermek, modelin nötrlüğü öğrendiğini göstermez. Ayrıca yalnız z korunurken cache veya readout değişirse sonuç yine kayabilir. Bu karşı-örnek bağımsız test paketinde özellikle yer alır.

## 2. İkinci düzeltme: eğitimde kesilen yolu restore etmek

A5 dosyasında TBPTT8-final koşulunda başlangıç parametresinin güncellenmediği belirtiliyor. Geri yayılım yolu sekizinci adımda gerçekten kesiliyorsa, son etiketin daha eski başlangıç kaydını düzeltmemesi beklenen bir yazılım sonucudur. Ortak parametreler son segmentten hâlâ gradyan alabilir; bu, ilk gizli duruma kredi ulaştığı anlamına gelmez.

Öneri: küçük referansta tam BPTT; üretim adayında segmentli yeniden hesaplama ama sınırda detach olmaması. Belleğe her ara aktivasyonu kalıcı tutmak yerine gereken segment aynı ağırlık, girdi ve rastgelelik durumuyla yeniden yürütülür. Bütün differentiable recurrent durumlar ve encoder bağımlılıkları sözleşmeye katılır [S2]. Optimizer segmentlerin arasında ağırlık değiştirmez.

Burada üç ayrı iddia vardır: 1) gradyan yolu bağlıdır, 2) gradyan sayısal olarak kararlıdır, 3) model uzak sonuca neden olan davranışı gerçekten öğrenir. Bu paketin deneyi birinciyi ve referans gradyan eşliğini sınar; ikincisi ve üçüncüsü asıl görevlerde ölçülmelidir. Yardımcı etiketler ve doğrulanmış ara işlemler değerlidir ama son-etiketli deneyin yerine sessizce konmaz.

## 3. Başlık belleği: canlı devamı özetin insafına bırakmamak

Başlık düğümü; amaç, açık koşullar, aday işlemler, kaynak parçaları, komşu sürümleri, yürütme snapshot referansı, ham kayıt ve kompakt gösterim taşır. Bir sayfa dolduğu için yeni konu açılmaz. Aynı başlığın sayfa numarası artar; kimliği aynı kalır.

B6’daki yalnız son z ile devamın özetten devamdan daha iyi olması, metinsel özetin henüz yeterli bir yürütme temsili olmadığını düşündürüyor. Önerilen akış bu eksikliği inkâr etmek yerine üç temsili ayırır: insan-okunur compact, yürütülebilir/türlendirilmiş ara kayıt ve mevcut modelin canlı tam durumu. Compact üretmek canlı durumu sıfırlamaz. Çok uzun genel bağlam için CPU/NVMe paging veya yeni bellek mimarisi ayrıca tasarlanır; başlık compact’ı global KV’yi kendiliğinden küçültmez.

Vektör indeksi ham düşüncenin kendisi değildir. Ham kayıt baytları ve gerektiğinde model-sürümüne bağlı latent snapshot kayıpsız depoda saklanır. Embedding ve graph index yeniden üretilebilir yardımcı adreslerdir. Yazılmış işlem izinin modelin bütün gizli düşünce mekanizmasını eksiksiz temsil ettiği varsayılmaz. Sonraki eğitim için problem seçimi, kaynak kullanımı, revizyon, test sonucu ve başarısız rota birlikte tutulur.

Karşılıklı düzeltmede A ve B aynı eski snapshot üzerinden öneriler üretir. A’nın yeni sürümü B’nin dayandığı kaydı geçersiz kılmışsa B eski sonuçla commit edemez. Grafın ters bağımlılık kapanımı yeniden hesaplanır. Döngüdeki düğümlerin birlikte sürümlenmesi tasarım hedefidir; bu teslimde yalnız bağımlılık kapanımı/freshness mekanizması kodlanmıştır, tam paralel transaction kontrolörü değil.

## 4. Anlamsal erişim ve ayrıştırma: motorun öncesindeki sorunu çözmek

B2/B3’teki düşük skorları sadece daha güçlü reranker veya daha fazla graph işlemiyle kapatmayı varsaymıyoruz. İlk kanıt parçasından ikinci arama için varlık/ilişki çıkaran çok-atlamalı erişim öneriyoruz. Bir dal yanlışsa alternatif kanıt cephesi korunur. Son top10 yine on parçadır; içerde yüz parça görmüş olmak, recall@10 hesabında yüz parça saymak değildir. IRCoT bu tür erişim–araçıkarım döngüsü için dış araştırma dayanağıdır [S4]; bizim hedeflerin deneysel çözümü değildir.

Ayrıştırıcı yalnız geçerli JSON yazmamalı. Her iddiayı kaynak aralığına, işlem türüne, sayı/birime, olumsuzluğa ve varsayımlara bağlamalı. Birden çok olası okuma CANDIDATE olarak kalabilir; yapısal motor yalnız uygun alt parçaları işler. Böylece doğal düşünce yeteneği sembolik gösterime sığan görevlerle sınırlandırılmaz.

Symbolica/Graphica’nın ana hatta park kararı korunur. Uygun IR hazır olduğunda Graphica yapısal eşleme/kanonikleştirme, Symbolica cebirsel hesap, basit deterministik motor referans sağlar [S6–S7]. Önbellek anahtarı sadece başlık adı değil; girdi, kaynak, varsayım, model/indeks/rule-engine sürümleridir. Soğuk hazırlama maliyeti dahil net yarar ölçülür.

## 5. Hız: düşünceyi kısaltmadan bağımlılık ve yürütme düzenini değiştirmek

Aynı nonlinear token bağımlılığını sadece farklı isim vererek paralel yapamayız. Önerilen yollar birbirinden ayrı tutulur: bağımsız dizileri/başlık işlerini toplu yürütmek; kernel ve host overhead’i düşürmek; bağımsız taşıma alt yolunu associative scan ile yürütmek; chunk-recurrent yeni mimariyi ayrı karşılaştırmak.

Resmi RLT sayfası 17 Eylül’de RLT-2 chunk feedback önerisini yayımlamış. Chunk içinde feedback sabit, sınırda güncelleniyor; boyut bir olduğunda RLT-1’e dönüyor. Aynı sayfa henüz RLT-2 eğitim veya throughput sonucu olmadığını söylüyor [S1]. Dolayısıyla eski "hiç alternatif paralelleştirme yolu yok" anlatımı güncellenmeli, ama RLT-1’e eşdeğer hızlanma varmış gibi de konuşulmamalı. Yardımcı scan yolu için paralelleştirilebilir recurrent mimariler dış dayanak sağlar [S5]; nonlinear düşünce çekirdeği yerinde kalır.

Paket içindeki scan, aynı sırayla "ölçekle ve ekle" türü taşıma işlemlerini yeniden gruplar. Önceki gizli duruma bağımlı kapıları olan genel RLT’nin yerine geçmez. Float64 parite ölçümü vardır; GPU hız sonucu yoktur. Referans protected_update fonksiyonu debug doğruluk kontrolleri nedeniyle CPU senkronizasyonu yapabilir; üretim GPU kernel’i olarak sunulmaz.

8 GB VRAM için memory ledger; weights, gradients, optimizer, activations, encoder KV, decoder KV, logits, allocator/compile buffer ve CPU arşivini ayrı sayar. Mini-batch/accumulation ve offload, işin toplam mantığını küçültmeden yerleşimi değiştirir. Kontrol edilmeyen OOM veya host swap’ı hız kazancı diye raporlanmaz.

## 6. Deney ve operasyon: kapsamı değil tekrar yapılan işi azaltmak

Bütün problem×kol×bütçe×tohum işleri baştan manifestte yer alır. Bir işi parçalara bölmek kapsam azaltmak değildir. Başarılı transaction tekrar çalıştırılmaz; kesik transaction doğrulanmış eğitim/generation sınırından yeniden yürütülür. Tam çalışma bitmeden final durum COMPLETE olmaz.

Fail-closed korunur. Eğitim checkpointi sırf hidden state değildir; optimizer, RNG, sampler/cursor, accumulation, model ve cache durumlarını birlikte taşır. Otograd grafiğini serialize etmeden ara yerden devam etmek yeni gizli gradyan kesintisi doğurabilir; güvenli optimizer-step sınırından replay gerekir.

İddialar üç düzeyde etiketlenir: mekanizma sözleşmesi, ölçülmüş görev davranışı, yeni dağılıma aktarım. Arşiv/makbuz testlerinin geçmesi ikinci ve üçüncü düzeylere kanıt değildir. Yeni test uzayları, tüm tohumlar, negatif sonuçlar, nicemleme koşulu ve Türkçe/çeviri ayrımı görünürdür. Sonuç için doğru cevap + kapsam + yanlış kabul + toplam maliyet birlikte izlenir.

## 7. İş paketleri ve bağımlılıkları

| Paket | Çıktı | Önce gerekli olan | Asıl başarı ölçümü |
|---|---|---|---|
| P1 | Korunan anlam / çalışma ayrımı ve dinamik bellek | Tam durum tanımı, U2 ölçümü | Bilgi taşıyan nötr koşu + sonraki gerçek işlem |
| P2 | Kesintisiz eğitim yolu ve segment remat | Saf step, tüm differentiable durum | Tam BPTT paritesi, sonra terminal etiketten öğrenme |
| P3 | Başlık sürümleme, ham arşiv, devam ve revizyon | Kaynak/sürüm sözleşmesi | Canlı devam, tam vs seçici yeniden hesaplama paritesi |
| P4 | Çok-atlamalı erişim, typed IR, aday düzeltme | Kaynak/provenance ve veri ayrımı | Recall/kapsam/risk, yeni ifade ve derinlik |
| P5 | Profiling, kernel/scan/chunk/offload yolları | Sayısal referans ve P6 | Aynı başarıda toplam süre/enerji/bellek |
| P6 | İş defteri, makbuz, kesinti, yedek ve tek test girişi | Sabit veri/model/kod hash’leri | Restore/tekrar üretim ve tüm hücreler |
| P7 | Tüm ölçüm matrisi ve iddia kalibrasyonu | P6, her paketin ayrı kolu | Taze test, tüm tohumlar, hata ve maliyet birlikte |

Bu sırada bir paketin başarısız olması diğerlerini kavramsal olarak kapatmaz. Örneğin P6 hemen üretilebilir; P1 ve P2 mekanizmaları birbirinden ayrıştırılarak ölçülür; P4’e başlanması uzun çekirdek eğitiminin bitmesine bağlı değildir. Ancak bunların hepsi birlikte eklenip tek başarılı skor gösterilmez: katkı ve regresyonu ayıran kollar korunur.

## 8. Madde madde 64 problem ve önerilen çözüm

Aşağıdaki her "kapanış testi" gelecek proje doğrulamasıdır. Bir önerinin yazılması o problemin kapandığını göstermez. `problem_mapping.json` aynı eşleştirmenin makine-okunur tam sürümüdür.

### A1. Kurulmuş durum boş koşuda kayıyor (sürüklenme) — MEVCUT, en büyük.

**Çözüm / geliştirme:** Korunan anlam durumu ile değişebilir hesaplama alanını ayır. Token okuma ve iç düzeltme sürerken, doğrulanmış bir değişiklik yoksa korunan alana sayısal güncelleme uygulama; tam kopya geçişi kullan. Kalıcı durumdan okuyan çıktı yolu da bu sözleşmeye uymalı. KV, encoder belleği ve konum üzerinden dolaylı yeniden bozulmayı ayrıca ölç.

**Kapanış testi:** Nötr ön/orta/son eklerde 128, 512, 4096, 16384, 65536 adım; parite, beş-durum, permütasyon ve yığın. Bilinen nötr-maskeli mekanizma kolu ile nötrlüğü öğrenen kol ayrı. Sonraki gerçek işlem doğruluğu da korunmalı; yalnız sabit çıktıya bakılmaz.

**Açık kalan / yanlış iddiayı önleme:** Tam kopya, yalnız doğru HOLD kararı için garanti verir. Doğal dilde nötrlüğü tanıma ve doğru sonucu seçme hâlâ öğrenilmelidir. Bir vektörü korumak RLT bütün durumunu korumak değildir.

**Paket:** P1 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A1.

### A2. Üç ucuz çare işe yaramadı — MEVCUT.

**Çözüm / geliştirme:** Başarısız üç müdahaleyi yeniden adlandırıp kullanma. Pencere büyütme ve genel nötr kayıp yerine, değişmemesi gereken bilgi yönleri ile düzeltilebilen hesaplama yönlerini farklı güncelle. Her bilginin tek çekiciye çekilmesi yerine farklı geçerli kayıtlar ayrı korunur.

**Kapanış testi:** Eski üç müdahale dondurulmuş negatif kontroller olarak kalsın. Aynı eğitim/test aileleri, yeni tohumlar ve aynı ek maliyet raporuyla yeni mekanizma karşılaştırılsın.

**Açık kalan / yanlış iddiayı önleme:** Yanlış çapa da kalıcılaşabilir. Çapa geçerliliği içerik/sürüm bağımlılıklarıyla takip edilir; çapa sayısı veya kayıp katsayısı için ölçülmemiş optimum önerilmez.

**Paket:** P1 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A2.

### A3. Zararın görünürlüğü göreve bağlı (permütasyon vs üzerine yazma) — MEVCUT.

**Çözüm / geliştirme:** Onarımın mümkün olduğu üzerine-yazma ve hatanın sonraki işlemlerle görünmez hâle gelmediği permütasyon koşullarını birlikte çalıştır. Yığının yalnız tepesini değil, sonra pop ile açığa çıkan derin hücreleri de sorgula.

**Kapanış testi:** Her müdahale her görev ailesinde ve aynı baş/orta/son yerleşimlerde raporlanır. Doğru son etiket yanında eski bilgiye bağlı karşı-olgusal devam testi gerekir.

**Açık kalan / yanlış iddiayı önleme:** İki ailede başarı bütün algoritmalar için güvence değildir; yeni aileler mevcut hedefleri değiştirmeden eklenir.

**Paket:** P1/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A3.

### A4. Başlangıç durumu kararlı, bilgi taşıyan durum değil — MEVCUT (yorum).

**Çözüm / geliştirme:** Başlangıç noktasına değil, bilgi taşıyan çok sayıda erişilmiş duruma küçük pertürbasyon uygula. Tek-adım özdeğer yanında çok-adımlı büyümeyi ve en fazla genişleyen yönü ölç. Amaç bütün farklı anlamları birbirine yaklaştırmak değil; anlamı korurken ilgisiz sapmayı bastırmak.

**Kapanış testi:** Aynı nötr koşuda durum farkı, çıktı farkı, sınıf yığılması, çok-adım duyarlılık ve sonraki işlem doğruluğunu birlikte kaydet. Etiketli sınıf merkezleri yalnız tanı için kullanılır.

**Açık kalan / yanlış iddiayı önleme:** Bir adımın özdeğerinin küçük olması, değişen veya normal olmayan geçişlerde geçici büyümeyi dışlamaz. Altuzayın öğrenilmesi ayrı araştırmadır.

**Paket:** P1 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A4.

### A5. Uzak kredi ataması yok — MEVCUT.

**Çözüm / geliştirme:** Son etiketli ana kolda TBPTT sınırında gradyan kesme yerine, aynı ağırlıklarla segment yeniden hesaplama ve kesintisiz geri yayılım kullan. Bütün farklılaştırılabilir durum bileşenleri sınırdan geçsin. Yardımcı ara etiket/kanıt kaybını ayrı bir eğitim kolunda tut.

**Kapanış testi:** Kısa dizide tam BPTT ile çıktı, kayıp, giriş, başlangıç durumu, cache ve parametre gradyanları eşleşmeli. Sonra yalnız terminal etiketle gerçek görev başarısı ölçülmeli. Rastgele konumlu tek etiket ayrı kontrol olarak kalır.

**Açık kalan / yanlış iddiayı önleme:** TBPTT8 sonrası ilk duruma ulaşmayan yol, öğrenme başarısızlığından önce hesaplama grafiği kesintisidir. Yolu geri bağlamak gradyanın yararlı olmasını veya gecikmiş ödülle öğrenmeyi garanti etmez.

**Paket:** P2 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A5.

### A6. Büyüyen bellek görevi öğrenilmiyor — MEVCUT (bütçeye koşullu).

**Çözüm / geliştirme:** Sabit genişlikte vektörün yanına büyüyebilen adresli bellek ekle. Yığın için push/pop benzeri işlemlerin hedefini öğrenen denetleyici ve kalıcı hücre kayıtları kullan; eski hücreyi silmek yerine etkin işaretçiyi değiştir. Daha uzun eğitim ve genişlik taraması da korunur.

**Kapanış testi:** Birbirinden ayrı iki kol: doğru işlemleri verilen bellek motoru ve işlemleri modelin seçtiği uçtan uca sistem. Derin pop, farklı sembol kümesi ve 4096/16k/64k uzunluklarında ölçüm.

**Açık kalan / yanlış iddiayı önleme:** Deterministik yığının doğru çalışması modelin yığın algoritmasını öğrendiği anlamına gelmez. Büyüyen kayıtların disk/bellek maliyeti gerçek olarak raporlanır.

**Paket:** P1/P3 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A6.

### A7. Tohum piyangosu — MEVCUT.

**Çözüm / geliştirme:** Tohum, veri örnekleme tohumu ve test üretim tohumu ayrı alanlar olsun. En az beş başlangıç tohumu, tohum başına sonuç ve başarısız koşuların eksiksiz kaydı kullan. Seçilen en iyi tohum ana sonuç yapılmaz.

**Kapanış testi:** Tüm planlanan tohumların hesap verme kaydı, başarı oranı ve dağılımı; aynı test problemleri üzerinde eşlenik karşılaştırma. Eksik/çöken koşular görünür kalır.

**Açık kalan / yanlış iddiayı önleme:** Beş tohum istatistiksel kesinlik sihri değildir. Bağımlı örnekler bağımsızmış gibi çoğaltılmaz.

**Paket:** P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A7.

### A8. Gradyan sıçramaları — MEVCUT.

**Çözüm / geliştirme:** Optimizer adımı öncesinde kayıp, durum, gradyan ve normların sonluluğunu denetle. AMP varsa ölçek geri alındıktan sonra kontrol/kırpma yap. NaN/inf halinde güncellemeyi commit etme; son doğrulanmış eğitim checkpointinden yeni deneme aç.

**Kapanış testi:** Enjekte edilmiş NaN/inf optimizer ağırlığına bulaşmamalı. Ham kırpma-öncesi norm, atlanan adım, yeniden başlatma ve başarısızlık oranı bütün koşullarda raporlanmalı.

**Açık kalan / yanlış iddiayı önleme:** Kırpma nedenin giderildiği anlamına gelmez. LR geri çekme gibi politika değişikliği önceden tanımlanır veya yeni protokol sürümüdür.

**Paket:** P2/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A8.

### A9. Öğrenme oranı duyarlılığı ölçeğe göre değişiyor — MEVCUT.

**Çözüm / geliştirme:** Her genişlikte öğrenme oranı, warmup ve güncelleme ölçeğini validation üzerinde tara; state yazma projeksiyonu ile diğer ağırlıkların normlarını ayrı izle. Eski ölçekteki oranı otomatik taşımama.

**Kapanış testi:** Tarama bütçesi ve adayları dondurulsun; test performansına göre LR seçilmesin. Optimize edilmiş kolların arama maliyeti eşitlik tablosunda yer alsın.

**Açık kalan / yanlış iddiayı önleme:** Mevcut w256 sonucunu evrensel ölçek yasası gibi kullanma. Ortak başlangıç LR kuralı ancak yeni ölçekte doğrulanır.

**Paket:** P2/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A9.

### A10. Geç öğrenme (plato) — MEVCUT.

**Çözüm / geliştirme:** Sabit ana bütçeyi koru; bütün ilgili kollara uygulanacak önceden tanımlı uzatma evreleri ekle. Öğrenmeye-kadar-süreyi ve bütçe sonunda öğrenmemiş/yarım kalmış koşuları ayrı raporla.

**Kapanış testi:** Uzun plato ve ani düşüş örnekleri kayıtlı olmalı. Teste bakmadan belirlenen uzatma ölçütü bütün tohumlara uygulanmalı; erken başarısız tohum atılmamalı.

**Açık kalan / yanlış iddiayı önleme:** Düşen validation kaybı sonunda mutlaka çözeceğinin kanıtı değildir. Uzatma maliyeti gizlenmez.

**Paket:** P2/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / A10.

### B1. Doğrulayıcı kapsamı düşük — MEVCUT.

**Çözüm / geliştirme:** Doğrulayıcıyı gevşetmek yerine aday üretimini hata geri bildirimiyle bağla: eksik koşul, yanlış kaynak ve başarısız işlem ayrı karşı-örnekler üretir. Doğru aday yok / aday var ama ispat üretilemiyor / doğrulayıcı kapsayamıyor durumlarını ayır.

**Kapanış testi:** Doğru-aday-varlığı, adaylar içinden seçme doğruluğu, yanlış kabul ve toplam kapsama ayrı. Depth-8 ve yeni topolojiler aynı sertifika koşulunda yeniden çalıştırılır.

**Açık kalan / yanlış iddiayı önleme:** Çıkarım kuralları doğru olsa bile kaynak varsayımı yanlış olabilir. Koşullu sertifikayı koşulsuz doğruluk diye sunma.

**Paket:** P3/P4 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B1.

### B2. Anlamsal erişim hedefi çok uzak — MEVCUT.

**Çözüm / geliştirme:** Tek sorgu-top10 yerine kanıt cephesi kur: ilk erişimden çıkan varlık/ilişki ve kaynak parçası ikinci sorguyu oluştursun. Birden fazla dalı tut; başarısız dal arşivde kalır. Dense, sözcüksel ve varlık erişimlerini önce aday birleşiminde tut; füzyon ağırlığını validation ile seç. Ara arama genişleyebilir ama final top10 tanımı değişmez.

**Kapanış testi:** Aynı MuSiQue diliminde recall@10, bütün gerekli kanıtları birlikte bulma, atlama başına kaçırma, cevap doğruluğu ve toplam erişim/yeniden-sıralama süresi. Oracle kanıt kolu yalnız hatanın yerini ayırır.

**Açık kalan / yanlış iddiayı önleme:** IRCoT bu yönde bir literatür dayanağıdır [S4]; bizim %95/%90 hedeflerinin sağlandığını göstermez. Fazladan aday maliyeti ve final10 kuralı açık tutulur.

**Paket:** P4 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B2.

### B3. Öğrenilmiş ayrıştırma yeni kalıba aktarılmıyor — MEVCUT (Faz 1).

**Çözüm / geliştirme:** Serbest metinden tek kesin ayrıştırma yerine, kaynak aralığına bağlı türlendirilmiş ara gösterim üret: varlık, işlem, nicelik, birim, koşul, olumsuzluk ve bağımlılık. Belirsizlikte birden fazla aday sakla; doğrulayıcı geri bildirimiyle düzelt. Yeni cümle dizilişi ve yapısal eşanlam dönüşümleri eğitim/validation/test aileleri ayrılarak üretilir.

**Kapanış testi:** Şema geçerliliği ile anlam doğruluğu ayrı ölçülür. Hiç görülmemiş cümle aileleri, yeni derinlik, Türkçe olumsuzluk/birim/sayı değişimleri ve yanlış öncül testleri gerekir.

**Açık kalan / yanlış iddiayı önleme:** Geçerli JSON doğru anlam demek değildir. Programatik/LLM dönüşümlü veri ayrı etiketlenir; insan referansının yerini sessizce almaz.

**Paket:** P4 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B3.

### B4. Seçici revizyon negatif — MEVCUT (Faz 1).

**Çözüm / geliştirme:** Her sonuç tam bağımlılık sürümlerini taşısın. Bir kaynak değişince ters-bağımlılık kapanımını STALE işaretle; bağımsız dalları koru. Döngüdeki düğümler aynı eski snapshot üzerinden aday üretip tutarlı bir sürüm grubu olarak yayınlansın.

**Kapanış testi:** Seçici yeniden hesaplama, tüm grafiği sıfırdan hesaplama referansıyla aynı sonucu vermeli. Değişen/etkilenmeyen cevaplar, ≥%10 kazanç kapısı ve süre birlikte raporlanmalı.

**Açık kalan / yanlış iddiayı önleme:** Eksik bağımlılık çıkarmak yanlış negatif invalidation yaratır. Bağ belirsizse etki alanını genişletmek gerekir; hız garantisi yoktur.

**Paket:** P3 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B4.

### B5. Sözcük-karma erişimi negatif — MEVCUT.

**Çözüm / geliştirme:** Sözcük-karma %4 tabanını kontrol olarak tut. Yeni erişimde gerçek sözcüksel eşleşme, varlık/alias eşleştirmesi ve öğrenilmiş dense adaylarını ayrı ölç. Başlık adı benzerliğini tek başına içerik eşitliği sayma.

**Kapanış testi:** Aynı sorgularda recall@5 ve @10, Türkçe karakter/ek/alias dönüşümleri, dense üzerine ek yarar ve ek maliyet. Başarısız taban sonuç tablosundan kaldırılmaz.

**Açık kalan / yanlış iddiayı önleme:** Hibrit sistem ekleyince otomatik iyileşme olmaz; B2’deki negatif füzyon sonucu geçerliliğini korur.

**Paket:** P4 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B5.

### B6. Sıkıştırma: son z'yi saklamak çalışıyor, anlamsal özet değil — MEVCUT.

**Çözüm / geliştirme:** İnsan-okunur kompakt özet, yürütülebilir ara gösterim ve canlı latent/checkpoint üçlüsünü ayır. Pencere dolunca ham sayfa kalıcılaşır ve aktif metin sıkışır; çalışan model durumu sırf özet üretildi diye özetle değiştirilmez. Aynı ağırlıkla tam devam için z yanında KV/encoder/konum/RNG sözleşmesi de gerekir.

**Kapanış testi:** Sıkıştırmasız devam, aynı ağırlıkta tam snapshot devamı ve yalnız özetten devam farklı kollar. Kritik koşul, çelişki ve eski bilgi geri çağırma testleri; aynı başlık kimliği korunmalı.

**Açık kalan / yanlış iddiayı önleme:** Bu tercih özetin anlamsal yeterlilik sorununu sihirli çözmez; başarısız özetin canlı hesabı bozmasını önler. Genel encoder KV büyümesi ayrı olarak yönetilir.

**Paket:** P3/P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B6.

### B7. Başlık grafiği tek başına yeni derinlikte yetersiz — MEVCUT.

**Çözüm / geliştirme:** Başlık grafiğini etiket listesi değil, sürümlü mesajlaşma sistemi yap. Komşular amaç/koşul/kanıt/öneri iletir; güncelleme eski snapshot üzerinden yapılır. Karşılıklı düzeltme turları, yanlışta sabitlenme ve salınım durumlarını kaydeder; çözülmemiş aday silinmez.

**Kapanış testi:** k düzeltme turu ile d bağımlılık derinliği ayrı eksenler. Aynı sayıda adsız slot ve aynı FLOP’lu tek yön kolu karşısında başlık/çift yön katkısı ayrıştırılır.

**Açık kalan / yanlış iddiayı önleme:** Yerel tutarlılık genel doğruluk değildir. Ara gözetim kullanılan kol, yalnız son etiketli kolla aynı öğrenme iddiasını taşımaz.

**Paket:** P3/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B7.

### B8. Symbolica/Graphica park — KARAR.

**Çözüm / geliştirme:** Park kararını ana sonuç hattında koru; yan tarafta motor arayüzünü geliştir. Typed IR, açık varsayımlar ve kaynak sürümleri hazır olduğunda Graphica’ya yapısal kanonikleştirme, Symbolica’ya geçerli cebirsel alt işlemler gönder. Önce basit deterministik motor aynı işi yapsın.

**Kapanış testi:** 1) Motorsuz, 2) basit kayıt/önbellek, 3) Symbolica/Graphica; hem oracle IR hem öğrenilmiş IR altında. Eşitliği bozan varsayım, birim, değişken kapsamı ve sürüm değişimleri önbellek isabeti vermemeli.

**Açık kalan / yanlış iddiayı önleme:** Graf izomorfizmi doğal dil anlam eşitliği değildir; Symbolica kaynak koşulunu icat etmez. Genel motor lisansı ile Graphica MIT lisansı farklıdır [S6–S7].

**Paket:** P4/P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / B8.

### C1. Test kümesi kampanyalar arası yeniden kullanıldı — MEVCUT (açıklandı).

**Çözüm / geliştirme:** Kampanya, eğitim tohumu, veri üretici sürümü ve problem içeriği için ayrı kimlik/hash tut. Yeni test uzayını kilitle; eski testlerdeki bütün keşif sonuçlarını tarihçesiyle sakla.

**Kapanış testi:** Kampanyalar arasında tohum ve içerik çakışma denetimi. Eski test içeriği final taze test manifestine girememeli.

**Açık kalan / yanlış iddiayı önleme:** Dosya hash’i semantik eşdeğer örnek sızıntısını tek başına yakalamaz; aile düzeyi ayrım da gerekir.

**Paket:** P7/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C1.

### C2. Üç tohum + keşif aralıkları — MEVCUT.

**Çözüm / geliştirme:** Tohum başına eşlenik sonuçları ve problem-aile gruplarını raporla. Önceden belirlenmiş karşılaştırma ailesinde Holm gibi çokluk kontrolü kullan; örnek ve tohum belirsizliğini ayrı göster.

**Kapanış testi:** Tüm p değerleri/karşılaştırma kimlikleri çıktı dosyasında olmalı; düzeltilmiş ve ham sonuçlar birlikte. Kod paketindeki Holm testi yalnız aritmetik sözleşmesini sınar.

**Açık kalan / yanlış iddiayı önleme:** Az sayıda tohumla dar güven aralığı vaat edilmez. Aynı problemdeki çok sayıda çıktı bağımsız örnek değildir.

**Paket:** P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C2.

### C3. Türkçe dilimi yok — MEVCUT.

**Çözüm / geliştirme:** Türkçeyi kaldırma: doğal Türkçe insan-kaynaklı değerlendirme, etiketli çeviri ve programatik türetilmiş stres verisini üç ayrı dilim yap. TurkishMMLU gibi uzman yazımlı kaynak bir okuma/QA dayanağı olabilir [S10]; başlık-revizyon görevleri için ek insan referansı yine gerekir.

**Kapanış testi:** Türkçe diliminde ayrı başarı, erişim, ayrıştırma, olumsuzluk ve birim doğruluğu. Veri erişim/lisans kontrolü; olmayan insan anotasyonu varmış gibi raporlanmaz.

**Açık kalan / yanlış iddiayı önleme:** Bazı kaynakların erişim izni istenir. Mevcut İngilizce skorların Türkçe karşılığı çıkarılamaz.

**Paket:** P4/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C3.

### C4. Model-yazımı veri riski — OLASI.

**Çözüm / geliştirme:** Her veri satırına origin, parent_id, transformation, author/model sürümü ve doğrulama türü ekle. Sentetik/çeviri satırlar ayrı train/test gruplarında izlenir; altın referans etiketine otomatik yükseltilmez.

**Kapanış testi:** Kaynağı eksik satır final sete alınırken hata üretir. Ana sonuç ile sentetik stres sonucu karışmamalı; yakın paraphrase train/test sızıntısı taranmalı.

**Açık kalan / yanlış iddiayı önleme:** Programatik dönüşüm de anlamı bozabilir. Özellikle olumsuzluk ve nicelik değişimi doğrulanır.

**Paket:** P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C4.

### C5. Tek model hakemi — OLASI.

**Çözüm / geliştirme:** Açık uçlu yanıtta hakemlerin uzlaşısını yalnız görüş birliği olarak kaydet. Kod/test/kanıt varsa nesnel ölçütü kullan; olmayan görevlerde insan-kalibrasyon dilimi ve belirsiz sonuç statüsü tut.

**Kapanış testi:** İnsan referansına karşı yanlış kabul/ret, hakemler arası uyuşmazlık ve yanıt sırası etkisi ölçülür. Tek modelin olumlu puanı CERTIFIED üretmez.

**Açık kalan / yanlış iddiayı önleme:** İnsan referansı henüz yoksa o katmanın sertifikasyonu açık kalır; görev kapsamdan çıkarılmaz.

**Paket:** P7/P4 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C5.

### C6. Nicemleme sadakati ölçülemedi — MEVCUT.

**Çözüm / geliştirme:** Aynı modelin nicemlenmemiş referansını katmanlı CPU/GPU yükleme ile ayrı oturumda değerlendir; iki modeli aynı anda belleğe alma. Önce çıktı ve katman bazlı nicemleme duyarlılığı, sonra aynı görev örneklerinde fark. Offload büyük-model inference için var olan bir yöntemdir [S9].

**Kapanış testi:** Aynı ağırlık sürümü, tokenizer, prompt ve örnekleme koşulu; nf4/8-bit/referans skor ve logits farkı. CPU/GPU kernel farkını nicemleme farkı diye adlandırmamak için mümkün olduğunda ortak cihazlı ara test.

**Açık kalan / yanlış iddiayı önleme:** 16 GB RAM’de tüm modelin yükleme tepe maliyeti ölçülmelidir; katman paritesi uçtan uca görev paritesinin yerine geçmez. Bu inference planı training offload garantisi değildir.

**Paket:** P5/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C6.

### C7. "Sabit sınıf çöküşü" gibi anomalilerin kısmi okunması — YAŞANDI.

**Çözüm / geliştirme:** Rapor üretimini bütün kayıtları tarayan makine toplamlarına bağla. Beklenen örnek sayısı, görülen kimlikler, sınıf histogramı ve eksik kayıt listesi bir arada çıksın.

**Kapanış testi:** İlk 16 satırı kalanlardan kasıtlı farklı olan fixture doğru tam histogramı vermeli. Eksik/tekrar kayıtlar COMPLETE sayılamamalı.

**Açık kalan / yanlış iddiayı önleme:** Otomatik toplam yanlış etiket anlamını düzeltmez; veri şemasının yorumu da denetlenir. Paket bu tam-tarama sözleşmesini test ediyor.

**Paket:** P6/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C7.

### C8. Faz 1 test paketi worktree'de 7 testte ham dosya ister — MEVCUT.

**Çözüm / geliştirme:** Testleri birim, varlık-gerektiren entegrasyon ve mühürlenmiş tekrar-üretim olarak ayır. Her dış varlık hash ve kaynak yoluyla manifestte tanımlansın. Eksik checkpoint MISSING_ASSET olarak raporlansın.

**Kapanış testi:** Yeni kökte birim paketi tek komutla; gerçek entegrasyon için gerekli ağırlık yoksa genel proje başarısı yeşil gösterilmesin. Mühürlü kökteki 108/108 tarihsel kayıt değişmeden kalsın.

**Açık kalan / yanlış iddiayı önleme:** Beklenen başarısızlık testin çalıştığı anlamına gelmez. Bu teslim mevcut extractor checkpointini içermiyor.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C8.

### C9. Çoklu son nokta / seçici raporlama — OLASI.

**Çözüm / geliştirme:** 71 karşılaştırmanın tam kimlik ve yönlerini preregistration dosyasında koru; olumlu olmayanlar dahil tek sonuç matrisi üret. Yeni hipotezler yeni keşif etiketiyle eklenir.

**Kapanış testi:** Planlanan/gerçekleşen/eksik karşılaştırma eşleştirmesi. Birincil son nokta ve alternatif son noktalar farklı alanlarda; testten sonra seçim algılanır.

**Açık kalan / yanlış iddiayı önleme:** Tüm tabloyu göstermek tek başına çokluk sorununu çözmez; C2 ile birlikte uygulanır.

**Paket:** P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C9.

### C10. İki kampanyayı birleştirme cazibesi — OLASI.

**Çözüm / geliştirme:** Kampanyaları sürüm, veri ailesi, protokol ve tohum hiyerarşisiyle sakla. Aynı ailedeki tekrarları bağımsız yeni kanıt gibi toplamak yerine eşlenik veya hiyerarşik analiz yap.

**Kapanış testi:** Birleştirme komutu uyumsuz protokolleri varsayılan olarak reddetsin; keşif amaçlı birleştirme açık etiket ve ayrı tablo taşısın.

**Açık kalan / yanlış iddiayı önleme:** Farklı protokollerin bir araya gelmesi otomatik olarak daha güçlü örneklem üretmez.

**Paket:** P7/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / C10.

### D1. Zaman ekseni paralelleşmez — YAPISAL.

**Çözüm / geliştirme:** Üç yolu ayrı geliştir: mevcut nonlinear RLT’yi bağımsız diziler/başlıklar boyunca batchle; chunk-recurrent kolunu ayrı mimari olarak dene; önceki gizli duruma bağımlı olmayan kapılardan oluşan yardımcı affine taşıma yolunu associative scan ile paralelleştir. Genel nonlinear düşünme çekirdeğini kaldırma.

**Kapanış testi:** Aynı-semantik optimizasyonda çıktı/gradyan paritesi; yeni mimaride görev başarısı ve toplam latency/FLOP karşılaştırması. B=1 chunk sınır testi. Paket affine yol için değer/gradyan paritesi sağlar.

**Açık kalan / yanlış iddiayı önleme:** Arbitrary nonlinear token zincirinin sıradan parallel scan ile aynı hesap olarak hızlanacağı iddia edilmez. RLT-2 resmi önerisi de henüz throughput sonucu sunmuyor [S1].

**Paket:** P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D1.

### D2. Kernel-başlatma darboğazı — MEVCUT.

**Çözüm / geliştirme:** Küçük kernel zincirlerini statik biçimli yürütme adımlarında birleştir; CPU referansı, eager GPU, compile ve CUDA Graphs ayrı profillensin. Bağımsız diziler ve hazır başlık işleri batchlensin; her tokenda CPU .item()/print/senkronizasyonu kaldır.

**Kapanış testi:** Isınma ve derleme maliyeti ayrı, toplam duvar saati dahil. Doğru dtype/mask ve sayısal tolerans paritesi; CPU’nun hızlı kaldığı iş boyutunda CPU seçeneği korunur.

**Açık kalan / yanlış iddiayı önleme:** CUDA Graphs bellek tüketimini artırabilir ve her modeli yakalayamaz [S3]. CPU prototip hızlarını kullanıcı GPU’suna taşımıyoruz.

**Paket:** P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D2.

### D3. Bellek: detach-only TBPTT hiç tasarruf etmiyor — MEVCUT.

**Çözüm / geliştirme:** Detach ile bellek serbest bırakmayı aynı şey sayma. Segment içi aktivasyon, dış referans tutan listeler, encoder grafiği, cache ve çıktı logits sahipliğini ölç. P2’deki kesintisiz segment remat ile iç aktivasyonları yeniden hesapla; gerektiğinde encoder katmanlarını da ayrı remat et.

**Kapanış testi:** Kısa tam-BPTT gradyan paritesi; gerçek peak allocated/reserved, CPU RAM, tekrar hesaplama sayısı ve duvar saati. Başarısız eski checkpoint konfigürasyonu negatif kontrol olarak kalır.

**Açık kalan / yanlış iddiayı önleme:** Checkpoint her yerleşimde tasarruf sağlamaz. Büyüyen global KV, geri-yayılım için gereken encoder yolu ve Python referansları hâlâ bellek tutabilir [S2].

**Paket:** P2/P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D3.

### D4. 8 GB VRAM / 4 çekirdek / 16 GB RAM — MEVCUT.

**Çözüm / geliştirme:** 30M→300M hedefini koru. Microbatch/gradient accumulation, optimizer durumlarını RAM’e taşıma, katman/segment remat ve ağırlık paylaşımı ayrı kombinasyonlar olarak planlansın. 30M pilotu 100 güncellemede süre ve bellek kalibre etsin; 300M için aynı hesap bileşenleriyle projeksiyon çıkar.

**Kapanış testi:** Parametre, gradyan, optimizer, activation, cache, derleme ve geçici buffer ayrı bellek kalemleri. En az bir güncelleme ve bir evaluation tam başarılı olmadan fizibilite onayı verme.

**Açık kalan / yanlış iddiayı önleme:** Dört float32 parametre-eşdeğeri varsayımında 300M için yalnız ağırlık/gradyan/Adam momentleri 4,8 GB (ondalık) eder; aktivasyon/cache hariçtir. Sığma veya eğitim bitiş süresi garanti edilmez. Kiralık GPU kararı eklenmez.

**Paket:** P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D4.

### D5. Değerlendirme maliyeti — MEVCUT.

**Çözüm / geliştirme:** Problemleri elemek yerine her problem/kol/bütçeyi idempotent işe çevir. Tokenizasyon, sabit doküman indeksi ve uygun sabit-ağırlık hesapları paylaş; model/indeks değişince cache geçersizleşsin. Retrieval, rerank ve generation maliyetini ayrı profil et.

**Kapanış testi:** Tüm 3000 problem ve tüm kollar manifestte tamamlanmalı. Optimizasyon öncesi/sonrası aynı kayıt çıktıları veya önceden belirlenen sayısal parite; toplam süre, p95 ve yeniden deneme maliyeti.

**Açık kalan / yanlış iddiayı önleme:** Eğitim sırasında değişen encoder embeddinglerini eski hâliyle yeniden kullanmak yanlış gradyan/sonuç üretir; sadece geçerli sabit artefaktlar paylaşılır.

**Paket:** P5/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D5.

### D6. Bant genişliği/disk — MEVCUT.

**Çözüm / geliştirme:** Ham arşivi kayıpsız sıkıştırılmış ve içerik-adresli parçalarda tut; aynı baytların birden çok kopyası yerine kaynak referanslarını çoğalt. Veri/checkpoint sürümleri için yer planı, hash doğrulamalı indirme ve eksik parça devamı kullan.

**Kapanış testi:** Sıkıştır-aç bayt eşitliği, bütün kayıtlara erişim, tam restore ve disk dolma enjeksiyonu. Boşluk biterse kayıt silmek yerine işi doğrulanmış noktada beklet.

**Açık kalan / yanlış iddiayı önleme:** Sürekli artan benzersiz ham kayıt sonlu diskte sınırsız saklanamaz. Kapasite artışı/harici arşiv kararı gerçek ihtiyaçtır; performans uğruna silme uygulanmaz.

**Paket:** P3/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D6.

### D7. CPU çekişmesi ölçümü kirletiyor — YAŞANDI.

**Çözüm / geliştirme:** Performans koşularına CPU/GPU kaynak kilidi ekle; veri hazırlama ve ağır değerlendirmeyi aynı ölçüm aralığında koşturma. İşçi PID zinciri, host yükü, thread sayısı, sıcaklık ve saat hızlarını logla.

**Kapanış testi:** Temiz ve kontrollü çekişmeli koşuları ayrı etiketle. GPU süreleri uygun senkronizasyonla ölçülsün; kirli ölçüm ana hız iddiasında kullanılmasın.

**Açık kalan / yanlış iddiayı önleme:** Gürültü eşiği sonuç görüldükten sonra seçilmez. Affinity/thread ayarı ortam bağımlıdır; kullanıcı makinesinde uygulanmadı.

**Paket:** P5/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D7.

### D8. 16.384 / 65.536 uzunlukları açılmadı — MEVCUT.

**Çözüm / geliştirme:** 16.384 ve 65.536 uzunluklarını resmi manifestte aç; k/d eksenleri ve gerçek işlem sayısı ayrı alanlar olsun. Uzun nötr ek ile uzun gerçek işlem zincirini karıştırma.

**Kapanış testi:** Her uzunluk/görev/tohum hücresinin beklenen kimliği ve makbuzu. Tamamlanmayan hücre görünür INCOMPLETE; küçük uç test genel ölçüm yerine geçmez.

**Açık kalan / yanlış iddiayı önleme:** Önceki yaklaşık 40 dakika tahmini bütün görev/model boyutlarına taşınmaz; kalibrasyon yenilenir.

**Paket:** P1/P5/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D8.

### D9. Reranker ileri geçişi bütün konumlar için sözlük-boyutlu logits üretiyor — YAŞANDI.

**Çözüm / geliştirme:** Reranker’da bütün konumların sözlük logits’ini üretme. Model destekliyorsa yalnız gereken konum logits’i; değilse taban hidden-state + son geçerli konuma lm_head. Sol/sağ padding ve çok-token etiketler için ayrı doğru indeksleme.

**Kapanış testi:** Eski tam-logits skoruyla örnek bazında parite; uzun dizi/batchte tahsis şekli ve peak bellek. Generation optimizasyonu tüm-token eğitim kaybına aynen uygulanmamalı.

**Açık kalan / yanlış iddiayı önleme:** logits_to_keep modern Qwen3 arayüzünde mevcut [S8]; yerel sürüm/modelde destek doğrulanır. Etiket normalizasyonunu değiştirmek farklı reranker olur.

**Paket:** P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / D9.

### E1. Windows venv kurulumu — YAŞANDI.

**Çözüm / geliştirme:** Ortam hazırlığını atomik aşamalara ayır: kurulum tamamlandı sinyali ancak doğru Python, torch sürümü, CUDA kullanılabilirliği ve küçük GPU forward/backward kontrolünden sonra. İndirilen wheel/hash ve TEMP yolu makbuzda olsun.

**Kapanış testi:** CPU torch yanlışlıkla kurulduğunda GPU kampanyası başlamamalı; Windows dosya-kilidi hatası açık başarısızlık üretmeli.

**Açık kalan / yanlış iddiayı önleme:** Buradaki test ortamı Linux CPU’dur; Windows/cu130 kurulumu bu teslimde denenmedi.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E1.

### E2. venv `python.exe` yönlendirici — MEVCUT.

**Çözüm / geliştirme:** Launcher PID’sini gerçek işçi yerine kullanma. İşçi kendi PID, parent PID, başlangıç zamanı ve run_id ile heartbeat yazsın; kontrolör tüm zinciri izlesin.

**Kapanış testi:** PID yeniden kullanımı, venv yönlendiricisi ve aniden ölen işçi senaryolarında yanlış sürece bağlanma olmamalı.

**Açık kalan / yanlış iddiayı önleme:** Salt PID yeterli kimlik değildir; aynı PID başka işlem olabilir. Yerel runner adaptörü gerekli.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E2.

### E3. Satır sonu / kodlama olayları — YAŞANDI.

**Çözüm / geliştirme:** Makine kayıtlarını açık UTF-8 ve belirlenmiş LF politikasıyla bayt olarak yaz; ham kullanıcı/düşünce kayıtlarının özgün baytlarını değiştirme. Manifest hash’i en son diskteki bayttan hesapla.

**Kapanış testi:** Türkçe İ/ı/ğ, CRLF/LF, yarım dosya ve yeniden açma testleri. Ham içerik aynen kalmalı, JSON çıktısı belirlenen kodlamayı kullanmalı.

**Açık kalan / yanlış iddiayı önleme:** Konsoldaki bozuk görüntü ile diskteki bozuk içerik ayrı olaydır. Paket ikisinin dosya tarafını sınar.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E3.

### E4. API kota kesintisi — YAŞANDI.

**Çözüm / geliştirme:** Önce makine-okunur sonuç ve işlem makbuzu, sonra insan-okunur rapor. API 429/timeout için idempotent anahtar ve artan beklemeli yeniden deneme; her deneme ayrı kayıt.

**Kapanış testi:** Model raporu yarıda kalsa bile tamamlanmış sayısal işlerden rapor yeniden üretilebilmeli. Başarısız API yanıtı boş geçerli sonuç sayılamamalı.

**Açık kalan / yanlış iddiayı önleme:** Kota veya dış servis erişimi yerel mekanizmayla ortadan kalkmaz; kayıp ve yeniden iş yapma azaltılır.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E4.

### E5. Kontrolör devir teslimi — YAŞANDI.

**Çözüm / geliştirme:** El yazısı ledger yerine append-only events kaydından üretilen görünüm kullan. Devir teslim anında model/kod/veri pinleri, tamamlanan koşular, sıradaki işler ve eksikler otomatik özetlensin.

**Kapanış testi:** Eski ledger silinmeden güncel görünüm yeniden oluşturulsun. Yeni kontrolör aynı açık işleri ve tamamlanma sayısını bulmalı.

**Açık kalan / yanlış iddiayı önleme:** LLM kontrolörün özeti kaynak-of-truth değildir; makine manifesti ve makbuzlarıdır.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E5.

### E6. Mühürleme ve sonradan ekleme — MEVCUT.

**Çözüm / geliştirme:** Değişmez içerik parçaları ve kampanya-kapsamlı manifestler kullan. Kapanış kök manifesti bunların hash’lerini içerir; yeni çalışma yeni kök sürümü üretir, eski mühre dosya eklemez.

**Kapanış testi:** Eski kampanya manifestleri yeni dosyalardan etkilenmemeli. Final manifest kendi dosya hash’ini kendine dahil etmemeli; restore doğrulaması geçmeli.

**Açık kalan / yanlış iddiayı önleme:** Hash zinciri kötü niyetli sahip silmesine karşı yedek değildir. Değişmezlik sözleşmesi ve fiziksel yedek farklıdır.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E6.

### E7. Runner'lar kesintide devam etmiyor (fail-closed) — MEVCUT/KARAR.

**Çözüm / geliştirme:** Fail-closed davranışı koruyarak doğrulanmış transaction sınırından devam ekle. Eğitimde ağırlık/optimizer/RNG/data-cursor/accumulation ve bütün recurrent cache birlikte checkpointlensin. Yarım transaction yeniden oynatılır; belirsiz devam başarı diye yazılmaz.

**Kapanış testi:** Kesintisiz koşu ile kill→restore koşusunun aynı sonraki örnekleri, güncellemeleri ve sonuçları üretmesi. Kesinti optimizer güncellemesinin önünde/arkasında ve dosya yazımında enjekte edilir.

**Açık kalan / yanlış iddiayı önleme:** Python autograd grafiği sıradan checkpoint dosyası değildir. Eğitim güvenli optimizer-step sınırından veya tam yeniden hesaplama ile devam etmeli; gizli TBPTT kesintisi eklenmemeli.

**Paket:** P2/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E7.

### E8. Heredoc/tırnak hataları — YAŞANDI.

**Çözüm / geliştirme:** Yamaları ayrı sürümlü dosya/diff olarak üret; çalıştırmadan sözdizimi, encoding ve hedef hash kontrolü yap. Heredoc içinden iç içe yamalama yerine tek patch aracı ve tek giriş noktası.

**Kapanış testi:** Yanlış tırnak, REPL başlığı ve yanlış hedef kaynak üzerinde patch uygulanmadan hata. Mühürlü kaynak hash’i değişirse ayrı deney sürümü zorunlu.

**Açık kalan / yanlış iddiayı önleme:** Bu teslim bağımsız paket; kullanıcının dondurulmuş dosyalarına otomatik patch uygulanmadı.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E8.

### E9. Git remote yok — MEVCUT.

**Çözüm / geliştirme:** Önce yerel arşivin tutarlı yedeğini ve restore testini otomatikleştir. SQLite WAL açıkken dosyayı kopyalamak yerine backup API/snapshot kullan. Farklı fiziksel disk ve/veya özel uzak yedek için hedef ayrıca tanımlansın.

**Kapanış testi:** Yedekten kaynak hash’leri, kayıt zinciri ve testler geri getirilmeli. Asıl diski erişilemez sayan restore provası.

**Açık kalan / yanlış iddiayı önleme:** Bu paket backup fonksiyonunu geçici yerel dosyalarla sınadı; gerçek bilgisayarının yedeği alınmış veya remote kurulmuş değildir. Harici paylaşım yapılmadı.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E9.

### E10. Kod tekrarı — MEVCUT.

**Çözüm / geliştirme:** Dondurulmuş deney kaynaklarına dokunmadan yeni sürümlü experiment_core çıkar. Eski kampanyalar eski vendored sürümü kullanmaya devam etsin; yeni kampanyalar ortak makbuz/freeze/runner şemasını alsın.

**Kapanış testi:** Ortaklaştırma öncesi/sonrası aynı fixture üzerinde çıktı paritesi; geçmiş kampanyalar eski ortamla açılmalı. Görev mantığı ile ortak altyapı bağımlılığı ayrılmalı.

**Açık kalan / yanlış iddiayı önleme:** Eski kaynakları yerinde refactor etmek tekrar üretilebilirliği bozar. Yeni ortak çekirdek entegrasyonu bu teslimde asıl repoya yapılmadı.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E10.

### E11. Test paketleri dağınık — MEVCUT.

**Çözüm / geliştirme:** Tüm test köklerini toplayan tek komut ekle; sonuçta hangi katmanın çalıştığı ve hangi asset’in eksik olduğu makine-okunur çıksın. Birim test, GPU entegrasyon ve tam deney doğrulaması ayrı sonuç alanları olsun.

**Kapanış testi:** Bir alt klasördeki test hatası genel çıkış kodunu bozmalı. Atlanan/eksik testler açık sayılmalı, bütün testlerin adları listelenmeli.

**Açık kalan / yanlış iddiayı önleme:** Bu paketin tek komutu 46 bağımsız testi çalıştırır; orijinal dağınık testleri keşfedip çalıştırmış sayılmaz.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E11.

### E12. Zincir loglarında yanlış çıkış kodu — YAŞANDI.

**Çözüm / geliştirme:** Kabuk kodunu komut bitiminde hemen yakala; başarı kararı için doğru run_id, sıfır çıkış kodu, tam örnek sayısı, test başarısı ve artifact hash’lerinin hepsini iste. Dosya varlığı tek başına yeterli değildir.

**Kapanış testi:** Çöken programın eski sonuç dosyası, yanlış run_id, eksik örnek ve bozulmuş hash durumlarının tamamı başarısız olmalı.

**Açık kalan / yanlış iddiayı önleme:** Makbuzun doğruladığı şey yürütme bütünlüğüdür, bilimsel hipotezin doğruluğu değildir.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / E12.

### F1. B1 hedefleri (recall@10 ≥ %95, kapsama ≥ %90) bu yöntemlerle ulaşılmaz görünüyor — MEVCUT.

**Çözüm / geliştirme:** recall@10 ≥%95 ve kapsama ≥%90 hedefleri korunur. B2/B3/B1’i aynı kanıt zincirinin farklı başarısızlık kapıları olarak bağla; iyileşmeyi gerçek taze sorgularda doğrula.

**Kapanış testi:** Aynı final top10, aynı örnekler, koşulsuz kapsama ve yanlış kabul; oracle üst sınır ile öğrenilmiş uçtan uca sonuç ayrı.

**Açık kalan / yanlış iddiayı önleme:** Mevcut yöntemlerin düşük skorunu, hedefin matematiksel imkânsızlığı diye yorumlama; önerilen sistemin hedefe ulaştığı da söylenmez.

**Paket:** P4/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F1.

### F2. D1 hedefleri (F1 ≥ 0,95, risk ≤ %1, kapsam ≥ %80) — OLASI zor.

**Çözüm / geliştirme:** F1≥0,95, risk≤%1 ve kapsam≥%80 hedeflerini birlikte tut. Ayrıştırma/aday/erişim/sertifika kalibrasyonunu validationda yap; testte eşik dondurulur. Çekimserlik ile yanlış cevap ayrı ölçülür.

**Kapanış testi:** Risk için yalnız nokta tahmini değil üst güven sınırı. Sıfır yanlışta bile %95 tek-taraflı <%1 sınırı için en az 299 bağımsız kabul gerekir; bu hesap paket içinde testli.

**Açık kalan / yanlış iddiayı önleme:** 299 sayısı yalnız sıfır hata, bağımsız kabul ve sabit politika varsayımlarındadır. Tüm dağılımlara sertifika veya çoklu sınamalar için genel sayı değildir.

**Paket:** P4/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F2.

### F3. D2 3.000 problem × kollar × bütçeler = 12.000+ bölüm — OLASI.

**Çözüm / geliştirme:** 3000 problem ve 12000+ bölüm korunur. Tam manifesti model/uzunluk/grup bazında iş kuyruğuna böl; kesilen kampanya biten bölümleri tekrar çalıştırmasın. Bütün kol ve bütçeler tamamlanıncaya kadar final rapor eksik bayrağı taşır.

**Kapanış testi:** Her problem×kol×bütçe×tohum hücresi tek geçerli makbuzla kapanmalı; eski yarım denemeler saklanmalı. Toplam enerji/süre/yeniden deneme maliyeti görünür.

**Açık kalan / yanlış iddiayı önleme:** Kuyruklama mevcut donanımın işlem gücünü artırmaz. Süre kalibrasyona bağlıdır; daha küçük örneklemle hedefi değiştirme yapılmaz.

**Paket:** P5/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F3.

### F4. D4 (gerçek RLT dil eğitimi) yerelde fizibilite belirsiz — OLASI.

**Çözüm / geliştirme:** D4 gerçek RLT dil eğitimini D2’deki hazır model değerlendirmesinden ayrı tut. 30M gerçek çekirdek pilotunda forward/backward/cache ve checkpoint paritesi önce kurulur; aynı yöntemle 300M maliyet modeli çıkarılır.

**Kapanış testi:** 100 güncellemelik ölçüm; token throughput, tepe bellek, kayıp eğrisi, LR/tohum ve kısa dil görevleri. D2 başarısı D4 çekirdek başarısı diye taşınmamalı.

**Açık kalan / yanlış iddiayı önleme:** Pilot bütçesi bitince düşük skor kabiliyet tabanı kaynaklı olabilir. Fizibilite ve öğrenme kalitesi ayrı sorulardır.

**Paket:** P2/P5/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F4.

### F5. U3 (biçimsel güvence) — OLASI.

**Çözüm / geliştirme:** Teorem hedefini bütün durumun daralması değil, doğrulanmış kimlik işleminde korunan alanın aynılığı olarak kur. Token/scratch yolu çalışmaya devam edebilir. Ardışık kimlik geçişlerinin kayıt durumunu değiştirmediği tümevarımla gösterilir; uygulama alias/cache koşulları ayrıca sınanır.

**Kapanış testi:** Bağımsız referans kod, tam kopya testleri, yanlış kapı karşı-örneği ve cache üzerinden bozulma karşı-örneği. Lean kullanılacaksa bu dar sözleşmenin gerçek proof dosyası ayrıca derlenmeli.

**Açık kalan / yanlış iddiayı önleme:** Bu teslim Lean doğrulaması yapmadı. Dar mekanizma teoremi genel akıl yürütmenin doğruluğu veya doğal dil nötrlüğünün öğrenildiği anlamına gelmez.

**Paket:** P1/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F5.

### F6. U1 k ve d eksenleri ertelendi — MEVCUT.

**Çözüm / geliştirme:** U1’in ertelenen k ve d eksenlerini geri ekle. Token sayısı, gerçek anlamlı işlem sayısı, iç düzeltme turu ve başlık bağımlılık derinliği farklı alanlar olsun.

**Kapanış testi:** Aynı token uzunluğunda değişen k/d, aynı d’de nötr token ekleme ve aynı k’de yeni bağımlılıklar. Hesap bütçesi ve son doğruluk birlikte.

**Açık kalan / yanlış iddiayı önleme:** Tek bir uzunluk grafiği yapısal derinliği ölçmez. Tam faktöriyel pahalıysa takvim evrelere ayrılabilir ama planlı hücreler silinmez.

**Paket:** P1/P3/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F6.

### F7. "Genel doğal dil / sınırsız doğru düşünme" iddiaları için kanıt yok — MEVCUT.

**Çözüm / geliştirme:** Genel dil ve açık uçlu düşünme hedeflerini koru, iddiayı ölçüm alanına bağla. Kontrollü dil, doğal Türkçe/İngilizce, kod, uzun bellek ve yeni problem aileleri için ayrı sonuç kartları oluştur.

**Kapanış testi:** İddia kartında eğitim alanı, test alanı, kabul ölçütü, bellek açık/kapalı ve motor açık/kapalı koşulları olmalı.

**Açık kalan / yanlış iddiayı önleme:** Sonlu deney sınırsız doğruluk kanıtı değildir. Bu, hedefi bırakmak değil elde edilen kanıtın kapsamını dürüst belirtmektir.

**Paket:** P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F7.

### F8. LongMemEval-V2, Huginn karşılaştırması, enerji ölçümü — YAPILMADI.

**Çözüm / geliştirme:** LongMemEval-V2, Huginn ve enerji ölçümünü plana dahil et. LME-V2 ile daha eski LongMemEval-M aynı veri seti sayılmasın [S11]. Huginn karşılaştırmasında parametre/FLOP ve inference koşulları raporlanır [S12].

**Kapanış testi:** Veri/model sürümü ve indirme hash’i; bellek açık/kapalı; GPU kart enerjisi ile bütün sistem enerjisi ayrı. Boşta tüketim ve ölçüm aralığı açık.

**Açık kalan / yanlış iddiayı önleme:** Bu teslim bu veri setlerini indirmedi veya Huginn çalıştırmadı. Telemetri desteklenmiyorsa enerji sayısı uydurulmaz; uygun ölçüm yolu gerekir.

**Paket:** P5/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / F8.

### G1. 

**Çözüm / geliştirme:** Önerilen karar: U2 ölçümüyle birlikte korunan-anlam/değişebilir-hesap mimarisi ana araştırma kolu; yardımcı eğitim hedefi ayrı kol; dış çapa referans/onarım kolu. Birini seçip diğer olasılıkları kapatmak yerine katkıları ayrıştır.

**Kapanış testi:** Her kol aynı ailelerde ve aynı rapor şemasında; ortak hesap arttığında ek maliyet görünür.

**Açık kalan / yanlış iddiayı önleme:** Karar bir başarı ilanı değildir. Orijinal kayıtta olmayan varsayımlar yeni protokol olarak etiketlenir.

**Paket:** P1/P2 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G1.

### G2. 

**Çözüm / geliştirme:** Önerilen karar: 3000 problem yerelde ve tam kapsam; kiralık GPU eklemeden transactionlı çok-oturumlu yürütme. Çalışma süresi kalibrasyondan çıkarılır.

**Kapanış testi:** F3 manifestinde bütün hücrelerin kapanışı; örnek/kol/bütçe silinmesi test tarafından reddedilir.

**Açık kalan / yanlış iddiayı önleme:** Yerel süreyi keyfi bir güne indirme sözü verilmez. Hedef kalır, takvim ölçümle netleşir.

**Paket:** P5/P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G2.

### G3. 

**Çözüm / geliştirme:** Önerilen karar: doğal Türkçe insan-kaynaklı veri ve etiketli çeviri iki ayrı hat; mevcut olmayan insan kontrolü tamamlanana kadar o hücre UNMEASURED. İngilizce hat durdurulmaz.

**Kapanış testi:** C3/C4 kaynak etiketleri ve test aileleri; insan onayı gerektiren satır eksikse rapor bunu gösterir.

**Açık kalan / yanlış iddiayı önleme:** İnsan doğrulaması kullanıcı adına icat edilemez; veri erişim/katkı gerekir.

**Paket:** P4/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G3.

### G4. 

**Çözüm / geliştirme:** Önerilen karar: aynı diskte geçici yedek değil, doğrulanmış farklı fiziksel hedefe snapshot ve restore; özel remote ikinci koruma. Şimdiki teslim yalnız yerel backup/verify mekanizmasını sağlıyor.

**Kapanış testi:** Gerçek hedef tanımlanınca tam restore denemesi ve hash eşliği.

**Açık kalan / yanlış iddiayı önleme:** Harici diske erişim veya uzak paylaşım bu oturumda kurulmadı. Kimlik bilgisi/araştırma verisi bir dış servise gönderilmedi.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G4.

### G5. 

**Çözüm / geliştirme:** Önerilen karar: geçmiş dondurulmuş kaynakları aynen tut, yeni ortak altyapıyı sürümle. Deney manifesti kaynak çekirdeğinin sürüm/hash’ini belirtir.

**Kapanış testi:** Eski ve yeni kampanyaları aynı makinede ayrı ortamdan yeniden açma; çıktılarda kaynak kökü karışmamalı.

**Açık kalan / yanlış iddiayı önleme:** Ortaklaştırma bilimsel yöntem değişikliği gizlememeli.

**Paket:** P6 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G5.

### G6. 

**Çözüm / geliştirme:** Önerilen karar: 16k/64k ve uzun yığın eğitimini tam manifestte koru. Kısa kalibrasyon yalnız çalışma sırasını ve tahmini süreyi belirler; uzun koşuları iptal etmez.

**Kapanış testi:** Uzunluk ve uzatma evrelerinin ayrı makbuzları, bütün tohumların görünürlüğü.

**Açık kalan / yanlış iddiayı önleme:** Kalibrasyon sonucu çok uzun süre çıkabilir; bunu çözülmüş fizibilite diye gizleme.

**Paket:** P1/P5/P7 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G6.

### G7. 

**Çözüm / geliştirme:** Önerilen karar: Symbolica/Graphica için geliştirici arayüzü şimdi tasarlanır; bilimsel ana hatta yeniden açılış, öğrenilmiş IR’nin yeterli kalitesi ve basit motor üzerine ölçülen ek yarara bağlanır. Hiçbir genel düşünce başlığı bu motora çevrilemediği için reddedilmez.

**Kapanış testi:** B8 üç motor kolu, kaynak/sürüm/varsayım-fingerprint ve sıcak/soğuk önbellek maliyeti.

**Açık kalan / yanlış iddiayı önleme:** Motor kullanmak zorunlu şema daraltması olmamalı. Temsil edilemeyen parça doğal/latent başlık yolunda kalır; durum açıkça CANDIDATE olarak saklanır.

**Paket:** P4/P5 — **Durum:** asıl projede doğrulanmadı. Kaynak: problem_havuzu.md / G7.

## 9. Bu oturumda gerçekten yapılanlar

- Kaynak dosyadaki **64/64 madde** kendi kimlikleri ve mevcut durumlarıyla çözüm/test eşleştirmesine alındı.
- Bağımsız Python/PyTorch referans mekanizmaları yazıldı: tam korumalı seçici yazma, tüm durumu açık geçen segment remat, affine prefix scan, append-only ham arşiv, sürümlü bağımlılık denetimi, eşikte sayfa geçişi, hash’li makbuz, tam-log agregasyonu, çokluk düzeltmesi ve tam iş manifesti.
- **46/46 test geçti; 0 hata, 0 başarısızlık, 0 atlama.** Test ortamı Python 3.13.5, PyTorch 2.10.0+cpu, CPU. Windows/CUDA testi yapılmadı.
- Tam nötr maskesi verilmiş mekanizmada çalışma alanı güncellenirken korunan kayıt 65.536 adımda bit-bit aynı kaldı. Bu **öğrenilmiş nötr tanıma veya RLT başarısı değil**, açık koruma sözleşmesinin uygulama testidir.
- 128 adımlık, iki-durumlu küçük nonlinear recurrent hesapta son kayıp için tam BPTT ile segment8 remat’ın kayıp ve bütün karşılaştırılan gradyan farkı sıfır çıktı. TBPTT8 kolunda ilk durum/girdi gradyanı yoktu. **Gradyanın ulaşması öğrenmenin başarıldığı anlamına gelmez.**
- 4096 adımlık yardımcı affine scan ile seri referansın en büyük mutlak farkı yaklaşık **6,66×10⁻¹⁶** oldu. GPU hızlanması ölçülmedi.
- 64 ham arşiv sürümü byte-for-byte geri okundu; tutarlı yerel SQLite backup doğrulandı. Kullanıcının diski veya uzak yedeği üzerinde işlem yapılmadı.

`results/tests.json` bütün test adlarını, `results/mechanisms.json` bütün probe sonuçlarını içerir. Paket kaynakları ve bu rapor için manifest ayrıca üretilir. Orijinal RLT eğitim kodu, learned retrieval controller, doğal dil parser, tam graph scheduler, CUDA kernel, end-to-end training ve büyük deney matrisi uygulanmış değildir. Dosyadaki eski 108/108 test veya kampanya başarılarını bu 46 testle birleştirmiyoruz.

## 10. Kaynaklar ve kullanım sınırı

**[S1] RLT resmi proje sayfası, 17 Eylül güncellemesi.** https://yifanzhang-pro.github.io/recurrent-looped-tranformer/

Tam durum, gated merge, RLT-2 önerisi; bağımsız dış kaynak, yerel kampanyanın sonucu değil. Erişim: 18 Eylül 2026.

**[S2] PyTorch 2.10 checkpoint belgeleri.** https://docs.pytorch.org/docs/2.10/checkpoint.html

Segment yeniden hesaplama, RNG ve aynı fonksiyonun tekrar yürütülmesi gereği. Erişim: 18 Eylül 2026.

**[S3] PyTorch Performance Tuning Guide.** https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

CUDA Graphs, host overhead, ek bellek ve senkronizasyon uyarıları. Erişim: 18 Eylül 2026.

**[S4] IRCoT, ACL 2023.** https://aclanthology.org/2023.acl-long.557/

Çok atlamalı erişim ile ara çıkarımın birbirini yönlendirmesi. Erişim: 18 Eylül 2026.

**[S5] Were RNNs All We Needed?.** https://arxiv.org/abs/2410.01201

Geçiş biçimi uygun tasarlanınca paralel eğitilebilir recurrent yollar; mevcut nonlinear RLT’ye eşitlik iddiası değil. Erişim: 18 Eylül 2026.

**[S6] Symbolica 1.0 stable release.** https://symbolica.io/posts/stable_release/

Sembolik/sayısal motor; Numerica/Graphica ve lisans ayrımı. Erişim: 18 Eylül 2026.

**[S7] Graphica resmi depo.** https://github.com/symbolica-dev/graphica

Etiketli graf kanonikleştirme; doğal dil anlam çözümleme değil. Erişim: 18 Eylül 2026.

**[S8] Qwen3 Transformers forward arayüzü.** https://huggingface.co/docs/transformers/main/en/model_doc/qwen3

logits_to_keep ve işlenmemiş token/cache sözleşmesi. Erişim: 18 Eylül 2026.

**[S9] Hugging Face Accelerate Big Model Inference.** https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling

Inference için CPU/GPU/disk ağırlık yerleşimi; eğitim garantisi değil. Erişim: 18 Eylül 2026.

**[S10] TurkishMMLU, EMNLP 2024.** https://aclanthology.org/2024.findings-emnlp.413/

Uzmanlarca yazılmış Türkçe QA; bizim özel revizyon görevlerimizde sonuç yok. Erişim: 18 Eylül 2026.

**[S11] LongMemEval-V2.** https://arxiv.org/abs/2605.12493

2026 tarihli ayrı uzun-süreli ajan bellek değerlendirmesi. Erişim: 18 Eylül 2026.

**[S12] Scaling up Test-Time Compute with Latent Reasoning.** https://arxiv.org/abs/2502.05171

Huginn/recurrent-depth karşılaştırmasının birincil kaynağı. Erişim: 18 Eylül 2026.

## 11. Nihai karar

Kapsamı daraltan bir "daha küçük problem seç" çözümü önerilmiyor. Önerilen değişiklik; uzun düşünmenin korunan kaydı sürekli yeniden yazmasını engellemek, son hatanın önceki hesaplara ulaşacağı eğitim yolunu açık tutmak, başlıkların yalnız etiket değil sürümlü çalışma birimleri olmasını sağlamak ve tüm deneyleri yeniden başlatılabilir bir yürütme düzenine taşımak.

Mühendislik sözleşmelerinin bir kısmı burada kodlandı ve test edildi. Çekirdek zekâ, doğru yazma kararı, semantik erişim, genel aktarım ve gerçek donanım kazancı için sunulan şey ayrıntılı, yanlışlanabilir bir çözüm tasarımıdır; elde olmayan sonuçlar uydurulmamıştır.
