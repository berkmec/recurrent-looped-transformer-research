# RLT ve başlık belleği: iddiaların matematiksel sınırı

Bu not, ölçüm sonuçlarının hangi iddiayı destekleyebileceğini tanımlar. Deney sayısının artması bu sınırları kendiliğinden değiştirmez. Kullanıcının `RLT KUSUR ÇÖZÜM.txt` dosyası araştırılacak otuz risk başlığıdır; dosyadaki her ifade kanıtlanmış bir RLT kusuru olarak kabul edilmedi. Kaynak dosya SHA256: `8D242CADC26F787A122E9D6FCB66C830FB9E193ADC10D078A107B8AE1F67B649`.

## Üç farklı derinlik

Token zamanı t, başlık güncelleme turu k ve problemin bağımlılık derinliği d ayrı değişkenlerdir. RLT çekirdeğinde her tüketilen token bir durum geçişi yapar. Başlık güncellemesinde ise mevcut problem temsili üzerinde ek hesaplama yapılır; yeni token veya SWA girdisi eklenmez. Daha uzun bir girdi daha zor bir bağımlılık zinciri içerebilir, fakat gereksiz kimlik işlemleri eklemek d'yi artırmadan t'yi artırır.

Kaynağın nedensel çekirdek sözleşmesi şu biçimdedir:

\[
H_t=(s_t,C_t^D),\qquad H_t=F_\theta(e_t,H_{t-1},M_{\le t}).
\]

Burada s son decoder gizli durumu, C katmanların yerel dikkat önbelleği, M nedensel encoder belleğidir. Yalnız s'yi izlemek bütün model belleğini izlemek değildir. Yerel pencere sabit olsa da M girdiyle büyür. Kaynak ve deney ayrımı: [sabitlenmiş resmî README](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md). Bizim çekirdek bağımsız, küçük bir uygulamadır; resmî büyük model sonucunun yeniden üretimi değildir.

Başlık mekanizmasında bir güncelleme yaklaşık olarak

\[
z_i^{k+1}=U_\theta(z_i^k,a_i,\{m_{j\rightarrow i}^k\}_{j\in N(i)})
\]

şeklindedir. a_i gözlenen kısıtları, N(i) kullanılan yönlü veya çift yönlü ilişkiyi temsil eder. Bu ifade düşüncenin kendisini açıklayan bir teori değildir; uygulamadaki mesaj geçişinin sözleşmesidir. Çift yönlü güncelleme, sonraki kanıt geldiğinde önceki başlığın gizli durumunu günceller. Başlık sicilindeki anlamsal sürüm yalnız doğrudan gözlenen bir kısıt değiştiğinde artar. Daha önce üretilmiş bir token geçmişe dönük değiştirilmez.

## “Sonsuz derinlik” hangi iddiadır?

Sabit sayıda ağırlıkla yinelenen bir dönüşümün yürütme zinciri, matematiksel tanımında seçilen adım sayısı boyunca uzatılabilir. Bu, mimaride sabit bir ileri geçiş katman sayısının bütün hesaplama yolunu sınırlamadığını anlatır. Fiziksel makinede süre, bellek ve sayısal hassasiyet sonludur.

Bu yapıdan şu sonuçlar çıkmaz: her ek adımın doğruluğu artırması; sonsuz bağımlılığın öğrenilebilmesi; hata birikiminin durması; bilgi kapasitesinin sınırsızlaşması; eğitimin veya çıkarımın ücretsiz olması. Bunlar ayrı hipotezlerdir. Sonlu örneklerde ve en fazla1024tokenlık bir ızgarada ölçülen başarı, bütün uzunluklarda başarı teoremi değildir.

Operasyonel etkin ufuk, önceden belirlenmiş uzunluk ızgarasında bir modelin bütün önceki uzunluklarla birlikte en az%95 nihai doğruluk sağladığı en büyük ölçülmüş uzunluktur. İlk noktada bile bu eşiğe erişmeyen modele bu protokol altında etkin ufuk atanmamalıdır. Izgaradaki son noktayı geçen model için yalnız “1024'e kadar bu testte geçti” denebilir; üst sınır keşfedilmiş olmaz.

Parite iki durumlu, diğer çekirdek görevi beş durumlu bir süreçtir. Doğru güncelleme kuralı öğrenilirse geçmişin sonraki adım için yeterli bilgisi sırasıyla bir bit veya beş olasılıklı bir durumla taşınabilir. Uzun girdide başarı bu sonlu durum kuralının genellemesini ölçer; taşınması gereken bilginin uzunlukla sınırsız büyüdüğü bir problemi çözmüş olmaz. Uzun bağımlılık, büyük bilgi kapasitesi ve zor kavramsal akıl yürütme aynı deney değişkeni değildir.

## Daha fazla hesaplama neden bozulabilir?

Bir geçişin yerel doğrusal yaklaşımında küçük bir durum farkı

\[
\delta H_T\simeq J_TJ_{T-1}\cdots J_{t+1}\delta H_t
\]

olarak taşınır. Jacobian çarpımları bazı yönleri bastırabilir, bazılarını büyütebilir. Modelde genel dikkat yolları bulunduğundan yalnız gizli durum zincirindeki türeve bakıp bütün geçmiş bilgisinin silindiği söylenemez. Aynı nedenle gizli durum/SWA bağlantılarını sekiz adımda kesmek, encoder belleğinden geçen bütün uzun menzilli gradyanları kesmek değildir.

Prefix kaybı, her adımın doğru hedefini verir; son adım kaybı ise gecikmiş hata atfını daha doğrudan sınar. Bu yüzden iki eğitim karşılaştırması ayrıldı. Sıfır olmayan bir son-kayıp türevi, sonucun doğru öğrenildiği anlamına gelmez. Sonlu fark, otomatik türevin sayısal kontrolüdür; öğrenme kanıtı ayrı test doğruluğudur.

Küçük yerel türev de tek başına bilginin unutulduğu anlamına gelmez. Bunu gösteren, elle tanımlanmış bir matematik örneği kurulabilir: `s_next=tanh(2*(-1)^x*s)`, `x∈{0,1}`. Pozitif `a=tanh(2a)` çözümü yaklaşık0,957504'tür. Başlangıç durumu+a ise giriş0 işareti korur, giriş1 işareti değiştirir; ideal modelde durum her adımda±a üzerinde kalır ve parite işaretle saklanır. Buna karşılık durumun yerel türev büyüklüğü `2*(1-a²)≈0,166372` olur;31adımda çarpım yaklaşık7,14e-25'e iner. Ayrık iki durumun kimliği korunurken sonsuz küçük pertürbasyonlara duyarlılık sönmüştür.

Bu örnek eğitim sonucu, RLT'nin gerçek iç mekanizması veya gürültü altında sınırsız dayanıklılık iddiası değildir. Yalnız “türev küçük ⇒ bilgi yok” çıkarımına bir karşı örnektir. Öğrenilmiş doğru ve doygun bir sınıflandırıcının küçük kayıp türevi de beklenebilir. Bu yüzden gradyan büyüklüğü, ayrık girdide karşı-olgusal etki, elde tutulan bilgi ve sondaki hatadan kuralı öğrenme başarısı ayrı değerlendirilir.

## Başlık, sürüm ve sıkıştırma sözleşmesi

Sonlu deneyde H(task,entity) sabit başlık kimliğidir. Bir değişkenin mutlak gözlemi ile başka bir değişkene göre fark bağıntısı ayrı anlamsal alanlarda tutulur. Örneğin x1=x0+0.25 ve x1=0.75 birlikte korunmalıdır; ikinci bilgi ilkini silerse ters çıkarım kaybolur. Doğrudan gözlenen bir kısıt değiştiğinde başlık sicilinde yeni anlamsal sürüm üretilir; aynı gözlemin tekrarında sürüm değişmez. Bu sicil sayacı, her sinirsel güncelleme turunun veya modelin iç olay sayacının eşanlamlısı değildir. Eski sürüm ve ham metin kalır.

Aktif kapasite, modelin toplam RAM'i değil, denetleyicinin tuttuğu ham kayıt kimliklerinin sayısıdır. Başlıklar bu ortak kapasiteyi paylaşır. Kapasite dolduğunda güncel kısıtların disk özeti oluşturulur. Ham gözlemler ve üretilen yerel model izleri arşivden silinmez. Yeni kanıt eski özetin kapsamını değiştirdiğinde yeniden açma gerekir.

Genel kayıpsız anlamsal sıkıştırma için, izin verilen her gelecekteki ek bilgi u ve sorgu q bakımından geçmiş h ile özeti C(h)'nin aynı cevaba yetmesi gerekir. Burada yalnız afin kısıt ailesinin güncel denklemleri korunur. Bu, öğrenilmiş RLT gizli durumunun veya serbest dilde bütün düşüncenin yeterli istatistiği olduğunu kanıtlamaz. Tam geçmişi yeniden oynatma ile güncel kısıtlardan modeli yeniden başlatma bu nedenle ayrı deney koşullarıdır.

Üretilen düşünce durumları için ayrıca sürümlü bir devamlılık deneyi vardır. Sabit bağlam K altında güncelleme z_next=U(z,K) yalnız son z'ye bakıyorsa, geçmişteki dört z kaydının aktif tamponunu son z sınırıyla değiştirmek bir sonraki yerel güncellemeyi değiştirmez. Ham dört kayıt yine arşivde kalır. Uygulama sınırdan z'yi gerçekten yeniden yükler, adayı yeniden hesaplar ve ancak sonra yeterliliği denetler. K; ağırlık, geçiş yönü/ayarları, gözlem, grafik, token durumu, önbellek ve ilgili sürüm yükümlülüklerini içerir. Bunlar değişirse eski sınır kabul edilmez. Bu deney bütün başlıkların son z tensörünü korur; diğer çalışma durumu bellekte kalır. Aktif geçmişteki kayıt veya serileştirilmiş yük azalması, toplam süreç RAM'inde aynı oranda azalma değildir.

## Doğrulama ve durma

Doğrulayıcı, modelin ürettiği sayısal adayları gözlenmiş denklemlere koyar. Ankora bağlanan bir yol üzerinde üçgen eşitsizliği, sorgudaki hatayı ankora ve kenarlara ait artıkların toplamıyla sınırlar. Bağlı bileşendeki bütün artıkların toplamı bu yola ait toplamdan büyük veya eşittir; koruyucu bir üst sınır verir. Kod ayrıca kanıtın tutarlılığını ve ankora bağlı olmayı denetler.

Bu, dar problem ailesi için güçlü haricî bilgi içeren bir denetleyicidir. Cevabı adayın yerine yazmaz. Dahası, çıkartılan metin yanlışsa doğru çalışan denklem denetleyicisi de özgün metne göre yanlış bir cevabı kabul edebilir. Bu yüzden `VERIFIED_PARSED_CONSTRAINTS`, kaynak metnin doğru yorumlandığını iddia etmez. Kabul kapsamı, kabul edilenlerde kaynak hatası ve gerçekten yürütülen tur sayısı birlikte raporlanır. Sıfır kabul, sıfır risk kanıtı değildir.

## Bellek ve sonraki eğitim

Ham kayıt, vektör dizini ve aktif özet farklı nesnelerdir. Benzerlik sırası bir kaydın aynı görev, başlık ve güncel sürüm için geçerli olduğunu kanıtlamaz. Bu deneydeki karma tabanlı sözcük vektörü, öğrenilmiş anlamsal embedding olarak sunulmaz. Ana bağlı sistemin tam geçmiş erişimi anahtar/sürüm temellidir; vektör aramasının başarısızlığını gizleyen bir vektör başarı sonucu değildir.

Doğru ara değerlerin eğitim hedefi olması ile modelin gerçekten ürettiği başarılı izlerin sonraki eğitimde kullanılması da ayrı deneylerdir. Birinci deney prosedürel ara gözetimi; ikinci deney üretilmiş, doğrulanmış yerel izlerden sonraki ağırlık güncellemesini sınar. Hiçbiri tek başına genel bir düşünme üslubunun her probleme aktarılacağını göstermez.

## Otuz risk başlığının kapsam karşılığı

| Dosya maddeleri | Bu araştırmanın doğrudan ölçümü veya sınırı |
|---|---|
|1,9,12,29: sürüklenme, kendi hatasını büyütme, uzun ufuk, gürültü | Uzunluk ve kontrollü durum gürültüsü testleri; belirli bir bozunma nedenini tek başına teşhis etmez. |
|2,10,11,20: ezilme, unutma, kapasite, girişim | Başlık sürümleri, ham arşiv, etkilenen/etkilenmeyen revizyon ve pencere deneyleri; genel bilgi kapasitesi kanıtı değil. |
|3,4,5: sönme, patlama, hata atfı | Gradyan/sonlu fark, eğitim normları, prefix ve final-only tam/kesilmiş eğitim. |
|6,8,14,26: ardışıklık, hesaplama, prefill, donanım | Gerçek çalıştırma süreleri, eşlenik no-feedback kontrolü, encoder/decoder maliyetleri; iş yükü ve paralellik farkı açık. |
|7,27: eğitim belleği ve yeniden hesaplama | Gerçek backward/AdamW eşdeğerliği, işlem tepe belleği ve zaman ölçümü; işlem belleği aktivasyon belleğiyle eşitlenmez. |
|13,30: fazla düşünme ve durma | Kimlik tokenları, gerçek başlık turları, kabul kapsamı/hatası ve yürütülen iş. |
|15,16,18: önbellek, eğitim/çıkarım, reset | Akış/toplu çıktı-gradyan eşliği, KV saatleri, istek izolasyonu ve yeniden oynatma; üretim RL önbellek yönetimi değil. |
|17: RL yeniden oynatma | RL optimizasyonu ve değişen politika ağırlıkları bu deneyde ölçülmedi. |
|19: temsil çökmesi | Uzun ufuk hatası tek başına çökme kanıtı değildir; genel temsil-geometri teşhisi ölçülmedi. |
|21,22,28: feedback konumu, bant genişliği, başlatma | Birkaç belirli feedback müdahalesi var; bu tasarımlar üzerinde kapsamlı eniyileme taraması yapılmadı. |
|23,24,25: ölçek yasası, bilgi kapasitesi, büyük ölçek | Küçükten1.5Mparametreye yerel duyarlılık ölçümü; ölçek yasası, yeni bilgi kapasitesi ve büyük dil modeli sonucu yok. |

Bu tablo, bir araştırma riskini araştırmaya dahil etmek ile o riski ortadan kaldırmayı ayrı tutar. Sonuç raporundaki olumlu, olumsuz ve karışık hükümler yalnız ilgili ölçüm kapsamı için geçerlidir.
