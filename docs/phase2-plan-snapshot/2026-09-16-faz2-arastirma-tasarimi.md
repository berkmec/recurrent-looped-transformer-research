# RLT Faz 2 araştırma tasarımı

**Durum:** PLAN_HAZIR; yeni model indirme, eğitim, veri kümesi çalıştırma veya teorem kanıtlama yapılmadı.
**Amaç:** Üç ayrı açık iddiayı, başarısız olabilecek hipotezler ve önceden belirlenen karar kurallarıyla sınamak.
**Kapsam:** 3 ana hat, 12 deney paketi; ayrıca ortak G0 hazırlığı.
**Başlangıç kanıtı:** E:/rlt-research/2026-09-16-e2e. Kapanış commit'i 3c118c7, teslimat manifest SHA256 değeri 146eb693e3303e17b1caa6bba2b1eb7662e20d87bb0ff3e71bbdaeb1e54484af. Bu plan hazırlanırken manifest ve doğrulama kaydının eşleşmesi yeniden okundu; önceki 9.073 dosyanın tamamı bu tur yeniden hashlenmiş gibi sunulmaz.

## 1. Üç iddia ve uygun kanıt türü

| İddia | Faz 2'de sınanabilir karşılık | Yetersiz kanıt |
|---|---|---|
| Serbest doğal dilde genel akıl yürütme | Birbirinden farklı beş görev ailesinde, yeni ifade/konu/yazar/bileşimlerde; aynı kaynak bilgisi ve hesap bütçesiyle; dış doğrulamaya dayalı aktarım | Birkaç iyi sohbet, yalnız sentetik şablon, tek benchmark, değerlendiren modelin beğenisi |
| Güvenilir anlamsal bellek | Gerekli kanıtları bulmak, hangi sürümün ve varsayımların uygulanacağını ayırmak, kaynak üzerinden doğru cevap vermek; risk ve kapsama birlikte | Yakın vektör, güzel bir özet, anahtarla bulunan doğru kaydı vektör başarısı saymak |
| Sınırsız doğru düşünme | Önce sonlu ufukta güvenilirlik eğrisi; ayrıca tanımlı durum/görev alanında tüm adımlar için korunan invariant veya kabul edilen cevap için biçimsel kanıt | 65.536 adımı geçmek, küçük Jacobian, değişmeyen cevap, genel bir LLM'nin kendini doğrulaması |

Doğal dilde başarı ifadesi her zaman test edilen görev/dil/dağılım kapsamını taşır. Evrensel görev yeterliliği sonucu çıkarılmaz. Biçimsel doğruluk, her sorunun çözüleceğini veya çözümün sonlu sürede bulunacağını otomatik olarak kapsamaz.

## 2. Önceki sonuçların belirlediği araştırma açıkları

1. Yeni cümle düzenlerinde ayrıştırma başarısızdı. D1'in ilk işi metnin anlamını ve yanlış anlamayı ayırt etmektir.
2. Karma sözcük dizininin recall@5 değeri %4,17'ydi. B1, gerçek anlamsal embedding ve yeniden sıralamayı doğrudan ölçer.
3. Başlık grafiği benzer görevde yararlıydı; yeni bağımlılık derinliğinde tek başına yeterli değildi. D2/D3 eşit bütçe ve yapısal aktarımı ayırır.
4. Doğru ara hedefler ile modelin ürettiği izlerin sonraki eğitime katkısı aynı şey değildi. D4 öğretmen hatalarını ve hedef bütçesini ayrı kontrol eder.
5. Son z durumunu saklamak işe yaradı; kısıt özetinden yeni sinirsel başlangıç kayıp verdi. B3 semantik özetin kaybını geri açma ve ham kaynakla ölçer.
6. Rastgele uzun parite dizilerinde başarı, nötr son eklerde dayanıklılığı garanti etmedi. U1/U2 bu ayrımı temel başarısızlık testi yapar.
7. Son-cevap gözetimi öğrenemedi. D4'te gecikmiş hata atfı, yoğun ara hedefler ve eğitim bütçesi ayrı faktörler olur.

## 3. Mimari yön ve alternatifler

**Öneri: ölçüm köprüsü, ardından gerçek mimari deneyi.** İlk aşamada mevcut bir yerel dil modeli sabit tutularak başlık, grafik, bellek ve doğrulayıcının katkısı ölçülür. Sonra gerçek RLT geçişi ve başlık geri beslemesi içeren bir dil modeli ayrı eğitilir. Biçimsel güvence dar görevler üzerinde ayrı yürür.

Yalnız hazır dil modelini bellekle birleştirmek ucuz bir sistem deneyi sağlar, fakat RLT'nin dil başarısını göstermez. Hemen büyük bir RLT'yi sıfırdan eğitmek mimari iddiaya daha doğrudan gider, ancak ayrıştırma/erişim/gözetim hatalarını pahalı bir eğitim içinde karıştırır. Yalnız biçimsel sonlu durum görevleriyle ilerlemek güçlü teorem sağlayabilir, fakat doğal dil genellemesi açık kalır.

### Üç ayrı nesne

- **Sistem katmanı:** önceden eğitilmiş dil modeli + sürümlü başlık grafiği + arşiv + erişim + görev doğrulayıcısı. Buradaki başlıklar modelin gerçek gizli durumunu değiştirmiyorsa bu açıkça dış düzenleyici olarak adlandırılır.
- **Gerçek RLT modeli:** H_t=(s_t,C_t^D), nedensel encoder belleği, gerçek token geçişi; ayrı başlık durumu Z ve k güncellemesi. Başlık müdahalesi sonraki model durumunu gerçekten etkiler. D4 bu iddiayı sınar.
- **Kanıt sistemi:** modelden bağımsız olarak biçimselleştirilmiş iddia ve kanıtı kontrol eder. Yalnız modelin kendi güven puanını okuyarak VERIFIED vermez.

Huginn, önceden eğitilmiş yinelenen derinlik dil modeli olarak ilgili bir karşılaştırmadır; özgün RLT ile aynı mimari değildir. Hazır Huginn başarısı bizim RLT eğitimimizin yerine geçmez. [Model kartı](https://huggingface.co/tomg-group-umd/huginn-0125), [makale](https://arxiv.org/abs/2502.05171).

### Başlık ve bellek sözleşmesi

Başlık B_i=(id,v,goal,claims,evidence,deps,raw_refs,state_ref,summary,open_obligations,status) olarak izlenir. id sabit kimlik, v sürüm, deps tipli bağımlılıktır. state_ref yalnız gerçekten saklanan model durumu varsa kullanılır; metin özetine latent durum adı verilmez.

Ham arşiv eklemelidir. Özet ve vektör dizini ham arşivden türeyen ayrı nesnelerdir. Yeni kanıt A^0→B^0→A^1 biçiminde yeni sürüm doğurur; önceki gözlem veya token çıktısı geriye dönük yeniden yazılmaz. Döngülü anlam ilişkileri sürüm ve tur üzerinden yürütülür.

Bellek kaydının türü fact veya method olabilir. Bir yöntemin önceki göreve benzemesi, yeni görevin varsayımlarına uygun olduğunu kanıtlamaz. Yöntem bağlanırken giriş türleri, birimler, amaç, öncüller ve geçerlilik alanı denetlenir.

### Ortak veri arayüzü

Her Problem kaydı: id, family, language, source_group, split, question, observations, as_of, allowed_tools. Gold cevaplar, kanıt etiketleri ve değerlendirme kuralları model girdisinden ayrı tutulur.

Her Evidence kaydı: record_id, raw_sha256, source_span, heading_id, semantic_version, valid_from, valid_until, record_kind, scope, prerequisites. Yakınlık puanı ayrı alandır; applicability veya doğruluk etiketi olarak kullanılmaz.

Her Run kaydı: protocol_sha256, model/tokenizer/quantization hashes, problem_id, arm, seed, t, k, d, candidate, accepted, status, citations, actual_model_work, wall_seconds, peak_RAM/VRAM, raw_trace_refs. Yeniden denemeler, yeniden okuma ve doğrulayıcı maliyeti hesaba dahil edilir.

Durumlar CANDIDATE, VERIFIED_WITH_SCOPE, CONTINUE, UNRESOLVED_BUDGET, STALLED, COMPACTION_REJECTED, INVALID_EVIDENCE. Sıfır kabulde kabul edilenlere ait hata oranı null'dır.

## 4. Ortak G0: deney açılmadan önce

- [ ] Eski manifesti ve bilimsel kaynak otoritesini doğrula; yeni çalışmayı ayrı E: kökünde kur.
- [ ] Model, tokenizer, veri sürümü, lisans, kod ve nicemleme karmalarını sabitle. Bir model kartının güncel ana dalını değişmez sürüm sanma.
- [ ] Her veri üreticisinde kaynak grup/kişi/konu/kanıt örtüşmesini önleyen train/dev/validation/final ayrımı oluştur. Çeviri ve parafrazlar aynı grupta kalır.
- [ ] Değerlendirme cevapları ve has_answer benzeri alanları model girdisinden kaldır; oracle erişimini ayrı tanı kolunda tut.
- [ ] 32 geliştirme problemiyle zaman/RAM/VRAM kalibrasyonu; 200 geliştirme problemiyle veri ve ölçüm pilotu yap. Bunlar doğrulayıcı son test değildir.
- [ ] Yeni yapı için ayrık anlamlı negatif kontrolleri, model çağrısı/t/k sayaçlarını ve tam iş maliyetini doğrula.
- [ ] A/A tekrarlarıyla sistem oynaklığını ölç; eşlenik koşulları AB/BA sırayla çalıştır. Üç eğitim tohumu gerekirse 17/43/91; aynı görevde tekrar üretilen cevaplar bağımsız veri sayılmaz.
- [ ] Son test açılmadan iddia, birincil karşılaştırma, karar eşikleri, süre sınırı, örnek sayısı ve başarısızlık kaydı dondurulsun.

Bu turda G0 çalıştırılmadı; yalnız plan için donanım ve eski manifest kimliği okundu.

## 5. On iki deney ve bağımlılıklar

| ID | Deney | Ön koşul | Temel çıktı |
|---|---|---|---|
| D1 | Serbest ifade → başlık, iddia ve kaynak bağı | G0 | Anlam koruma / yanlış yorumlama ölçümü |
| D2 | Beş doğal dil görev ailesinde eşit bütçe karşılaştırması | D1, B2, B3 | Genel görev aktarımı için kapsamlı sistem kanıtı |
| D3 | Yeni konu, dil, ifade ve bileşim | D2 | Dağılım dışı genelleme ve kötüleşme sınırı |
| D4 | Gerçek RLT dil eğitimi ve başlık geri beslemesi | D1, B2, U1 | Sistem katmanından ayrı mimari/eğitim kanıtı |
| B1 | Gerçek anlamsal erişim | G0 | Recall, bütün kanıt kümesi ve maliyet |
| B2 | Sürüm, zaman ve uygulanabilirlik | B1 | Benzer kaydı geçerli kanıttan ayırma |
| B3 | Sıkıştırma, geri açma ve seçici revizyon | D1, B2 | Kaynak anlamını ve eski cevapları koruma |
| B4 | Kalıcı bellek ölçeği ve yeniden başlatma | B3 | Veri bütünlüğü ve uzun oturum performansı |
| U1 | t/k/d ve bilgi ihtiyacını ayrı büyütme | G0 | Doğruluk–hesaplama ve etkin ufuk eğrisi |
| U2 | Sürüklenme nedeni ve kontrollü düzeltme | U1 | Karşı-olgusal ve nedensel müdahale kanıtı |
| U3 | İnvariant veya denetlenebilir cevap kanıtı | U1 | Dar alanda bütün adımları kapsayan koşullu teorem |
| U4 | Bütün sistem için son bağımsız doğrulama | D3, D4, B4, U2, U3 | Üç iddia için ayrı son hükümler |

~~~mermaid
flowchart TD
 G0[Ortak veri ve ölçüm hazırlığı] --> D1[D1: Doğal dil anlamı]
 G0 --> B1[B1: Anlamsal erişim]
 G0 --> U1[U1: Uzun ufuk ölçümü]
 B1 --> B2[B2: Geçerli kanıt]
 D1 --> D2[D2: Doğal dil görevleri]
 B2 --> D2
 D2 --> D3[D3: Yeni dağılımlar]
 D1 --> B3[B3: Özet ve revizyon]
 B2 --> B3
 B3 --> D2
 B3 --> B4[B4: Bellek ömrü]
 D1 --> D4[D4: Gerçek RLT dil eğitimi]
 B2 --> D4
 U1 --> D4
 U1 --> U2[U2: Sürüklenme düzeltmesi]
 U1 --> U3[U3: Biçimsel güvence]
 D3 --> U4[U4: Bağımsız son doğrulama]
 D4 --> U4
 B4 --> U4
 U2 --> U4
 U3 --> U4
~~~

## 6. Veriler, karşılaştırmalar ve eşit bütçe

Beş birincil aile: çok belgeli çıkarım; sayısal/kısıtlı problemler; doğal dil belirtiminden test edilebilir kod; çok oturumlu zamansal bilgi güncelleme; doğrulanabilir eylem sırasına sahip planlama. Türkçe ve İngilizce dilimler ayrı raporlanır.

Her aile için geliştirme ve seçime ayrılmış 100+100 kaynak grubu; bağımsız son test için hedef 600 kaynak grubu. Toplam son test hedefi 3.000 bağımsız problemdir. Yazar, kaynak ve parafraz grupları ayrılır. Beş ailenin Türkçe/İngilizce yarıları ayrı birer %1 risk garantisi almaz; alt dilimlerin örnek sayısı ayrıca yeterli olmalıdır.

Doğal dil özelliği yalnız kelime değiştiren bir üreticiden gelmez. Yeni yazarların serbest ifadeleri, gerçek belge soruları, belirsizlik/eksik bilgi ve çözümü değişen karşı-olgusal çiftler bulunur. Model yazımı veriler ayrı etiketlenir. İnsan doğrulaması gereken örneklerde iki modelin anlaşması insan anotasyonu diye raporlanmaz. Gerekli insan çalışması ve zamanı maliyete dahildir; otomatik testin bunu tamamladığı iddia edilmez.

MuSiQue, BigCodeBench ve temizlenmiş LongMemEval dış çapalar olur. Eğitim geçmişine karışmış olabilecek açık benchmark başarısı taze kör testin yerine geçmez. Güncel LongMemEval-V2-small daha sonraki bellek/entegrasyon dış doğrulamasında kullanılır; görüntüyü veya uzun geçmişi çıkarılmış bir sürüm, tam V2 skoruyla karşılaştırılmaz. Resmî değerlendirme bir dış model hakemi kullanıyorsa eşdeğer bir yerel hakem sonucu resmî skor sayılmaz. [MuSiQue](https://github.com/StonyBrookNLP/musique), [BigCodeBench](https://github.com/bigcode-project/bigcodebench), [LongMemEval](https://github.com/xiaowu0162/LongMemEval), [V2](https://github.com/xiaowu0162/LongMemEval-V2).

Birincil sistem karşılaştırması:
- C0: aynı kanıt paketiyle güçlü düz çözüm/öz denetim.
- C1: sözcüksel ve anlamsal karma erişim, düz çözüm.
- C2: karma erişim + yeniden sıralama, düz çözüm; **ana rakip**.
- C3: C2 ile aynı kanıt paketi, başlık grafiği + sürümlü revizyon + kontrollü pencere; **aday sistem**.
- Ablasyonlar: yanlış grafik; geri besleme kapalı; yeniden okuma kapalı; ham geçmişe dönüş kapalı. Tek müdahale başına ayrı karşılaştırma.
- Oracle kanıt/ayrıştırma: yalnız arızanın hangi modülden geldiğini bulmak için; ana başarı tablosuna girmez.

C3/C2 karşılaştırmasında model, nicemleme, görülen kanıt, üretim ayarları ve toplam süre tavanı eşlenir. Birincil karşılaştırmada kaynak paketleri ve varsa zamana bağlı güncellemeler iki kola aynı sırada verilir; C3'e özel ek kanıt verilmez. Canlı arama/ham geri açma seçimlerinin farklılaştığı bütünleşik kol ayrıca raporlanır ve kazancı yalnız grafiğe atfedilmez. C2/C1 yeniden sıralama katkısını, B1 erişim bileşenlerini, B3 ise sıkıştırma/geri açma katkısını ayırır.

İki bütçe noktası: geliştirmede C2'nin ölçülen medyan uçtan uca süresinin 1 ve 2 katı; sayısal saniye değerleri son testten önce sabitlenir. Birincil üstünlük ve beş ailelik ortak risk iddiası 1× bütçede sabit C3 sistemi içindir; 2× bütçe ikincil kalite–maliyet eğrisidir. İki bütçede birden ortak risk güvencesi istenecekse on hücreye göre hata bütçesi/örnek sayısı G0'da yeniden belirlenir; testte daha iyi görünen bütçe seçilmez. İlk yerel pilotta çağrı başına bağlam üst sınırı 8.192, toplam üretilen token üst sınırı 1.024; bütün çağrıların tekrar işlediği tokenlar da kaydedilir. Bunlar modelin ilan edilen azami bağlamı değil, başlangıç deney sınırlarıdır. Yerel tur eşitliği tek başına FLOP eşitliği değildir. Erişim, yeniden sıralama, yeniden okuma, doğrulama ve disk maliyeti toplam süreye dahildir.

## 7. Önerilen karar eşikleri — henüz ölçülmedi

**Doğal dil ilerleme hedefi:** C3, C2'ye karşı eşit bütçede makro son-cevap doğruluğunda en az 5 yüzde puanı kazanç; eşlenik güven aralığının alt sınırı sıfırın üstünde; hiçbir ailede 2 puandan büyük kötüleşmenin dışlanamadığı durum başarı diye kapatılmaz. En az %80 cevap kapsamı ile kabul edilen cevap riskinin üst güven sınırı en çok %1 ayrıca aranır. Başarı ve güvenilirlik ayrı kapılardır; biri geçip diğeri kalabilir.

**Anlamsal erişim hedefi:** recall@10 en az %95; çok kanıtlı sorularda gerekli kanıtların tamamını ilk 10 içinde bulma en az %90. Her iki metriğin de kaynak gruplu belirsizliği raporlanır. Güncel/uygulanabilir kanıtla cevapta risk üst sınırı %1 ve kapsam en az %80. Yapısal kapsam/sürüm ihlalinin herhangi bir gözlenmiş kabulü, ilgili mühendislik kapısını kapatır.

**Sıkıştırma hedefi:** eşit aktif bütçede tam ham erişim karşısında son-cevap farkının tek taraflı güven alt sınırı -1 yüzde puanının üstünde; hedef bağımsız problemler üzerinden. Yalnız son z'yi geri yüklemenin eşliği, serbest dil özeti için bu kapıyı geçirmez. Seçici revizyonda etkilenen cevapların hata kaybında en az %10 göreli azalma, her tohumda olumlu yön, etkilenmeyenlerin hata kaybında en çok 1,05 oranı ve toplam hata kaybında azalma aranır. Bunlar süre kazancı değildir. Eski MSE görevi MSE ile; dil görevi önceden sabit 0/1 hata veya görev kaybıyla ölçülür ve iki ölçek birleştirilmez. Referans hata sıfırsa oran tanımsızdır; yeni hata olmaması raporlanır, %10 iyileşme iddiası kurulmaz.

**Uzun ufuk hedefi:** önceden sabit görev ailesinde eğitim ufku 128 iken 4.096'a kadar her test noktasında en az %95 son-cevap doğruluğu; nötr uzatmada en çok 2 puan kayıp; k büyüdükçe en iyi cevaptan geri dönüş oranı ve ek iş başına kazanç ayrıca raporlanır. 16.384 ve 65.536 noktaları kalibrasyonda yapılabilir bulunursa son testten önce genişletilmiş ızgaraya alınır; sonradan başarı görülen nokta seçilmez.

Bu yüzdeler doğa yasası veya yayımlanmış optimum değildir; araştırma için önerilen güçlü hedeflerdir. Pilot fizibiliteyi belirler; final sonuçlarına bakılarak değiştirilmez.

### Risk iddiası için örnek sayısı

n bağımsız kabulde sıfır hata görülürse tek taraflı binom üst sınır p_U=1-alpha^(1/n). %95 güven ve %1 üst sınır için en az 299 kabul gerekir. Beş aile birlikte değerlendirildiğinde aile başına alpha=0,01 kullanılır; her ailede en az 459 hatasız kabul gerekir. 600 problemin %80'ini kabul etmek 480 kabul verir; sıfır hata koşulunda üst sınır yaklaşık %0,955'tir.

Bu hesap bağımsız/problem-gruplu ve aynı dağılımlı örneklem varsayımına dayanır. Aynı belgenin parafrazları veya aynı görevin üç model tohumu n'yi üç kat yapmaz. Birden çok bağımlı soru varsa kaynak grubu üzerinden koruyucu hata olayı ve grup analizi kullanılır. Hata görülürse sıfır-hata formülü kullanılmaz; genel binom/uygun grup aralığı hesaplanır. Beş-aile düzeltmesi bütün araştırmadaki tüm sonradan seçilmiş hipotezler için toplu güvence değildir.

Birincil üstünlük karşılaştırmaları ve aileler testten önce dondurulur; çoklu karşılaştırmalar için Holm düzeltmesi uygulanır. Üç eğitim tohumu tek başına güçlü nüfus genellemesi sağlamaz; tohumlar ayrı verilir, geliştirmede ölçülen değişkenliğe göre gerekli tekrar sayısı ve azami bütçe son testten önce belirlenir.

## 8. Sonsuz adımlara yaklaşan koşullu matematik hedefi

Referans doğru durum q_t, gerçek durum h_t ve hata e_t=dist(h_t,Enc(q_t)) olsun. Her izin verilen giriş ve doğrulanmış durum bölgesinde

e_(t+1) ≤ rho e_t + delta, 0 ≤ rho < 1

kanıtlanabiliyorsa, indüksiyonla

e_T ≤ rho^T e_0 + delta(1-rho^T)/(1-rho)

elde edilir. Her adımda doğru karar bölgesine uzaklık marjı m ise max(e_0,delta/(1-rho)) < m yeterli bir hedef koşuldur.

Burada koşulun **her izin verilen giriş ve erişilebilir bölge** için geçerli olduğu kanıtlanmalıdır; birkaç örnekte ölçülmüş Jacobian buna yetmez. Bölgeler doğru ayrık durumlara göre ayrılabilir. Kimlik geçişi iki farklı doğru durumu saklıyorsa, ikisini kapsayan tek bir global daraltıcı dönüşüm varsaymak uygun değildir. delta sayısal yuvarlama/izin verilen gürültüyü, durum tanımı da KV/global bellek ve dış arşiv etkilerini kapsamına uygun biçimde hesaba katmalıdır.

U3'ün ilk hedefi sonlu durum ailesi veya biçimsel kısıt sistemidir. Kanıt hedefi öğrenilmiş RLT'nin sabit ağırlık/geçiş kimliğine bağlanmalıdır. Yalnız referans otomata veya elle tasarlanmış başka bir geçiş kanıtlanırsa sonuç REFERENCE_ONLY olur; öğrenilmiş RLT için bütün-adım doğruluğu açık kalır. Parite için kurulan invariant, belleği problemle büyüyen yığın veya genel doğal dil için teorem olmaz. Kabul edilen bir ispatın denetlenmesi ayrıca çözüm bulmayı ve doğal dil öncüllerinin doğruluğunu garanti etmez. Lean çıktısında doğru hedef, bağımlı aksiyomlar ve bağımsız yeniden kontrol birlikte gerekir. [Lean doğrulama belgesi](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

## 9. Kaynaklar ve yürütme sınırları

16 Eylül okumamız: RTX 2060 SUPER, toplam 8.192 MiB VRAM; anlık boş 7.142 MiB; yaklaşık 16 GiB sistem RAM'i; C: boş 2,54 GiB, E: boş 58,77 GiB. Boşluk ve yük yürütmeden önce yeniden okunur.

Yerel başlangıç adayları: Qwen3-4B için doğrulanmış düşük-bit çıkarım; Qwen3-Embedding-0.6B ve Qwen3-Reranker-0.6B için ayrı yükleme. Bunlar en iyi model iddiası değil, ölçülebilir başlangıç adaylarıdır. Reranker/embedding/cevap modelini aynı anda VRAM'de tutmak varsayılmaz. Nicemleme ve batch değişikliği ayrı sayısal kontrol gerektirir. [Cevap modeli](https://huggingface.co/Qwen/Qwen3-4B), [embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

100.000 adet 1.024 boyutlu float32 vektörün yalnız ham yükü 409.600.000 bayt, yaklaşık 390,6 MiB'dir. Metin, HNSW/grafik, metadata, çalışma tamponu ve model ağırlıkları buna dahil değildir. B1'de önce tam benzerlik araması referansı; yaklaşık indeksin kendi kaybı ayrıca ölçülür.

Huginn kartı bfloat16 değerlendirme ve özel model kodu belirtir. Bu donanımda destek/VRAM/performans canlı ön kontrol olmadan varsayılmaz; farklı hassasiyet veya offload deneyi özgün kart sonucunun tekrarı sayılmaz. Doğal dil eğitimi için 30M→100M→300M gibi bir yerel boyut merdiveni yalnız fizibilite adayıdır; sonraki boyut öncekinin dil/yeterlilik ve kaynak kapısını geçmesine bağlıdır. Bu büyüklükler genel dil başarısı garantisi değildir.

Saat/gün tahmini ölçülmeden verilmez. T_toplam yaklaşık N_sorgu × N_kol × N_tekrar × ölçülen_saniye + indeksleme + eğitim + değerlendirme/insan anotasyonu olur. Yalnız D2'nin 3.000 problem ×2 ana kol ×2 bütçesi 12.000 bölüm yürütmesidir; C0/C1, ablasyon, tohum ve dış benchmark tekrarları buna eklenecektir. Bu büyüklük, birkaç örneklik yerel gösterimle aynı maliyette değildir. 32 örneklik pilot ölçümüyle süre ve disk tahmini hazırlanır. GPU işleri seri, hafif CPU ölçümleri kaynak elverdiğinde ayrı yürür. Modeller ve yeni cache E: üzerinde tutulur; eski Python ortamları değiştirilmez. Büyük eğitim veya ücretli altyapı için gerçek model/veri/bütçe paketi oluşmadan belirsiz kaynak taahhüdü verilmez.

## 10. Kapanış koşulu ve başarısızlıktan sonra karar

Her paket için uygulama durumu, deney tamamlanması, hipotez sonucu ve dış geçerlilik ayrı tutulur. Measured-positive, mixed, measured-negative, inconclusive ve unmeasured ayrımı korunur.

D1 kötü ise D2'de oracle ayrıştırma yalnız arıza tespiti içindir; bu çıktı doğal dil başarısı olmaz. B1/B2 kötü ise doğru anahtar erişimiyle elde edilen sonuç anlamsal erişim iddiasını kurtarmaz. D4'nin dil yeteneği taban eşiğe erişmezse grafik yararı hakkında güçlü olumsuz mimari sonuç da çıkarılmaz: sonuç INCONCLUSIVE_CAPABILITY_FLOOR olur. U3 kanıtlanamazsa deney eğrileri korunur, sonsuz-adım iddiası UNPROVED kalır.

Faz 2'nin teslimatı yeni protokoller, veri/model/source karmaları, kaynak gruplu ham çıktılar, tüm maliyetler, bağımsız yeniden hesaplama, modelden ayrı doğrulayıcı ve üç iddia için ayrı rapordur. Son rapor ancak gerçekte geçilen kapıları desteklenmiş sayar.
