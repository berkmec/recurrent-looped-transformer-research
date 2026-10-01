# Gerçek üretilmiş durum penceresi

**Mekanizma sonucu: IMPLEMENTED_TESTED; yerel devam eşdeğerliği geçti.** Üç model ve iki problem ailesinde128'er ortak yeni örnek, iki bellek koşulu altında çalıştı:256 ayrı kaynak problemi,1.536 bölüm yürütmesi. Bu tamamlayıcı deney ana6.144 çalıştırmadan ayrı veri ve protokol kullanır. Protokol SHA256: `54e37e67b107af932ceb2a22381547885944cc1139e83210a2bd6d30107c9de9`.

Her sinirsel güncellemenin gerçek gizli durum yükü aktif geçmiş tamponuna girer. Tampon4'e ulaştığında son z durumu bir sınır kaydına yazılır ve gerçekten yeniden okunur. Model cevap adayını yeniden yüklenen durumdan hesaplar; yeterlilik kontrolü bundan sonra yapılır. Yetersizse aynı başlıkta devam edilir. Ham dört kayıt diskten silinmez. Her iki karşılaştırma kolunda da kontrol yalnız4/8turda yapılır; ana deneyin her turdaki uyarlanır durmasıyla aynı politika değildir.

## Bağımsız kanıt

Denetim12 koşulu ve6 eşleşmiş çifti tamamıyla kontrol etti.1.536 arşivde94.656 ham kayıt,21.504 kaynak gözlemi ve70.080 gerçek model izi var. Sıkıştırmalı kolda8.760 üretilmiş durum sınırı ve8.760 geri okuma bulunuyor. Eşleşmiş kolların **bütün üretilmiş iz yükleri, sayısal cevapları, durumları, bütçeleri ve tüketilen token sayıları aynı**. Sınırlar tam dört ardışık izden oluşuyor; başlık kimliği ve son z durumu korunuyor.

| Ölçü | Sıkıştırmasız | Dört kayıtlık pencere |
|---|---:|---:|
| Bir sorguda tepe aktif üretilmiş kayıt |8 |4 |
| IID ortalama tepe serileştirilmiş aktif iz yükü |54.648 bayt |28.005 bayt |
| Derinlik8 ortalama tepe serileştirilmiş aktif iz yükü |55.901 bayt |28.012 bayt |
| IID doğru aday |%98,44 |%98,44 |
| IID doğru ve doğrulamadan geçen son cevap |%31,77 |%31,77 |
| Derinlik8 doğru aday |%38,02 |%38,02 |
| Derinlik8 doğru ve doğrulamadan geçen son cevap |%0 |%0 |
| IID gözlenen ortalama bölüm süresi |0,530s |0,571s |
| Derinlik8 gözlenen ortalama bölüm süresi |0,546s |0,597s |

Bu durum koruma mekanizması doğruluğu artırmadı; mevcut modelin devamını korudu. Derinlik8'de adayların bir kısmı doğru olsa da yeterlilik sertifikası veren son cevap çıkmadı. Yanlış veya yetersiz durumun eksiksiz korunması, problemi çözmekle eşdeğer değildir.

## Ne azaldı, ne artabildi?

Aktif üretilmiş geçmişin tepe yükü yaklaşık yarıya indi. Bu sayı JSON ile serileştirilmiş iz yüküdür; süreç RAM'i değildir. Model ağırlıkları, canlı bütün-başlık z tensörü, grafik, token durumu, encoder/global bellek, KV önbellekleri, yükleme/kopyalama ve diğer çalışma nesneleri ayrı kalır. Karşı kol da geçmişi her sorgu başında sıfırlar; sınırsız bölüm geçmişiyle karşılaştırılmadı.

Ham izlere ek sınır kayıtları yazıldığı için bölüm arşivleri yaklaşık%21 büyüdü. Bu, ham geçmişi silmeme tercihinin gerçek maliyetlerinden biridir. Gözlenen tekil süre oranları0,951–1,219 aralığında; diğer yerel işlerle yarışma vardı, yalıtılmış hızlanma sonucu yok. Toplam deney868,19s, bağımsız denetim94,30s sürdü.

7.992 aynı-başlık devam sayacı, daha önce karşılaşılan sorgu başlığının sonraki sınırına erişmeyi sayar. Önceki sorguların yeniden oynatılması dahildir; yeni keşif veya yeni çözülmüş alt problem sayısı değildir.

## Eşdeğerliğin matematiksel kapsamı

Sabit K bağlamında yerel geçiş `z_next=U(z,K)` yalnız son z'ye dayanıyorsa, eski z geçmişini son z sınırıyla değiştirmek sonraki yerel geçişi koruyabilir. Deney bu dar sözleşmeyi uyguladı. K; ağırlıkları, geçiş ayarlarını, gözlemleri, grafiği, ilgili sürümleri, token/encoder/decoder durumunu ve önbellekleri içerir. Bunların değişmesi eski sınırın kabulünü engeller.

Bu, serbest dilde bütün düşüncenin kayıpsız anlamsal özeti veya bağımsız bir dosyadan sıfırdan bütün modeli yeniden başlatma kanıtı değildir. Tam K çalışma durumu arşivlenmiş z içinde bulunmaz. Bağımsız son denetim bütün K parmak izini yalnız z'den yeniden üretmedi; bu korumanın kapsamı ve doğru çağrı sırası sabit kaynak kodu ile ayar/sürüm mutasyon testleriyle ayrıca destekleniyor. Geri okuma sayacı sınırlarla eşleşir; ayrı bir donanım okuma olay kaydı yoktur.

Tam bağımsız denetim: `reports/reasoning-window-independent-audit.md`. Ham sonuçlar: `integration/reasoning-companion/results/metrics.json`. Bu sonuç, güncel afin kısıt özetiyle sinirsel geçmişi yeniden başlatan ana deneyin derinlik8 kaybını ortadan kaldırmış sayılmaz; iki bellek sözleşmesi ayrıdır.
