# RLT Faz 2 — üç iddia için doğrulama planı

Durum: **PLAN_HAZIR / DENEYLER_BASLAMADI**. Tarih: 16 Eylül 2026.

Bu paket, tamamlanan on dört deneyin ardından üç açık iddiayı araştırılabilir hale getirir: doğal dilde görevler arası genelleme; doğru ve uygulanabilir geçmiş bilgiyi bulma; uzun hesaplama boyunca doğruluğu koruma. On iki yeni deney paketi vardır: D1–D4, B1–B4 ve U1–U4. Ortak G0 hazırlığı bu sayıya dahil değildir.

**Önerilen ilk dalga: G0 → D1 + B1 + U1.** Yeni dil kalıplarını anlayıp anlamadığımızı, gerçek anlamsal erişimin zayıf sözcük dizininden iyi olup olmadığını ve önceki uzun-ufuk başarısının nerede bozulduğunu önce ölçeriz. Bu ölçümler pahalı dil eğitiminin hedefini belirler.

- [Ana araştırma tasarımı, matematik ve karar eşikleri](docs/superpowers/specs/2026-09-16-faz2-arastirma-tasarimi.md)
- [Doğal dil hattı: dört deney](docs/superpowers/plans/2026-09-16-dogal-dil.md)
- [Anlamsal bellek hattı: dört deney](docs/superpowers/plans/2026-09-16-anlamsal-bellek.md)
- [Uzun ufuk ve biçimsel güvence hattı: dört deney](docs/superpowers/plans/2026-09-16-uzun-ufuk.md)
- [Birincil kaynaklar ve seçim gerekçeleri](KAYNAKLAR.md)
- [Makinece okunabilir deney matrisi](deney-plani.json)

Başarı eşikleri aşağıda **önerilen, henüz ölçülmemiş hedeflerdir**. Kalibrasyon yalnız geliştirme verisinde yapılır; son test açılmadan model, veri, bütçe ve ölçütler dondurulur. Bir eşiği sonradan düşürmek yeni protokol ve yeni test gerektirir.

Hazır bir dil modeline başlık/bellek katmanı eklemek, gerçek RLT mimarisini doğal dile taşımak ve biçimsel bir doğruluk teoremi kurmak ayrı çıktılardır. Birinin başarısı diğerinin başarısı olarak raporlanmaz.

Sonlu testlerle bütün doğal dil görevlerinde ve sonsuz zamanda kusursuzluk gösterilemez. U3'te hedef, açık varsayımlar altında belirli bir görev ailesi için her adımı kapsayan bir koruma teoremi veya kabul edilen cevaplar için kontrol edilebilir kanıttır. Kaynak metnin doğru yorumlandığı ve kullanılan öncüllerin doğru olduğu ayrıca incelenir.

Önceki teslimat E:/rlt-research/2026-09-16-e2e içinde korunmuştur. Plan belgeleri ayrı köktedir; eski sonuç dizininde yeni deney çalıştırılmaz.
