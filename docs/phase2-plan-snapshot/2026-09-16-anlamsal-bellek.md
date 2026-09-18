# Anlamsal bellek araştırma hattı — uygulamaya geçiş planı

> Agentic workers: uygulama istendiğinde superpowers:subagent-driven-development veya superpowers:executing-plans ile görevleri sırayla yürütün. Bu belge araştırma planıdır; indeks/eğitim henüz başlamadı.

**Goal:** Anlamsal benzerliği, gerekli kanıtı bulma ve mevcut soruya geçerli biçimde uygulamadan ayrı ölçmek.
**Architecture:** Eklemeli ham arşiv; türetilmiş vektör/sözcük dizini; sürümlü başlık grafiği; uygulanabilirlik ve cevap doğrulama; dolunca sıkıştırma ve gerektiğinde ham kaynağa dönüş.
**Tech Stack:** Ayrı E: kökünde SQLite/ham dosyalar, çok dilli embedding, sözcüksel arama, yeniden sıralayıcı, tam arama referansı ve ölçüldükten sonra yaklaşık indeks.
**Spec:** [Ortak araştırma tasarımı](../specs/2026-09-16-faz2-arastirma-tasarimi.md)

## Global constraints

Ham kayıt fiziksel olarak silinmez. İndeks kaynak otoritesi değildir. Her aday kaynak/sürüm/varsayım kimliğini taşır. Yeni embedding veya tokenizer sürümü eski indeksle sessizce karıştırılmaz. Erişim ve son cevap doğruluğu ayrı raporlanır. Gold kanıt veya cevap etiketleri indekslenecek metinden çıkarılır. Eski araştırma kanıtı değişmez.

**Arayüz:** query(question, task_context, as_of, k, evidence_token_budget) bir EvidencePack döndürür. Paket retrieved_candidates, applicable_evidence, rejected_candidates_with_reason ve source_refs alanlarına sahiptir. Bu kavramsal veri sözleşmesi D2 tarafından tüketilir; henüz uygulanmış fonksiyon değildir.

## B1 — sözcük benzerliğinden gerçek anlamsal erişime

**Girdi:** Çok dilli, kaynak gruplu sorular; ilgili/eksik/çelişen belgelerin modelden gizli altın etiketleri.
**Çıktı:** Sıralamalar, gerekli kanıt kümesinin bulunma oranı ve maliyet.
**Planlanan dosyalar:** semantic_memory/retrieval.py, semantic_memory/embedding.py, semantic_memory/rerank.py, tests/test_retrieval_contract.py, protocols/B1.json.

- [ ] İlk geliştirme havuzunda aynı anlam/farklı sözcük, aynı sözcük/farklı anlam, birim farkı, yeni isim, ters ilişki ve Türkçe/İngilizce eşlemeler kur.
- [ ] Önceki 128 boyutlu sözcük karması, BM25, çok dilli embedding, karma arama ve karma+yeniden sıralama kollarını aynı parça bölme/kanıt bütçesinde karşılaştır.
- [ ] Başlangıç adayı Qwen3-Embedding-0.6B ve Qwen3-Reranker-0.6B. Sürüm, yönerge biçimi, pooling, normalizasyon, uzunluk kesimi ve nicemleme kaynak kartına bağlansın; isim kullanmak doğru entegrasyonu kanıtlamaz.
- [ ] Bir örneği tekli/batch kodlama ve tekrar çalıştırmada karşılaştır; sayısal farkın sıralama/cevap etkisini ölç. Exact cosine sıralaması yaklaşık indeks için referans olsun.
- [ ] 1.000 /10.000 /100.000 kayıt merdiveninde aynı soru gruplarını eşle. Korpus büyümesi ve çoklu tekrarları yeni bağımsız soru sayma.
- [ ] Geliştirmede yapılandırmayı seç; en az 1.000 ayrı kaynak gruplu taze son sorgu hedefle. Sıralama metriklerini cevap modeli çalışmadan hesapla.
- [ ] Recall@1/5/10, nDCG, gereken tüm kanıtı kapsama, bellek/indeksleme/sorgu zamanı ölç. Hedef recall@10 ≥%95 ve çok kanıtlı tam kapsama ≥%90; dil ve zor negatif dilimleri ayrıca ver.
- [ ] Rastgele kayıt, yanlış başlık bağı ve oracle kanıt kollarını ayrı göster. Oracle üst sınırı vektör erişimi başarısı sayılmaz.

## B2 — doğru sürüm ve uygulanabilir bilgi

**Girdi:** B1 adayları; zaman, görev, başlık sürümü ve yöntem öncülleri.
**Çıktı:** Uygulanabilir kanıt paketi, ret nedeni, kaynakla desteklenen cevap veya çözülmemiş durum.
**Planlanan dosyalar:** semantic_memory/applicability.py, semantic_memory/versioning.py, tests/test_applicability.py, protocols/B2.json.

- [ ] Güncel sürüm, iptal edilmiş sürüm, başka görev, çelişen kaynak, birim/öncül uyuşmazlığı ve hiç cevap bulunmayan durumu ayrı etiketle.
- [ ] Zor örnek: geçmişte “limit 5”, yeni kayıtta “limit 3”; benzerliği yüksek eski kaydın cevapta kullanılmasını reddet. Eski sorunun tarihine cevap isteniyorsa doğru geçmiş sürümü koru.
- [ ] Yöntem belleği için yeni göreve Bind adımını zorunlu kıl: varsayımları taşımayan benzer yöntemi yalnız örnek/aday olarak sun, çözüm kanıtı olarak değil.
- [ ] Alıntılanmış bellek metnindeki talimatı yeni kullanıcı isteği veya kontrol komutu sayma; kaynak içeriği ve görev yetkisi ayrı olsun.
- [ ] Salt benzerlik, metadata filtresi ve metadata+iddia/öncül kontrolünü ayrı kollar yap. Çok reddederek riski düşüren sistemin kapsamını saklama.
- [ ] En az 600 bağımsız zor negatif ve 600 cevaplanabilir olumlu grup hedefle; alt türleri dengeli raporla. Bu karışım gerçek kullanım prevalansı değildir.
- [ ] Yanlış kanıt kabul oranı, olumlu kapsam ve kaynakla doğru son cevap riskini ayrı hesapla. Kabul koşullu riskinde sıfır kabul null'dır.
- [ ] Her gözlenmiş kapsam/sürüm ihlali mühendislik kapısını kapatır. İstatistiksel güvenilirlik hedefi spec'teki %1 üst risk ve %80 kapsamdır; alt dilimlere yetersiz n ile aynı garanti verilmez.

## B3 — semantik sıkıştırma, geri açma ve revizyon

**Girdi:** D1 başlık/iddia grafiği, B2 geçerli kaynaklar, uzun oturum ham kaydı.
**Çıktı:** Sürüm bağımlı özet, açık yükümlülükler, ham kaynak geri açma ve yeni cevap.
**Planlanan dosyalar:** semantic_memory/compact.py, semantic_memory/reopen.py, semantic_memory/revise.py, tests/test_compaction_reopen.py, protocols/B3.json.

- [ ] Pencere dolana kadar gerçek üretilmiş iş kayıtları korunur; eşik doluluk temellidir. Token bağlamı, aktif başlık geçmişi, KV ve arşiv büyüklüğü ayrı ölçülür.
- [ ] Özet sonuçla birlikte varsayım, reddedilen yol, karşı kanıt, kaynak kimliği ve açık yükümlülükleri taşısın. Ham kaynak korunur; özetin kendi güveni yeterlilik sayılmaz.
- [ ] Tam ham erişim, erken/sürekli özet, yalnız dolunca özet ve dolunca özet+ham geri açma kollarını aynı aktif kanıt bütçesinde karşılaştır.
- [ ] Sonradan gelen kanıtın eski özeti geçersiz kıldığı çiftler kur; özetin atladığı küçük ayrıntıya bağlı yeni soru sor. Doğru davranış stale özeti reddedip ilgili ham aralığı okumaktır.
- [ ] Aynı başlıkta yeni sürümle devam ve gerçek yeni alt problemde yeni başlık oluşturmayı ayrı say. Gereksiz bölme ve farklı sorunları tek başlıkta toplama hataları raporlanır.
- [ ] Etkilenen ve etkilenmeyen soruların final hata kayıplarını birlikte ölç: etkilenenlerde en az %10 göreli hata azalması, her tohumda olumlu yön, etkilenmeyenlerin hata kaybı/baz hata kaybı ≤1,05 ve toplam hata azalması. Eski deneyin MSE'si ile dil görevlerinin 0/1 hatası ayrı raporlanır. Baz hata sıfırsa oran üretilmez; yeni hata olmaması denetlenir ve göreli iyileşme UNMEASURABLE olur. Bu kalite ölçütü süre tasarrufu değildir.
- [ ] Semantik özet için tam ham erişime karşı -1 puan non-inferiority sınırını gruplu aralıkla sına. Geri açma maliyeti kalite kazancından ayrı gösterilir.
- [ ] Son z snapshot eşliği bu deneyin kontrolüdür; doğal dil özeti başarısına çevrilmez.

## B4 — bellek ömrü, ölçek ve yeniden başlatma

**Girdi:** B3 bütünleşik bellek; tekrarlar, sürüm değişimleri, yeniden embedding ve program kesintileri içeren kayıt akışı.
**Çıktı:** Kalıcılık, güncellik, indeks kaybı/rebuild, toplam kaynak ve uzun oturum cevap kalitesi.
**Planlanan dosyalar:** semantic_memory/persistence_eval.py, semantic_memory/reindex.py, tests/test_recovery.py, protocols/B4.json.

- [ ] Ham arşivi yazıp indeks güncellenmeden kesilme; indeks yazılıp işlem yinelenme; metadata sürümü değişme koşullarını ayrı uygula.
- [ ] Yeniden başlatma sonrası ham kayıt kaybı0, yinelenen gözlemden sahte yeni sürüm0, eski indeksin güncelmiş gibi kullanımı0 hedefle. İdempotent replay sonucu raw SHA ve kaynak/sürüm kimlikleriyle karşılaştır.
- [ ] Arşiv büyüklüğü1k→10k→100k ve oturum sayısı10→100→1.000 merdivenini ayrı boyutlar olarak planla; her kesitte erişim ve son cevap kalitesini yeniden ölç.
- [ ] Öğrenilmiş embedding sürümü değiştiğinde ayrı namespace ve yeniden indeksleme yap. Aynı uzaymış gibi eski/yeni vektörleri karıştırma.
- [ ] LongMemEval temizlenmiş metin sürümünü dış çapa olarak kullan; LongMemEval-V2-small için veri/modality ve model/hakem maliyetini G0'da ölç. Kırpılmış geçmişi tam benchmark diye sunma.
- [ ] İndeksleme maliyeti, sorgu p50/p95, disk, RAM, VRAM ve cevap kalitesi birlikte raporlansın; kayıt sayısını sabitlemek toplam bellek sabitliği sayılmasın.
- [ ] Bağımsız denetçi kaynak/özet/indeks zincirini ve rastgele seçilmiş geri çağırmaları yeniden kurup bütünlük ile anlam başarısını ayrı kapatsın.

## Hattın teslimatı

Çalışan kanıt paketleme arayüzü, dondurulmuş erişim ve uygulanabilirlik testleri, korunan ham arşiv, yeniden başlatma kayıtları ve anlam temelli son cevap sonuçları. Şu an NOT_STARTED; yüksek retrieval skoru veya çalışan veri tabanı sonucu henüz yoktur.
