# Faz 2 için kontrol edilen birincil kaynaklar

Kontrol tarihi: 16 Eylül 2026. Bunlar seçim ve deney tasarımı kaynaklarıdır; bizim modelimizin başarısına kanıt oluşturmaz. Çalıştırma öncesinde kaynak/model/veri revision kimlikleri ayrıca sabitlenecek. Bu tur model veya veri indirilmedi.

1. [Özgün RLT, sabit kaynak commit'i](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md). Faz1'in mimari otoritesi; token boyunca yinelenen yolu başlık turundan ayırma gerekçesi.
2. [Qwen3-4B model kartı](https://huggingface.co/Qwen/Qwen3-4B). Sabit yerel cevap modeli için başlangıç adayı; 4B ve çok dilli model. Donanımımıza sığma, seçilen nicemlemenin doğruluğu veya bizim görevlerde başarı henüz ölçülmedi.
3. [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Çok dilli, 1.024'e kadar boyut; B1 için gerçek öğrenilmiş embedding adayı. Doğru query yönergesi/pooling ve model sürümü koşulları uygulanmalı.
4. [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Aday kayıtların yeniden sıralanması için ayrı kol. Embedding+reranker maliyeti ek olarak ölçülür.
5. [Qwen3 Embedding araştırma makalesi](https://arxiv.org/abs/2506.05176). Embedding/yeniden sıralama ailesinin yöntem kaynağı; yayımlanan benchmark sıralaması kendi veri dağılımımıza aktarılmış sonuç değildir.
6. [MuSiQue resmî depo](https://github.com/StonyBrookNLP/musique) ve [makale](https://arxiv.org/abs/2108.00573). Çok kaynaklı/çok adımlı sorular için dış çapa. Alt soruyu veya gold destek kimliğini modele sızdırmamak gerekir.
7. [BigCodeBench resmî depo](https://github.com/bigcode-project/bigcodebench). Doğal dil belirtimi, çoklu fonksiyon kullanımı ve test edilebilir kod sonuçları. Birim test kapsamı bütün program doğruluğuyla eşitlenmez.
8. [LongMemEval resmî depo](https://github.com/xiaowu0162/LongMemEval). Bilgi çıkarımı, çok oturumlu çıkarım, bilgi güncelleme, zamansal akıl yürütme ve çekimserliği ölçen metin çapasının temizlenmiş sürümü. Oracle/has_answer alanları yalnız denetçiye aittir.
9. [LongMemEval-V2 resmî depo](https://github.com/xiaowu0162/LongMemEval-V2). Erişim tarihinde 451 soru, web/enterprise alanları ve agent geçmişlerinden bellek değerlendirmesi bildiriliyor. Ölçekli ve çok kipli verisi nedeniyle yerel ilk pilotun otomatik varsayımı değil; B4/U4 için kaynak kontrollü küçük-tier dış çapa. Resmî hakem ve model seçimiyle yerel değiştirilmiş protokol ayrılır.
10. [Huginn model kartı](https://huggingface.co/tomg-group-umd/huginn-0125) ve [Recurrent Depth makalesi](https://arxiv.org/abs/2502.05171). 3,5B parametreli, token başına değiştirilebilen yineleme derinliği olan ilgili model. Kartın bfloat16, özel kod ve KV-cache ayrıntıları donanım ön kontrolünü gerektirir. Yifanzhang RLT ile aynı mimari değildir.
11. [Lean — Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/). Doğru formal hedef, bağımlı aksiyomlar ve bağımsız checker kapsamı için otorite. Bir yapının derlenmesi tek başına bütün doğal dil yorumunu doğrulamaz.
12. [Lean — Induction and Recursion](https://docs.lean-lang.org/theorem_proving_in_lean4/induction_and_recursion.html). Bütün adımlar için indüksiyonla invariant hedefinin yöntem kaynağı.

Ana tasarımdaki risk sınırı ve örnek sayısı hesabı doğrudan binom formülünden türetildi. 299, 459 ve 2.995 kabul sayıları bu tur sayısal olarak kontrol edildi; bunlar herhangi bir modelin ölçülmüş doğruluk sonucu değildir.
