"""Assemble the final Turkish research report only after independent closure."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def mean(values):return statistics.mean(values)
def pct(value):return f'%{100*value:.2f}'.replace('.',',')
def link(name,label):return f'[{label}]({(ROOT/name).as_posix()})'


def main():
    core=read('core/results/metrics.json');final=read('core/results/final-only-metrics.json')
    core_audit=read('audit/core-independent-audit.json')
    scale=read('integration/scale/metrics.json');scale_audit=read('audit/scale-independent-audit.json')
    connected=read('integration/connected/results/metrics.json')
    window=read('integration/reasoning-companion/results/metrics.json')
    learning=read('learning/results/closure.json');distill=read('learning/distillation/results/closure.json')
    audits=[read('audit/learning-independent-audit.json'),read('audit/distillation-independent-audit.json'),
            read('audit/connected-independent-20260916T175138.json'),
            read('audit/reasoning-companion-independent-20260916T180847.json')]
    assert core['status']==final['status']==scale['status']==window['status']=='COMPLETE'
    connected_events=[json.loads(line) for line in (ROOT/'integration/connected/results/execution.jsonl').read_text().splitlines() if line.strip()]
    assert connected_events[-1]['event']=='complete' and sum(e['event']=='complete' for e in connected_events)==1
    assert connected['condition_seed_episode_runs']==6144 and connected['independent_source_cases']==512
    assert len({(r['seed'],r['split'],r['condition']['name']) for r in connected['rows']})==24
    assert core_audit['status']==scale_audit['status']=='PASS'
    assert all(str(a['status']).startswith('PASS') for a in audits)
    assert len(core['training'])==len(core['evaluation'])==21 and len(final['records'])==6
    assert len(scale['training'])==len(scale['evaluation'])==6 and len(connected['rows'])==24
    assert len(window['rows'])==12 and len(window['comparisons'])==6
    assert all(all(r[k] for k in ['candidate_status_budget_identical','actual_refinements_identical',
                                  'actual_tokens_identical','all_reloads_bitwise_equal']) for r in window['comparisons'])
    assert distill['path_status']=='IMPLEMENTED_TESTED'
    assert learning['primary_corrected_models']+learning['combined_models']==21
    upstream=read('audit/upstream_postflight.json');assert upstream['status']=='PASS' and upstream['total_verified']==424
    suites=ET.parse(ROOT/'audit/final-tests-closure.xml').getroot()
    suite_rows=list(suites) if suites.tag=='testsuites' else [suites]
    assert sum(int(r.attrib.get(k,0)) for r in suite_rows for k in ['failures','errors','skipped'])==0
    tests=sum(int(r.attrib['tests']) for r in suite_rows)
    assert tests==108 and len(list(suites.iter('testcase')))==108,'full suite including the final report-encoding regression is required'
    assert not any(list(suites.iter(k)) for k in ['failure','error','skipped']),'non-passing testcase'
    final_accuracies=[e['final_accuracy'] for r in final['records'] for e in r['evaluation']]
    assert len(final_accuracies)==12
    final_range=f'{pct(min(final_accuracies))}–{pct(max(final_accuracies))}'
    benchmark=read('core/results/benchmark.json')['rows']
    peak_groups={}
    for row in benchmark:
        peak_groups.setdefault((row['loss'],row['mode']),[]).append(row['after']['peak_working_set_bytes']/1024**2)
    assert len(benchmark)==18 and len(peak_groups)==6 and all(len(v)==3 for v in peak_groups.values())
    peak_range=f'{min(mean(v) for v in peak_groups.values()):.1f}–{max(mean(v) for v in peak_groups.values()):.1f}'.replace('.',',')
    from audit.closure_pins import verify
    fresh_pins=verify(ROOT)
    assert fresh_pins['status']=='PASS'
    groups={}
    for row in core['evaluation']:
        c=row['condition'];groups.setdefault((c['task'],c['width'],c['mode']),[]).append(row)
    core_table=[];core_summary=[];neutral_table=[]
    for (task,width,mode),rows in groups.items():
        values={length:[next(x['final_accuracy'] for x in r['long_horizon'] if x['length']==length) for r in rows] for length in [32,128,1024]}
        label={'rlt':'RLT / tam','tbptt':'RLT / kesilmiş, 8 adım','nofeedback':'Geri beslemesiz'}[mode]
        name='Parite' if task=='parity' else 'Beş durum'
        core_table.append(f'| {name} / {width} / {label} | {pct(mean(values[32]))} | {pct(mean(values[128]))} | {pct(mean(values[1024]))} | {pct(min(values[1024]))}–{pct(max(values[1024]))} |')
        core_summary.append({'task':task,'width':width,'mode':mode,'mean_accuracy':{str(k):mean(v) for k,v in values.items()},'L1024_per_seed':values[1024]})
        if mode!='nofeedback':
            by_seed={r['condition']['seed']:r for r in rows}
            assert set(by_seed)=={17,43,91}
            assert all(next(n['final_accuracy'] for n in r['neutral_slots'] if n['slots']==0)==1 for r in rows)
            values128=[next(n['final_accuracy'] for n in by_seed[s]['neutral_slots'] if n['slots']==128) for s in [17,43,91]]
            neutral_table.append(f'| {name} / {width} / {label} | '+ ' | '.join(pct(v) for v in values128)+' |')
    packages=[
      (1,'RLT çekirdeği ve kontrol','IMPLEMENTED_TESTED','Gizli durum, yerel KV ve global encoder belleği; nedensellik ve akış/toplu eşlik. Bağımsız küçük uygulama.'),
      (2,'Uzun ufuk','MIXED','32 işlem uzunluğunda eğitimden sonra 32–1.024 testi; parite, beş durum ve eğitim tohumu ayrı. Sonlu test üst sınırı.'),
      (3,'Ek hesaplamanın yararı','MIXED','t/k/d ayrıldı. Bazı ek turlar yararlı; kimlik son ekleri ve tur eğrileri tekdüze kazancı desteklemiyor.'),
      (4,'Gecikmiş hata ve eğitim maliyeti','MIXED','Prefix öğrenimi ile son-cevap gözetimi ayrıldı; son-cevap koşulları şans düzeyine yakın. Gerçek backward/update ve süreç belleği ölçüldü.'),
      (5,'Metinden başlık ve ilişki','MIXED','Öğrenilmiş çıkarıcı görülen kalıpta başarılı; yeni kalıpta aktarım zayıf. Başlık örgütleme sonlu varlıklar için deterministik.'),
      (6,'İleri/çift yönlü ağ','MIXED','Aynı tur sayısında eğitim benzeri görevde büyük yarar; görülmemiş derinlikte çift yön tek başına yeterli değil.'),
      (7,'Seçici revizyon','MEASURED_NEGATIVE','Etkilenen cevapta en az%10 kazanç ve etkilenmeyeni koruma ölçütü iki test ailesinin ikisinde de geçmedi.'),
      (8,'Başlık penceresi','IMPLEMENTED_TESTED','Kaynak ve gerçek üretilmiş durum pencereleri ayrı uygulandı; üretilmiş aktif geçmişin tepe büyüklüğü 4/8 kayıt.'),
      (9,'Sıkıştırma ve yeniden okuma','MIXED','Sabit bağlamda son z sınırı eşdeğer; güncel kısıtlardan sinirsel yeniden başlatma derin görevde kayıp verdi.'),
      (10,'Ham arşiv ve vektör erişimi','MEASURED_NEGATIVE','Ham/sürüm/özet korundu; sözcük-vektörü recall@5 yalnız %4,17. Anahtar erişimi bunun başarısı sayılmadı.'),
      (11,'Yeterlilik ve durma','MIXED','Ana IID testte %24,14 daha az gerçek tur ve %38,93 doğru sertifikalı kapsam; yeni derinlikte kapsam çok düşük.'),
      (12,'Gerçek RLT/başlık entegrasyonu','MIXED','Metin→başlık→model→arşiv→denetim uçtan uca çalıştı. Ayrı geri besleme kazancı küçük/belirsiz; yeni cümlelerde cevap yok.'),
      (13,'Üretilmiş izleri ağırlıklara aktarma','MIXED','Gerçek öğretmen izlerinden altı öğrenci eğitildi, arşiv kapalı çıktı eşliği geçti. Kolay görevde zarar, derinde koşullu kazanç.'),
      (14,'Ölçek, aktarım ve maliyet','MIXED','Yaklaşık 24 bin ile 1,51 milyon parametre arasında duyarlılık; büyük modelde 1/3 tohum kısa problemi öğrendi. Enerji/RL/büyük dil modeli ölçülmedi.')]
    training_count=21+6+learning['primary_corrected_models']+learning['combined_models']+len(read('memory/results/metrics.json')['extractor'])+6+6
    assert training_count==63
    text=f'''# RLT ve başlık belleği — uçtan uca sonuç raporu

**On dört paketin tanımlanmış yerel deney döngüsü tamamlandı. Çalışan bir araştırma prototipi ve denetlenmiş olumlu, olumsuz, karışık sonuçlar var; genel ve sınırsız doğru düşünme iddiası doğrulanmadı.** Bu kapanış, yirmi fikrin her alanda başarılı olduğu veya otuz riskin ortadan kaldırıldığı anlamına gelmez.

Ana sonuçlara dahil edilen 63 eğitim/devam eğitimi koşulu, bağlı sistemde 6.144 yürütme ve gerçek durum penceresinde 1.536 ek yürütme tamamlandı. Ana bağlı sistem 512, pencere deneyi 256 ayrı kaynak problemi kullanır; koşul ve model tekrarları bağımsız yeni problem sayılmaz. Üç öğretmenden türetilen altı öğrencinin ağırlık ve çıktı kanıtları korunmuştur. {tests} yazılım testi geçti. Önceki üç araştırmadaki 424 kanıt dosyası değişmeden doğrulandı.

## RLT'nin sonsuz derinlik iddiası

Kaynak mimaride t token sonrasında yinelenen yol t×L_D decoder bloğundan geçer; token başına kullanılan blok sayısı sabittir. Bu, hesaplama yolunun uzatılabilmesini ifade eder. Her ek adımda doğruluğun artması veya sınırsız bilginin korunması sonucu çıkmaz. [Sabitlenmiş resmî kaynak](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md).

Biz t=32…1.024 tüketilen işlem, ayrı k=1…8 başlık güncellemesi ve d=8'e kadar yeni bağımlılık derinliği ölçtük. Parite/beş durum görevleri küçük, sonlu yeterli durum taşır. Bu görevlerde uzun ufuk başarısı, geniş kavramsal akıl yürütme veya sınırsız bilgi kapasitesi değildir. Kaynak yazarlarının 26–29M modellerinin yeniden üretimi yapılmadı.

## Araştırma hattındaki14 paket

MIXED, ölçümün eksik olduğu anlamına gelmez; farklı koşulların farklı sonuç verdiğini belirtir. MEASURED_NEGATIVE, tanımlanan başarı ölçütünün karşılanmadığı geçerli bir deney sonucudur.

| No | Paket | Sonuç | Kanıtın kapsamı |
|---|---|---|---|
'''+ '\n'.join(f'| {i} | {name} | {status} | {scope} |' for i,name,status,scope in packages)+f'''

## Küçük çekirdek: uzunluk gerçekten neyi değiştirdi?

Aşağıdaki değerler üç eğitimin ortalama son durum doğruluğu. Her model/uzunluk hücresinde512 yeni örnek; en sağda üç ayrı eğitimin1.024 sonucunun aralığı var. Aralık güven aralığı değildir. Kontrol, RLT'nin global ve yerel dikkati korunup gizli durum geri beslemesi kapatılan sürümüdür.

| Görev / genişlik / model |32 |128 |1.024 |1.024 tohum aralığı |
|---|---:|---:|---:|---:|
'''+ '\n'.join(core_table)+f'''

Her adımda hedef verilen öğrenme ile yalnız son cevaptan hedef verilen öğrenme aynı sonuç değildi. Ayrı son-cevap deneyinde tam ve kesilmiş gradyanla eğitilen altı koşulun 32 ve 128 uzunluktaki doğrulukları {final_range} arasında, iki sınıflı görevin şans düzeyine yakın kaldı. Bu yalnız sabit2.000güncellemelik protokol için olumsuz bulgudur; şansla istatistiksel eşdeğerlik iddiası değildir. Prefix deneyinde32kat fazla gözetim hedefi vardır; iki rejim eşit etiket bütçesi karşılaştırması değildir.

Kesilmiş gradyan gizli durum ve yerel decoder KV yollarını sekiz adımda keser; encoder/global bellek yollarını kesmez. Küçük veya sıfır yerel türev tek başına unutma, sıfır olmayan türev de öğrenme kanıtı değildir. Gerçek checkpoint yeniden hesaplamasında backward/AdamW eşliği kontrol edildi. Kısa süreç benchmark'ında koşul başına üç tekrarın ortalama toplam tepe çalışma belleği {peak_range} MiB arasında kaldı. Bu ölçüm belirgin toplam bellek tasarrufu göstermedi; aktivasyon belleği ayrı ölçülmedi.

![Çekirdeğin uzun ufuk sonuçları]({(ROOT/'reports/figures/core-long-horizon.png').as_posix()})

Ayrı kimlik-son-eki deneyinde aynı 32 işlemin sonuna cevabı değiştirmeyen 128 işlem eklendi. Aşağıdaki modellerin son ek yokken doğruluğu üç tohumda da %100'dü. Tablo son ekten sonraki doğrulukları gösteriyor. Rastgele uzun dizilerde başarı, farklı işlem düzeninde gereksiz adımlara dayanıklılık sağlamadı; etki model ve tohuma göre değişti.

| Görev / genişlik / model | Tohum 17 | Tohum 43 | Tohum 91 |
|---|---:|---:|---:|
'''+ '\n'.join(neutral_table)+f'''

## Senin başlık/ağ fikrinde işe yarayan ve yetersiz kalan parçalar

Sabit yerel tur sayısında çift yönlü ağ, eğitim benzeri yeni görevlerde son cevap doğruluğunu ileri yönün %26,11'inden %95,70'e çıkardı. Ancak gerçek yeni bağımlılık derinliği 8'de tek başına çift yönlü ağ%9,90'da kaldı; ileri yön%11,98 idi. Doğru ara değerlere ek gözetim vermek aynı derinlikte%33,46'ya çıkardı. Ek hedefler gerçek bir eğitim avantajıdır; sonuç yalnız ağ biçimine veya eşit hesaplama maliyetine yüklenemez.

Gerçek model geri beslemesi sonraki token durumuna bağlandı; ayrı ek yararı küçük ve belirsiz kaldı. Seçici revizyonun önceden belirlenen güçlü ölçütü geçmedi. Sözcük karma vektörüyle erişim@5 yalnız%4,17 idi; kalıcı arşivin doğru çalışması bunu başarılı anlamsal erişim yapmaz.

![Başlık ağı ve ara gözetim]({(ROOT/'reports/figures/learning-generalization.png').as_posix()})

Ana bağlı sistemde eğitim benzeri yeni problemlerde doğru sayısal aday%98,18, doğru ve doğrulayıcıdan geçen son cevap kapsamı%38,93 çıktı. Sabit8tura göre gerçek güncelleme sayısı%24,14 azaldı. Yeni derinlikte aday doğruluğu%37,63'e, kabul kapsamı%0,13'e indi. Yeni cümle düzenlerinde ayrıştırma reddi sonrası son cevapların tamamı eksik kaldı.

Güncel kısıt özetinden modeli yeniden başlatmak, derinlik 8 aday doğruluğunu%37,63'ten%20,57'ye düşürdü. Buna karşılık sabit bağlamda **gerçek son z durumunu saklayıp yeniden yükleyen** ayrı pencere deneyi bütün eşlenik izleri ve çıktıları korudu:70.080gerçek iz,8.760sınır/okuma, aktif tepe geçmiş8→4. Bu ikinci mekanizma genel anlamsal özet değildir. Serileştirilmiş aktif yük yaklaşık yarıya inerken ek sınır kayıtları diski yaklaşık%21 büyüttü; toplam RAM tasarrufu gösterilmedi.

![Bağlı sistemde aday ve kabul kapsamı]({(ROOT/'reports/figures/connected-quality-coverage.png').as_posix()})

## Geçmişteki gerçek düşünme yolunu sonraki eğitime aktarma

İlk ara-gözetim deneyi doğru ara hedefler kullanıyordu. Ayrı deneyde modelin gerçekten ürettiği latent/sayısal yollar saklandı ve altı öğrenci devam eğitiminde kullanıldı. Bütün bölümün sertifikalanabildiği örnek bulunmadı; önceden belirlenen sorgulanan-bileşen seçeneği üç tohumda313/418/333bölüm sağladı. Bu seçim sığ örneklere kayıyor. Başarılı son aday, her ara adımın doğruluğunu kanıtlamıyor.

MSE ortalama karesel hatadır; daha düşük değer daha iyi sonucu gösterir.

| Yeni test | Yalnız öğretmen son cevabı: MSE | Gerçek ara izler de kullanıldığında: MSE |
|---|---:|---:|
| Eğitim benzeri |0,002615 |0,011706 |
| Gerçek derinlik 8 |0,697890 |0,557985 |

Kolay görevde üç tohumda da zarar, derin görevde cevap-hedefli kola göre üçünde de MSE iyileşmesi var. Özgün eşli karşılaştırmanın tohum-t%95aralıkları iki bölümde de sıfırı içeriyor. Ayrı, sonradan eklenen başlangıç-modeli referansı da kolayda kötüleşme/derinde iyileşme yönünü gösterdi; bu özgün önceden kayıtlı üçüncü kol değildir. Yöntemin genel üstünlüğü sonucu çıkarılmadı.

Altı öğrencinin512'şerörnekte arşiv ve üretici modülleri bulunmayan çıkarım dışa aktarımı kayıtlı çıktılarla tam eşleşti. Bu, çıkarımın saklanan öğretmen izlerini okuyarak cevap vermediğini gösteren dar bir paket kontrolüdür; işletim sistemi erişim izolasyonu değildir.

## Büyütünce çözüldü mü?

1,51Mparametreli üç RLT eğitiminden yalnız biri bu ayarlarla kısa görevi öğrendi. Başarılı tohumun 1.024 adımdaki doğruluğu %99,61 oldu; diğer iki tohum1.024'te%48,44'tü. Başarılı modelde, ayrı aynı32işlemin sonuna cevabı değiştirmeyen128kimlik işlemi eklemek doğruluğu%99,41'den%50,59'a düşürdü. Ek, girdinin işlem düzenini değiştirir; saf başlık-turu deneyi değildir.

Genişlikle dikkat başı sayısı ve küçük çekirdeğe göre cihaz değişti. Öğrenme oranı ölçek başına eniyilenmedi. Bu, belirtilen protokol altında duyarlılık ölçümüdür; ölçek yasası veya büyük modelin evrensel kapasite sınırı değildir.

![Ölçek, tohum ve kimlik işlemleri]({(ROOT/'reports/figures/scale-seeds-and-neutral.png').as_posix()})

## Teslimat ve açık geçerlilik sınırları

Çalışma ayrı E: kökünde yürütüldü. Önceki424kanıt değişmedi. Ham gözlemler, gerçek model izleri, başarısız ilk denemeler, model ağırlıkları, tahminler, protokoller ve bağımsız denetimler korundu. Ana bağlı sistem ve pencere deneyinin toplam242.162gerçek güncelleme izi ayrı SQLite arşivlerindedir; öğretmen eğitim arşivi ayrıca saklanır.

Kapanış öncesinde mevcut kaynak, protokol, ağırlık, tahmin ve arşiv dosyaları kaydedilmiş karmalarına karşı tekrar okundu. Bağımsız bellek deneyinin bazı eski dosyalarında önceden kaydedilmiş bütün-dosya karması yok; bu eksikler kapanış doğrulamasında ayrı listeleniyor. Veri tabanı içindeki kayıt karmaları bunun yerine bütün dosyanın geçmişteki durumunu kanıtlıyor sayılmadı. Son teslimat manifesti mevcut dosyaların bütünlüğünü korur; haricî imza veya eksik eski kayıtların yerine geçen kanıt değildir.

Serbest doğal dil, öğrenilmiş genel konu bölme, güvenilir anlamsal vektör erişimi, geniş model/RL eğitimi, enerji, toplam RAM için sabit üst sınır ve sonsuz doğru akıl yürütme **doğrulanmadı veya ölçülmedi**. Bu açık sınırlar, yerel deneylerin bitmediği anlamına gelmez. Yeni başarı iddiaları yeni deney gerektirir.

Kanıt yolları:

- {link('reports/core-report.md','Çekirdek ayrıntıları')} ve {link('reports/core-independent-audit.md','bağımsız çekirdek denetimi')}.
- {link('reports/learning-report.md','Ağ, revizyon ve ara gözetim')} ve {link('reports/learning-independent-audit.md','bağımsız öğrenme denetimi')}.
- {link('reports/connected-report.md','Metinden sonuca bağlı sistem')} ve {link('reports/connected-independent-audit.md','6.144 arşivin denetimi')}.
- {link('reports/reasoning-window-report.md','Üretilmiş durum penceresi')} ve {link('reports/reasoning-window-independent-audit.md','1.536 arşivin denetimi')}.
- {link('learning/distillation/report.md','Gerçek izlerden sonraki eğitim')} ve {link('reports/distillation-independent-audit.md','bağımsız aktarım ve başlangıç referansı')}.
- {link('reports/scale-report.md','Büyük yerel modelin sonuçları')}, {link('reports/user-requirement-map.md','yirmi fikrin ayrı karşılığı')}, {link('reports/mathematical-interpretation.md','matematik ve otuz risk kapsamı')}.
- {link('audit/campaign-closure.json','Kapanış kaydı')}, {link('audit/closure-pins.json','kapanış öncesi güncel karma kontrolü ve kapsam sınırları')}, {link('audit/final-manifest.json','bütün dosyaların SHA256 manifesti')}, {link('README.md','doğrulama yönergesi')}.
'''
    out=ROOT/'reports/RLT_SONUC_RAPORU.md'
    closure_path=ROOT/'audit/campaign-closure.json'
    pins_path=ROOT/'audit/closure-pins.json'
    assert all(not p.exists() for p in [out,closure_path,pins_path]),'refuse to overwrite a completed delivery'
    for name in ['core-long-horizon','learning-generalization','connected-quality-coverage','scale-seeds-and-neutral']:
        assert (ROOT/f'reports/figures/{name}.png').is_file(),'missing final figure: '+name
    assert (ROOT/'reports/core-independent-audit.md').is_file(),'missing independent core report'
    for old,new in {
        't=32…1.024':'t = 32…1.024','k=1…8':'k = 1…8','d=8':'d = 8',
        '26–29M modellerinin':'26–29M parametreli modellerinin','hattındaki14':'hattındaki 14',
        'hücresinde512':'hücresinde 512',"eğitimin1.024":"eğitimin 1.024",
        'sabit2.000güncellemelik':'sabit 2.000 güncellemelik','deneyinde32kat':'deneyinde 32 kat',
        'yönün%26,11':'yönün %26,11',"tek başına çift yönlü ağ%9,90":"tek başına çift yönlü ağ %9,90",
        'yön%11,98':'yön %11,98','derinlikte%33,46':'derinlikte %33,46',
        'erişim@5 yalnız%4,17':'erişim@5 yalnız %4,17','aday%98,18':'aday %98,18',
        'kapsamı%38,93':'kapsamı %38,93','Sabit8tura':'Sabit 8 tura',
        'sayısı%24,14':'sayısı %24,14','doğruluğu%37,63':'doğruluğu %37,63',
        'kapsamı%0,13':'kapsamı %0,13',"doğruluğunu%37,63'ten%20,57":"doğruluğunu %37,63'ten %20,57",
        'korudu:70.080gerçek':'korudu: 70.080 gerçek','iz,8.760sınır':'iz, 8.760 sınır',
        'geçmiş8→4':'geçmiş 8→4','yaklaşık%21':'yaklaşık %21',
        'tohumda313/418/333bölüm':'tohumda 313 / 418 / 333 bölüm',
        'tohum-t%95aralıkları':'tohum-t %95 aralıkları',"öğrencinin512'şerörnekte":"öğrencinin 512'şer örnekte",
        '1,51Mparametreli':'1,51M parametreli','tohum1.024adıma':'tohum 1.024 adıma',
        'kadar%99,61doğruluk':'kadar %99,61 doğruluk',"tohum1.024'te%48,44":"tohum 1.024'te %48,44",
        'aynı32işlemin':'aynı 32 işlemin','değiştirmeyen128kimlik':'değiştirmeyen 128 kimlik',
        "doğruluğu%99,41'den%50,59":"doğruluğu %99,41'den %50,59",
        'Önceki424kanıt':'Önceki 424 kanıt','toplam242.162gerçek':'toplam 242.162 gerçek',
        'en az%10':'en az %10','doğrulaması.512':'doğrulaması. 512',
    }.items():text=text.replace(old,new)
    with pins_path.open('x',encoding='utf8') as f:json.dump(fresh_pins,f,indent=2,ensure_ascii=False)
    with out.open('x',encoding='utf8') as f:f.write(text)
    closure={'status':'COMPLETE_BOUNDED_RESEARCH','utc':datetime.now(timezone.utc).isoformat(),
             'packages':[{'id':i,'name':name,'outcome':status,'scope':scope} for i,name,status,scope in packages],
             'training_conditions':training_count,'tests_passed':tests,'previous_files_unchanged':424,
             'connected_episode_executions':6144,'connected_distinct_source_tasks':512,
             'window_episode_executions':1536,'window_distinct_source_tasks':256,
             'primary_core_summary':core_summary,'all_independent_audits_passed_with_stated_scope':True,
             'fresh_pin_verification_sha256':hashlib.sha256(pins_path.read_bytes()).hexdigest(),
             'scientific_boundary':'Finite controlled local research complete; general-purpose or infinite correct reasoning is not established.',
             'report':str(out),'manifest_phase':'Built and byte-verified only after this receipt and every remaining writer is quiescent.'}
    with closure_path.open('x',encoding='utf8') as f:json.dump(closure,f,indent=2,ensure_ascii=False)
    print(json.dumps({'report':str(out),'packages':len(packages),'training_conditions':training_count,'tests':tests}))


if __name__=='__main__':main()
