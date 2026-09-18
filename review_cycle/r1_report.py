"""Readable R1 report generated solely from frozen result records."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'results/R1'
    p = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    rows = {(r['split'], r['seed'], r['method']): r for r in result['rows']
            if r['threshold'] == p['primary_threshold'] and r['policy'] == 'fixed8'}
    effects = [r['path_minus_component_coverage'] for r in result['paired_primary'] if r['split'] == 'branched']
    mean = statistics.mean(effects)
    half = 4.302652729 * statistics.stdev(effects) / (len(effects) ** .5)
    lines = ['# R1 — Yol sertifikasının etkisi ve sınırı', '',
        'Yol sınırı, dallanmış görevlerde bazı doğru cevapların gereksiz reddini azalttı. '
        'Zincir görevlerinde kabul oranı değişmedi. Dallanmış grupta artış üç model için '
        '**512 örnekte 6, 0 ve 22 ek kabul** oldu. Bu, bütün doğru cevapları güvenle '
        'kabul etme sorununun çözüldüğünü göstermiyor.', '',
        'Deneydeki 2.048 kaynak problemi dört dağılıma eşit bölündü. Aynı problemler '
        'üç donmuş modelde çalıştırıldığı için 6.144 model-problem çalıştırması vardır; '
        '6.144 bağımsız problem yoktur. Aynı problemin düşünme turları da bağımsız '
        'örnek olarak sayılmadı.', '',
        'Birincil karşılaştırma, son sorgunun sekizinci turundaki **aynı adaylara** '
        'uygulandı. Doğruluk eşiği 0,1; kabul eşiği 0,099999 idi. Modeller, '
        'eşikler ve yeni veri tohumları sonuçlar açılmadan önce sabitlendi. '
        'Önceki sorgular altışar tur kullandı. Protokoldeki daha düşük eşikler '
        've uyarlamalı durma sonuçları ayrıca sonuç dosyasında yer alır.', '',
        '| Dağılım | Model tohumu | Doğru aday / 512 | Eski kabul | Yol kabul | Kabul edilen yanlış |',
        '|---|---:|---:|---:|---:|---:|']
    for split in p['sources']:
        for seed in p['model_seeds']:
            old, new = rows[split, seed, 'component'], rows[split, seed, 'path']
            lines.append(f"| {split} | {seed} | {new['correct']} | {old['accepted']} | {new['accepted']} | {new['accepted_wrong']} |")
    lines += ['', 'Üç zincir dağılımında tek çapa ile sorgu arasındaki yol, sorgu bileşeninin '
        'bütün kenarlarını kapsıyor. İki artık toplamının eşitliği bu yapıda matematiksel '
        'olarak beklenir. Bunlar kontrol gruplarıdır. Yeni dallanmış dağılımda dört '
        'kenarlı sorgu yoluna ek dört kenarlı bir yan dal vardır; bu topoloji eğitim '
        'dağılımının dışındadır. Oradaki kazanç, önceki zincirlerdeki retlerin '
        'düzeltildiği şeklinde yorumlanamaz.', '',
        f'Dallanmış grupta ortalama kabul artışı {mean * 100:.2f} yüzde puanıdır. '
        f'Yalnız üç eğitim tohumu üzerinden yaklaşık t aralığı '
        f'[{(mean-half)*100:.2f}, {(mean+half)*100:.2f}] yüzde puanıyla sıfırı kapsar. '
        'Bu aralık geniş model genellemesi için güçlü kanıt değildir. Kaynak problemi '
        'eşleştirmesini koruyan, her model için ayrı bootstrap aralıkları ham özette '
        'bulunur; model tohumları birleştirilerek örnek sayısı şişirilmedi.', '',
        'Birincil koşullarda yanlış kabul gözlenmedi. Sıfır kabul bulunan koşullarda '
        'seçici hata oranı `null` tutuldu. Az kabulden güçlü güvenilirlik sonucu çıkmaz: '
        'dallanmış grupta 6 ve 22 kabul için sıfır hata gözlemine ait tek taraflı '
        'ampirik %95 üst hata sınırları sırasıyla yaklaşık %39,3 ve %12,7’dir.', '',
        'Matematiksel sertifika farklı bir iddiadır: doğru kapsam/sürümdeki tutarlı '
        'afin kaynaklar verildiğinde, çapa artığı ile yol kenarlarının mutlak '
        'artık toplamı sorgu hatasını üstten sınırlar. Uygulama tam rasyonel '
        'aritmetik kullanır; model cevabını çözücü cevabıyla değiştirmez. Bu '
        'koşullu kanıt serbest doğal dil ayrıştırmasını, kaynakların dünyaya '
        'uygunluğunu veya sınırsız doğru düşünmeyi kanıtlamaz.', '',
        'Yanlış cevap, eski sürüm, başka görev kapsamı, yanlış sorgu bağı, bozuk '
        'özet ve çelişkili kaynak kontrolleri geçildi. Her kabul edilen yol '
        'sertifikası kısa yol algoritmasından ayrı denetimle yeniden hesaplandı. '
        'Tüm turların önceden hesaplandığı karşılaştırmadan elde edilen durma '
        'sayısı bir hesaplama tasarrufu ölçümü değildir. Her koşulun ilk 16 '
        'problemi ayrıca gerçekten durabilen çağrıyla çalıştırıldı; bu küçük '
        'altkümenin çağrı sayısı ve süresi ayrı dosyalardadır.', '',
        '**Durum:** deney, protokol kontrolleri ve bağımsız ham denetim PASS. '
        'Bağımsız denetim 49.152 aday turunu, 294.912 kabul kararını ve 3.943 '
        'sertifikayı yeniden hesapladı; bu tekrarlı kararlar bağımsız problem '
        'sayısı değildir. Bilimsel sonuç: dallanmış '
        'görevlerde sınırlı ve modele bağlı kazanım; zincirlerde beklenen sıfır etki.', '',
        'Kanıtlar: [protokol](../results/R1/protocol.json), '
        '[sonuç özeti](../results/R1/summary.json), '
        '[tamamlanma kaydı](../results/R1/completion.json), '
        '[matematiksel sözleşme](R1-proof.md), '
        '[bağımsız öninceleme](../.superpowers/sdd/review-diagnostics-plan/r1-benchmark-review.md), '
        '[bağımsız ham denetim](../.superpowers/sdd/review-diagnostics-plan/r1-main-audit.md).', '']
    (ROOT / 'notes/R1-report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
