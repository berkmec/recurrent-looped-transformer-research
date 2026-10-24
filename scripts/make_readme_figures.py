"""README tables and figures, computed from the raw result files.

No plotting library: every figure is a plain SVG whose colours follow the reader's light / dark theme (CSS media query),
so it renders on GitHub without a second image. Only results that were read with their registered rule and recorded in
notes/ANA_BELGE_RLT.md are used; the numbers are recomputed here from the JSON / JSONL files with the repository's own
reading code (horizon.readings, horizon.sk_run), and the script stops if a recomputed number disagrees with the recorded one.

  python -B scripts/make_readme_figures.py                      # writes results/figures/*.svg and the README section
  python -B scripts/make_readme_figures.py --check              # recompute and compare only
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

BEGIN, END = '<!-- TABLES:BEGIN -->', '<!-- TABLES:END -->'

# ---------------------------------------------------------------- numbers in Turkish notation
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def dec(x, nd=3):
    return ('%.*f' % (nd, x)).replace('.', ',')


def num(n):
    return '{:,}'.format(int(n)).replace(',', '.')


def sci(x, nd=2):
    if x == 0:
        return '0'
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, nd - 1) >= 10:
        m, e = m / 10, e + 1
    return '%s×10%s' % (dec(m, nd - 1).replace('-', '−'), str(e).translate(SUP))


def poisson_ci(events, exposure):
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * events) / 2 / exposure if events else 0.0
    hi = chi2.ppf(0.975, 2 * (events + 1)) / 2 / exposure
    return lo, hi


# ---------------------------------------------------------------- SVG
CSS = """
.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.tm{fill:#898781}.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}.ring{stroke:#fcfcfb}
.l1{stroke:#2a78d6}.l2{stroke:#eb6834}.l3{stroke:#1baf7a}.l4{stroke:#eda100}
.f1{fill:#2a78d6}.f2{fill:#eb6834}.f3{fill:#1baf7a}.f4{fill:#eda100}
@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}
.ring{stroke:#1a1a19}.l1{stroke:#3987e5}.l2{stroke:#d95926}.l3{stroke:#199e70}.l4{stroke:#c98500}
.f1{fill:#3987e5}.f2{fill:#d95926}.f3{fill:#199e70}.f4{fill:#c98500}}
text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
"""
W, H = 760, 380


def esc(s):
    return html.escape(str(s), quote=True)


def text_width(s, size):
    return len(s) * size * 0.56


class Svg:
    def __init__(self, title, subtitle, desc, height=H):
        self.h, self.items = height, []
        self.title, self.desc = title, desc
        self.text(24, 30, title, 'title')
        self.text(24, 50, subtitle, 'subtitle')

    def text(self, x, y, s, role='label', anchor='start'):
        cls, size, weight = {'title': ('t1', 16, 600), 'subtitle': ('t2', 12.5, 400), 'label': ('t2', 12, 400),
                             'value': ('t1', 12, 600), 'tick': ('tm', 11, 400), 'note': ('tm', 11, 400)}[role]
        self.items.append('<text class="%s" x="%.1f" y="%.1f" font-size="%s" font-weight="%d" text-anchor="%s">%s</text>'
                          % (cls, x, y, size, weight, anchor, esc(s)))

    def line(self, x1, y1, x2, y2, cls='grid', width=1):
        self.items.append('<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="%s"/>' % (cls, x1, y1, x2, y2, width))

    def raw(self, s):
        self.items.append(s)

    def save(self, path):
        body = '\n'.join(self.items)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="t d">\n'
               '<title id="t">%s</title><desc id="d">%s</desc>\n<style>%s</style>\n<rect class="bg" width="%d" height="%d" rx="8"/>\n%s\n</svg>\n'
               % (W, self.h, W, self.h, esc(self.title), esc(self.desc), CSS, W, self.h, body))
        Path(path).write_bytes(svg.encode('utf8'))


def nice_ticks(lo, hi, n=5):
    """Clean ticks that COVER [lo, hi] (the last tick is at or above the largest value)."""
    raw = (hi - lo or 1.0) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)     # no 2.5: its ticks need one more decimal than the axis shows
    t, end, ticks = math.floor(lo / step) * step, math.ceil(hi / step - 1e-9) * step, []
    while t <= end + step * 1e-6:
        ticks.append(round(t, 10))
        t += step
    return ticks


def legend(svg, series, y):
    x = 24
    for s in series:
        svg.line(x, y - 4, x + 18, y - 4, 'l%d' % s['slot'], 2.5)
        svg.text(x + 24, y, s['name'], 'label')
        x += 24 + text_width(s['name'], 12) + 22


def line_chart(path, title, subtitle, desc, xlabels, series, ylab, yfmt, ymin=None, ymax=None, xlab=None, note=None, end_labels=False):
    svg = Svg(title, subtitle, desc)
    if len(series) > 1:
        legend(svg, series, 74)
    left, right, top, bottom = 70, 150 if end_labels else 40, 92 if len(series) > 1 else 72, H - 72
    vals = [v for s in series for v in s['values'] if v is not None]
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    ticks = nice_ticks(lo, hi)
    lo, hi = ticks[0], ticks[-1]
    xs = [left + (right_x - left) * i / max(1, len(xlabels) - 1) for i, right_x in enumerate([W - right] * len(xlabels))]

    def Y(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)
    for t in ticks:
        svg.line(left, Y(t), W - right, Y(t), 'grid', 1)
        svg.text(left - 8, Y(t) + 4, yfmt(t), 'tick', 'end')
    svg.line(left, bottom, W - right, bottom, 'axis', 1)
    for x, lab in zip(xs, xlabels):
        svg.text(x, bottom + 18, lab, 'tick', 'middle')
    if xlab:
        svg.text((left + W - right) / 2, bottom + 36, xlab, 'note', 'middle')
    svg.raw('<text class="tm" font-size="11" transform="translate(16 %.1f) rotate(-90)" text-anchor="middle">%s</text>' % ((top + bottom) / 2, esc(ylab)))
    ends = []
    for s in series:
        pts = [(x, Y(v)) for x, v in zip(xs, s['values']) if v is not None]
        svg.raw('<polyline class="l%d" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="%s"/>'
                % (s['slot'], ' '.join('%.1f,%.1f' % p for p in pts)))
        for x, y in pts:
            svg.raw('<circle class="f%d ring" cx="%.1f" cy="%.1f" r="4" stroke-width="2"/>' % (s['slot'], x, y))
        last = [v for v in s['values'] if v is not None][-1]
        ends.append((pts[-1][1], s, last))
    if end_labels:
        ends.sort(key=lambda e: e[0])
        if all(b[0] - a[0] >= 15 for a, b in zip(ends, ends[1:])):
            for y, s, v in ends:
                svg.text(W - right + 12, y + 4, '%s  %s' % (s['short'], yfmt(v)), 'label')
    if note:
        svg.text(24, H - 12, note, 'note')
    svg.save(path)


def bar_chart(path, title, subtitle, desc, rows, xfmt, xmax, xlab, note=None):
    """Horizontal bars, one series: rows = [(label, value, value_text)]."""
    height = 146 + 34 * len(rows)
    svg = Svg(title, subtitle, desc, height)
    left = max(190, int(24 + max(text_width(lab, 12) for lab, _, _ in rows) + 12))     # the longest label must fit left of the axis
    right, top = 90, 78
    def X(v):
        return left + v / xmax * (W - left - right)
    for t in nice_ticks(0, xmax):
        svg.line(X(t), top - 6, X(t), top + 34 * len(rows), 'grid', 1)
        svg.text(X(t), top + 34 * len(rows) + 16, xfmt(t), 'tick', 'middle')
    for i, (lab, v, vt) in enumerate(rows):
        y, h = top + 34 * i + 5, 22
        x0, x1, r = left, max(X(v), left + 5), 4
        svg.raw('<path class="f1" d="M%.1f %.1f H%.1f Q%.1f %.1f %.1f %.1f V%.1f Q%.1f %.1f %.1f %.1f H%.1f Z"/>'
                % (x0, y, x1 - r, x1, y, x1, y + r, y + h - r, x1, y + h, x1 - r, y + h, x0))
        svg.text(left - 10, y + 15, lab, 'label', 'end')
        svg.text(x1 + 8, y + 15, vt, 'value')
    svg.line(left, top - 6, left, top + 34 * len(rows), 'axis', 1)
    svg.text((left + W - right) / 2, top + 34 * len(rows) + 36, xlab, 'note', 'middle')
    if note:
        svg.text(24, height - 12, note, 'note')
    svg.save(path)


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, root):
        self.root = Path(root)
        sys.path.insert(0, str(self.root))

    def json(self, rel):
        return json.loads((self.root / rel).read_text(encoding='utf8'))


def s5_rows(d):
    from horizon.readings import pooled_rate, pooled_windows, _load
    z = d.root / 'results/Z1-pilot'
    snap = {Path(m['weights']).name: m for m in d.json('results/Z1-pilot/self-snap.json')}
    for s in (2, 3):
        for m in d.json('results/Z1-pilot/ext-perm5-core-w128-s%d-cur-snap.json' % s):
            snap[Path(m['weights']).name] = m
    spec = [('Core w128', 'tohum 1, sabit lr', 'perm5-core-w128-s1-cur'),
            ('Core w128', 'tohum 2, sabit lr', 'perm5-core-w128-s2-cur'),
            ('Core w128', 'tohum 3, sabit lr', 'perm5-core-w128-s3-cur'),
            ('Core w128', 'tohum 1, lr azaltmalı', 'perm5-core-w128-s1-cur-lrdecay0.3'),
            ('Core w128', 'tohum 1, durum gürültüsü σ = 0,2', 'perm5-core-w128-s1-cur-lrdecay0.3-noise0.2'),
            ('Core w128', 'tohum 1, durum gürültüsü σ = 0,1', 'perm5-core-w128-s1-cur-lrdecay0.3-noise0.1'),
            ('Core w256', 'tohum 1, lr azaltmalı (genişlik ×2)', 'perm5-core-w256-s1-cur-lrdecay0.3'),
            ('Core + öğrenilen temizleme', 'tohum 1, lr azaltmalı, σ = 0,1', 'perm5-core_clean-w128-s1-cur-lrdecay0.3-noise0.1-open0.3'),
            ('GRU w256', 'tohum 1, sabit lr', 'perm5-gru-w256-s1-cur'),
            ('GRU w256', 'tohum 1, lr azaltmalı', 'perm5-gru-w256-s1-cur-lrdecay0.3'),
            ('LSTM w208', 'tohum 1, sabit lr', 'perm5-lstm-w208-s1-cur'),
            ('LSTM w208', 'tohum 1, lr azaltmalı', 'perm5-lstm-w208-s1-cur-lrdecay0.3')]
    rows = []
    for model, recipe, stem in spec:
        r = _load(z / (stem + '.json'))
        rate, events, exposure = pooled_rate(r)
        lo, hi = poisson_ci(events, exposure)
        sn = snap.get(stem + '.pt')
        c16 = sn['conditions']['class-16'] if sn else None
        rows.append(dict(model=model, recipe=recipe, stem=stem, in_range=r['curve']['16']['final'], final1024=r['curve']['1024']['final'],
                         hazard=rate, events=events, exposure=exposure, lo=lo, hi=hi, windows=pooled_windows(r),
                         snap=(c16['events'], c16['person_steps'], c16['final']) if c16 else None,
                         base=(sn['conditions']['base']['events'], sn['conditions']['base']['person_steps']) if sn else None))
    return rows


def drift_rows(d):
    import numpy as np
    mids = {'0-1': 0.5, '1-4': 2.5, '4-16': 10.0, '16-32': 24.0, '32-64': 48.0, '64-128': 96.0}
    out = []
    for e in d.json('results/Z1-pilot/drift-profile.json') + d.json('results/Z1-pilot/drift-profile-s2.json'):
        D = {}
        for t0 in (8, 128, 256):
            prof = e['profile']['injected@%d' % t0]
            bins = ['4-16', '16-32', '32-64', '64-128']
            D[t0] = float(np.polyfit([mids[b] for b in bins], [prof[b]['total'] ** 2 for b in bins], 1)[0])
        prof = e['profile']['injected@128']
        out.append(dict(name=Path(e['weights']).stem, task=e['task'], width=e['width'], seed=e['seed'], D=D,
                        curve=[prof[b]['total'] ** 2 for b in mids], first=prof['1-4']['total'], last=prof['64-128']['total']))
    return out, list(mids)


def es_reading(d):
    from horizon.sk_run import reading
    files = [f for f in sorted((d.root / 'results/ES-pilot').glob('sk3-*.json')) if not f.name.endswith('.manifest.json')]   # RUNMAN manifests sit beside the results
    return reading([json.loads(f.read_text(encoding='utf8')) for f in files]), len(files)


# every sk4 run of E-S stage B (ES-pilot), E-S-LR, E-S-B2X (twice the updates), E-S-PE (positional transformer) and E-S-W (4-16x parameters)
ES4_GROUPS = [
    ('Core (RLT, w64)', [('ES-pilot', 'core', 64)]),
    ('Core, geri beslemesiz (A0)', [('ES-pilot', 'core_nofb', 64)]),
    ('GRU w128', [('ES-pilot', 'gru', 128), ('ES-lr', 'gru', 128), ('ES-b2x', 'gru', 128)]),
    ('GRU w256 / w512 (4–16× parametre)', [('ES-width', 'gru', 256), ('ES-width', 'gru', 512)]),
    ('LSTM w112', [('ES-pilot', 'lstm', 112), ('ES-lr', 'lstm', 112), ('ES-b2x', 'lstm', 112)]),
    ('LSTM w224 / w448 (4–16× parametre)', [('ES-width', 'lstm', 224), ('ES-width', 'lstm', 448)]),
    ('Transformer, NoPE, w64', [('ES-pilot', 'tfm', 64), ('ES-lr', 'tfm', 64), ('ES-b2x', 'tfm', 64)]),
    ('Transformer, konumlu, w64', [('ES-pe', 'tfm_pe', 64)]),
    ('Transformer, konumlu, w128', [('ES-width', 'tfm_pe', 128)]),
]


def es4_rows(d):
    """One row per ES4_GROUPS entry (runs, mean in-range and extrapolation, best extrapolation, learned by sk_run's 0.95 rule) and the
    registered reading of E-S stage B recomputed with horizon.sk_run.reading."""
    from horizon.sk_run import LEARNED, reading
    runs = {}
    for directory in {k for _, keys in ES4_GROUPS for k, _, _ in keys}:
        runs[directory] = [json.loads(f.read_text(encoding='utf8')) for f in sorted((d.root / 'results' / directory).glob('sk4-*.json'))
                           if not f.name.endswith('.manifest.json')]
    rows = []
    for label, keys in ES4_GROUPS:
        rs = [r for directory, arm, width in keys for r in runs[directory] if r['arm'] == arm and r['width'] == width]
        ins, ext = [r['in_range'] for r in rs], [r['extrapolation'] for r in rs]
        rows.append(dict(label=label, runs=len(rs), in_range=sum(ins) / len(ins), extrapolation=sum(ext) / len(ext), best=max(ext),
                         learned=sum(ins) / len(ins) >= LEARNED - 1e-6))
    return rows, reading(runs['ES-pilot'])


def l1_rows(d, run='L1-scratch-w928-s401'):
    rows = []
    for line in (d.root / 'results' / run / 'probe.jsonl').read_text(encoding='utf8').splitlines():
        if '"eval"' in line:
            r = json.loads(line)
            if isinstance(r.get('eval'), dict) and 'bits_per_byte' in r['eval']:
                rows.append(dict(step=r['step'], **r['eval']))
    return rows


def l1x(d):
    import statistics as st
    j = d.json('results/L1-long-doc/step006000-long-doc.json')
    bins = list(j['per_doc']['full'])
    mean = {c: [st.fmean(j['per_doc'][c][b]) for b in bins] for c in j['per_doc']}
    diff = []
    for b in bins:
        ds = [f - c for f, c in zip(j['per_doc']['full'][b], j['per_doc']['chunk512'][b])]
        diff.append((st.fmean(ds), st.stdev(ds) / math.sqrt(len(ds))))
    return j, bins, mean, diff


# ---------------------------------------------------------------- recorded values (notes/ANA_BELGE_RLT.md) the recomputation must match
RECORDED = [
    ('S5 seed 1 hazard', lambda R: R['s5'][0]['hazard'], 1.41e-3, 0.01e-3),
    ('S5 seed 2 hazard', lambda R: R['s5'][1]['hazard'], 1.48e-5, 0.01e-5),
    ('S5 seed 3 hazard', lambda R: R['s5'][2]['hazard'], 2.91e-3, 0.01e-3),
    ('S5 lr-decay hazard', lambda R: R['s5'][3]['hazard'], 1.65e-3, 0.01e-3),
    ('S5 w256 hazard', lambda R: R['s5'][6]['hazard'], 8.58e-4, 0.01e-4),
    ('S5 core_clean hazard', lambda R: R['s5'][7]['hazard'], 2.15e-3, 0.01e-3),
    ('drift D seed 1 t0=8', lambda R: R['drift'][0]['D'][8], 9.48e-4, 0.01e-4),
    ('drift D seed 2 t0=128', lambda R: [x for x in R['drift'] if x['seed'] == 2][0]['D'][128], 3.7e-6, 0.1e-6),
    ('E-S GRU extrapolation', lambda R: R['es']['arms']['gru']['extrapolation'], 0.993, 0.0006),
    ('E-S Core extrapolation', lambda R: R['es']['arms']['core']['extrapolation'], 0.889, 0.0006),       # two seeds, 0.866 / 0.912 (item 60)
    ('L1 bpb step 8000', lambda R: [r for r in R['l1'] if r['step'] == 8000][0]['bits_per_byte'], 1.617, 0.0006),
    ('L1 bpb end (E-L1C)', lambda R: R['l1'][-1]['bits_per_byte'], 1.3745, 0.0006),
    ('A0F bpb end (E-L1C)', lambda R: R['a0f'][-1]['bits_per_byte'], 1.4408, 0.0006),
    ('E-S sk4 Core extrapolation, 3 seeds', lambda R: R['es4'][0][0]['extrapolation'], 0.984, 0.0006),
    ('E-S sk4 A0 in-range, 3 seeds', lambda R: R['es4'][0][1]['in_range'], 0.178, 0.0006),
    ('E-L1X D_full 2048-4096', lambda R: R['l1x'][3][-1][0], 0.494, 0.0006),
]


def compute(root):
    d = Data(root)
    es, n_es = es_reading(d)
    drift, ages = drift_rows(d)
    R = dict(s5=s5_rows(d), drift=drift, ages=ages, es=es, n_es=n_es, es4=es4_rows(d), l1=l1_rows(d), a0f=l1_rows(d, 'L1-a0-w928-s401'),
             l1x=l1x(d))
    if [r['step'] for r in R['l1']] != [r['step'] for r in R['a0f']]:
        raise SystemExit('ABORT: L1 and A0F were measured at different updates')
    bad = [(name, f(R), want) for name, f, want, tol in RECORDED if abs(f(R) - want) > tol]
    return R, bad


# ---------------------------------------------------------------- figures + README section
def figures(R, out):
    out.mkdir(parents=True, exist_ok=True)
    s1, s2, s3 = R['s5'][0], R['s5'][1], R['s5'][2]
    wins = [w for w in s1['windows'] if w[0] >= 16]
    lab = ['%d–%d' % w for w in wins]
    series = []
    for slot, row, name in ((1, s1, 'tohum 1'), (2, s2, 'tohum 2'), (3, s3, 'tohum 3')):
        series.append(dict(slot=slot, name=name, short=name, values=[1e4 * row['windows'][w][0] / row['windows'][w][1] for w in wins]))
    line_chart(out / 's5-hazard-seeds.svg', 'S₅ durum izleme: kopuş tohuma bağlı',
               'Core w128, aynı tarif (sabit lr, müfredat 1…16); konum penceresi başına ilk-hata hazard’ı',
               'Tohum 1 ve 3 eğitim uzunluğunun (16) 4-8 katından sonra kopuyor, tohum 2 kopmuyor.',
               lab, series, 'ilk-hata hazard’ı (×10⁻⁴ / adım)', lambda v: dec(v, 0) if v == int(v) else dec(v, 1),
               ymin=0, xlab='konum penceresi (eğitim uzunluğu 16)', note='Kaynak: results/Z1-pilot/perm5-core-w128-s{1,2,3}-cur.json (E-R5).')

    picks = [(1, 'perm5-core-w128-s1-cur', 'S₅ tohum 1'), (2, 'perm5-core-w128-s2-cur', 'S₅ tohum 2'), (3, 'perm4-core-w64-s1-cur', 'S₄ w64')]
    by = {x['name']: x for x in R['drift']}
    line_chart(out / 's5-state-drift.svg', 'Taşınan durumun sapması yaşla büyüyor mu?',
               'Kahin enjeksiyonundan sonra durumun doğru sınıf merkezine uzaklığının karesi (M² birimi), t0 = 128',
               'S₅ tohum 1 difüzyon gibi doğrusal büyüyor; S₅ tohum 2 ve S₄ düz.',
               R['ages'], [dict(slot=s, name=n, short=n, values=by[k]['curve']) for s, k, n in picks],
               'sapma² (M²)', lambda v: dec(v, 2), xlab='enjeksiyondan beri geçen adım (yaş)',
               note='Kaynak: results/Z1-pilot/drift-profile.json, drift-profile-s2.json (E-A5x, E-R5).')

    l1, a0f = R['l1'], R['a0f']
    steps = [('%dK' % (r['step'] // 1000)) if r['step'] % 1000 == 0 else dec(r['step'] / 1000, 1) + 'K' for r in l1]
    steps = [s if i == 0 or i == len(l1) - 1 or (l1[i]['step'] % 5000 == 0 and len(l1) - 1 - i >= 3) else ''
             for i, s in enumerate(steps)]      # 31 points: label every 5K, the first and the last (no crowding)
    line_chart(out / 'l1-bits-per-byte.svg', 'L1 dil modeli (20M, bayt düzeyi): geri beslemeli ve geri beslemesiz',
               'Doğrulama bit/bayt, her 1.000 güncellemede ölçüm pili; iki koşu aynı veri, sıra ve programla 30.518 güncelleme',
               'Geri beslemeli L1 her ölçümde geri beslemesiz ikizinden (A0F) iyi; sonda fark %s bit/bayt.'
               % ('+' + dec(a0f[-1]['bits_per_byte'] - l1[-1]['bits_per_byte'], 3)),
               steps, [dict(slot=1, name='L1 (geri beslemeli)', short='L1', values=[r['bits_per_byte'] for r in l1]),
                       dict(slot=2, name='A0F (geri beslemesiz ikiz)', short='A0F', values=[r['bits_per_byte'] for r in a0f])],
               'bit / bayt', lambda v: dec(v, 2), xlab='güncelleme', end_labels=True,
               note='Kaynak: results/L1-scratch-w928-s401 ve results/L1-a0-w928-s401, probe.jsonl (E-L1C). İkisi de CNN haber metniyle eğitildi.')
    met = [(1, 'train_k4_first', 'eğitim anahtarları'), (2, 'held_k4_first', 'görülmemiş anahtarlar'),
           (3, 'natural_first', 'doğal anahtarlar'), (4, 'single_fact_exact@384', 'tek olgu @384')]
    last = l1[-1]
    line_chart(out / 'l1-retrieval.svg', 'L1: dikkat belleğiyle getirme',
               'Anahtar eşleştirme (ilk hane doğruluğu) ve 384 bayt uzaktan tek olgu getirme (tam doğruluk)',
               'Son ölçümde eğitim anahtarları %s, görülmemiş %s, doğal %s, tek olgu @384 %s.'
               % tuple(dec(last[k], 2) for k in ('train_k4_first', 'held_k4_first', 'natural_first', 'single_fact_exact@384')),
               steps, [dict(slot=s, name=n, short=n, values=[r[k] for r in l1]) for s, k, n in met],
               'doğruluk', lambda v: dec(v, 1), ymin=0, ymax=1, xlab='güncelleme', end_labels=False,
               note='Eşikler: anahtar eşleştirme ≥ 0,8; genelleme (görülmemiş ve doğal) ≥ 0,6; uzaklık @384 ≥ 0,9.')

    j, bins, mean, diff = R['l1x']
    line_chart(out / 'l1-long-document.svg', 'L1, eğitim penceresinin (512 bayt) ötesinde',
               '32 doğrulama makalesi × 4.096 bayt; bit/bayt, konum kutusuna göre (adım 6.000)',
               'Kesintisiz koşu 512 baytın ötesinde kötüleşiyor; belleği son 512 konuma sınırlamak düzeltiyor.',
               ['%s–%s' % tuple(b.split('-')) for b in bins],
               [dict(slot=1, name='kesintisiz', short='kesintisiz', values=mean['full']),
                dict(slot=2, name='bellek penceresi 512', short='pencere 512', values=mean['win512'])],
               'bit / bayt', lambda v: dec(v, 2), xlab='konum (bayt)', end_labels=True,
               note='Durumu 512’de sıfırlamak kesintisizle aynı (1,9×10⁻⁸ bit); bağımsız 512’likler pencere 512 ile aynı (≤ 0,002).')

    es = R['es']['arms']
    names = [('core', 'Core (RLT)'), ('gru', 'GRU'), ('lstm', 'LSTM'), ('core_nofb', 'Core, geri beslemesiz'), ('tfm', 'Transformer (NoPE)')]
    bar_chart(out / 'es-extrapolation.svg', 'Durum-anahtarlı getirme, kolay sürüm (E-S, sk3: 6 olgu)',
              'Yürüyüş 32-127 derinliğinde sorgu doğruluğu (eğitim azami 16); şans 0,125',
              'GRU Core’a yetişiyor ve geçiyor; geri beslemesiz ikiz ve transformer öğrenemiyor.',
              [(n, es[k]['extrapolation'], dec(es[k]['extrapolation'], 3)) for k, n in names], lambda v: dec(v, 1), 1.0,
              'dışdeğerleme doğruluğu', note='Her kol 2 tohum ortalaması. Kaynak: results/ES-pilot/sk3-*.json.')
    rows4, _ = R['es4']
    bar_chart(out / 'es-sk4.svg', 'Durum-anahtarlı getirme, zor sürüm (E-S, sk4: 24 olgu)',
              'Yürüyüş 32-127 derinliğinde sorgu doğruluğu (eğitim azami 16); şans 0,125',
              'Yalnız geri beslemeli Core öğreniyor; GRU, LSTM ve transformer 16 kata kadar büyütülse de şans civarında.',
              [(r['label'], r['extrapolation'], dec(r['extrapolation'], 3)) for r in rows4], lambda v: dec(v, 1), 1.0,
              'dışdeğerleme doğruluğu', note='Her satır o koldaki bütün koşuların ortalaması. Kaynak: results/ES-{pilot,lr,b2x,pe,width}/sk4-*.json.')


def section(R):
    L = [BEGIN, '', '## Özet sonuçlar: tablolar ve grafikler', '',
         'Tablolar ve grafikler `scripts/make_readme_figures.py` ile ham sonuç dosyalarından hesaplanır; yalnızca kayıtlı okuma kuralıyla okunmuş '
         've `notes/ANA_BELGE_RLT.md`’de kayda geçmiş sonuçlar yer alır. Okuma kuralları, öngörüler ve sınırlar: `notes/ANA_BELGE_RLT.md` ve '
         '`notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md`. Grafikler GitHub’ın açık / koyu temasına uyar. '
         '*English: summary tables and figures recomputed from the raw result files; only results read with their registered rule are shown.*', '']
    L += ['### 1. S₅ durum izleme: kopuş tohuma bağlı, test-zamanı yakalama onu kaldırıyor', '',
          '![S₅ hazard, üç tohum](results/figures/s5-hazard-seeds.svg)', '',
          'S₅ grubunda kelime izleme (120 durum), eğitim uzunluğu 16, değerlendirme 1.024 konum (n = 100 dizi / uzunluk; yakalama n = 200). '
          '"Yakalama": taşınan durumu her 16 adımda bir, modelin kendi tahmin ettiği sınıfın merkezine çekmek (eğitimsiz, test zamanında).', '',
          '| model | tarif | aralık-içi (16) | son-durum @1.024 | ilk-hata hazard’ı / adım [Poisson %95] | yakalamayla @1.024 (olay / kişi-adım) |',
          '|---|---|---|---|---|---|']
    for r in R['s5']:
        snap = '—'
        if r['snap']:
            snap = '%s / %s%s' % (num(r['snap'][0]), num(r['snap'][1]), ' (kesin)' if r['snap'][0] == 0 else '')
        L.append('| %s | %s | %s | %s | %s [%s–%s] | %s |' % (r['model'], r['recipe'], dec(r['in_range']), dec(r['final1024'], 2),
                 sci(r['hazard']), sci(r['lo']), sci(r['hi']), snap))
    L += ['', 'Okuma: aynı tarifle üç tohumun ikisi 1.024’ten önce kopuyor, biri kopmuyor; eğitim tarafındaki dört müdahale '
          '(lr azaltma, durum gürültüsü, genişlik, öğrenilen temizleme) kopuşu gidermedi; yakalama kopuşlu iki tohumda da '
          '1.024 konumda hatasız (tohum 1’in iki modelinde 131.072 konuma kadar). Yakalama mimariye özgü değil: GRU / LSTM de her adım yakalamayla hatasız.', '']
    L += ['### 2. Neden: taşınan durum yaşla sapıyor (S₅ tohum 1’de difüzyon)', '',
          '![durum sapması](results/figures/s5-state-drift.svg)', '',
          '| model | D (M² / adım), t0 = 8 · 128 · 256 | sapma, yaş 1–4 → 64–128 (M) |', '|---|---|---|']
    label = {'perm5-core-w128-s1-cur': 'S₅ Core w128, tohum 1 (sabit lr)', 'perm5-core-w128-s1-cur-lrdecay0.3': 'S₅ Core w128, tohum 1 (lr azaltmalı)',
             'perm5-core-w128-s1-cur-lrdecay0.3-noise0.2': 'S₅ Core w128, tohum 1 (σ = 0,2)', 'perm4-core-w32-s1-cur': 'S₄ Core w32',
             'perm4-core-w64-s1-cur': 'S₄ Core w64', 'perm5-core-w128-s2-cur': 'S₅ Core w128, tohum 2 (sabit lr)'}
    for x in R['drift']:
        L.append('| %s | %s | %s → %s |' % (label.get(x['name'], x['name']), ' · '.join(sci(x['D'][t]) for t in (8, 128, 256)), dec(x['first']), dec(x['last'])))
    L += ['', 'D: yaş ≥ 4 kutularında toplam sapmanın karesinin yaşa doğrusal eğimi; M: en yakın diğer sınıf merkezine ortalama uzaklık.', '']
    es, n_es = R['es'], R['n_es']
    rows4, stage_b = R['es4']
    L += ['### 3. Durum + bellek birleşimi: kolay sürümde fark yok, zor sürümde yalnız RLT öğreniyor', '',
          'Görev (E-S, `horizon/sk_task.py`): dizinin başında her grup elemanı için bir "anahtar → değer" olgusu, ardından uzun bir yürüyüş. '
          'Model yürüyüşün o anki durumunu hatasız izlemeli ve soru işaretinde o durumun anahtarına ait değeri söylemeli: hem durum (izleme) '
          'hem bellek (geriye bakma) gerekiyor. GRU ve LSTM’in yalnız durumu, transformer’ın yalnız belleği var; RLT’de (Core) ikisi de var.', '',
          '#### Kolay sürüm (sk3: 6 olgu, aşama A)', '',
          '![E-S dışdeğerleme](results/figures/es-extrapolation.svg)', '',
          '| kol | tohum | aralık-içi (yürüyüş 16) | dışdeğerleme (yürüyüş 32–127) | durum başı dışdeğerleme | öğrendi |', '|---|---|---|---|---|---|']
    names = {'core': 'Core (RLT, w64)', 'core_nofb': 'Core, geri beslemesiz (A0)', 'gru': 'GRU w128', 'lstm': 'LSTM w112', 'tfm': 'Transformer, NoPE, 4 katman'}
    for k in ('core', 'gru', 'lstm', 'core_nofb', 'tfm'):
        a = es['arms'][k]
        L.append('| %s | %d | %s | %s | %s | %s |' % (names[k], a['seeds'], dec(a['in_range']), dec(a['extrapolation']), dec(a['state_extrapolation']),
                 'evet' if a['learned'] else 'hayır'))
    L += ['', 'Kayıtlı hüküm: **%s** (%d / 10 koşu, kesin; ana belge madde 60). 6 olgu GRU’nun tek durum vektörüne sığıyor; asıl sınav zor sürüm.'
          % (es['verdict'].replace('provisional: fewer than 2 seeds for some arm', 'geçici: bir kolda < 2 tohum'), n_es), '',
          '#### Zor sürüm (sk4: 24 olgu)', '',
          '![E-S sk4](results/figures/es-sk4.svg)', '',
          '| kol | koşu | aralık-içi (yürüyüş 16) | dışdeğerleme (yürüyüş 32–127) | en iyi koşunun dışdeğerlemesi | öğrendi |', '|---|---|---|---|---|---|']
    for r in rows4:
        L.append('| %s | %d | %s | %s | %s | %s |' % (r['label'], r['runs'], dec(r['in_range']), dec(r['extrapolation']), dec(r['best']),
                 'evet' if r['learned'] else 'hayır'))
    L += ['', 'Ortalamalar o koldaki bütün koşular üzerinden (tohumlar; lr 10⁻³ ve 3×10⁻³; iki kat bütçe); şans 0,125 (8 değer); “öğrendi”: aralık-içi ≥ 0,95. '
          'Kayıtlı hükümler: aşama B **%s** (bu sayfada yeniden hesaplandı; ana belge madde 83 ve 105), lr denetimi **ROBUST TO LR** (madde 83), '
          'iki kat bütçe **ROBUST TO BUDGET** (madde 96), konumlu transformer **ADVANTAGE HOLDS** (madde 104), 4-16 kat parametre '
          '**ADVANTAGE HOLDS AT 16x PARAMETERS** (madde 121). Olgular çoğalınca durum ve belleği birlikte kullanan tek yapı geri beslemeli Core; '
          'aynı Core’un geri beslemesiz ikizi de öğrenemiyor. Kampanyanın şimdiye kadarki en sağlam olumlu sonucu.' % stage_b['verdict'], '']
    l1, a0f = R['l1'], R['a0f']
    by_a0 = {r['step']: r for r in a0f}
    pick = [r for r in l1 if r['step'] in (1000, 4000, 8000, 16000, 24000)] + [l1[-1]]
    L += ['### 4. L1 dil modeli (20M parametre, bayt düzeyi, sıfırdan) ve geri beslemesiz ikizi', '',
          '![L1 bit/bayt](results/figures/l1-bits-per-byte.svg)', '', '![L1 getirme](results/figures/l1-retrieval.svg)', '',
          '| adım | L1 bit/bayt | A0F bit/bayt | fark (A0F − L1) | eğitim anahtarları | görülmemiş anahtarlar | doğal anahtarlar | tek olgu @384 |',
          '|---|---|---|---|---|---|---|---|']
    for r in pick:
        gap = by_a0[r['step']]['bits_per_byte'] - r['bits_per_byte']
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (num(r['step']), dec(r['bits_per_byte']), dec(by_a0[r['step']]['bits_per_byte']),
                 ('+' if gap >= 0 else '−') + dec(abs(gap)), dec(r['train_k4_first']), dec(r['held_k4_first']), dec(r['natural_first']),
                 dec(r['single_fact_exact@384'])))
    L += ['', 'Anahtar sütunları L1’in ilk hane doğruluğu (şans 0,1), tek olgu tam doğruluk. L1 500M baytın tamamını okudu (%s güncelleme); '
          'geri beslemesiz ikiz A0F aynı veri, sıra, başlangıç ağırlıkları ve programla eğitildi. Kayıtlı hüküm: **FEEDBACK HELPS (AT END)** (madde 123). '
          'Doğrulamanın eğitimde de geçen belgeleri çıkarılınca fark +0,069 (madde 140); BLiMP dilbilgisi testinde L1 %%64,83, A0F %%61,90 (madde 135). '
          'Bedeli: ekran kartı süresi ≈ 7,6 kat. İkisi de CNN haber metniyle eğitildi; kullanıcı kuralıyla (madde 138) yeni eğitimler FineWeb-Edu’da.'
          % num(l1[-1]['step']), '']
    j, bins, mean, diff = R['l1x']
    L += ['### 5. L1, eğitim penceresinin ötesinde (E-L1X, adım 6.000)', '', '![L1 uzun belge](results/figures/l1-long-document.svg)', '',
          '| konum (bayt) | kesintisiz | bellek penceresi 512 | durum 512’de sıfırlanıyor | bağımsız 512’likler | kesintisiz − bağımsız (±SE) |', '|---|---|---|---|---|---|']
    for i, b in enumerate(bins):
        lo, hi = b.split('-')
        L.append('| %s–%s | %s | %s | %s | %s | %s ± %s |' % (num(lo), num(hi), dec(mean['full'][i]), dec(mean['win512'][i]), dec(mean['reset512'][i]),
                 dec(mean['chunk512'][i]), ('+' if diff[i][0] >= 0 else '−') + dec(abs(diff[i][0])), dec(diff[i][1])))
    L += ['', 'Kayıtlı hüküm: **%s, %s**. Bozulma dikkat belleğinde; çıkarımda belleği eğitim uzunluğuyla sınırlamak ücretsiz ve tam bir düzeltme; '
          'taşınan durum 256 adımdan öteye bilgi taşımıyor.' % (j['reading']['verdict'], j['reading']['attribution']), '',
          'Sonraki ölçüm (E-L1X-F, madde 129 ve 132): eğitimin her noktasında taşınan durum ≈ 6 adımda unutuyor, etkin bağlam 512 bayt. '
          'Hata avında nedeni bulundu (madde 139): L1’in eğitiminde her 512 baytlık satır aynı başlangıç durumuyla başlıyordu ve kodlayıcı satırın '
          'tamamını görüyordu; durumun satırın ötesini taşıması eğitimde hiç istenmedi. Durumun pencereler arası tek kanal olduğu eğitim E-SC1’de '
          'sınanıyor (madde 145).', '', END]
    return '\n'.join(L) + '\n'


def write_readme(readme, block):
    text = readme.read_text(encoding='utf8') if readme.exists() else ''
    if BEGIN in text and END in text:
        text = re.sub(re.escape(BEGIN) + '.*?' + re.escape(END) + r'\n?', lambda m: block, text, flags=re.S)
    else:
        text = text.rstrip('\n') + '\n\n' + block
    readme.write_bytes(text.encode('utf8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='results/figures')
    ap.add_argument('--readme', default='README.md')
    ap.add_argument('--check', action='store_true', help='recompute and compare with the recorded values only')
    a = ap.parse_args()
    R, bad = compute(a.root)
    for name, got, want in bad:
        print('MISMATCH %s: recomputed %r, recorded %r' % (name, got, want))
    if bad:
        sys.exit('ABORT: a recomputed number disagrees with the recorded one; nothing written')
    print('recomputed numbers match the %d recorded values' % len(RECORDED))
    if a.check:
        return
    root = Path(a.root)
    out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    figures(R, out)
    readme = Path(a.readme) if Path(a.readme).is_absolute() else root / a.readme
    write_readme(readme, section(R))
    print('wrote %d figures to %s and the README section to %s' % (len(list(out.glob('*.svg'))), out, readme))


if __name__ == '__main__':
    main()
