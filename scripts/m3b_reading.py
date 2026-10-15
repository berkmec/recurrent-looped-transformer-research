"""M-3b seed split: the registered decision rule (notes/_hive_M3b_TASARIM.md section 4, commit 8857f99), applied mechanically to the 9 result files.

  python -B -m scripts.m3b_reading                 # reads results/M3b/perm4-gru-w64-w<W>d<D>.json, prints the table and the verdict
  python -B -m scripts.m3b_reading --json out.json # also writes the reading (refuses to overwrite)

Rule, letter of 8857f99 (nothing added):
  cell metric      y = log10(pooled first-error hazard) (`readings.pooled_rate`, the E-D/E-P/es5seeds machine); a cell with 0 events gets +0.5 event
  exclusion        in-range accuracy (curve['16']['final']) < 0.99 -> cell out of the analysis, reported by name; > 2 cells out -> INCOMPLETE
  decomposition    3x3 two-way without replication: MS_w (init, df 2), MS_d (data, df 2), MS_res (df 4);
                   s2w = (MS_w - MS_res)/3, s2d = (MS_d - MS_res)/3, negative -> 0 (+ note)
  verdict          s2d >= s2w -> ÇÜRÜTÜLDÜ;  s2d < s2w/4 -> DESTEKLENİR;  otherwise KISMEN
The 0.99 comparison uses the repository tolerance (`readings.ge`, 1e-6, blinded review F6): accuracies are float32 means, 198/200 is stored as 0.99000001.
Rule gap (not filled here): the formulas assume the full 3x3; with 1 or 2 cells excluded the registration gives no unbalanced variant, so the reading
returns verdict None with reason RULE_GAP instead of inventing one (decision belongs to the rule owner).
"""
import argparse
import json
import math
import sys
from pathlib import Path

from horizon.readings import ge, pooled_rate

WSEEDS = (1, 2, 3)
DSEEDS = (101, 102, 103)
IN_RANGE_MIN = 0.99
MAX_EXCLUDED = 2
CONTINUITY = 0.5
BAND = 4.0
RESULTS = Path('results/M3b')

SUPPORTED, REFUTED, PARTIAL, INCOMPLETE = 'DESTEKLENİR', 'ÇÜRÜTÜLDÜ', 'KISMEN', 'INCOMPLETE'


def cell_path(w, d, root=RESULTS):
    return Path(root) / f'perm4-gru-w64-w{w}d{d}.json'


def cell_metric(result):
    """(y, events, exposure, in_range) for one result json; y = log10 hazard with +0.5 event when there are none."""
    _, events, exposure = pooled_rate(result)
    if not exposure:
        raise ValueError('no hazard exposure in the result')
    y = math.log10((events if events > 0 else CONTINUITY) / exposure)
    return y, events, exposure, float(result['curve']['16']['final'])


def anova(y):
    """Two-way ANOVA without replication on a full {(w, d): y} 3x3 grid."""
    g = sum(y[w, d] for w in WSEEDS for d in DSEEDS) / 9
    rows = {w: sum(y[w, d] for d in DSEEDS) / 3 for w in WSEEDS}
    cols = {d: sum(y[w, d] for w in WSEEDS) / 3 for d in DSEEDS}
    ss_w = 3 * sum((rows[w] - g) ** 2 for w in WSEEDS)
    ss_d = 3 * sum((cols[d] - g) ** 2 for d in DSEEDS)
    ss_t = sum((y[w, d] - g) ** 2 for w in WSEEDS for d in DSEEDS)
    ms_w, ms_d, ms_res = ss_w / 2, ss_d / 2, (ss_t - ss_w - ss_d) / 4
    s2w_raw, s2d_raw = (ms_w - ms_res) / 3, (ms_d - ms_res) / 3
    return dict(grand_mean=g, row_means=rows, col_means=cols, MS_w=ms_w, MS_d=ms_d, MS_res=ms_res,
                s2w_raw=s2w_raw, s2d_raw=s2d_raw, s2w=max(0.0, s2w_raw), s2d=max(0.0, s2d_raw))


def verdict(s2w, s2d):
    if s2d >= s2w:
        return REFUTED
    if s2d < s2w / BAND:
        return SUPPORTED
    return PARTIAL


def reading(cells):
    """cells: {(w, d): result json or None (missing)}. Returns the reading dict (verdict + everything it rests on)."""
    table, excluded, missing = {}, [], []
    for w in WSEEDS:
        for d in DSEEDS:
            r = cells.get((w, d))
            if r is None:
                missing.append(f'w{w}d{d}')
                continue
            y, ev, ex, inr = cell_metric(r)
            table[w, d] = dict(y=y, events=ev, exposure=ex, in_range=inr, continuity=ev == 0)
            if not ge(inr, IN_RANGE_MIN):
                excluded.append(f'w{w}d{d}')
    out = dict(rule='8857f99 notes/_hive_M3b_TASARIM.md s4', cells={f'w{w}d{d}': v for (w, d), v in table.items()},
               excluded=excluded, missing=missing, notes=[])
    if missing:
        out.update(verdict=INCOMPLETE, reason='missing cells')
        return out
    if len(excluded) > MAX_EXCLUDED:
        out.update(verdict=INCOMPLETE, reason=f'{len(excluded)} cells below in-range {IN_RANGE_MIN} (> {MAX_EXCLUDED})')
        return out
    if excluded:
        out.update(verdict=None, reason='RULE_GAP: registered ANOVA assumes the full 3x3; no rule for 1-2 excluded cells')
        return out
    a = anova({k: v['y'] for k, v in table.items()})
    for name in ('s2w', 's2d'):
        if a[name + '_raw'] < 0:
            out['notes'].append(f'{name} negative ({a[name + "_raw"]:.4f}) -> 0')
    out.update(anova={k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in a.items()},
               verdict=verdict(a['s2w'], a['s2d']), reason='')
    return out


def load_cells(root=RESULTS):
    cells = {}
    for w in WSEEDS:
        for d in DSEEDS:
            p = cell_path(w, d, root)
            cells[w, d] = json.loads(p.read_text(encoding='utf8')) if p.exists() else None
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=str(RESULTS))
    ap.add_argument('--json', default=None, help='write the reading here (refuses to overwrite)')
    args = ap.parse_args(argv)
    out = reading(load_cells(args.root))
    for k, v in out['cells'].items():
        print(f"{k}: y={v['y']:.4f} events={v['events']} exposure={v['exposure']} in16={v['in_range']:.4f}" + (' (+0.5 event)' if v['continuity'] else ''))
    if 'anova' in out:
        a = out['anova']
        print(f"MS_w={a['MS_w']:.4f} MS_d={a['MS_d']:.4f} MS_res={a['MS_res']:.4f} s2w={a['s2w']:.4f} s2d={a['s2d']:.4f} (raw {a['s2d_raw']:.4f}) s2w/4={a['s2w'] / BAND:.4f}")
    for n in out['notes']:
        print('note:', n)
    print('excluded:', out['excluded'] or '-', '| missing:', out['missing'] or '-')
    print('VERDICT:', out['verdict'], out['reason'])
    if args.json:
        p = Path(args.json)
        if p.exists():
            sys.exit(f'refusing to overwrite {p}')
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf8')
    return out


if __name__ == '__main__':
    main()
