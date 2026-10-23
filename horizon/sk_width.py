"""E-S-W reading (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.57): do the sk4 baselines that failed at the Core's size
learn with about 4x-16x its parameters? A cell (arm, width, peak lr) learns when its seed-mean in_range reaches the stage-B threshold.

    python -B -m horizon.sk_width results/ES-width/sk4-*.json
"""
import argparse
import json
import os
import re
from collections import defaultdict

CORE_PARAMS = 100960                                                    # Core w64 and its no-feedback twin (sk_run.make_arm)
CELLS = {('gru', 256, 0.003): 417568, ('lstm', 224, 0.003): 423168,     # about 4x, seeds 1-2
         ('gru', 512, 0.001): 1621536, ('lstm', 448, 0.001): 1649120,   # about 16x, seed 1
         ('tfm_pe', 128, 0.001): 804768}                                 # about 8x, seed 1
SMALL = (('gru', 256, 0.003), ('lstm', 224, 0.003))
LEARNS, PARTIAL = 0.95, 0.50
NAME = re.compile(r'^sk4-(?P<arm>[a-z_]+)-w(?P<w>\d+)-s(?P<s>\d+)-lr(?P<lr>[0-9.]+)\.json$')


def _label(cell):
    return '%s-w%d-lr%g' % cell


def load(paths):
    """{(arm, width, lr): [(in_range, extrapolation), ...]} from result files; manifests and other names are skipped."""
    runs = defaultdict(list)
    for p in paths:
        m = NAME.match(os.path.basename(p))
        if m is None:
            continue
        with open(p, encoding='utf8') as f:
            r = json.load(f)
        runs[(m['arm'], int(m['w']), float(m['lr']))].append((r['in_range'], r['extrapolation']))
    return dict(runs)


def reading(runs):
    table = {}
    for cell, rows in runs.items():
        ir, ex = sum(a for a, _ in rows) / len(rows), sum(b for _, b in rows) / len(rows)
        table[cell] = dict(seeds=len(rows), in_range=ir, extrapolation=ex, ratio=CELLS.get(cell, 0) / CORE_PARAMS,
                           status='LEARNS' if ir >= LEARNS else 'PARTIAL' if ir >= PARTIAL else 'NOT LEARNED')
    learned = [c for c in table if table[c]['status'] == 'LEARNS']
    partial = [c for c in table if table[c]['status'] == 'PARTIAL']
    missing = [c for c in CELLS if c not in table]
    if any(c in SMALL for c in learned):
        verdict = 'CAPACITY EXPLAINS IT AT 4x'
    elif learned:
        verdict = 'CAPACITY EXPLAINS IT AT 8-16x'
        if any(table[c]['seeds'] < 2 for c in learned):
            verdict += ' (seed 2 needed before the verdict)'
    elif not missing:
        verdict = 'ADVANTAGE HOLDS AT 16x PARAMETERS'
    elif not any(c in missing for c in SMALL):
        verdict = 'ADVANTAGE HOLDS AT 4x (larger cells pending)'
    else:
        verdict = 'INCOMPLETE'
    if partial:
        verdict += '; PARTIAL: ' + ', '.join(_label(c) for c in sorted(partial))
    return dict(verdict=verdict, cells={_label(c): v for c, v in sorted(table.items())}, missing=[_label(c) for c in missing])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+')
    print(json.dumps(reading(load(p.parse_args().paths)), indent=1))


if __name__ == '__main__':
    main()
