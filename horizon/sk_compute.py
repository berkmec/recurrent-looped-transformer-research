"""E-S-C reading (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.58): do the sk4 arms that failed at 15,000 updates learn
when they get about the Core's CPU wall-clock instead of its update count? A cell (arm, width, updates) learns when its seed-mean
in_range reaches the stage-B threshold.

    python -B -m horizon.sk_compute results/ES-compute/sk4-*.json
"""
import argparse
import json
import os
import re
from collections import defaultdict

CORE_UPDATES = 15000
# Stage-B seed-1 wall-clock for 15,000 updates (descriptive; the queues differed): Core 6,877 s, core_nofb 1,412 s, GRU w128 794 s,
# LSTM w112 647 s, positional transformer w64 3,043 s (E-S-PE). The budgets round the Core-to-arm ratio up.
CELLS = {('gru', 128, 150000): 6877 / 794, ('lstm', 112, 165000): 6877 / 647,
         ('core_nofb', 64, 75000): 6877 / 1412, ('tfm_pe', 64, 60000): 6877 / 3043}
LEARNS, PARTIAL = 0.95, 0.50
NAME = re.compile(r'^sk4-(?P<arm>[a-z_]+)-w(?P<w>\d+)-s(?P<s>\d+)-u(?P<u>\d+)\.json$')


def _label(cell):
    return '%s-w%d-u%d' % cell


def load(paths):
    """{(arm, width, updates): [(in_range, extrapolation), ...]} from result files; manifests and other names are skipped."""
    runs = defaultdict(list)
    for p in paths:
        m = NAME.match(os.path.basename(p))
        if m is None:
            continue
        with open(p, encoding='utf8') as f:
            r = json.load(f)
        runs[(m['arm'], int(m['w']), int(m['u']))].append((r['in_range'], r['extrapolation']))
    return dict(runs)


def reading(runs):
    table = {}
    for cell, rows in runs.items():
        ir, ex = sum(a for a, _ in rows) / len(rows), sum(b for _, b in rows) / len(rows)
        table[cell] = dict(seeds=len(rows), in_range=ir, extrapolation=ex, budget=cell[2] / CORE_UPDATES,
                           wall_ratio=CELLS.get(cell), status='LEARNS' if ir >= LEARNS else 'PARTIAL' if ir >= PARTIAL else 'NOT LEARNED')
    learned = sorted(c for c in table if table[c]['status'] == 'LEARNS')
    partial = sorted(c for c in table if table[c]['status'] == 'PARTIAL')
    missing = [c for c in CELLS if c not in table]
    if learned:
        verdict = 'COMPUTE EXPLAINS IT: ' + ', '.join(_label(c) for c in learned)
        if any(table[c]['seeds'] < 2 for c in learned):
            verdict += ' (seed 2 needed before the verdict)'
    elif not missing:
        verdict = 'ADVANTAGE HOLDS AT EQUAL WALL-CLOCK'
    else:
        verdict = 'INCOMPLETE'
    if partial:
        verdict += '; PARTIAL: ' + ', '.join(_label(c) for c in partial)
    return dict(verdict=verdict, cells={_label(c): v for c, v in sorted(table.items())}, missing=[_label(c) for c in missing])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+')
    print(json.dumps(reading(load(p.parse_args().paths)), indent=1))


if __name__ == '__main__':
    main()
