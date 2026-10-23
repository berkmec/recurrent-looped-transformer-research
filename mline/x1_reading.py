"""E-X1 reading (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.59): L1, the RLT language model, against a standard
transformer of its parameter budget (mline.tfm, arm tfm6) on the same recipe, data, seed and schedule, at E-L1C's points
12,000 ... 30,518.

The rule is E-A0's (mline.a0_reading.reading) with the transformer in the reference seat: delta_k = bpb_T(k) - bpb_L1(k), so a
positive delta means L1 is better; effect 0.020 bit/byte at the last point and the sign at 5 of the 6 points. The transformer
against the no-feedback twin A0F is reported the same way, descriptively.

    python -B -m mline.x1_reading [--t ...] [--l1 ...] [--a0 ...] [--out reading-x1.json]
"""
import argparse
import json
from pathlib import Path

from mline import a0_reading as ar

LABELS = {'FEEDBACK HELPS': 'RLT BEATS TRANSFORMER', 'FEEDBACK HURTS': 'TRANSFORMER BEATS RLT'}
TWIN_LABELS = {'FEEDBACK HELPS': 'A0 BEATS TRANSFORMER', 'FEEDBACK HURTS': 'TRANSFORMER BEATS A0'}


def reading(t, l1, a0=None):
    r = ar.reading(t, l1, ar.END_POINTS, ar.END_MAJORITY)
    r['verdict'] = LABELS.get(r['verdict'], r['verdict'])
    if a0 is not None:
        d = ar.reading(t, a0, ar.END_POINTS, ar.END_MAJORITY)
        r['transformer_vs_a0'] = dict(verdict=TWIN_LABELS.get(d['verdict'], d['verdict']), deltas=d.get('deltas'),
                                      delta_last=d.get('delta_last'))
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--t', default='results/L1-tfm6-w528-s401')
    p.add_argument('--l1', default='results/L1-scratch-w928-s401')
    p.add_argument('--a0', default='results/L1-a0-w928-s401')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    r = reading(ar.battery(a.t), ar.battery(a.l1), ar.battery(a.a0))
    text = json.dumps(r, indent=1, default=float)
    if a.out:
        Path(a.out).write_bytes((text + '\n').encode('utf8'))
    print(text)


if __name__ == '__main__':
    main()
