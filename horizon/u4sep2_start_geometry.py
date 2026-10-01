"""What the gap/drift ratio actually was BEFORE U4-sep2 fine-tuned anything, measured rather than assumed.

`results/U4sep2/analysis.json` carries a field `ratio_u2c` that is meant to be the starting point of each pair -
the geometry of the frozen U2c checkpoint both arms were initialised from. It is not. The sealed analyser sets it
with one line:

    entry['ratio_u2c'] = entry.get('ratio_none')   # the pair starts from the same weights, so the control's first ratio is the start

and `ratio_none` is `train.json['ratio_last']`, the ratio at the LAST fine-tuning step of the control arm. Two
errors compound in that comment. The control arm was itself fine-tuned for 512 steps, so its geometry has moved;
and even at step one the value would be `ratio_first`, not `ratio_last`. The field is also measured over the 64
neutral slots the fine-tuner uses, while `eval_ratio_none` and `eval_ratio_hinge` - the numbers it is printed
beside - come from the 200-sequence evaluation. Comparing them is the same units mistake that produced the
U4-sep2 headline correction.

Nothing in the cycle's conclusions depends on `ratio_u2c`: it is read by no verdict, no sign test, no audit check
and no sentence of the report. That is why this module computes the honest number in a new file rather than
reaching into a sealed one. `horizon/u4sep2_analyze.py`, `results/U4sep2/analysis.json` and the report are all
hashed in the U4-sep2 r2 manifest, and a wrong-but-unused field is not worth breaking a seal over; a silent one
is worth recording.

The measurement here is the same call `sep_finetune.evaluate_one` makes, on the model `load_source` returns
before any fine-tuning step has run.

Usage: python -B -m horizon.u4sep2_start_geometry --out results/U4sep2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from horizon import sep_finetune as sf
from horizon import sweep as sw
from horizon import u2d_compare as ud


def measure_start(out='results/U4sep2'):
    """Per pair: the frozen U2c checkpoint's own gap, drift and ratio, on the evaluation measurement."""
    torch.set_num_threads(1)                      # a background job must not compete with whatever holds the GPU
    out = Path(out); manifest = sf.verify_frozen(out); p = manifest['protocol']
    analysis = json.loads((out / 'analysis.json').read_text(encoding='utf8'))
    rows = {}
    for c in sf.conditions(p):
        if c['arm'] != 'none':                    # both arms of a pair share one source, so measure it once
            continue
        key = f"{c['source']['arm']}/s{c['source']['seed']}"
        model, sc, u2c_p, source_sha = sf.load_source(c, p)
        m = ud.measure(model, sc, u2c_p, n_seq=p['n_geometry'])
        gap, drift = m['anchors']['min_center_distance'], m['displacement']['end']
        recorded = analysis['pairs'].get(key, {})
        start = (gap / drift) if drift > 1e-9 else None
        claimed = recorded.get('ratio_u2c')
        rows[key] = dict(source_sha256=source_sha, gap=gap, drift=drift, start_ratio=start,
                         recorded_ratio_u2c=claimed,
                         factor=(claimed / start) if (start and claimed) else None,
                         eval_ratio_none=recorded.get('eval_ratio_none'),
                         eval_ratio_hinge=recorded.get('eval_ratio_hinge'))
    return manifest, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/U4sep2')
    args = parser.parse_args()
    manifest, rows = measure_start(args.out)
    print('%-22s %10s %10s %10s %10s %8s' % ('pair', 'gap', 'drift', 'TRUE start', 'recorded', 'factor'))
    for key, r in rows.items():
        print('%-22s %10.4f %10.4f %10.4f %10.4f %7.2fx'
              % (key, r['gap'], r['drift'], r['start_ratio'], r['recorded_ratio_u2c'], r['factor']))
    worst = max((abs(r['factor']) for r in rows.values() if r['factor']), default=None)
    report = dict(kind='U4SEP2_START_GEOMETRY', utc=time.time(), protocol_hash=manifest['protocol_hash'],
                  pairs=rows, worst_factor=worst,
                  finding='results/U4sep2/analysis.json records ratio_u2c as the control arm LAST fine-tuning-step '
                          'ratio over 64 neutral slots, not the frozen checkpoint geometry over the 200-sequence '
                          'evaluation. The field is read by no verdict, sign test, audit check or report sentence.')
    sw.write_json(Path(args.out, 'start-geometry.json'), report)
    print('\nworst factor %.2fx  ->  %s' % (worst, Path(args.out, 'start-geometry.json')))


if __name__ == '__main__':
    main()
