"""E-WS reading (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.48), written and tested before the runs: feedback switched
on after a full no-feedback run. From A0's final weights, two arms run the same 2,000-update schedule on the same batches - the
A0 twin continued (feedback off) and the same weights with feedback on (RLT-1). delta_k = bpb_continue(k) - bpb_feedback(k) at the
batteries 500 ... 2,000 (positive: feedback better).
  WARM START HELPS   delta_2000 >= +0.010 and delta > 0 at 1,500 and 2,000
  WARM START HURTS   delta_2000 <= -0.010 and delta < 0 at 1,500 and 2,000
  NO DIFFERENCE      |delta_2000| < 0.010
  MIXED              otherwise;  INCOMPLETE if a battery is missing

  python -B -m mline.ews [--cont results/EWS/a0-cont] [--fb results/EWS/fb-ft] [--out FILE]
"""
import argparse
import json
from pathlib import Path

from mline.a0_reading import battery

POINTS = (500, 1000, 1500, 2000)
EFFECT = 0.010
TOL = 1e-9


def reading(cont, fb, points=POINTS):
    missing = [k for k in points if k not in cont or k not in fb]
    if missing:
        return dict(verdict='INCOMPLETE', missing=missing)
    deltas = {k: cont[k]['bits_per_byte'] - fb[k]['bits_per_byte'] for k in points}
    last, prev = deltas[points[-1]], deltas[points[-2]]
    if last >= EFFECT - TOL and prev > 0:
        verdict = 'WARM START HELPS'
    elif last <= -EFFECT + TOL and prev < 0:
        verdict = 'WARM START HURTS'
    elif abs(last) < EFFECT - TOL:
        verdict = 'NO DIFFERENCE'
    else:
        verdict = 'MIXED'
    k = points[-1]
    return dict(verdict=verdict, deltas={str(s): d for s, d in deltas.items()}, delta_last=last,
                bpb=dict(cont={str(s): cont[s]['bits_per_byte'] for s in points}, fb={str(s): fb[s]['bits_per_byte'] for s in points}),
                retrieval={m: dict(cont=cont[k].get(m), fb=fb[k].get(m)) for m in ('train_k4_first', 'held_k4_first', 'natural_first',
                                                                                   'single_fact_exact@448')})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cont', default='results/EWS/a0-cont')
    p.add_argument('--fb', default='results/EWS/fb-ft')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    r = reading(battery(a.cont), battery(a.fb))
    text = json.dumps(r, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_bytes((text + '\n').encode('utf8'))


if __name__ == '__main__':
    main()
