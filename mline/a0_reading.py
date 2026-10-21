"""E-A0 reading (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.41), written and tested before the run ended.

The A0 twin (feedback off) against L1 (feedback on) at updates 1,000 ... 8,000, from the battery rows of each run's probe.jsonl.
delta_k = bpb_A0(k) - bpb_L1(k) (positive: the feedback run is better).
  FEEDBACK HELPS            delta_8000 >= +0.020 and delta_k > 0 at >= 6 of the 8 points
  FEEDBACK HURTS            delta_8000 <= -0.020 and delta_k < 0 at >= 6 of the 8 points
  NO MEANINGFUL DIFFERENCE  |delta_8000| < 0.020
  MIXED                     otherwise
  INCOMPLETE                a battery point is missing in either run
Retrieval metrics are compared only when both runs reach train_k4_first >= 0.95 at 8,000 (else VACUOUS); descriptive, no verdict.

  python -B -m mline.a0_reading [--a0 results/L1-a0-w928-s401] [--l1 results/L1-scratch-w928-s401] [--out FILE]
"""
import argparse
import json
from pathlib import Path

POINTS = tuple(range(1000, 8001, 1000))
EFFECT = 0.020
MAJORITY = 6
VALID_IN_RANGE = 0.95
TOL = 1e-9
RETRIEVAL = ('train_k4_first', 'held_k4_first', 'natural_first', 'single_fact_exact@64', 'single_fact_exact@128',
             'single_fact_exact@256', 'single_fact_exact@384', 'single_fact_exact@448', 'attention_conditionality_mean')


def battery(run_dir):
    """{step: eval dict} from a run's probe.jsonl (the last row of a step wins: a resumed run rewrites its tail)."""
    rows = {}
    for line in (Path(run_dir) / 'probe.jsonl').read_text(encoding='utf8').splitlines():
        if line.strip():
            r = json.loads(line)
            if isinstance(r.get('eval'), dict) and 'bits_per_byte' in r['eval']:
                rows[int(r['step'])] = r['eval']
    return rows


def reading(a0, l1, points=POINTS):
    missing = [k for k in points if k not in a0 or k not in l1 or a0[k].get('bits_per_byte') is None or l1[k].get('bits_per_byte') is None]
    if missing:
        return dict(verdict='INCOMPLETE', missing=missing)
    deltas = {k: a0[k]['bits_per_byte'] - l1[k]['bits_per_byte'] for k in points}
    last = deltas[points[-1]]
    positive = sum(d > 0 for d in deltas.values())
    negative = sum(d < 0 for d in deltas.values())
    if last >= EFFECT - TOL and positive >= MAJORITY:
        verdict = 'FEEDBACK HELPS'
    elif last <= -EFFECT + TOL and negative >= MAJORITY:
        verdict = 'FEEDBACK HURTS'
    elif abs(last) < EFFECT - TOL:
        verdict = 'NO MEANINGFUL DIFFERENCE'
    else:
        verdict = 'MIXED'
    k = points[-1]
    valid = (a0[k].get('train_k4_first') or 0) >= VALID_IN_RANGE - TOL and (l1[k].get('train_k4_first') or 0) >= VALID_IN_RANGE - TOL
    retrieval = {m: dict(a0=a0[k].get(m), l1=l1[k].get(m)) for m in RETRIEVAL}
    return dict(verdict=verdict, deltas={str(s): d for s, d in deltas.items()}, delta_last=last, positive=positive, negative=negative,
                retrieval_valid=valid, retrieval=retrieval if valid else 'VACUOUS (train_k4_first < 0.95 at %d in a run)' % k,
                retrieval_raw=retrieval)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a0', default='results/L1-a0-w928-s401')
    p.add_argument('--l1', default='results/L1-scratch-w928-s401')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    r = reading(battery(a.a0), battery(a.l1))
    text = json.dumps(r, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
