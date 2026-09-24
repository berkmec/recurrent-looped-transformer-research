"""M1: score the 30M arms against predictions registered before any of them was trained.

The question is transfer, not novelty: the exact-preservation gate is the one thing the campaign replicated across
3 seeds x 2 task families x 2 learning rates on toy automata. Does it still do anything at 30M on real text?

Predictions, fixed here in code:
  (a) hard_st does not cost language ability - bits per byte within 0.02 of plain;
  (b) hard_st improves needle exact-match over plain by more than 0.05 at BOTH long fillers (512 and 2048);
  (c) soft stays below hard_st on the long fillers, because language state carries contents the way `stack` did and
      U2c measured soft mixing leaking on exactly that kind of state.
Falsification: if (b) fails, the gate is a property of the toy setting and the M-line stops at rung one.

(c) is recorded but does not change the status: it is a prediction about WHY, not about whether the gate transfers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

THRESHOLDS = dict(bpb_regression=.02, needle_gain=.05, long_filler=[512, 2048], measurable_floor=.02)


def _mean_needle(row, n, kinds=('repeat', 'text')):
    vals = [row['needle'][f'{k}@{n}'] for k in kinds if f'{k}@{n}' in row['needle']]
    return sum(vals) / len(vals) if vals else None


def verdict(arms, seeds=1, thresholds=None):
    """`arms` maps arm name -> {bits_per_byte, needle: {kind@N: exact_match}}."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    if not {'plain', 'hard_st', 'soft'} <= set(arms):
        return dict(status='INCOMPLETE', meaning='not all three arms have finished', arms=sorted(arms), provisional=True)
    plain, hard, soft = arms['plain'], arms['hard_st'], arms['soft']
    bpb_cost = hard['bits_per_byte'] - plain['bits_per_byte']
    gains = {n: (_mean_needle(hard, n) - _mean_needle(plain, n)) for n in t['long_filler']}
    soft_below = all(_mean_needle(soft, n) < _mean_needle(hard, n) for n in t['long_filler'])
    a = bpb_cost <= t['bpb_regression']
    b = all(g > t['needle_gain'] for g in gains.values())
    # If no arm can do the task at the long fillers, the comparison has no resolution: a gate that helps would be
    # invisible. Reporting that as GATE_DOES_NOT_TRANSFER converts "we could not measure it" into "it does not work",
    # which is the error an INTERVENTION_INEFFECTIVE branch caught twice in the two preceding cycles.
    at_floor = all(max(_mean_needle(arms[k], n) or 0. for n in t['long_filler']) <= t['measurable_floor'] for k in arms)
    if at_floor:
        status = 'NOT_MEASURABLE_AT_THIS_BUDGET'
    elif not b:
        status = 'GATE_DOES_NOT_TRANSFER'
    elif not a:
        status = 'GATE_AT_A_COST'
    else:
        status = 'GATE_TRANSFERS'
    notes = []
    if not soft_below:
        notes.append('soft matched or beat hard_st on a long filler, which contradicts the U2c stack result that '
                     'motivated prediction (c); the reason for the gate working would then be different from the one assumed')
    if seeds < 2:
        notes.append('single seed: this is a first look, not a replicated result')
    return dict(status=status, thresholds=t, provisional=seeds < 2, seeds=seeds,
                prediction_a_no_bpb_cost=a, prediction_b_needle_gain=b, prediction_c_soft_below_hard=soft_below,
                bits_per_byte={k: v['bits_per_byte'] for k, v in arms.items()}, bpb_cost_of_gate=bpb_cost,
                needle_gain_over_plain=gains,
                needle_by_arm={k: {str(n): _mean_needle(v, n) for n in t['long_filler']} for k, v in arms.items()},
                notes='; '.join(notes) or 'none',
                meaning=dict(
                    GATE_TRANSFERS='the exact-preservation gate helps at 30M on real text and costs no language ability: '
                                   'the toy result transfers and the M-line has a rung to build on',
                    GATE_AT_A_COST='the gate improves the needle but costs bits per byte; a trade-off, not a free win',
                    GATE_DOES_NOT_TRANSFER='no needle gain at 30M: the gate is a property of the toy setting and the '
                                           'M-line stops at rung one',
                    NOT_MEASURABLE_AT_THIS_BUDGET='no arm, including the control, does better than the floor at the long '
                                                  'fillers, so this budget could not have detected a gate benefit either '
                                                  'way; the result is about the budget, not about the gate',
                    INCOMPLETE='not all three arms have finished')[status])


def collect(out):
    from mline import train as mt
    out = Path(out); manifest = mt.verify_frozen(out); p = manifest['protocol']
    arms, rows = {}, {}
    for c in mt.conditions(p):
        path = out / mt.ident(c) / 'train.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding='utf8'))
        rows[mt.ident(c)] = r
        arms[c['arm']] = dict(bits_per_byte=r['selected_validation']['bits_per_byte'],
                              needle={k: v['exact_match'] for k, v in r['needle'].items()},
                              needle_bytes={k: v['byte_accuracy'] for k, v in r['needle'].items()},
                              parameters=r['parameters']['total'], guard=r['guard'], selected_step=r['selected_step'])
    return manifest, arms, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/M1')
    args = parser.parse_args()
    manifest, arms, rows = collect(args.out)
    p = manifest['protocol']
    print('%-9s %10s %s' % ('arm', 'bpb', ' '.join('%11s' % f'{k}@{n}' for k in p['needle_kinds'] for n in p['needle_filler'])))
    for arm in p['arms']:
        if arm not in arms:
            print('%-9s %10s' % (arm, 'incomplete')); continue
        a = arms[arm]
        print('%-9s %10.4f %s' % (arm, a['bits_per_byte'],
                                  ' '.join('%11.3f' % a['needle'].get(f'{k}@{n}', float('nan')) for k in p['needle_kinds'] for n in p['needle_filler'])))
    v = verdict(arms, seeds=len(p['seeds']))
    report = dict(kind='M1_ANALYSIS', utc=time.time(), protocol_hash=manifest['protocol_hash'],
                  corpus=manifest['corpus'], arms=arms, verdict=v)
    Path(args.out, 'analysis.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
    print()
    print('VERDICT %s%s' % (v['status'], '  (provisional: single seed)' if v.get('provisional') else ''))
    print('  %s' % v['meaning'])
    if v['status'] != 'INCOMPLETE':
        print('  (a) no bpb cost: %s (gate costs %+.4f bpb) | (b) needle gain: %s %s | (c) soft below hard: %s'
              % (v['prediction_a_no_bpb_cost'], v['bpb_cost_of_gate'], v['prediction_b_needle_gain'],
                 {k: round(g, 3) for k, g in v['needle_gain_over_plain'].items()}, v['prediction_c_soft_below_hard']))
        print('  notes: %s' % v['notes'])
    print('written', Path(args.out, 'analysis.json'))


if __name__ == '__main__':
    main()
