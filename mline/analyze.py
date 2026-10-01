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
import math
import sys
import time
from pathlib import Path

THRESHOLDS = dict(bpb_regression=.02, needle_gain=.05, long_filler=[512, 2048], measurable_floor=.02, items=256)

STATUSES = ('GATE_TRANSFERS', 'GATE_AT_A_COST', 'GATE_DOES_NOT_TRANSFER',
            'NOT_MEASURABLE_AT_THIS_LENGTH', 'NOT_MEASURABLE_AT_THIS_BUDGET', 'INCOMPLETE')

MEANING = dict(
    GATE_TRANSFERS='the exact-preservation gate helps at 30M on real text and costs no language ability: the toy '
                   'result transfers and the M-line has a rung to build on',
    GATE_AT_A_COST='the gate improves the needle but costs bits per byte; a trade-off, not a free win',
    GATE_DOES_NOT_TRANSFER='no needle gain at 30M, on sequences the length-matched control shows the arms CAN '
                           'process: the gate is a property of the toy setting and the M-line stops at rung one',
    NOT_MEASURABLE_AT_THIS_LENGTH='no arm clears the floor even in the length-matched control, where the fact sits '
                                  'next to the question; the sequence length defeated every arm at any distance, so '
                                  'the far scores are not evidence about the gate. The remedy is a longer training '
                                  'length, not a longer run',
    NOT_MEASURABLE_AT_THIS_BUDGET='no arm, including the control, does better than the floor at the long fillers, and '
                                  'no length-matched control was recorded, so this budget could not have detected a '
                                  'gate benefit either way; the result is about the budget, not about the gate',
    INCOMPLETE='not all three arms have finished')

assert set(MEANING) == set(STATUSES), 'every outcome the rule can return must be named and explained'


def _mean_needle(row, n, kinds=('repeat', 'text'), placement='far'):
    suffix = '' if placement == 'far' else '@' + placement
    vals = [row['needle'][f'{k}@{n}{suffix}'] for k in kinds if f'{k}@{n}{suffix}' in row['needle']]
    return sum(vals) / len(vals) if vals else None


def _required_gain(t, items, plain_p, hard_p):
    """The pre-registered constant, or two standard errors of the difference, whichever is larger.

    `needle_gain = 0.05` is 12.8 items out of 256. The standard error of a difference of two proportions is
    sqrt(2p(1-p)/n), which at n=256 is 0.041 near p=0.3 - so at mid-range scores a bare 0.05 sits inside its own
    sampling noise and would report a coin flip as a transfer. Taking the maximum makes the rule self-calibrating:
    near the floor the constant binds, mid-range the noise does, and neither direction can make a positive claim
    EASIER than the registered threshold.
    """
    p = max(0., min(1., (plain_p + hard_p) / 2))
    return max(t['needle_gain'], 2 * math.sqrt(2 * p * (1 - p) / max(items, 1)))


def verdict(arms, seeds=1, thresholds=None, items=None):
    """`arms` maps arm name -> {bits_per_byte, needle: {kind@N: exact_match}}."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    # `soft` became optional when it was dropped from M1 for budget. It only ever served prediction (c), which
    # this rule has always treated as recorded-but-not-status-changing, so its absence must not read as INCOMPLETE
    # - that would report a missing arm as a missing result.
    if not {'plain', 'hard_st'} <= set(arms):
        return dict(status='INCOMPLETE', meaning=MEANING['INCOMPLETE'], arms=sorted(arms), provisional=True)
    plain, hard = arms['plain'], arms['hard_st']
    soft = arms.get('soft')
    bpb_cost = hard['bits_per_byte'] - plain['bits_per_byte']
    items = items or t['items']
    gains = {n: (_mean_needle(hard, n) - _mean_needle(plain, n)) for n in t['long_filler']}
    required = {n: _required_gain(t, items, _mean_needle(plain, n) or 0., _mean_needle(hard, n) or 0.)
                for n in t['long_filler']}
    soft_below = (None if soft is None
                  else all(_mean_needle(soft, n) < _mean_needle(hard, n) for n in t['long_filler']))
    a = bpb_cost <= t['bpb_regression']
    b = all(g > required[n] for n, g in gains.items())
    # If no arm can do the task at the long fillers, the comparison has no resolution: a gate that helps would be
    # invisible. Reporting that as GATE_DOES_NOT_TRANSFER converts "we could not measure it" into "it does not work",
    # which is the error an INTERVENTION_INEFFECTIVE branch caught twice in the two preceding cycles.
    def at_floor(placement):
        return all(max(_mean_needle(arms[k], n, placement=placement) or 0. for n in t['long_filler'])
                   <= t['measurable_floor'] for k in arms)

    # Both decisive fillers put the model on sequences LONGER than it was trained on - 1.10x at 512 and 4.10x at
    # 2048 - so a zero there has two possible causes. The length-matched `near` control holds the width fixed and
    # moves the fact next to the question: if no arm can do it even at distance zero, the sequence length defeated
    # them and the far score is not evidence about memory. This distinction is prior to the budget floor, because
    # the two call for different remedies: more steps cannot fix a length the model cannot process.
    near_available = all(_mean_needle(arms[k], n, placement='near') is not None for k in arms for n in t['long_filler'])
    far_floor = at_floor('far')
    near_floor = near_available and at_floor('near')
    if far_floor and near_floor:
        status = 'NOT_MEASURABLE_AT_THIS_LENGTH'
    elif far_floor and not near_available:
        status = 'NOT_MEASURABLE_AT_THIS_BUDGET'
    elif not b:
        status = 'GATE_DOES_NOT_TRANSFER'
    elif not a:
        status = 'GATE_AT_A_COST'
    else:
        status = 'GATE_TRANSFERS'
    notes = []
    if soft_below is None:
        notes.append('prediction (c) is NOT MEASURED: the soft arm was dropped from this run for budget, so '
                     'nothing here speaks to WHY the gate would work, only to whether it transfers')
    elif not soft_below:
        notes.append('soft matched or beat hard_st on a long filler, which contradicts the U2c stack result that '
                     'motivated prediction (c); the reason for the gate working would then be different from the one assumed')
    if seeds < 2:
        notes.append('single seed: this is a first look, not a replicated result')
    if not near_available:
        notes.append('no length-matched control was recorded, so a floor result cannot distinguish "the distance '
                     'defeated the model" from "the sequence length did"; the budget branch is used instead')
    elif far_floor and not near_floor:
        notes.append('the arms score above the floor at distance zero and at the floor across the filler, at the '
                     'SAME sequence length: the failure is about distance, not about length')
    return dict(status=status, thresholds=t, provisional=seeds < 2, seeds=seeds,
                prediction_a_no_bpb_cost=a, prediction_b_needle_gain=b, prediction_c_soft_below_hard=soft_below,
                bits_per_byte={k: v['bits_per_byte'] for k, v in arms.items()}, bpb_cost_of_gate=bpb_cost,
                needle_gain_over_plain=gains, required_gain=required, needle_items=items,
                needle_by_arm={k: {str(n): _mean_needle(v, n) for n in t['long_filler']} for k, v in arms.items()},
                needle_near_by_arm={k: {str(n): _mean_needle(v, n, placement='near') for n in t['long_filler']}
                                    for k, v in arms.items()},
                length_control=dict(available=near_available, far_at_floor=far_floor, near_at_floor=near_floor),
                notes='; '.join(notes) or 'none',
                meaning=MEANING[status])


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
    v = verdict(arms, seeds=len(p['seeds']), items=p['needle_items'])
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
