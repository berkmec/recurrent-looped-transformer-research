"""The reading rules of E-D and E-C (program note sections 5.2 and 5.3), written BEFORE the runs and applied here mechanically to the result files.

  python -B -m horizon.readings ed            # noise-trained S4 GRU / LSTM against the best baseline seed
  python -B -m horizon.readings ec            # scale-diverse excursion counter (count8_32 and count8_64, LSTM and GRU)

E-D (per arm): the baseline is the LOWEST pooled first-error hazard over the seeds that exist (provisional while only seed 1 does). SUPPORTED when some
training noise sigma gives in-range final accuracy >= 0.99 AND hazard <= baseline / 5; REFUTED when for every sigma the hazard is >= baseline / 2 or the
in-range accuracy is < 0.95; MIXED otherwise.
E-C (per run): valid when the in-range counter accuracy (D = 8, `prefix`) >= 0.95. O1: first descent >= 0.9 at every D in (8, R_max]. O2 (LSTM): first
descent >= 0.5 at 2 R_max and < 0.5 at 4 R_max (refuted if >= 0.95 at 4 R_max). The factor rho is the largest D, reached through consecutive evaluated depths
from 16 on with first descent >= 0.95, divided by R_max. O3: the GRU's rho is <= the LSTM's.
"""
import json
import sys
from pathlib import Path

from .depth_run import meets
from .hazard_summary import pooled_windows

RESULTS = Path('results/Z1-pilot')


def pooled_rate(result):
    cells = pooled_windows(result)
    events, exposure = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
    return (events / exposure if exposure else None), events, exposure


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def ed_arm(arm, sigmas=(0.1, 0.05, 0.2), directory=RESULTS, width=64, baseline_seeds=(1, 2, 3)):
    base = {}
    for seed in baseline_seeds:
        path = Path(directory) / f'perm4-{arm}-w{width}-s{seed}-cur.json'
        if path.exists():
            base[seed] = pooled_rate(_load(path))[0]
    if not base:
        return dict(arm=arm, verdict='NO BASELINE')
    best = min(v for v in base.values() if v is not None)
    runs = {}
    for sigma in sigmas:
        path = Path(directory) / f'perm4-{arm}-w{width}-s1-cur-noise{sigma}.json'
        if path.exists():
            result = _load(path)
            rate = pooled_rate(result)[0]
            runs[sigma] = dict(in_range=result['curve']['16']['final'], hazard=rate, ratio=(rate / best if best else None))
    if not runs:
        return dict(arm=arm, verdict='NO RUNS', baseline=base)
    supported = any(r['in_range'] >= 0.99 and r['hazard'] <= best / 5 for r in runs.values())
    refuted = len(runs) == len(sigmas) and all(r['hazard'] >= best / 2 or r['in_range'] < 0.95 for r in runs.values())
    return dict(arm=arm, baseline_hazards=base, best_baseline=best, provisional=set(base) == {1}, runs=runs, complete=len(runs) == len(sigmas),
                verdict='SUPPORTED' if supported else 'REFUTED' if refuted else 'MIXED' if len(runs) == len(sigmas) else 'INCOMPLETE')


def rho_of(result, r_max, first=16):
    """The extrapolation factor: the largest depth reached through consecutive evaluated depths (from `first` on) whose first descent is >= 0.95, over r_max."""
    best = 0
    for depth in sorted(int(d) for d in result['curve']):
        if depth < first:
            continue
        value = result['curve'][str(depth)].get('reentry')
        if value is None or not meets(value, 0.95):
            break
        best = depth
    return best / r_max


def ec_run(path, r_max, lstm):
    result = _load(path)
    curve = {int(d): row for d, row in result['curve'].items()}
    reentry = {d: row.get('reentry') for d, row in curve.items()}
    valid = 8 in curve and meets(curve[8]['prefix'], 0.95)
    inside = [d for d in reentry if 8 < d <= r_max]
    out = dict(valid=valid, in_range=curve.get(8, {}).get('prefix'), rho=rho_of(result, r_max), first_descent=reentry,
               O1=(all(reentry[d] is not None and reentry[d] >= 0.9 for d in inside) if inside else None))
    if lstm:
        two, four = reentry.get(2 * r_max), reentry.get(4 * r_max)
        out['O2'] = (two is not None and four is not None and two >= 0.5 and four < 0.5)
        out['O2_refuted'] = four is not None and four >= 0.95
    return out


def ec_task(task, directory=RESULTS):
    r_max = {'count8_32': 32, 'count8_64': 64}[task]
    out = {}
    for arm in ('lstm', 'gru'):
        path = Path(directory) / f'{task}-{arm}-w64-s1.json'
        if path.exists():
            out[arm] = ec_run(path, r_max, lstm=(arm == 'lstm'))
    if 'lstm' in out and 'gru' in out and out['lstm']['valid'] and out['gru']['valid']:
        out['O3'] = out['gru']['rho'] <= out['lstm']['rho']
    return dict(task=task, r_max=r_max, runs=out)


def ec0(directory=RESULTS, seeds=range(1, 9), arms=('gru', 'lstm'), width=64):
    """E-C0 (program note 5.6): is the saturating counter of the default design universal over training seeds? A run is valid when the in-range counter
    accuracy (D = 8, `prefix`) is >= 0.95. UNIVERSAL SATURATION: >= 90 % of the valid runs have first descent (D = 16) <= 0.10. LOTTERY: >= 2 valid runs
    have first descent (D = 16) >= 0.5. MIXED otherwise."""
    runs = {}
    for arm in arms:
        for seed in seeds:
            path = Path(directory) / f'count-{arm}-w{width}-s{seed}.json'
            if not path.exists():
                continue
            curve = _load(path)['curve']
            runs[f'{arm}-s{seed}'] = dict(valid=meets(curve['8']['prefix'], 0.95), in_range=curve['8']['prefix'], first_descent_16=curve['16'].get('reentry'))
    valid = {k: r for k, r in runs.items() if r['valid'] and r['first_descent_16'] is not None}
    saturated = [k for k, r in valid.items() if r['first_descent_16'] <= 0.10]
    counting = [k for k, r in valid.items() if r['first_descent_16'] >= 0.5]
    complete = len(runs) == len(arms) * len(list(seeds))
    if not valid:
        verdict = 'NO VALID RUNS'
    elif len(counting) >= 2:
        verdict = 'LOTTERY'
    elif len(saturated) >= 0.9 * len(valid):
        verdict = 'UNIVERSAL SATURATION'
    else:
        verdict = 'MIXED'
    return dict(verdict=verdict if complete else verdict + ' (INCOMPLETE: ' + str(len(runs)) + ' runs)', runs=runs, valid=len(valid), saturated=len(saturated),
                counting=counting)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ''
    if which == 'ed':
        print(json.dumps([ed_arm('gru'), ed_arm('lstm')], indent=1, default=str))
    elif which == 'ec':
        print(json.dumps([ec_task('count8_32'), ec_task('count8_64')], indent=1, default=str))
    elif which == 'ec0':
        print(json.dumps(ec0(), indent=1, default=str))
    else:
        raise SystemExit('usage: python -B -m horizon.readings ed|ec|ec0')


if __name__ == '__main__':
    main()
