"""The reading rules of E-D and E-C (program note sections 5.2 and 5.3), written BEFORE the runs and applied here mechanically to the result files.

  python -B -m horizon.readings ed            # noise-trained S4 GRU / LSTM against the best baseline seed
  python -B -m horizon.readings ep            # E-P: S4 GRU / LSTM trained with the learning rate decayed over the last 30 % (same rule as E-D; SUPPORTED = DECAY FIXES)
  python -B -m horizon.readings ep3           # E-P3: lr floor dose-response (0.3 / 0.1 / 0.03 / 0.01 x the base rate) on the S4 GRU / LSTM w64; the shallowest sufficient floor
  python -B -m horizon.readings ep4           # E-P4: S4 GRU / LSTM w64, seeds 1-5, constant vs annealed: is the floor typical (ARTEFACT CONFIRMED) or a seed lottery?
  python -B -m horizon.readings ep5           # E-P2: the same for the S5 Core w128, GRU w256 and LSTM w208 (seed 1)
  python -B -m horizon.readings ed5           # E-D5: S5 Core w128, state-noise training (sigma 0.1 / 0.2) on top of the lr annealing, against the constant-lr baseline
  python -B -m horizon.readings epseeds       # E-P2 replicates: S4 GRU / LSTM w64, seeds 2 and 3 against their own seed's constant-lr baseline
  python -B -m horizon.readings ec            # scale-diverse excursion counter (count8_32 and count8_64, LSTM and GRU)

E-D (per arm): the baseline is the LOWEST pooled first-error hazard over the seeds that exist (provisional while only seed 1 does). SUPPORTED when some
training noise sigma gives in-range final accuracy >= 0.99 AND hazard <= baseline / 5; REFUTED when for every sigma the hazard is >= baseline / 2 or the
in-range accuracy is < 0.95; MIXED otherwise.
E-C (per run): valid when the in-range counter accuracy (D = 8, `prefix`) >= 0.95. O1: first descent >= 0.9 at every D in (8, R_max]. O2 (LSTM): first
descent >= 0.5 at 2 R_max and < 0.5 at 4 R_max (refuted if >= 0.95 at 4 R_max). The factor rho is the largest D, reached through consecutive evaluated depths
from 16 on with first descent >= 0.95, divided by R_max. O3: the GRU's rho is <= the LSTM's.
"""
import json
import math
import sys
from pathlib import Path

from .depth_run import meets
from .hazard_summary import pooled_windows

RESULTS = Path('results/Z1-pilot')
TOL = 1e-6      # accuracies are float32 means over n = 100 sequences: 90/100 is stored as 0.89999998, 95/100 as 0.94999999 and 10/100 as 0.10000000149; a comparison at an exact
                # threshold must not depend on that rounding (blinded review finding F6)


def ge(value, threshold):
    return value >= threshold - TOL


def le(value, threshold):
    return value <= threshold + TOL


def pooled_rate(result):
    cells = pooled_windows(result)
    events, exposure = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
    return (events / exposure if exposure else None), events, exposure


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rate_ratio_p(events_base, exposure_base, events_run, exposure_run):
    """One-sided exact p-value that the run's hazard is not lower than the baseline's: conditional on the total number of events, the run's share is binomial with
    probability exposure_run / (exposure_base + exposure_run) under equal hazards; p = P(X <= events_run). Descriptive (not part of any verdict)."""
    total = events_base + events_run
    if total == 0:
        return 1.0
    q = exposure_run / (exposure_base + exposure_run)
    log_terms = [math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1) + k * math.log(q) + (total - k) * math.log1p(-q) for k in range(events_run + 1)]
    top = max(log_terms)
    return float(min(1.0, math.exp(top) * sum(math.exp(t - top) for t in log_terms)))


def ed_arm(arm, sigmas=(0.1, 0.05, 0.2), directory=RESULTS, width=64, baseline_seeds=(1, 2, 3), tag='noise', task='perm4', run_seed=1):
    base = {}
    for seed in baseline_seeds:
        path = Path(directory) / f'{task}-{arm}-w{width}-s{seed}-cur.json'
        if path.exists():
            base[seed] = pooled_rate(_load(path))[0]
    if not base:
        return dict(arm=arm, verdict='NO BASELINE')
    best = min(v for v in base.values() if v is not None)
    best_seed = min((s for s, v in base.items() if v == best))
    base_events, base_exposure = pooled_rate(_load(Path(directory) / f'{task}-{arm}-w{width}-s{best_seed}-cur.json'))[1:]
    runs = {}
    for sigma in sigmas:
        path = Path(directory) / f'{task}-{arm}-w{width}-s{run_seed}-cur-{tag}{sigma}.json'
        if path.exists():
            result = _load(path)
            rate, events, exposure = pooled_rate(result)
            runs[sigma] = dict(in_range=result['curve']['16']['final'], hazard=rate, ratio=(rate / best if best else None), events=events, exposure=exposure,
                               p_lower=rate_ratio_p(base_events, base_exposure, events, exposure))
    if not runs:
        return dict(arm=arm, verdict='NO RUNS', baseline=base)
    supported = any(ge(r['in_range'], 0.99) and r['hazard'] <= best / 5 for r in runs.values())
    refuted = len(runs) == len(sigmas) and all(r['hazard'] >= best / 2 or not ge(r['in_range'], 0.95) for r in runs.values())
    return dict(arm=arm, baseline_hazards=base, best_baseline=best, provisional=set(base) == {1}, runs=runs, complete=len(runs) == len(sigmas),
                verdict='SUPPORTED' if supported else 'REFUTED' if refuted else 'MIXED' if len(runs) == len(sigmas) else 'INCOMPLETE')


FLOORS = (0.3, 0.1, 0.03, 0.01)


def ep3_arm(arm, floors=FLOORS, directory=RESULTS, width=64, baseline_seeds=(1, 2, 3)):
    """E-P3 (note 5.12): the dose-response of the S4 error floor to how far the learning rate is annealed. Each run decays linearly over the last 30 % to `floor` x the
    base rate. `sufficient_floor` = the LARGEST floor whose run meets the E-P rule (in-range accuracy >= 0.99 and hazard <= best baseline / 5): the shallowest annealing
    that removes the floor. The curve (hazard per floor) is reported whole; None when no floor meets it."""
    out = ed_arm(arm, sigmas=floors, directory=directory, width=width, baseline_seeds=baseline_seeds, tag='lrfloor')
    runs = out.get('runs', {})
    ok = [f for f, r in runs.items() if ge(r['in_range'], 0.99) and r['hazard'] <= out['best_baseline'] / 5]
    out['sufficient_floor'] = max(ok) if ok else None
    out['curve'] = {f: (runs[f]['hazard'], runs[f]['events']) for f in sorted(runs, reverse=True)}
    return out


FLOOR_HAZARD = 2e-4


def ep4(directory=RESULTS, arms=('gru', 'lstm'), seeds=(1, 2, 3, 4, 5), width=64):
    """E-P4 (note 5.14): is the S4 constant-lr floor TYPICAL (and annealing removes it) or a seed lottery? Per arm, over the training seeds, each run's pooled first-error
    hazard (all windows, as E-D / E-P) is compared with FLOOR_HAZARD = 2e-4 (>= 41 first errors in 203,200 sequence-steps); a run counts only if its in-range accuracy is
    >= 0.95. c = constant-lr runs with a floor, d = annealed runs (`-lrdecay0.3`) with a floor. NOT REMOVED: d >= 2; LOTTERY: c <= 2; ARTEFACT CONFIRMED: c >= 3 and d <= 1.
    A verdict is INCOMPLETE (suffix) until every seed of both conditions exists."""
    out = {}
    for arm in arms:
        rows = {'constant': {}, 'decayed': {}}
        for cond, suffix in (('constant', ''), ('decayed', '-lrdecay0.3')):
            for seed in seeds:
                path = Path(directory) / f'perm4-{arm}-w{width}-s{seed}-cur{suffix}.json'
                if path.exists():
                    result = _load(path)
                    rate, events, exposure = pooled_rate(result)
                    rows[cond][seed] = dict(hazard=rate, events=events, exposure=exposure, in_range=result['curve']['16']['final'],
                                            valid=ge(result['curve']['16']['final'], 0.95), floor=(rate is not None and rate >= FLOOR_HAZARD))
        c = sum(r['floor'] for r in rows['constant'].values() if r['valid'])
        d = sum(r['floor'] for r in rows['decayed'].values() if r['valid'])
        complete = all(len(rows[k]) == len(seeds) for k in rows)
        verdict = 'NOT REMOVED' if d >= 2 else 'LOTTERY' if c <= 2 else 'ARTEFACT CONFIRMED' if c >= 3 and d <= 1 else 'MIXED'
        out[arm] = dict(verdict=verdict if complete else verdict + ' (INCOMPLETE)', constant_with_floor=c, decayed_with_floor=d, runs=rows)
    return out


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
               O1=(all(reentry[d] is not None and ge(reentry[d], 0.9) for d in inside) if inside else None))
    if lstm:
        two, four = reentry.get(2 * r_max), reentry.get(4 * r_max)
        out['O2'] = (two is not None and four is not None and ge(two, 0.5) and not ge(four, 0.5))
        out['O2_refuted'] = four is not None and ge(four, 0.95)
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
    saturated = [k for k, r in valid.items() if le(r['first_descent_16'], 0.10)]
    counting = [k for k, r in valid.items() if ge(r['first_descent_16'], 0.5)]
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
    elif which == 'ep':
        print(json.dumps([ed_arm('gru', sigmas=(0.3,), tag='lrdecay'), ed_arm('lstm', sigmas=(0.3,), tag='lrdecay')], indent=1, default=str))
    elif which == 'ep4':      # E-P4: seeds 1-5, constant vs annealed (note 5.14)
        print(json.dumps(ep4(), indent=1, default=str))
    elif which == 'ep3':      # E-P3: lr floor dose-response on the S4 GRU / LSTM w64 (note 5.12)
        print(json.dumps([ep3_arm('gru'), ep3_arm('lstm')], indent=1, default=str))
    elif which == 'ep5':      # E-P2: the S5 arms (note 5.11), seed 1 against the stored seed-1 baselines
        print(json.dumps([ed_arm('core', sigmas=(0.3,), tag='lrdecay', task='perm5', width=128, baseline_seeds=(1,)),
                          ed_arm('gru', sigmas=(0.3,), tag='lrdecay', task='perm5', width=256, baseline_seeds=(1,)),
                          ed_arm('lstm', sigmas=(0.3,), tag='lrdecay', task='perm5', width=208, baseline_seeds=(1,))], indent=1, default=str))
    elif which == 'ed5':      # E-D5: the S5 Core w128 trained with carried-state noise on top of the lr annealing (note 5.21), against the constant-lr S5 Core baseline
        print(json.dumps(ed_arm('core', sigmas=(0.1, 0.2), tag='lrdecay0.3-noise', task='perm5', width=128, baseline_seeds=(1,)), indent=1, default=str))
    elif which == 'epseeds':  # E-P2 replicates: S4 seeds 2 and 3, each against its OWN seed's constant-lr baseline
        print(json.dumps({f'{arm}-s{seed}': ed_arm(arm, sigmas=(0.3,), tag='lrdecay', baseline_seeds=(seed,), run_seed=seed) for arm in ('gru', 'lstm') for seed in (2, 3)},
                         indent=1, default=str))
    elif which == 'ec':
        print(json.dumps([ec_task('count8_32'), ec_task('count8_64')], indent=1, default=str))
    elif which == 'ec0':
        print(json.dumps(ec0(), indent=1, default=str))
    else:
        raise SystemExit('usage: python -B -m horizon.readings ed|ep|ep3|ep4|ep5|ed5|epseeds|ec|ec0')


if __name__ == '__main__':
    main()
