"""U4-sep: score the separation-loss sweep against the prediction that was pre-registered before it ran.

The scoring rule lives here as tested code (tests/test_u4sep_compare.py) and was written before any U4-sep number
existed, so the verdict cannot be shaped by the results. It compares the U4-sep soft_sep arm with the U2c soft arm
on identical frozen endpoints, at both learning rates, and reads the U2d separation ratios from the diagnostic that
produced the correlational finding in the first place.

Diagnostic only: trains nothing, freezes no protocol, records the sha256 of every checkpoint and source it reads.

Usage: python -B -m horizon.u4sep_compare --out results/U4sep
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

from horizon import gated_sweep as gs
from horizon import sep_sweep as ss
from horizon import u2d_compare as ud

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/gated.py', 'horizon/gated_sweep.py', 'horizon/gated_trajectory.py',
           'horizon/sep_loss.py', 'horizon/sep_sweep.py', 'horizon/u2d_compare.py', 'horizon/u4sep_compare.py')
LEVELS = {'lr0.003': dict(u4sep='results/U4sep-lr003', u2c='results/U2c-v2'),
          'lr0.0003': dict(u4sep='results/U4sep-lr0003', u2c='results/U2c-lr0003')}
REFERENCE = 'results/U2b-intervene'
ENDPOINT = '512-suffix'
ORACLE_ARM = 'freeze_k2'

THRESHOLDS = dict(repair_pp=30., validation_min=.95, high_lr_drop_pp=10., ratio_rise=1.25)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def repair_fraction(arm_accuracy, none_accuracy, oracle_accuracy):
    """The U2c definition: how much of the repair the oracle had available did this arm actually achieve, in per cent.

    None when the oracle had no headroom (nothing to repair), so an undefined cell is never silently counted as zero.
    """
    headroom = oracle_accuracy - none_accuracy
    if headroom <= 0:
        return None
    return (arm_accuracy - none_accuracy) / headroom * 100.


def verdict(low, high, baseline, thresholds=None):
    """Score the pre-registered prediction. `low`/`high` are the U4-sep arms, `baseline` the U2c soft arm they replace.

    Prediction (a): at lr 0.0003 repair becomes clearly positive (> +30 pp) with validation still ~1.000.
    Prediction (b): at lr 0.003 little changes, because there is no room (no drop beyond 10 pp).
    Prediction (c): the separation ratio rises at both levels.
    Falsification: (c) without (a) means the U2d finding was a correlate of training progress, not a cause.
    """
    t = dict(THRESHOLDS, **(thresholds or {}))
    low_repair, high_repair = mean(low['repair']), mean(high['repair'])
    low_val = mean(low['validation'])
    ratio_low = mean(low['sep_ratio']) / mean(baseline['low_sep']) if mean(baseline['low_sep']) else None
    ratio_high = mean(high['sep_ratio']) / mean(baseline['high_sep']) if mean(baseline['high_sep']) else None
    rose = bool(ratio_low and ratio_low >= t['ratio_rise'])
    rescued = bool(low_repair is not None and low_repair > t['repair_pp'] and low_val is not None and low_val >= t['validation_min'])
    high_kept = bool(high_repair is not None and mean(baseline['high_repair']) is not None
                     and high_repair >= mean(baseline['high_repair']) - t['high_lr_drop_pp'])
    if not rose:
        status = 'INTERVENTION_INEFFECTIVE'
    elif low_repair is not None and low_repair > t['repair_pp'] and (low_val is None or low_val < t['validation_min']):
        status = 'RESCUE_AT_A_COST'
    elif rescued and high_kept:
        status = 'CAUSAL_SUPPORTED'
    elif rescued:
        status = 'CAUSAL_SUPPORTED_WITH_HIGH_LR_COST'
    else:
        status = 'CORRELATE_NOT_CAUSE'
    return dict(status=status, thresholds=t,
                prediction_a_low_lr_rescued=rescued, prediction_b_high_lr_unchanged=high_kept, prediction_c_ratio_rose=rose,
                low_repair_mean=low_repair, high_repair_mean=high_repair, low_validation_mean=low_val,
                ratio_gain_low=ratio_low, ratio_gain_high=ratio_high,
                baseline_low_repair_mean=mean(baseline['low_repair']), baseline_high_repair_mean=mean(baseline['high_repair']),
                meaning=dict(CAUSAL_SUPPORTED='widening the anchor separation CAUSES robustness: the U2d finding survives intervention',
                             CAUSAL_SUPPORTED_WITH_HIGH_LR_COST='the low-lr rescue worked but the high-lr arm paid for it',
                             CORRELATE_NOT_CAUSE='the ratio rose and robustness did not follow: U2d measured a correlate of training progress (pool item C17)',
                             RESCUE_AT_A_COST='repair improved but the task accuracy collapsed: a different failure, not a rescue',
                             INTERVENTION_INEFFECTIVE='the auxiliary loss did not widen the ratio, so the causal question was not tested')[status])


def reference_arms(task, seed):
    path = Path(REFERENCE) / f'u1-{task}-w32-s{seed}.json'
    d = json.loads(path.read_text(encoding='utf8'))
    return (d['arms']['none']['neutral'][ENDPOINT]['final_accuracy'],
            d['arms'][ORACLE_ARM]['neutral'][ENDPOINT]['final_accuracy'], sha(path))


def u2d_ratios():
    """The separation ratios U2d measured for the U2c soft arm, keyed by level."""
    path = Path('results/U2d/compare.json')
    if not path.exists():
        return {}, None
    rows = json.loads(path.read_text(encoding='utf8'))['rows']
    out = {}
    for key, row in rows.items():
        cid, level = key.split('@')
        if '-soft-' in cid:
            out.setdefault(level, []).append(row['anchors']['separation_ratio'])
    return out, sha(path)


def collect(level, dirs, task, seeds):
    arm, base = [], []
    for seed in seeds:
        none_acc, oracle_acc, ref_sha = reference_arms(task, seed)
        u4 = Path(dirs['u4sep']) / f'u4sep-soft_sep-{task}-w32-s{seed}'
        u2 = Path(dirs['u2c']) / f'u2c-soft-{task}-w32-s{seed}'
        e4 = json.loads((u4 / 'evaluation.json').read_text(encoding='utf8'))
        t4 = json.loads((u4 / 'train.json').read_text(encoding='utf8'))
        e2 = json.loads((u2 / 'evaluation.json').read_text(encoding='utf8'))
        arm.append(dict(seed=seed, level=level, repair=repair_fraction(e4['neutral'][ENDPOINT]['final_accuracy'], none_acc, oracle_acc),
                        accuracy=e4['neutral'][ENDPOINT]['final_accuracy'], base=e4['neutral']['0-base']['final_accuracy'],
                        validation=t4['selected_validation']['final_accuracy'], sep_ratio=e4['separation_ratio'],
                        clean={k: v['final_accuracy'] for k, v in e4['clean'].items()},
                        hold_rate=e4['neutral']['0-base']['gate']['hold_rate'], sep_ratio_last_step=t4.get('sep_ratio_last'),
                        checkpoint_sha256=sha(u4 / 'selected.pt'), reference_sha256=ref_sha))
        base.append(dict(seed=seed, level=level, repair=repair_fraction(e2['neutral'][ENDPOINT]['final_accuracy'], none_acc, oracle_acc),
                         accuracy=e2['neutral'][ENDPOINT]['final_accuracy'], clean={k: v['final_accuracy'] for k, v in e2['clean'].items()},
                         checkpoint_sha256=sha(u2 / 'selected.pt')))
    return arm, base


def drift_rows(level, dirs, task, seeds, n_seq):
    """The U2d measurement itself, on the new checkpoints: drift, norm growth, whether the class survives the run."""
    out = {}
    d = dirs['u4sep']
    manifest = ss.verify_frozen(d)
    for seed in seeds:
        cid = f'u4sep-soft_sep-{task}-w32-s{seed}'
        path = Path(d) / cid / 'selected.pt'
        if not path.exists():
            continue
        c = ss.known(manifest, cid)
        model = ss.make_model(c, manifest['protocol'])
        model.load_state_dict(torch.load(path, weights_only=True))
        m = ud.measure(model, c, manifest['protocol'], n_seq=n_seq)
        m.update(checkpoint_sha256=sha(path), level=level)
        out[f'{cid}@{level}'] = m
        print('  %s@%s: run acc %.3f | disp end %.2f | norm x%.2f | sep %.2f | class kept %.3f'
              % (cid, level, m['final_accuracy_run'], m['displacement']['end'], m['run_norm']['growth'],
                 m['anchors']['separation_ratio'], m['class_kept_through_run']), flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/U4sep')
    parser.add_argument('--task', default='five'); parser.add_argument('--seeds', default='117,143,191')
    parser.add_argument('--n-seq', type=int, default=200)
    parser.add_argument('--skip-drift', action='store_true')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    u2d, u2d_sha = u2d_ratios()
    report = dict(kind='DIAGNOSTIC_OVER_FROZEN_CHECKPOINTS', utc=time.time(), sources={s: sha(ROOT / s) for s in SOURCES},
                  u2d_compare_sha256=u2d_sha, environment=dict(python=platform.python_version(), torch=torch.__version__),
                  levels={k: v for k, v in LEVELS.items()}, endpoint=ENDPOINT, oracle_arm=ORACLE_ARM, arms={}, baseline={}, drift={})
    for level, dirs in LEVELS.items():
        report['arms'][level], report['baseline'][level] = collect(level, dirs, args.task, seeds)
        for r in report['arms'][level]:
            print('%s s%d: repair %+.1f pp (U2c soft %+.1f) | val %.3f | sep %.2f | clean4096 %.3f'
                  % (level, r['seed'], r['repair'], [b['repair'] for b in report['baseline'][level] if b['seed'] == r['seed']][0],
                     r['validation'], r['sep_ratio'], r['clean']['4096']), flush=True)
    low, high = 'lr0.0003', 'lr0.003'
    v = verdict(low=dict(repair=[r['repair'] for r in report['arms'][low]], validation=[r['validation'] for r in report['arms'][low]],
                         sep_ratio=[r['sep_ratio'] for r in report['arms'][low]]),
                high=dict(repair=[r['repair'] for r in report['arms'][high]], validation=[r['validation'] for r in report['arms'][high]],
                          sep_ratio=[r['sep_ratio'] for r in report['arms'][high]]),
                baseline=dict(low_repair=[b['repair'] for b in report['baseline'][low]], high_repair=[b['repair'] for b in report['baseline'][high]],
                              low_sep=u2d.get(low, []), high_sep=u2d.get(high, [])))
    report['verdict'] = v
    if not args.skip_drift:
        for level, dirs in LEVELS.items():
            report['drift'].update(drift_rows(level, dirs, args.task, seeds, args.n_seq))
    (out / 'compare.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
    print()
    print('VERDICT %s' % v['status'])
    print('  %s' % v['meaning'])
    print('  (a) low-lr rescued: %s (repair %+.1f pp vs U2c %+.1f, validation %.3f)'
          % (v['prediction_a_low_lr_rescued'], v['low_repair_mean'], v['baseline_low_repair_mean'], v['low_validation_mean']))
    print('  (b) high-lr unchanged: %s (repair %+.1f pp vs U2c %+.1f)' % (v['prediction_b_high_lr_unchanged'], v['high_repair_mean'], v['baseline_high_repair_mean']))
    print('  (c) ratio rose: %s (x%.2f low, x%.2f high)' % (v['prediction_c_ratio_rose'], v['ratio_gain_low'] or 0., v['ratio_gain_high'] or 0.))
    print('written', out / 'compare.json')


if __name__ == '__main__':
    main()
