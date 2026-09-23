"""U4-sep2: aggregate the five checkpoint pairs and apply the predictions that were registered before the run.

Each pair is one frozen U2c checkpoint fine-tuned twice from the same starting weights with the same budget and data:
`none` with the task loss alone, `hinge` with the task loss plus max(0, target - gap/drift). The pairing is what makes
the comparison causal - the two arms differ in exactly one term - so every prediction here is stated per pair and then
required to REPLICATE across checkpoints.

Usage: python -B -m horizon.u4sep2_analyze --out results/U4sep2
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ENDPOINT = '512-suffix'
ORACLE_ARM = 'freeze_k2'
REFERENCE = 'results/U2b-intervene'

THRESHOLDS = dict(vs_start_pp=30., vs_control_pp=20., validation_min=.95, control_drift_pp=10., ratio_target=1.5, min_pairs=3)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def repair_fraction(arm_accuracy, none_accuracy, oracle_accuracy):
    headroom = oracle_accuracy - none_accuracy
    return None if headroom <= 0 else (arm_accuracy - none_accuracy) / headroom * 100.


def verdict(pairs, thresholds=None):
    """Score the four pre-registered predictions. `pairs` carries repair in percentage points, per arm.

    (a) the hinge arm reaches the ratio target; (b) its repair beats both its own start and its paired control;
    (c) validation stays >= 0.95 in both arms; (d) the control barely moves, so "more training" is ruled out.
    Replication is required: a majority of pairs must be rescued, not one.
    """
    t = dict(THRESHOLDS, **(thresholds or {}))
    rescued = [p for p in pairs
               if p['hinge_repair'] - p['u2c_repair'] > t['vs_start_pp'] and p['hinge_repair'] - p['control_repair'] > t['vs_control_pp']]
    ratio_reached = all(p['ratio_hinge'] >= t['ratio_target'] for p in pairs)
    accuracy_kept = all(p['validation_hinge'] >= t['validation_min'] and p['validation_control'] >= t['validation_min'] for p in pairs)
    control_flat = all(abs(p['control_repair'] - p['u2c_repair']) < t['control_drift_pp'] for p in pairs)
    majority = len(rescued) > len(pairs) / 2
    if len(pairs) < t['min_pairs']:
        # A verdict machinery that exists to stop premature conclusions must not itself deliver one: with fewer than
        # min_pairs complete, "a majority of pairs" is satisfied trivially and means nothing.
        status = 'INCOMPLETE'
    elif not ratio_reached:
        status = 'INTERVENTION_INEFFECTIVE'
    elif not accuracy_kept:
        status = 'RESCUE_AT_A_COST'
    elif not control_flat:
        status = 'CONFOUNDED_BY_TRAINING'
    elif not rescued:
        status = 'CORRELATE_NOT_CAUSE'
    elif not majority:
        status = 'NOT_REPLICATED'
    else:
        status = 'CAUSAL_SUPPORTED'
    return dict(status=status, thresholds=t, pairs=len(pairs), pairs_rescued=len(rescued),
                rescued_checkpoints=[p['checkpoint'] for p in rescued],
                prediction_a_ratio_reached=ratio_reached, prediction_b_rescued=bool(rescued) and majority,
                prediction_c_accuracy_kept=accuracy_kept, prediction_d_control_flat=control_flat,
                mean_hinge_repair=mean([p['hinge_repair'] for p in pairs]),
                mean_control_repair=mean([p['control_repair'] for p in pairs]),
                mean_u2c_repair=mean([p['u2c_repair'] for p in pairs]),
                mean_gain_over_control=mean([p['hinge_repair'] - p['control_repair'] for p in pairs]),
                meaning=dict(
                    CAUSAL_SUPPORTED='moving gap/drift into the robust range CAUSES drift robustness: it replicates across checkpoints, '
                                     'the paired control rules out extra training, and task accuracy is untouched',
                    NOT_REPLICATED='one or a minority of checkpoints were rescued; the effect is real somewhere but not general',
                    CORRELATE_NOT_CAUSE='the ratio reached the target and repair did not follow: gap/drift is a correlate, not a cause',
                    CONFOUNDED_BY_TRAINING='the task-loss-only control improved too, so the effect is fine-tuning, not geometry',
                    RESCUE_AT_A_COST='repair improved but task accuracy fell: a different failure, not a rescue',
                    INTERVENTION_INEFFECTIVE='the hinge never reached its target, so the causal question was not tested',
                    INCOMPLETE='too few pairs have finished for replication to mean anything; no verdict yet')[status])


def reference_arms(task, seed):
    path = Path(REFERENCE) / f'u1-{task}-w32-s{seed}.json'
    d = json.loads(path.read_text(encoding='utf8'))
    return (d['arms']['none']['neutral'][ENDPOINT]['final_accuracy'],
            d['arms'][ORACLE_ARM]['neutral'][ENDPOINT]['final_accuracy'], sha(path))


def collect(out):
    from horizon import sep_finetune as sf
    out = Path(out); manifest = sf.verify_frozen(out); p = manifest['protocol']
    pairs, rows = [], {}
    for ck in p['checkpoints']:
        none_acc, oracle_acc, ref_sha = reference_arms(p['task'], ck['seed'])
        u2c_eval = json.loads((Path(ck['directory']) / f"u2c-{ck['arm']}-{p['task']}-w32-s{ck['seed']}" / 'evaluation.json').read_text(encoding='utf8'))
        entry = dict(checkpoint=f"{ck['arm']}/s{ck['seed']}", reference_sha256=ref_sha,
                     u2c_repair=repair_fraction(u2c_eval['neutral'][ENDPOINT]['final_accuracy'], none_acc, oracle_acc),
                     u2c_accuracy=u2c_eval['neutral'][ENDPOINT]['final_accuracy'])
        complete = True
        for arm in ('none', 'hinge'):
            cid = f"u4sep2-{arm}-from_{ck['arm']}-s{ck['seed']}"
            d = out / cid
            if not (d / 'evaluation.json').exists():
                complete = False; continue
            e = json.loads((d / 'evaluation.json').read_text(encoding='utf8'))
            t = json.loads((d / 'train.json').read_text(encoding='utf8'))
            entry[f'{arm}_repair'] = repair_fraction(e['neutral'][ENDPOINT]['final_accuracy'], none_acc, oracle_acc)
            entry[f'{arm}_accuracy'] = e['neutral'][ENDPOINT]['final_accuracy']
            entry[f'validation_{arm}'] = t['selected_validation']['final_accuracy']
            entry[f'ratio_{arm}'] = t['ratio_last']
            entry[f'eval_ratio_{arm}'] = e['geometry']['gap_over_drift']
            entry[f'clean4096_{arm}'] = e['clean']['4096']['final_accuracy']
            entry[f'endpoints_{arm}'] = {k: v['final_accuracy'] for k, v in e['neutral'].items()}
            entry[f'checkpoint_sha256_{arm}'] = sha(d / 'selected.pt')
        entry['ratio_u2c'] = entry.get('ratio_none')   # the pair starts from the same weights, so the control's first ratio is the start
        rows[entry['checkpoint']] = entry
        if complete:
            pairs.append(entry)
    return manifest, pairs, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/U4sep2')
    args = parser.parse_args()
    manifest, pairs, rows = collect(args.out)
    for e in rows.values():
        if 'hinge_repair' not in e or 'none_repair' not in e:
            print('%-18s incomplete' % e['checkpoint']); continue
        print('%-18s U2c %+7.1f | control %+7.1f | hinge %+7.1f | gain over control %+7.1f | gap/drift %.2f -> %.2f | val %.3f | clean4096 %.3f'
              % (e['checkpoint'], e['u2c_repair'], e['none_repair'], e['hinge_repair'], e['hinge_repair'] - e['none_repair'],
                 e['eval_ratio_none'] or 0., e['eval_ratio_hinge'] or 0., e['validation_hinge'], e['clean4096_hinge']))
    scored = [dict(p, control_repair=p['none_repair'], validation_control=p['validation_none'], ratio_control=p['ratio_none'])
              for p in pairs]
    v = verdict(scored) if scored else dict(status='NO_PAIRS', meaning='nothing complete yet')
    report = dict(kind='U4SEP2_ANALYSIS', utc=time.time(), protocol_hash=manifest['protocol_hash'],
                  endpoint=ENDPOINT, oracle_arm=ORACLE_ARM, pairs=rows, verdict=v)
    Path(args.out, 'analysis.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
    print()
    print('VERDICT %s  (%d/%d pairs rescued)' % (v['status'], v.get('pairs_rescued', 0), v.get('pairs', 0)))
    print('  %s' % v['meaning'])
    if pairs:
        print('  (a) ratio reached target: %s | (b) replicated rescue: %s | (c) accuracy kept: %s | (d) control flat: %s'
              % (v['prediction_a_ratio_reached'], v['prediction_b_rescued'], v['prediction_c_accuracy_kept'], v['prediction_d_control_flat']))
        print('  mean repair: U2c %+.1f -> control %+.1f -> hinge %+.1f' % (v['mean_u2c_repair'], v['mean_control_repair'], v['mean_hinge_repair']))
    print('written', Path(args.out, 'analysis.json'))


if __name__ == '__main__':
    main()
