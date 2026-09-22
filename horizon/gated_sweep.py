"""U2c: train the learned HOLD/WRITE gate (horizon.gated.GatedCore) with the U1 training loop and evaluate it on the U2b endpoints.

Arms (all trained from scratch, Core + gate head jointly): soft, hard_st, supervised (hard_st + gate_loss to the run-aware
target). Reference `none` = the U1 selected checkpoints measured in results/U2b-intervene (same data namespaces, 500 sequences).
Differences from the U1 loop, both recorded per step: readout from the committed state; the A8 guard skips (and counts)
non-finite steps instead of aborting. No oracle at evaluation time — the gate decides from [e_t, s_{t-1}].
"""
import argparse
import copy
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import gated as gd
from horizon import guards as gu
from horizon import metrics as mt
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/interventions.py', 'horizon/interventions_runaware.py',
           'horizon/gated.py', 'horizon/guards.py', 'horizon/gated_sweep.py')
ARMS = ('soft', 'hard_st', 'supervised')


def protocol():
    b = sw.protocol()
    return dict(version=1, purpose='U2c learned HOLD/WRITE gate over the Core: can a trained gate reproduce the U2b run-aware freeze ceiling without oracle neutral knowledge?',
                steps=b['steps'], batch=b['batch'], train_length=b['train_length'], lr=b['lr'], weight_decay=b['weight_decay'], clip=b['clip'],
                width=b['width'], heads=b['heads'], window=b['window'], alpha=b['alpha'], tasks=list(b['tasks']), primary_seeds=list(b['primary_seeds']),
                validation_every=b['validation_every'], validation_n=b['validation_n'], selection=b['selection'], source_seeds=dict(b['source_seeds']),
                arms=list(ARMS), gate_k=2, gate_loss_weight=1., reference='results/U2b-intervene', u1='results/U1',
                n_neutral=500, n_clean=500, neutral_slots=[128, 512], placements=['prefix', 'middle', 'suffix'], clean_lengths=[128, 1024, 4096], eval_batch=32,
                device='cpu', threads=1, targets=dict(neutral_loss_pp=2., clean_regression_pp=1., false_hold_max=.01, skipped_steps_max=.01))


def finite_or_none(x):
    """None for inf/NaN so no row can break the allow_nan=False writers (U2c hard_st, 2026-09-22)."""
    return x if (x is None or (isinstance(x, (int, float)) and math.isfinite(x))) else None


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, bool) or isinstance(value, str) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def gate_weight(arm, p):
    return float(p['gate_loss_weight']) if arm == 'supervised' else 0.


def mode_of(arm):
    return 'hard_st' if arm in ('hard_st', 'supervised') else 'soft'


def conditions(p):
    return [dict(arm=arm, task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'], lr=p['lr'], stage='u2c')
            for arm in p['arms'] for task in p['tasks'] for seed in p['primary_seeds']]


def ident(c):
    return f"u2c-{c['arm']}-{c['task']}-w{c['width']}-s{c['seed']}"


def known(manifest, cid):
    for c in conditions(manifest['protocol']):
        if ident(c) == cid:
            return c
    raise ValueError('unknown condition ' + cid)


def make_model(c, p):
    core = sw.make_model(c, p)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'] + 7)
        model = gd.GatedCore(core, mode=mode_of(c['arm']), k=p['gate_k'])
    return model


def source_hashes():
    return {name: sw.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol()); sw.configure(p)
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(), u1_protocol_hash=sw.verify_frozen(p['u1'])['protocol_hash'] if p.get('u1') else None,
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    sw.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sw.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sw.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != sw.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    sw.configure(manifest['protocol'])
    return manifest


@torch.no_grad()
def predict(model, x, y, p):
    model.eval(); logits, gates = [], []
    for at in range(0, len(x), p['eval_batch']):
        z, g, _ = model(x[at:at + p['eval_batch']]); logits.append(z); gates.append(g)
    logits = torch.cat(logits); gates = torch.cat(gates); pred = logits.argmax(-1)
    return dict(final_loss=float(F.cross_entropy(logits[:, -1], y[:, -1])), final_accuracy=float((pred[:, -1] == y[:, -1]).float().mean()),
                all_accuracy=float((pred == y).float().mean()), hold_rate=float((gates <= .5).float().mean())), pred, gates


def run_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']; known(manifest, ident(c))
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    (directory / 'active.lock').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall = time.perf_counter()
        model = make_model(c, p)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        torch.save(sw.cpu_state(model), directory / 'initial.pt')
        vx, vy, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', p['validation_n'], p['train_length'])
        counter = gu.GuardCounter(); best, selected, train_seconds = None, None, 0.
        w = gate_weight(c['arm'], p)
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train()
            logits, gates, glogits = model(batch['x'])
            task_loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
            g_loss = gd.gate_loss(glogits, gd.gate_target(batch['x'], p['gate_k'])) if w > 0 else torch.zeros(())
            loss = task_loss + w * g_loss
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'], task_loss=float(task_loss.detach()) if math.isfinite(float(task_loss.detach())) else None,
                       gate_loss=float(g_loss.detach()) if w > 0 else None, grad_norm=report['grad_norm_before_clip'], hold_rate=float((gates.detach() <= .5).float().mean()),
                       input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _, _ = predict(model, vx, vy, p); row['validation'] = json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'], selected_validation=json_safe(selected['validation']),
                      guard=counter.summary(), train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall, protocol_hash=manifest['protocol_hash'],
                      selected_sha256=sw.sha(directory / 'selected.pt'), initial_sha256=sw.sha(directory / 'initial.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', result)
        sw.write_json(directory / 'complete.json', dict(ident=ident(c), utc=time.time(), train_sha256=sw.sha(directory / 'train.json')))
        return result
    finally:
        (directory / 'active.lock').unlink(missing_ok=True)


def evaluate_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    directory = out / ident(c); path = directory / 'evaluation.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    if not (directory / 'complete.json').exists():
        raise ValueError('evaluation requires a complete condition: ' + ident(c))
    model = make_model(c, p); model.load_state_dict(torch.load(directory / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    row = dict(condition=c, ident=ident(c), protocol_hash=manifest['protocol_hash'], selected_sha256=sw.sha(directory / 'selected.pt'), neutral={}, clean={})
    x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', p['n_neutral'], p['train_length'])
    stats, base_pred, gates = predict(model, x, y, p)
    row['neutral']['0-base'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(x, p['gate_k'])), source_seed=seed)
    for slots in p['neutral_slots']:
        for placement in p['placements']:
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred, gates = predict(model, xx, yy, p)
            row['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(xx, p['gate_k'])),
                                                          **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    for L in p['clean_lengths']:
        cx, cy, cseed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L)
        stats, pred, gates = predict(model, cx, cy, p)
        row['clean'][str(L)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['all_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(cx, p['gate_k'])), source_seed=cseed)
    row['seconds'] = time.perf_counter() - t0
    sw.write_json(path, row)
    return row


def reference_rows(p):
    """The `none` reference: U1 selected checkpoints on the same endpoints, from results/U2b-intervene (arm 'none')."""
    ref = {}
    if not p.get('reference'):
        return ref
    for path in Path(p['reference']).glob('u1-*.json'):
        r = json.loads(path.read_text(encoding='utf8'))
        ref[(r['task'], r['seed'])] = r['arms']['none']
    return ref


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    rows = {ident(c): evaluate_one(out, c) for c in conditions(p)}
    trains = {ident(c): json.loads((out / ident(c) / 'train.json').read_text(encoding='utf8')) for c in conditions(p)}
    ref = reference_rows(p); summary = {}
    for task in p['tasks']:
        for arm in p['arms']:
            seeds = [(c['seed'], rows[ident(c)], trains[ident(c)]) for c in conditions(p) if c['task'] == task and c['arm'] == arm]
            entry = dict(neutral={}, clean={}, gate={}, guard={})
            for key in seeds[0][1]['neutral']:
                if key == '0-base':
                    continue
                loss = [(r['neutral']['0-base']['final_accuracy'] - r['neutral'][key]['final_accuracy']) * 100 for _, r, _ in seeds]
                gain = [(r['neutral'][key]['final_accuracy'] - ref[(task, s)]['neutral'][key]['final_accuracy']) * 100 for s, r, _ in seeds if (task, s) in ref]
                entry['neutral'][key] = dict(loss_vs_own_base_pp=mt.seed_summary(loss), gain_vs_none_pp=mt.seed_summary(gain) if gain else None,
                                             within_target=[l <= p['targets']['neutral_loss_pp'] for l in loss],
                                             false_write=[r['neutral'][key]['gate']['false_write'] for _, r, _ in seeds], false_hold=[r['neutral'][key]['gate']['false_hold'] for _, r, _ in seeds])
            for L in p['clean_lengths']:
                diff = [(r['clean'][str(L)]['final_accuracy'] - ref[(task, s)]['clean'][str(L)]['final_accuracy']) * 100 for s, r, _ in seeds if (task, s) in ref]
                entry['clean'][str(L)] = dict(final_accuracy=[r['clean'][str(L)]['final_accuracy'] for _, r, _ in seeds], diff_vs_none_pp=mt.seed_summary(diff) if diff else None,
                                              no_regression=[d >= -p['targets']['clean_regression_pp'] for d in diff], false_hold=[r['clean'][str(L)]['gate']['false_hold'] for _, r, _ in seeds])
            entry['guard'] = {str(s): t['guard'] for s, _, t in seeds}
            skips_ok = all(t['guard']['skipped'] <= p['targets']['skipped_steps_max'] * t['steps'] for _, _, t in seeds)
            all_neutral = all(all(v['within_target']) for v in entry['neutral'].values())
            all_clean = all(all(v['no_regression']) for v in entry['clean'].values()) and bool(ref)
            entry['status'] = ('MEASURED_POSITIVE' if all_neutral and all_clean and skips_ok else 'MIXED' if any(any(v['within_target']) for v in entry['neutral'].values()) else 'MEASURED_NEGATIVE')
            summary[f'{task}/{arm}'] = entry
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], conditions=rows, training={k: {kk: v[kk] for kk in ('selected_step', 'selected_validation', 'guard', 'wall_seconds')} for k, v in trains.items()},
                  summary=summary, caveat='learned gate, no oracle at test; reference none = U1 checkpoints on identical endpoints; three seeds; exploratory intervals')
    sw.write_json(out / 'analysis.json', result)
    return result


def worker(out, slot, slots):
    out = Path(out); manifest = verify_frozen(out)
    log = out / 'workers' / f'slot-{slot}.jsonl'; log.parent.mkdir(exist_ok=True)
    sw.append_line(log, dict(state='STARTED', pid=os.getpid(), utc=time.time(), protocol_hash=manifest['protocol_hash']))
    for c in conditions(manifest['protocol'])[slot::slots]:
        sw.append_line(log, dict(state='CONDITION_STARTED', condition=ident(c), utc=time.time()))
        r = run_one(out, c); e = evaluate_one(out, c)
        sw.append_line(log, dict(state='CONDITION_COMPLETE', condition=ident(c), selected_step=r['selected_step'], guard=r['guard'], wall_seconds=r['wall_seconds'],
                                 neutral={k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}, clean={k: round(v['final_accuracy'], 3) for k, v in e['clean'].items()}, utc=time.time()))
        print(ident(c), json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}), json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['clean'].items()}), f"{r['wall_seconds']:.0f}s", flush=True)
    sw.append_line(log, dict(state='COMPLETE', utc=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-one'); group.add_argument('--evaluate-one'); group.add_argument('--worker', type=int)
    group.add_argument('--analyze', action='store_true'); group.add_argument('--list', action='store_true')
    parser.add_argument('--out', default='results/U2c'); parser.add_argument('--slots', type=int, default=2)
    args = parser.parse_args()
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out)['protocol_hash'])
    elif args.list:
        result = [ident(c) for c in conditions(protocol())]
    elif args.run_one:
        result = {k: v for k, v in run_one(args.out, known(verify_frozen(args.out), args.run_one)).items() if k != 'condition'}
    elif args.evaluate_one:
        e = evaluate_one(args.out, known(verify_frozen(args.out), args.evaluate_one)); result = {k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}
    elif args.worker is not None:
        result = worker(args.out, args.worker, args.slots)
    else:
        a = analyze(args.out); result = {k: dict(status=v['status'], neutral={kk: round(vv['loss_vs_own_base_pp']['mean'], 1) for kk, vv in v['neutral'].items()},
                                                 clean={kk: [round(f, 3) for f in vv['final_accuracy']] for kk, vv in v['clean'].items()}) for k, v in a['summary'].items()}
    print(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
