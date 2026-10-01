"""U4-inject: does attention injection remove the explosion the additive feedback causes, and at what cost?

Three arms trained from scratch on identical data with identical Core initialisation (the arms differ only in how the
previous state reaches the current step):
  additive   - the sealed Core feedback, i.e. the U1 baseline, as the control
  inject     - the previous state queries the decoder window; its contribution is a softmax-weighted average
  inject_mix - the same, combined convexly with the input instead of added

Measured on both sides of the paper's claim: the residual side (state RMS growth across a 512-step neutral run, clean
norm curve) and the gradient side (the A8 guard's pre-clip norms, clipped and skipped steps), plus the campaign's
standard endpoints so robustness is comparable with U1/U2b/U2c and the anchor separation with U2d/U4-sep.

CLI: --freeze [--lr X] | --run-all   (--out results/U4inj)
"""
import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from horizon import gated_sweep as gs
from horizon import gated_trajectory as gt
from horizon import guards as gu
from horizon import inject as ij
from horizon import metrics as mt
from horizon import sep_loss as sl
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/gated_sweep.py',
           'horizon/gated_trajectory.py', 'horizon/guards.py', 'horizon/inject.py', 'horizon/inject_sweep.py')


def protocol():
    b = gs.protocol()
    p = {k: v for k, v in b.items() if k not in ('arms', 'gate_loss_weight', 'gate_k', 'purpose', 'version', 'targets')}
    p.update(version=1,
             purpose='U4-inject: replace the Core additive state feedback with attention injection (FLT, arXiv 2605.18797) '
                     'and measure residual growth, gradient explosion and drift robustness against the sealed additive control',
             arms=['additive', 'inject', 'inject_mix'], tasks=['five'], anchor_batches=64, norm_run_slots=512, norm_run_n=200,
             reference='results/U2b-intervene', u2d='results/U2d/compare.json',
             note='all three arms share the same seeded Core initialisation; in the injection arms the sealed gate/merge '
                  'parameters receive no gradient and stay at their initial values, and the window K/V come from the input alone',
             targets=dict(neutral_loss_pp=2., clean_regression_pp=1., skipped_steps_max=.01))
    return p


def conditions(p):
    return [dict(arm=arm, task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'], lr=p['lr'], stage='u4inj')
            for arm in p['arms'] for task in p['tasks'] for seed in p['primary_seeds']]


def ident(c):
    return f"u4inj-{c['arm']}-{c['task']}-w{c['width']}-s{c['seed']}"


def known(manifest, cid):
    for c in conditions(manifest['protocol']):
        if ident(c) == cid:
            return c
    raise ValueError('unknown condition ' + cid)


def make_model(c, p):
    return ij.InjectCore(sw.make_model(c, p), mode=c['arm'])


def source_hashes():
    return {name: sw.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol()); sw.configure(p)
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(),
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
    model.eval(); logits, states = [], []
    for at in range(0, len(x), p['eval_batch']):
        z, s = model(x[at:at + p['eval_batch']]); logits.append(z); states.append(s)
    logits = torch.cat(logits); states = torch.cat(states); pred = logits.argmax(-1)
    return dict(final_loss=float(F.cross_entropy(logits[:, -1], y[:, -1])), final_accuracy=float((pred[:, -1] == y[:, -1]).float().mean()),
                all_accuracy=float((pred == y).float().mean())), pred, states


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
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train()
            logits, states = model(batch['x'])
            loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'], grad_norm=report['grad_norm_before_clip'],
                       state_rms=float(states.detach().square().mean().sqrt()), input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _, _ = predict(model, vx, vy, p); row['validation'] = gs.json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        counts = model.core.counts()
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'], selected_validation=gs.json_safe(selected['validation']),
                      guard=counter.summary(), parameters=dict(core_allocated=counts['allocated'], injection=sum(q.numel() for q in model.injection_parameters())),
                      train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall, protocol_hash=manifest['protocol_hash'],
                      selected_sha256=sw.sha(directory / 'selected.pt'), initial_sha256=sw.sha(directory / 'initial.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', gs.json_safe(result))
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
    stats, base_pred, _ = predict(model, x, y, p)
    row['neutral']['0-base'] = dict(final_accuracy=stats['final_accuracy'], source_seed=seed)
    for slots in p['neutral_slots']:
        for placement in p['placements']:
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred, _ = predict(model, xx, yy, p)
            row['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=stats['final_accuracy'],
                                                          **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    for L in p['clean_lengths']:
        cx, cy, cseed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L)
        stats, _, cstates = predict(model, cx, cy, p)
        curve = gt.norm_curve(cstates)
        row['clean'][str(L)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['all_accuracy'], source_seed=cseed,
                                    norm=dict(first=curve[0], mid=curve[len(curve) // 2], last=curve[-1], max=max(curve)))
    row.update(**residual_growth(model, c, p), **anchors(model, c, p))
    row['seconds'] = time.perf_counter() - t0
    sw.write_json(path, gs.json_safe(row))
    return row


def residual_growth(model, c, p):
    """The FLT paper's residual-explosion quantity: how far the state norm grows across a long neutral run."""
    n = min(p['norm_run_n'], p['n_neutral'])
    x, _, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', n, p['train_length'])
    xx, yy = tf.insert_neutrals(x, c['task'], p['norm_run_slots'], 'middle')
    _, _, states = predict(model, xx, yy, p)
    curve = gt.norm_curve(states)
    start = p['train_length'] // 2
    end = min(start + p['norm_run_slots'] - 1, len(curve) - 1)
    disp = gt.displacement_curve(states, start=start)
    return dict(run_norm=dict(before=curve[start - 1], after=curve[end], max=max(curve), growth=curve[end] / max(curve[start - 1], 1e-9)),
                norm_growth=curve[end] / max(curve[start - 1], 1e-9),
                displacement=dict(k1=disp[0], k16=disp[15] if len(disp) > 15 else None, end=disp[-1]))


def anchors(model, c, p):
    """The U2d/U4-sep separation ratio, same definition, so the three cycles stay comparable."""
    zs, ls = [], []
    for step in range(1, p['anchor_batches'] + 1):
        b = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
        _, _, states = predict(model, b['x'], b['y'], p)
        zs.append(states.reshape(-1, states.shape[-1]).numpy()); ls.append(b['y'].reshape(-1).numpy())
    a = gt.anchor_summary(np.concatenate(zs), np.concatenate(ls), tf.CLASSES[c['task']])
    return dict(separation_ratio=a['separation_ratio'], anchors=dict(min_center_distance=a['min_center_distance'], mean_spread=a['mean_spread'], counts=a['counts']))


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true')
    parser.add_argument('--out', required=True); parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    if args.freeze:
        p = protocol()
        if args.lr:
            p['lr'] = args.lr
        m = freeze(args.out, p)
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], lr=m['protocol']['lr'], arms=m['protocol']['arms'])))
        return
    m = verify_frozen(args.out)
    for c in conditions(m['protocol']):
        r = run_one(args.out, c); e = evaluate_one(args.out, c)
        print(ident(c), 'val %.3f' % r['selected_validation']['final_accuracy'],
              'grad_max %.3g clipped %d skipped %d' % (r['guard']['max_grad_norm_before_clip'], r['guard']['clipped'], r['guard']['skipped']),
              'norm x%.2f' % e['norm_growth'], 'sep %.2f' % e['separation_ratio'],
              'neutral', json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}),
              'clean', json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['clean'].items()}),
              '%.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
