"""U4-sep2: take a model that already solves the task but is fragile, move ONLY its geometry, see if robustness follows.

U4-sep tried to answer this by training from scratch with an auxiliary loss and failed twice: the loss targeted
gap / within-class spread, which does not separate the outcome groups, and a gated model could satisfy it by closing
the gate, which collapsed the state and destroyed the task (validation 0.207 and 0.215 against a chance level of 0.200).

This runner fixes both. It starts from frozen U2c checkpoints that reached validation >= 0.95 and still repair badly,
so the task loss holds the model up and the collapse escape is closed; and it optimises gap / drift, the quantity that
does separate, through a hinge that stops at a target taken from the robust range.

The control arm matters as much as the intervention: `none` fine-tunes for exactly the same number of steps with the
task loss alone. Without it, any improvement could simply be more training.

CLI: --freeze | --run-all | --list   (--out results/U4sep2)
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

from horizon import gated as gd
from horizon import gated_sweep as gs
from horizon import geom_loss as gl
from horizon import guards as gu
from horizon import metrics as mt
from horizon import sweep as sw
from horizon import task_families as tf
from horizon import u2d_compare as ud

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/gated.py', 'horizon/gated_sweep.py',
           'horizon/gated_trajectory.py', 'horizon/guards.py', 'horizon/geom_loss.py', 'horizon/u2d_compare.py', 'horizon/sep_finetune.py')

# Selection rule, fixed in advance: U2c five-task checkpoints with validation >= 0.95 (the model solves the task, so the
# question "does geometry decide robustness" is even meaningful) and 512-suffix repair < 50 % (it is fragile, so there
# is something to rescue). Applied to results/U2c-lr0003, it yields exactly these five.
CHECKPOINTS = [
    dict(level='lr0.0003', directory='results/U2c-lr0003', arm='soft', seed=117, u2c_validation=1.000, u2c_repair=-40.7),
    dict(level='lr0.0003', directory='results/U2c-lr0003', arm='soft', seed=143, u2c_validation=1.000, u2c_repair=16.6),
    dict(level='lr0.0003', directory='results/U2c-lr0003', arm='supervised', seed=117, u2c_validation=0.988, u2c_repair=-40.7),
    dict(level='lr0.0003', directory='results/U2c-lr0003', arm='supervised', seed=143, u2c_validation=0.996, u2c_repair=-30.7),
    dict(level='lr0.0003', directory='results/U2c-lr0003', arm='supervised', seed=191, u2c_validation=0.992, u2c_repair=-99.5),
]


def protocol():
    b = gs.protocol()
    p = {k: v for k, v in b.items() if k not in ('arms', 'gate_loss_weight', 'purpose', 'version', 'targets', 'steps', 'lr')}
    p.update(version=1,
             purpose='U4-sep2: fine-tune fragile-but-working U2c checkpoints with a hinge on gap/drift - the only quantity '
                     'that separates robust from fragile when U2d conditions are regrouped by measured repair - against a '
                     'same-budget task-loss-only control',
             arms=['none', 'hinge'], checkpoints=copy.deepcopy(CHECKPOINTS), task='five',
             steps=512, lr=0.0003, hinge_weight=1.0, hinge_target=1.5, drift_slots=64, drift_placement='middle', n_geometry=200, drift_floor=0.5,
             selection_rule='U2c five checkpoints with validation >= 0.95 and 512-suffix repair < 50 %',
             target_rationale='regrouped by measured repair, gap/drift is fragile <= 0.57 and robust >= 0.92; the target '
                              'sits comfortably inside the robust group without demanding its extreme',
             drift_floor_rationale='a 64-step pilot showed a gated model can drive drift to exactly 0 by freezing the state, '
                                   'keeping a small non-zero gap (0.05) so the RAW ratio explodes and the collapse is rewarded. '
                                   'The loss therefore divides by (drift + 0.5); reported ratios keep the raw definition so they '
                                   'stay comparable with U2d. Working models in the pilot drift 1.7-2.0 over 64 slots, so the '
                                   'floor is far below the working range and far above the frozen value',
             pilot='lr 3e-4 with weight 1.0 moved gap/drift 1.18 -> 2.60 in 64 steps with validation 1.000; lr 1e-3 and 3e-3 '
                   'collapsed the model to 0.219 and 0.266, which is why the fine-tuning lr stays at 3e-4',
             reference='results/U2b-intervene', u2c_protocol='results/U2c-lr0003',
             targets=dict(repair_pp=30., validation_min=.95))
    return p


def conditions(p):
    return [dict(arm=arm, source=dict(c), task=p['task'], width=p['width'], heads=p['heads'], window=p['window'], stage='u4sep2')
            for arm in p['arms'] for c in p['checkpoints']]


def ident(c):
    return f"u4sep2-{c['arm']}-from_{c['source']['arm']}-s{c['source']['seed']}"


def known(manifest, cid):
    for c in conditions(manifest['protocol']):
        if ident(c) == cid:
            return c
    raise ValueError('unknown condition ' + cid)


def source_condition(c, u2c_protocol):
    src = c['source']
    return gs.known(gs.verify_frozen(src['directory']), f"u2c-{src['arm']}-{c['task']}-w{c['width']}-s{src['seed']}")


def load_source(c, p):
    """The frozen U2c model and its own protocol, loaded and hashed."""
    src = c['source']
    u2c = gs.verify_frozen(src['directory'])
    sc = source_condition(c, u2c)
    model = gs.make_model(sc, u2c['protocol'])
    path = Path(src['directory']) / f"u2c-{src['arm']}-{c['task']}-w{c['width']}-s{src['seed']}" / 'selected.pt'
    model.load_state_dict(torch.load(path, weights_only=True))
    return model, sc, u2c['protocol'], sw.sha(path)


def forward_with_states(model, x):
    """GatedCore.forward for soft AND hard_st, with the committed states kept on the graph (the sealed one detaches)."""
    core = model.core
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n))); mv = core.split(core.mv(n))
    committed = core.initial.expand(x.shape[0], -1)
    state, outs, gates, states = None, [], [], []
    for t in range(x.shape[1]):
        _, cand = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        raw = torch.sigmoid(model.head(torch.cat([e[:, t], committed], -1)).squeeze(-1))
        g = raw if model.mode == 'soft' else (raw > 0.5).to(raw.dtype) + raw - raw.detach()
        dk, dv = cand['dk'], cand['dv']
        if model.mode == 'soft':
            gv = g.view(-1, 1)
            s = gv * cand['s'] + (1 - gv) * committed
        else:
            m = (g.detach() > 0.5).view(-1, 1)
            s = torch.where(m, cand['s'], committed) + (g - g.detach()).view(-1, 1) * (cand['s'] - committed)
            if state is not None and state['dk'].shape == dk.shape:
                m4 = m.view(-1, 1, 1, 1)
                dk = torch.where(m4, dk, state['dk']); dv = torch.where(m4, dv, state['dv'])
        state = dict(cand, s=s, dk=dk, dv=dv); committed = s
        outs.append(core.read(core.outn(s))); gates.append(g); states.append(s)
    return torch.stack(outs, 1), torch.stack(gates, 1), torch.stack(states, 1)


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


def geometry_terms(model, batch, p, classes, states):
    """gap on the clean batch, drift across a neutral run inserted into the same batch, and their ratio.

    `states` is the clean forward the caller already ran, so only the neutral-run pass is extra work.
    """
    z = states.reshape(-1, states.shape[-1]); labels = batch['y'].reshape(-1)
    xx, _ = tf.insert_neutrals(batch['x'], p['task'], p['drift_slots'], p['drift_placement'])
    _, _, run_states = forward_with_states(model, xx)
    start = p['train_length'] // 2
    gap = gl.min_centre_distance(z, labels, classes)
    drift = gl.run_drift(run_states, start=start, end=start + p['drift_slots'] - 1)
    raw = gap / (drift + gl.EPS)                                   # reported: comparable with U2d
    floored = gap / (drift + p['drift_floor'] + gl.EPS)            # optimised: collapse cannot exploit drift -> 0
    return gap, drift, raw, floored


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
        model, sc, u2c_p, source_sha = load_source(c, p)
        torch.save(sw.cpu_state(model), directory / 'initial.pt')
        opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        vx, vy, _ = tf.heldout(p['source_seeds'], p['task'], sc['seed'], 'validation', p['validation_n'], p['train_length'])
        counter = gu.GuardCounter(); best, selected = None, None
        classes = tf.CLASSES[p['task']]; use_hinge = c['arm'] == 'hinge'
        for step in range(1, p['steps'] + 1):
            batch = tf.train_batch(p['source_seeds'], p['task'], sc['seed'], 10_000 + step, p['batch'], p['train_length'])
            model.train()
            logits, gates, states = forward_with_states(model, batch['x'])
            task_loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
            if use_hinge:
                gap, drift, ratio, floored = geometry_terms(model, batch, p, classes, states)
                penalty = gl.hinge(floored, p['hinge_target'])
                loss = task_loss + p['hinge_weight'] * penalty
            else:
                with torch.no_grad():
                    gap, drift, ratio, floored = geometry_terms(model, batch, p, classes, states.detach())
                penalty = torch.zeros(()); loss = task_loss
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            row = dict(step=step, applied=report['applied'], loss=report['loss'], task_loss=gs.finite_or_none(float(task_loss.detach())),
                       gap=float(gap.detach()), drift=float(drift.detach()), gap_over_drift=float(ratio.detach()),
                       gap_over_drift_floored=float(floored.detach()),
                       hinge=float(penalty.detach()), grad_norm=report['grad_norm_before_clip'],
                       hold_rate=float((gates.detach() <= .5).float().mean()), input_hash=batch['input_hash'])
            if step % p['validation_every'] == 0 or step == p['steps'] or step == 1:
                stats, _, _ = gs.predict(model, vx, vy, p); row['validation'] = gs.json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        rows = [json.loads(l) for l in (directory / 'updates.jsonl').read_text(encoding='utf8').splitlines()]
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'], selected_validation=gs.json_safe(selected['validation']),
                      guard=counter.summary(), source_sha256=source_sha, u2c_protocol_hash=sw.digest(u2c_p),
                      ratio_first=rows[0]['gap_over_drift'], ratio_last=rows[-1]['gap_over_drift'],
                      gap_first=rows[0]['gap'], gap_last=rows[-1]['gap'], drift_first=rows[0]['drift'], drift_last=rows[-1]['drift'],
                      wall_seconds=time.perf_counter() - wall, protocol_hash=manifest['protocol_hash'],
                      selected_sha256=sw.sha(directory / 'selected.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
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
    model, sc, u2c_p, _ = load_source(c, p)
    model.load_state_dict(torch.load(directory / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    row = dict(condition=c, ident=ident(c), protocol_hash=manifest['protocol_hash'], selected_sha256=sw.sha(directory / 'selected.pt'), neutral={}, clean={})
    x, y, seed = tf.heldout(p['source_seeds'], p['task'], sc['seed'], 'neutral', p['n_neutral'], p['train_length'])
    stats, base_pred, gates = gs.predict(model, x, y, p)
    row['neutral']['0-base'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(x, p['gate_k'])), source_seed=seed)
    for slots in p['neutral_slots']:
        for placement in p['placements']:
            xx, yy = tf.insert_neutrals(x, p['task'], slots, placement)
            stats, pred, gates = gs.predict(model, xx, yy, p)
            row['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(xx, p['gate_k'])),
                                                          **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    for L in p['clean_lengths']:
        cx, cy, cseed = tf.heldout(p['source_seeds'], p['task'], sc['seed'], 'test', p['n_clean'], L)
        stats, _, _ = gs.predict(model, cx, cy, p)
        row['clean'][str(L)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['all_accuracy'], source_seed=cseed)
    m = ud.measure(model, sc, u2c_p, n_seq=p['n_geometry'])
    row['geometry'] = dict(gap=m['anchors']['min_center_distance'], spread=m['anchors']['mean_spread'],
                           separation_ratio=m['anchors']['separation_ratio'], drift_end=m['displacement']['end'],
                           gap_over_drift=(m['anchors']['min_center_distance'] / m['displacement']['end']) if m['displacement']['end'] > 1e-9 else None,
                           class_kept_through_run=m['class_kept_through_run'], run_norm=m['run_norm'])
    row['seconds'] = time.perf_counter() - t0
    sw.write_json(path, gs.json_safe(row))
    return row


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true'); group.add_argument('--list', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.list:
        print(json.dumps([ident(c) for c in conditions(protocol())], indent=1)); return
    if args.freeze:
        m = freeze(args.out)
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], conditions=len(conditions(m['protocol'])),
                              target=m['protocol']['hinge_target'], steps=m['protocol']['steps'])))
        return
    m = verify_frozen(args.out)
    for c in conditions(m['protocol']):
        r = run_one(args.out, c); e = evaluate_one(args.out, c)
        print(ident(c), 'val %.3f' % r['selected_validation']['final_accuracy'],
              'ratio %.2f->%.2f' % (r['ratio_first'], r['ratio_last']),
              'gap %.2f->%.2f drift %.2f->%.2f' % (r['gap_first'], r['gap_last'], r['drift_first'], r['drift_last']),
              'eval gap/drift %s' % (('%.2f' % e['geometry']['gap_over_drift']) if e['geometry']['gap_over_drift'] else 'inf'),
              'neutral', json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}),
              '%.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
