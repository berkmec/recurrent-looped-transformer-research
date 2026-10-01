"""U4-sep: train the soft-gated Core with an auxiliary separation loss, to test the U2d finding causally.

U2d showed - correlationally - that robust and fragile gated solutions are separated with no overlap by the class-anchor
separation ratio, while absolute drift is identical in both groups. Here that ratio is optimised directly, everything
else held identical to U2c, and the same frozen endpoints are used. The prediction is pre-registered in the ledger.

`horizon/gated_sweep.py` and `horizon/gated.py` are hashed inside the sealed U2c manifests, so neither may be edited;
the training loop is therefore reimplemented here (deliberate duplication, forced by the immutability rule) while the
model builder, the predictor and the U2d anchor diagnostic are imported unchanged. `forward_with_states` reproduces
GatedCore.forward in soft mode but keeps the states attached to the graph; tests/test_sep_sweep.py asserts that its
logits and gates are bitwise identical to the sealed forward, so the duplication cannot drift silently.

CLI: --freeze [--lr X] | --run-all   (--out results/U4sep-lr003)
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
from horizon import gated_trajectory as gt
from horizon import guards as gu
from horizon import metrics as mt
from horizon import sep_loss as sl
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/gated.py', 'horizon/gated_sweep.py',
           'horizon/gated_trajectory.py', 'horizon/guards.py', 'horizon/sep_loss.py', 'horizon/sep_sweep.py')


def protocol():
    b = gs.protocol()
    p = {k: v for k, v in b.items() if k not in ('arms', 'gate_loss_weight', 'purpose', 'version', 'targets')}
    p.update(version=1,
             purpose='U4-sep: does directly widening the class-anchor separation ratio CAUSE drift robustness? (causal test of the U2d correlation)',
             arms=['soft_sep'], tasks=['five'], sep_weight=0.1, gate_k=b['gate_k'], anchor_batches=64,
             reference='results/U2c-v2', reference_lr0003='results/U2c-lr0003', u2d='results/U2d/compare.json',
             note='soft gate + auxiliary -log(min centre distance / mean within-class spread), the quantity U2d measured; '
                  'scale invariant by construction, so it cannot be satisfied by inflating the state norm',
             targets=dict(repair_pp=30., validation_min=.95))
    return p


def conditions(p):
    return [dict(arm=arm, task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'], lr=p['lr'], stage='u4sep')
            for arm in p['arms'] for task in p['tasks'] for seed in p['primary_seeds']]


def ident(c):
    return f"u4sep-{c['arm']}-{c['task']}-w{c['width']}-s{c['seed']}"


def known(manifest, cid):
    for c in conditions(manifest['protocol']):
        if ident(c) == cid:
            return c
    raise ValueError('unknown condition ' + cid)


def make_model(c, p):
    return gs.make_model(dict(c, arm='soft'), p)   # the gate is the plain U2c soft gate; only the loss differs


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


def forward_with_states(model, x):
    """GatedCore.forward in soft mode, with the committed states kept on the graph (the sealed one detaches them)."""
    core = model.core
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n))); mv = core.split(core.mv(n))
    committed = core.initial.expand(x.shape[0], -1)
    state, outs, gates, states = None, [], [], []
    for t in range(x.shape[1]):
        _, cand = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        g = torch.sigmoid(model.head(torch.cat([e[:, t], committed], -1)).squeeze(-1))
        gv = g.view(-1, 1)
        s = gv * cand['s'] + (1 - gv) * committed
        state = dict(cand, s=s); committed = s
        outs.append(core.read(core.outn(s))); gates.append(g); states.append(s)
    return torch.stack(outs, 1), torch.stack(gates, 1), torch.stack(states, 1)


def anchor_ratio(model, c, p):
    """The U2d separation ratio: anchors over `anchor_batches` training batches, float64, same code path as U2d."""
    zs, ls = [], []
    for step in range(1, p['anchor_batches'] + 1):
        b = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
        out = gt.run_states(model, b['x'], batch=p['eval_batch'])
        zs.append(out['states'].reshape(-1, out['states'].shape[-1]).numpy()); ls.append(b['y'].reshape(-1).numpy())
    a = gt.anchor_summary(np.concatenate(zs), np.concatenate(ls), tf.CLASSES[c['task']])
    return dict(separation_ratio=a['separation_ratio'], min_center_distance=a['min_center_distance'], mean_spread=a['mean_spread'], counts=a['counts'])


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
        classes = tf.CLASSES[c['task']]; w = float(p['sep_weight'])
        ratio_last, seploss_last = None, None
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train()
            logits, gates, states = forward_with_states(model, batch['x'])
            task_loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
            z = states.reshape(-1, states.shape[-1]); labels = batch['y'].reshape(-1)
            sep = sl.separation_loss(z, labels, classes)
            seploss_last = float(sep.detach()); ratio_last = float(sl.separation_ratio(z.detach(), labels, classes))
            report = counter.record(gu.guarded_step(opt, model.parameters(), task_loss + w * sep, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'],
                       task_loss=gs.finite_or_none(float(task_loss.detach())), sep_loss=gs.finite_or_none(seploss_last),
                       sep_ratio=gs.finite_or_none(ratio_last), grad_norm=report['grad_norm_before_clip'],
                       hold_rate=float((gates.detach() <= .5).float().mean()), input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _, _ = gs.predict(model, vx, vy, p); row['validation'] = gs.json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'], selected_validation=gs.json_safe(selected['validation']),
                      guard=counter.summary(), sep_ratio_last=gs.finite_or_none(ratio_last), sep_loss_last=gs.finite_or_none(seploss_last),
                      train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall, protocol_hash=manifest['protocol_hash'],
                      selected_sha256=sw.sha(directory / 'selected.pt'), initial_sha256=sw.sha(directory / 'initial.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', gs.json_safe(result))
        sw.write_json(directory / 'complete.json', dict(ident=ident(c), utc=time.time(), train_sha256=sw.sha(directory / 'train.json')))
        return result
    finally:
        (directory / 'active.lock').unlink(missing_ok=True)


def evaluate_one(out, c):
    """The U2c endpoints, unchanged, plus the U2d separation ratio of the selected checkpoint."""
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
    stats, base_pred, gates = gs.predict(model, x, y, p)
    row['neutral']['0-base'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(x, p['gate_k'])), source_seed=seed)
    for slots in p['neutral_slots']:
        for placement in p['placements']:
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred, gates = gs.predict(model, xx, yy, p)
            row['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=stats['final_accuracy'], gate=gd.gate_rates(gates, gd.gate_target(xx, p['gate_k'])),
                                                          **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    for L in p['clean_lengths']:
        cx, cy, cseed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L)
        stats, pred, gates = gs.predict(model, cx, cy, p)
        row['clean'][str(L)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['all_accuracy'],
                                    gate=gd.gate_rates(gates, gd.gate_target(cx, p['gate_k'])), source_seed=cseed)
    anchors = anchor_ratio(model, c, p)
    row['anchors'] = anchors; row['separation_ratio'] = anchors['separation_ratio']
    row['seconds'] = time.perf_counter() - t0
    sw.write_json(path, gs.json_safe(row))
    return row


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
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], lr=m['protocol']['lr'], sep_weight=m['protocol']['sep_weight'])))
        return
    m = verify_frozen(args.out)
    for c in conditions(m['protocol']):
        r = run_one(args.out, c); e = evaluate_one(args.out, c)
        print(ident(c), 'val %.3f' % r['selected_validation']['final_accuracy'],
              'sep %.2f' % e['separation_ratio'],
              'neutral', json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()}),
              'clean', json.dumps({k: round(v['final_accuracy'], 3) for k, v in e['clean'].items()}),
              '%.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
