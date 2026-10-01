"""U2 part 2: test-time interventions on the immutable Core at neutral positions, evaluated as a ceiling.

Arms (all use oracle knowledge of the neutral token 2, disclosed):
  none       — plain decode loop (bit-equal to Core.forward)
  freeze     — on neutral rows revert the decoder state to the pre-step value (s always; dk/dv when shapes match) and repeat the logits
  snap       — five only: on neutral rows replace s by the nearest measured class centroid; readout from the snapped state
  wrong_snap — control: replace s by a different centroid than the nearest
No parameter is trained; the Core weights are the U1 selected checkpoints.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from horizon import metrics as mt
from horizon import sweep as sw
from horizon import task_families as tf
from horizon import trajectory as tj

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL = 2
ARMS = ('none', 'freeze', 'snap', 'wrong_snap')
SOURCES = ('core/model.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/trajectory.py', 'horizon/interventions.py', 'drift/model.py')


def neutral_mask(x):
    return x == NEUTRAL


@torch.no_grad()
def run(model, x, arm='none', anchors=None, batch=None):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    model.eval()
    e, _, _ = model.encode(x)
    n = model.mn(e)
    mk = model.rope(model.split(model.mk(n))); mv = model.split(model.mv(n))
    centers = torch.as_tensor(np.asarray(anchors), dtype=e.dtype) if anchors is not None else None
    if arm in ('snap', 'wrong_snap') and centers is None:
        raise ValueError('snap arms need anchors')
    state, prev_logits = None, None
    logits, states = [], []
    for t in range(x.shape[1]):
        neutral = neutral_mask(x[:, t])
        z, new_state = model.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        if arm == 'freeze' and state is not None and bool(neutral.any()):
            m = neutral.view(-1, 1)
            s = torch.where(m, state['s'], new_state['s'])
            dk, dv = new_state['dk'], new_state['dv']
            if state['dk'].shape == dk.shape:
                m4 = neutral.view(-1, 1, 1, 1)
                dk = torch.where(m4, state['dk'], dk); dv = torch.where(m4, state['dv'], dv)
            new_state = dict(new_state, s=s, dk=dk, dv=dv)
            z = torch.where(m, prev_logits, z)
        elif arm in ('snap', 'wrong_snap') and bool(neutral.any()):
            s = new_state['s']
            nearest = torch.cdist(s, centers).argmin(1)
            target = nearest if arm == 'snap' else (nearest + 1) % centers.shape[0]
            m = neutral.view(-1, 1)
            s2 = torch.where(m, centers[target], s)
            new_state = dict(new_state, s=s2)
            z = torch.where(m, model.read(model.outn(s2)), z)
        state, prev_logits = new_state, z
        logits.append(z); states.append(state['s'])
    return torch.stack(logits, 1), torch.stack(states, 1)


def run_batched(model, x, arm, anchors, batch=32):
    logits, states = [], []
    for at in range(0, len(x), batch):
        z, s = run(model, x[at:at + batch], arm, anchors)
        logits.append(z); states.append(s)
    return torch.cat(logits), torch.cat(states)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def protocol():
    return dict(version=1, purpose='U2 test-time interventions at neutral positions (ceiling with oracle neutral knowledge)',
                u1='results/U1', arms=dict(five=list(ARMS), stack=['none', 'freeze']), n_neutral=500, neutral_slots=[128, 512],
                placements=['prefix', 'middle', 'suffix'], clean_lengths=[128, 1024, 4096], n_clean=500, anchor_batches=8,
                targets=dict(neutral_loss_pp=2., clean_regression_pp=1.), device='cpu', threads=1)


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(), u1_protocol_hash=sw.verify_frozen(p['u1'])['protocol_hash'],
                    environment=dict(python=platform.python_version(), torch=torch.__version__), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    torch.set_num_threads(1)
    return manifest


def anchors_for(model, c, p1, batches):
    zs, ls = [], []
    for step in range(1, batches + 1):
        b = tf.train_batch(p1['source_seeds'], c['task'], c['seed'], step, p1['batch'], p1['train_length'])
        _, s = tj.run_states(model, b['x']); zs.append(s.reshape(-1, s.shape[-1]).numpy()); ls.append(b['y'].reshape(-1).numpy())
    return tj.centroids(np.concatenate(zs), np.concatenate(ls), tf.CLASSES[c['task']])


def run_condition(out, cid):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    path = out / f'{cid}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    m1 = sw.verify_frozen(p['u1']); p1 = m1['protocol']; c = sw.known(m1, cid)
    model = sw.make_model(c, p1); model.load_state_dict(torch.load(Path(p['u1']) / cid / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    anchors = anchors_for(model, c, p1, p['anchor_batches']) if c['task'] == 'five' else None
    centers = anchors['centers'] if anchors else None
    result = dict(condition=cid, task=c['task'], seed=c['seed'], selected_sha256=sha(Path(p['u1']) / cid / 'selected.pt'), protocol_hash=manifest['protocol_hash'],
                  anchors=None if anchors is None else dict(counts=anchors['counts'].tolist(), spread=anchors['spread'].tolist(), centers_sha256=hashlib.sha256(centers.tobytes()).hexdigest()),
                  arms={})
    x, y, seed = tf.heldout(p1['source_seeds'], c['task'], c['seed'], 'neutral', p['n_neutral'], p1['train_length'])
    clean = {L: tf.heldout(p1['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L) for L in p['clean_lengths']}
    for arm in p['arms'][c['task']]:
        entry = dict(neutral={}, clean={})
        lb, _ = run_batched(model, x, arm, centers); base_pred = lb.argmax(-1)[:, -1]
        entry['neutral']['0-base'] = dict(final_accuracy=float((base_pred == y[:, -1]).float().mean()))
        for slots in p['neutral_slots']:
            for placement in p['placements']:
                xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
                lg, _ = run_batched(model, xx, arm, centers); pred = lg.argmax(-1)[:, -1]
                entry['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=float((pred == yy[:, -1]).float().mean()),
                                                                 **mt.flip_counts(base_pred.numpy(), pred.numpy(), yy[:, -1].numpy()))
        for L, (cx, cy, cseed) in clean.items():
            lg, _ = run_batched(model, cx, arm, centers)
            entry['clean'][str(L)] = dict(final_accuracy=float((lg.argmax(-1)[:, -1] == cy[:, -1]).float().mean()), prefix_accuracy=float((lg.argmax(-1) == cy).float().mean()), source_seed=cseed)
        result['arms'][arm] = entry
    result['seconds'] = time.perf_counter() - t0
    write_json(path, result)
    return result


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    m1 = sw.verify_frozen(p['u1']); cs = sw.conditions(m1['protocol'])
    rows = {sw.ident(c): run_condition(out, sw.ident(c)) for c in cs}
    summary = {}
    for task in ('five', 'stack'):
        seeds = [r for r in rows.values() if r['task'] == task]
        for arm in p['arms'][task]:
            if arm == 'none':
                continue
            entry = dict(neutral={}, clean={})
            for key in seeds[0]['arms'][arm]['neutral']:
                if key == '0-base':
                    continue
                diffs = [(r['arms'][arm]['neutral'][key]['final_accuracy'] - r['arms']['none']['neutral'][key]['final_accuracy']) * 100 for r in seeds]
                loss = [(r['arms'][arm]['neutral']['0-base']['final_accuracy'] - r['arms'][arm]['neutral'][key]['final_accuracy']) * 100 for r in seeds]
                entry['neutral'][key] = dict(gain_vs_none_pp=mt.seed_summary(diffs), loss_vs_own_base_pp=mt.seed_summary(loss),
                                             within_target=[l <= p['targets']['neutral_loss_pp'] for l in loss])
            for L in p['clean_lengths']:
                diffs = [(r['arms'][arm]['clean'][str(L)]['final_accuracy'] - r['arms']['none']['clean'][str(L)]['final_accuracy']) * 100 for r in seeds]
                entry['clean'][str(L)] = dict(diff_vs_none_pp=mt.seed_summary(diffs), no_regression=[d >= -p['targets']['clean_regression_pp'] for d in diffs])
            all_neutral = all(all(v['within_target']) for v in entry['neutral'].values())
            all_clean = all(all(v['no_regression']) for v in entry['clean'].values())
            entry['status'] = 'MEASURED_POSITIVE_CEILING' if all_neutral and all_clean else 'MIXED' if any(any(v['within_target']) for v in entry['neutral'].values()) else 'MEASURED_NEGATIVE'
            summary[f'{task}/{arm}'] = entry
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], u1_protocol_hash=manifest['u1_protocol_hash'], conditions=rows, summary=summary,
                  caveat='oracle neutral-token knowledge; ceiling for a learned no-op detector; three seeds; exploratory intervals')
    write_json(out / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-one'); group.add_argument('--run-all', action='store_true'); group.add_argument('--analyze', action='store_true')
    parser.add_argument('--out', default='results/U2-intervene')
    args = parser.parse_args()
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out)['protocol_hash'])
    elif args.run_one:
        r = run_condition(args.out, args.run_one); result = {arm: {k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()} for arm, e in r['arms'].items()}
    elif args.run_all:
        m = verify_frozen(args.out); m1 = sw.verify_frozen(m['protocol']['u1'])
        result = {}
        for c in sw.conditions(m1['protocol']):
            r = run_condition(args.out, sw.ident(c)); result[sw.ident(c)] = {arm: {k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()} for arm, e in r['arms'].items()}
            print(sw.ident(c), json.dumps(result[sw.ident(c)]), f"{r['seconds']:.0f}s", flush=True)
    else:
        a = analyze(args.out); result = {k: dict(status=v['status'], neutral={kk: round(vv['loss_vs_own_base_pp']['mean'], 1) for kk, vv in v['neutral'].items()}, clean={kk: round(vv['diff_vs_none_pp']['mean'], 2) for kk, vv in v['clean'].items()}) for k, v in a['summary'].items()}
    print(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
