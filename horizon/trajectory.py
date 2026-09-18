"""U2 measurement layer: hidden-state anchors, displacement under neutral runs, attractors, repair curves.

Training-free. Uses the U1 selected models and the drift decode-loop executor (read-only import) to obtain
per-position hidden states z_t. For the finite-state family the anchors are class centroids of z_t over
training-distribution positions; for the stack family only the distance to the aligned base trajectory is defined.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from drift.model import run_sequence
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run_states(model, x, batch=32):
    model.eval()
    logits, states = [], []
    for at in range(0, len(x), batch):
        z, s = run_sequence(model, x[at:at + batch], 'base')
        logits.append(z); states.append(s)
    return torch.cat(logits), torch.cat(states)


def centroids(z, labels, classes):
    z = np.asarray(z, dtype=np.float64); labels = np.asarray(labels)
    centers = np.full((classes, z.shape[1]), np.nan); counts = np.zeros(classes, dtype=int); spread = np.full(classes, np.nan)
    for c in range(classes):
        m = labels == c
        counts[c] = int(m.sum())
        if counts[c]:
            centers[c] = z[m].mean(0); spread[c] = float(np.linalg.norm(z[m] - centers[c], axis=1).mean())
    return dict(centers=centers, counts=counts, spread=spread)


def nearest(z, centers):
    z = np.asarray(z, dtype=np.float64); centers = np.asarray(centers, dtype=np.float64)
    d = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    return d.argmin(1)


def displacement(nearest_labels, oracle, run):
    start, end = run
    flags = (np.asarray(nearest_labels[start:end]) != np.asarray(oracle[start:end])).astype(int)
    return dict(displaced_in_run=flags, displaced_at_run_end=int(flags[-1]) if len(flags) else 0)


def repair_curve(nearest_labels, oracle, run, horizon):
    end = run[1]
    seg_n = np.asarray(nearest_labels[end:end + horizon]); seg_o = np.asarray(oracle[end:end + horizon])
    return (seg_n != seg_o).astype(int)


def run_positions(placement, base_length, slots):
    at = {'prefix': 0, 'middle': base_length // 2, 'suffix': base_length}[placement]
    return at, at + slots


def measure_condition(u1, cid, out, n_anchor_batches=8, n_eval=500, slots=(128, 512), placements=('prefix', 'middle', 'suffix'), repair_horizon=64):
    manifest = sw.verify_frozen(u1); p = manifest['protocol']; c = sw.known(manifest, cid)
    model = sw.make_model(c, p); model.load_state_dict(torch.load(Path(u1) / cid / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    result = dict(condition=cid, task=c['task'], seed=c['seed'], selected_sha256=sw.sha(Path(u1) / cid / 'selected.pt'), protocol_hash=manifest['protocol_hash'])
    base_len = p['train_length']
    x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', n_eval, base_len)
    logits_b, states_b = run_states(model, x)
    result['eval_source_seed'] = seed; result['n_eval'] = int(len(x))
    if c['task'] == 'five':
        zs, ls = [], []
        for step in range(1, n_anchor_batches + 1):
            b = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], base_len)
            _, s = run_states(model, b['x']); zs.append(s.reshape(-1, s.shape[-1]).numpy()); ls.append(b['y'].reshape(-1).numpy())
        anchors = centroids(np.concatenate(zs), np.concatenate(ls), tf.CLASSES['five'])
        result['anchors'] = dict(counts=anchors['counts'].tolist(), spread=anchors['spread'].tolist(),
                                 center_distances=np.linalg.norm(anchors['centers'][:, None] - anchors['centers'][None], axis=2).tolist(),
                                 sequences=n_anchor_batches * p['batch'])
        nb = nearest(states_b.reshape(-1, states_b.shape[-1]).numpy(), anchors['centers']).reshape(len(x), base_len)
        result['base_anchor_consistency'] = float((nb == y.numpy()).mean())
        result['base_final_accuracy'] = float((logits_b.argmax(-1)[:, -1] == y[:, -1]).float().mean())
        result['runs'] = {}
        for s_ in slots:
            for placement in placements:
                xx, yy = tf.insert_neutrals(x, c['task'], s_, placement)
                lg, st = run_states(model, xx)
                nn_ = nearest(st.reshape(-1, st.shape[-1]).numpy(), anchors['centers']).reshape(len(xx), xx.shape[1])
                start, end = run_positions(placement, base_len, s_)
                flags = np.stack([displacement(nn_[i], yy[i].numpy(), (start, end))['displaced_in_run'] for i in range(len(xx))])
                at_end = nn_[:, end - 1]
                attractor = np.bincount(at_end, minlength=tf.CLASSES['five']).tolist()
                curve_pts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                entry = dict(slots=s_, placement=placement, final_accuracy=float((lg.argmax(-1)[:, -1] == yy[:, -1]).float().mean()),
                             displaced_curve={str(k): float(flags[:, k - 1].mean()) for k in curve_pts if k <= s_},
                             displaced_at_run_end=float(flags[:, -1].mean()), attractor_histogram_at_run_end=attractor,
                             oracle_histogram_at_run_end=np.bincount(yy[:, end - 1].numpy(), minlength=tf.CLASSES['five']).tolist())
                if end < xx.shape[1]:
                    rc = np.stack([repair_curve(nn_[i], yy[i].numpy(), (start, end), repair_horizon) for i in range(len(xx))])
                    entry['repair_curve'] = {str(k): float(rc[:, k - 1].mean()) for k in curve_pts if k <= rc.shape[1]}
                    entry['repaired_within_horizon'] = float((rc.min(1) == 0).mean())
                result['runs'][f'{s_}-{placement}'] = entry
    else:
        rms = float(states_b.pow(2).mean().sqrt())
        result['base_state_rms'] = rms
        result['base_final_accuracy'] = float((logits_b.argmax(-1)[:, -1] == y[:, -1]).float().mean())
        result['runs'] = {}
        for s_ in slots:
            for placement in placements:
                xx, yy = tf.insert_neutrals(x, c['task'], s_, placement)
                lg, st = run_states(model, xx)
                start, end = run_positions(placement, base_len, s_)
                # aligned base position for a real-op position t after the run: t - slots; inside the run compare with the base state just before the run
                run_dist = np.linalg.norm((st[:, start:end] - (states_b[:, start - 1:start] if start > 0 else st[:, start:start + 1] * 0)).numpy(), axis=2) / rms if start > 0 else None
                entry = dict(slots=s_, placement=placement, final_accuracy=float((lg.argmax(-1)[:, -1] == yy[:, -1]).float().mean()))
                if run_dist is not None:
                    entry['distance_from_pre_run_state'] = {str(k): float(run_dist[:, k - 1].mean()) for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if k <= s_}
                if end < xx.shape[1]:
                    after = np.linalg.norm((st[:, end:] - states_b[:, end - s_:]).numpy(), axis=2) / rms
                    entry['distance_after_run'] = {str(k): float(after[:, k - 1].mean()) for k in [1, 2, 4, 8, 16, 32, 64] if k <= after.shape[1]}
                result['runs'][f'{s_}-{placement}'] = entry
    result['seconds'] = time.perf_counter() - t0
    path = Path(out) / f'{cid}.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--u1', default='results/U1'); parser.add_argument('--out', default='results/U2-measure'); parser.add_argument('--conditions', nargs='*')
    args = parser.parse_args()
    manifest = sw.verify_frozen(args.u1)
    ids = args.conditions or [sw.ident(c) for c in sw.conditions(manifest['protocol'])]
    summary = {}
    for cid in ids:
        r = measure_condition(args.u1, cid, args.out)
        summary[cid] = {k: (v.get('displaced_at_run_end'), v.get('repaired_within_horizon'), v['final_accuracy']) for k, v in r['runs'].items()}
        print(cid, json.dumps(summary[cid]), f"{r['seconds']:.0f}s", flush=True)
    Path(args.out, 'summary.json').write_text(json.dumps(dict(u1_protocol=manifest['protocol_hash'], conditions=ids, summary=summary), indent=1) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
