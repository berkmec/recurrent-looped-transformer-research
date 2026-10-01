"""U2d: run the gated-trajectory diagnostics over already-frozen U2c checkpoints and compare robust vs fragile solutions.

Diagnostic only — it trains nothing and freezes no protocol, but it records the sha256 of every checkpoint it reads
and of its own sources, so the numbers can be traced back (the lesson from U2-measure, which had no such record).

Usage: python -B -m horizon.u2d_compare --out results/U2d --arms soft --task five
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from horizon import gated_sweep as gs
from horizon import gated_trajectory as gt
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'horizon/gated.py', 'horizon/gated_sweep.py', 'horizon/gated_trajectory.py', 'horizon/u2d_compare.py')
LEVELS = {'lr0.003': 'results/U2c-v2', 'lr0.0003': 'results/U2c-lr0003'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(level_dir, cid, protocol):
    c = gs.known(gs.verify_frozen(level_dir), cid)
    model = gs.make_model(c, protocol)
    model.load_state_dict(torch.load(Path(level_dir) / cid / 'selected.pt', weights_only=True))
    return model, c


def measure(model, c, p, n_seq=200, n_anchor=64, slots=512):
    """Anchors on the training distribution, then a neutral run inserted in the middle."""
    torch.set_num_threads(1)
    zs, ls = [], []
    for step in range(1, n_anchor + 1):
        b = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
        out = gt.run_states(model, b['x'], batch=p['eval_batch'])
        zs.append(out['states'].reshape(-1, out['states'].shape[-1]).numpy()); ls.append(b['y'].reshape(-1).numpy())
    anchors = gt.anchor_summary(np.concatenate(zs), np.concatenate(ls), tf.CLASSES[c['task']])

    x, y, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', n_seq, p['train_length'])
    clean = gt.run_states(model, x, batch=p['eval_batch'])
    xx, yy = tf.insert_neutrals(x, c['task'], slots, 'middle')
    run = gt.run_states(model, xx, batch=p['eval_batch'])
    start = p['train_length'] // 2
    disp = gt.displacement_curve(run['states'], start=start)
    norms_clean = gt.norm_curve(clean['states'])
    norms_run = gt.norm_curve(run['states'])
    near_before = gt.nearest_class(run['states'][:, start - 1:start], anchors['centers'])[:, 0]
    near_after = gt.nearest_class(run['states'][:, start + slots - 1:start + slots], anchors['centers'])[:, 0]
    kept = float((near_before == near_after).float().mean())
    hist = np.bincount(near_after.numpy(), minlength=tf.CLASSES[c['task']]).tolist()
    return dict(
        anchors=dict(min_center_distance=anchors['min_center_distance'], mean_spread=anchors['mean_spread'],
                     separation_ratio=anchors['separation_ratio'], counts=anchors['counts']),
        clean_norm=dict(first=norms_clean[0], mid=norms_clean[len(norms_clean) // 2], last=norms_clean[-1], max=max(norms_clean)),
        run_norm=dict(before=norms_run[start - 1], after=norms_run[start + slots - 1], max=max(norms_run),
                      growth=norms_run[start + slots - 1] / max(norms_run[start - 1], 1e-9)),
        displacement=dict(k1=disp[0], k4=disp[3] if len(disp) > 3 else None, k16=disp[15] if len(disp) > 15 else None,
                          k64=disp[63] if len(disp) > 63 else None, end=disp[slots - 1] if len(disp) >= slots else disp[-1]),
        class_kept_through_run=kept, attractor_histogram=hist,
        gate_hold_rate_clean=float((clean['gates'] <= .5).float().mean()),
        gate_hold_rate_run=float((run['gates'] <= .5).float().mean()),
        final_accuracy_clean=float((clean['logits'].argmax(-1)[:, -1] == y[:, -1]).float().mean()),
        final_accuracy_run=float((run['logits'].argmax(-1)[:, -1] == yy[:, -1]).float().mean()))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/U2d'); parser.add_argument('--arms', default='soft,hard_st')
    parser.add_argument('--task', default='five'); parser.add_argument('--seeds', default='117,143,191')
    parser.add_argument('--n-seq', type=int, default=200)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = dict(kind='DIAGNOSTIC_OVER_FROZEN_CHECKPOINTS', utc=time.time(), sources={s: sha(ROOT / s) for s in SOURCES},
                  environment=dict(python=platform.python_version(), torch=torch.__version__), levels=LEVELS, rows={})
    for arm in args.arms.split(','):
        for seed in [int(s) for s in args.seeds.split(',')]:
            cid = f'u2c-{arm}-{args.task}-w32-s{seed}'
            for level, d in LEVELS.items():
                path = Path(d) / cid / 'selected.pt'
                if not path.exists():
                    continue
                manifest = gs.verify_frozen(d)
                model, c = load(d, cid, manifest['protocol'])
                t0 = time.perf_counter()
                m = measure(model, c, manifest['protocol'], n_seq=args.n_seq)
                m.update(checkpoint_sha256=sha(path), level=level, lr=manifest['protocol']['lr'], seconds=time.perf_counter() - t0)
                report['rows'][f'{cid}@{level}'] = m
                print(f"{cid}@{level}: run acc {m['final_accuracy_run']:.3f} | disp k1 {m['displacement']['k1']:.2f} k16 {m['displacement']['k16']:.2f} end {m['displacement']['end']:.2f} | "
                      f"norm {m['run_norm']['before']:.2f} to {m['run_norm']['after']:.2f} (x{m['run_norm']['growth']:.2f}) | sep {m['anchors']['separation_ratio']:.2f} | class kept {m['class_kept_through_run']:.3f}", flush=True)
    (out / 'compare.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
    print('written', out / 'compare.json')


if __name__ == '__main__':
    main()
