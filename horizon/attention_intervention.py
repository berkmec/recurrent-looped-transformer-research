"""E-A: why does the S5 Core break at 4-8x the trained length? Inference-time interventions on SAVED weights (no training, dev split only).

The per-window first-error hazard of the S5 Core (w128, curriculum to length 16) is 0 for positions < 64 and about 2.4e-3 per step from 128 on, while a
GRU/LSTM sits on a constant floor from the start. Two readings:

  H1 attention: past the trained context the softmax over n keys dilutes the logit margin the model learned at n <= 16 (and relative offsets beyond
     the trained range are new to RoPE). A window of the trained size, or a log-n temperature, then removes the break: the hazard stays near 0.
  H2 state: the recurrent state loses precision or leaves the region the transition was trained on. Attention interventions change nothing.

The sealed Core is never edited: `IntervenedCore` only overrides `attend`. Three places use attention and are told apart by shape:
  enc  the causal encoder self-attention over the whole prefix (`causal=True`),
  mem  the decoder's cross-attention over the encoder memory (non-causal, as many keys as positions so far, more than the decoder window),
  dec  the decoder's own window attention (at most `window` keys): never touched.

  python -B -m horizon.attention_intervention results/Z1-pilot/perm5-core-w128-s1-cur.pt --seeds 1,2,3 --out results/Z1-pilot/perm5-core-intervention.json
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from core.model import Core

from . import depth_run as dr
from .hazard_summary import pooled_windows, rate

N_REF = 16        # the trained context length: the temperature is 1 up to here and log n / log N_REF beyond

# name -> which attention to change; `window` keeps only the last W keys, `temp` scales the logits by max(1, log n / log N_REF)
INTERVENTIONS = {
    'base':     dict(window=0, temp=False, enc=False, mem=False),
    'w16':      dict(window=16, temp=False, enc=True, mem=True),
    'w32':      dict(window=32, temp=False, enc=True, mem=True),
    'w64':      dict(window=64, temp=False, enc=True, mem=True),
    'w16_enc':  dict(window=16, temp=False, enc=True, mem=False),
    'w16_mem':  dict(window=16, temp=False, enc=False, mem=True),
    'temp':     dict(window=0, temp=True, enc=True, mem=True),
    'temp_enc': dict(window=0, temp=True, enc=True, mem=False),
    'temp_mem': dict(window=0, temp=True, enc=False, mem=True),
}


class IntervenedCore(Core):
    """The sealed Core with one inference-time change to its attention; `iv` is set by `intervene`."""

    iv = None

    def attend(self, q, k, v, causal=False):
        iv = self.iv
        if causal:
            if not (iv['enc'] and (iv['window'] or iv['temp'])):
                return super().attend(q, k, v, causal)
            t = q.shape[2]
            idx = torch.arange(t, device=q.device)
            delta = idx[:, None] - idx[None, :]
            mask = delta >= 0
            n_keys = idx + 1
            if iv['window']:
                mask = mask & (delta < iv['window'])
                n_keys = torch.clamp(n_keys, max=iv['window'])
            if iv['temp']:
                q = q * torch.clamp(torch.log(n_keys.to(q.dtype)) / math.log(N_REF), min=1.0)[None, None, :, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.width)
        n_keys = k.shape[2]
        if not iv['mem'] or n_keys <= self.window or not (iv['window'] or iv['temp']):
            return super().attend(q, k, v, causal)
        if iv['window'] and n_keys > iv['window']:
            k, v, n_keys = k[:, :, -iv['window']:], v[:, :, -iv['window']:], iv['window']
        if iv['temp']:
            q = q * max(1.0, math.log(n_keys) / math.log(N_REF))
        return super().attend(q, k, v, False)


def intervene(model, name):
    """Turn a loaded Core into an IntervenedCore with the named change (in place: same parameters)."""
    if not isinstance(model, Core):
        raise ValueError('only the Core has this attention')
    if name not in INTERVENTIONS:
        raise ValueError('unknown intervention: ' + str(name))
    model.__class__ = IntervenedCore
    model.iv = dict(INTERVENTIONS[name])
    return model


def run(weights, names, points, n, seeds, batch):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if meta['arm'] != 'core':
        raise ValueError('the arm must be core, got ' + str(meta['arm']))
    task, out = meta['task'], {}
    for name in names:
        model = intervene(dr.load_model(weights), name)
        per_seed = {s: dr.evaluate(model, task, points, n, s, batch=batch) for s in seeds}
        windows = {}
        for curve in per_seed.values():
            for key, (errors, exposure) in pooled_windows(dict(curve=curve)).items():
                cell = windows.setdefault(key, [0, 0])
                cell[0] += errors
                cell[1] += exposure
        final = {p: sum(per_seed[s][p]['final'] for s in seeds) / len(seeds) for p in points}
        out[name] = dict(config=INTERVENTIONS[name], final=final, windows={f'{a}-{b}': cell for (a, b), cell in sorted(windows.items())},
                         per_seed=per_seed)
        print(describe(name, out[name]), flush=True)
    return dict(weights=str(weights), task=task, width=meta['width'], seeds=list(seeds), n=n, points=list(points), results=out)


def describe(name, row):
    finals = ' '.join(f'{p}:{v:.2f}' for p, v in row['final'].items())
    cells = '  '.join(f"[{k.replace('-', ',')}) {c[0]}/{c[1]}={(rate(c) * 1e4 if c[1] else float('nan')):.1f}" for k, c in row['windows'].items())
    return f'{name:<9} final accuracy by length {finals} | first-error hazard (x1e-4/step) {cells}'


def baseline_reproduces(result, stored_path, points):
    """The `base` row must equal the numbers stored when the model was trained (same draws): otherwise the harness, not the model, differs."""
    stored = json.loads(Path(stored_path).read_text(encoding='utf8'))['curve']
    seeds = result['seeds']
    if result['task'] != stored.get('task', result['task']) or not result['results'].get('base'):
        return None
    model_seed = json.loads(Path(stored_path).read_text(encoding='utf8'))['seed']
    if model_seed not in seeds:
        return None
    diffs = [abs(result['results']['base']['per_seed'][model_seed][p]['final'] - stored[str(p)]['final']) for p in points if str(p) in stored]
    return max(diffs) if diffs else None


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights')
    p.add_argument('--interventions', default=','.join(INTERVENTIONS))
    p.add_argument('--seeds', default=None, help='dev-draw seeds, several independent draws are pooled (default: the model seed)')
    p.add_argument('--points', default='16,32,64,128,256,512,1024')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--batch', type=int, default=10)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    names = [v for v in a.interventions.split(',') if v]
    points = [int(v) for v in a.points.split(',')]
    meta_seed = torch.load(a.weights, weights_only=True)['meta']['seed']
    seeds = [int(v) for v in a.seeds.split(',')] if a.seeds else [meta_seed]
    result = run(a.weights, names, points, a.n, seeds, a.batch)
    stored = Path(a.weights).with_suffix('.json')
    if stored.exists() and 'base' in names:
        diff = baseline_reproduces(result, stored, points)
        result['baseline_max_abs_difference'] = diff
        print('baseline vs the numbers stored at training time: max |difference| in final accuracy', diff,
              '(OK)' if diff is not None and diff <= 1e-6 else '<-- DOES NOT REPRODUCE: do not read the other rows', flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
