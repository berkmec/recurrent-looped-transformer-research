"""E-L1X-F (program note 5.66): does the trained L1's carried state bring context from beyond what the attention window shows, in the configuration the model is run in
(encoder and cross-attention restricted to the last 512 positions, E-L1X's free fix), and does that reach grow over training? Evaluation only, CPU, on validation articles
(never trained on). E-L1X's module (mline.long_doc) and its registered reading are not changed; this imports them.

E-L1X measured one early snapshot (6,000 of 30,518 updates): the state carried nothing past 256 steps. Its conditions do not isolate the state in the windowed
configuration either: win512 against chunk512 changes both the state (carried or restarted) and the attention context (the last 512 positions, or the 256-511 of a
chunk). This adds the missing arm, on the same bytes:

  win512       encoder and cross-attention restricted to the last 512 positions, the state carried over the whole text (E-L1X's condition)
  winreset512  the same restriction, the carried state replaced by the learned initial state every 512 steps (E-L1X's reset, t > 0 only)
  chunk512     independent 512-byte chunks (E-L1X's reference: the control, and the descriptive G)

The attention paths are identical in win512 and winreset512; only what the state remembers differs. At a position p with offset o = p mod 512 >= 256 the reset state
holds the last o steps, the carried state the whole text, so S = loss_winreset512 - loss_win512 is what the state's memory of the text before the last o steps is worth
on top of everything the 512-position window shows. For o near 511 that memory is almost all beyond the window; the offset bands below show how S depends on o.

Metric: per article, the mean per-byte loss (bits) over the positions of a bin with offset >= 256 (E-L1X's bins and offset rule); S(bin) the mean over articles of the
paired difference, standard error across articles. Bins starting at or above 1,024 carry the verdict ([1024,2048) and [2048,4096) in the 4,096-byte run).

Reading rule (written before the first run, program note 5.66; thresholds inclusive with E-L1X's 1e-9 tolerance):
  CONTROL FAILED         the three conditions differ on [0,512) by more than 1e-3 bits at any position (max over articles and positions)
  STATE EXTENDS CONTEXT  some bin >= 1,024 has S >= 0.010 bits and S >= 3 SE
  STATE HURTS            otherwise, some bin >= 1,024 has S <= -0.010 bits and S <= -3 SE
  NO STATE CONTEXT       otherwise, |S| <= 0.003 bits in every bin >= 1,024
  SMALL                  anything else
The verdict is the one of the final weights (30,518 updates, latest.pt); the earlier snapshots give the curve. Descriptive: G(bin) = loss_chunk512 - loss_win512 (the
windowed configuration against independent chunks: state and extra attention context together); S by offset band, pooled over the bins >= 1,024; the state-difference
profile (relative distance between the carried and the reset state, k steps after the reset at 512, on the first articles).

  python -B -m mline.state_reach results/L1-scratch-w928-s401/latest.pt --out results/L1-long-doc
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from core.model import Core
from mline import lm as ml
from mline import long_doc as ld

CONDITIONS = ('win512', 'winreset512', 'chunk512')
S_BITS, S_SE, NONE_BITS = 0.010, 3.0, 0.003
BANDS = ((0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0))        # offset bands as fractions of the training length: [256,320) ... [448,512)
PROFILE_DOCS = 4
PROFILE_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 511)


def configure(model, condition):
    """Set the condition in place on top of E-L1X's class swap; `chunk512` leaves the computation as trained."""
    if condition not in CONDITIONS:
        raise ValueError('unknown condition: ' + str(condition))
    if condition == 'winreset512':
        ld.configure(model, 'win512')
        model.reset_every = ld.TRAIN_LENGTH
    else:
        ld.configure(model, condition)
    return model


def load(path, width=928, heads=8, seed=401):
    """(model, step) from a snapshot (a plain state dict, step from the file name) or the run's checkpoint (latest.pt: {'step', 'model', ...})."""
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:                                   # the run's own checkpoint carries non-tensor metadata
        obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'model' in obj and 'step' in obj:
        state, step = obj['model'], int(obj['step'])
    else:
        match = re.search(r'step(\d+)', Path(path).stem)
        state, step = obj, int(match.group(1)) if match else None
    model = ml.build('plain', width=width, heads=heads, window=4, alpha=0.1, seed=seed)
    model.load_state_dict(state)
    return model.eval(), step


def band_means(diff, steps):
    """{band: (n,) per-article mean of diff over the positions at or above 2 x the training length whose offset lies in the band}."""
    positions = np.arange(steps)
    offset, high = positions % ld.TRAIN_LENGTH, positions >= 2 * ld.TRAIN_LENGTH
    out = {}
    for lo, hi in BANDS:
        a, b = int(round(lo * ld.TRAIN_LENGTH)), int(round(hi * ld.TRAIN_LENGTH))
        keep = high & (offset >= a) & (offset < b)
        if keep.any():
            out[f'{a}-{b}'] = diff[:, keep].mean(1)
    return out


def state_profile(model, x):
    """{k: mean over the articles of |s_carried - s_reset| / |s_carried| for the state returned k steps after the reset at the training length} (windowed attention)."""
    first = ld.TRAIN_LENGTH
    wanted = {first + k - 1: k for k in PROFILE_STEPS if k < ld.TRAIN_LENGTH and first + k - 1 < x.shape[1]}
    states = {}
    original = Core.decode

    def record(self_, e, s, mk, mv, t):
        out = original(self_, e, s, mk, mv, t)
        if t in wanted:
            states[(record.condition, t)] = out[1]['s'].detach().clone()
        return out

    try:
        Core.decode = record
        with torch.no_grad():
            for condition in ('win512', 'winreset512'):
                record.condition = condition
                configure(model, condition)
                model(torch.from_numpy(x))
    finally:
        Core.decode = original
    profile = {}
    for t, k in sorted(wanted.items()):
        carried, reset = states[('win512', t)], states[('winreset512', t)]
        profile[str(k)] = float(((carried - reset).norm(dim=-1) / carried.norm(dim=-1).clamp_min(1e-30)).mean())
    return profile


def run(path, docs=None, n=ld.DOCS, length=ld.LENGTH, batch=8, log=print, model=None, step=None, profile_docs=PROFILE_DOCS):
    data = ld.long_documents(n, length) if docs is None else docs
    x, y = data[:, :-1], data[:, 1:]
    steps = x.shape[1]
    if model is None:
        model, step = load(path)
    losses = {}
    for condition in CONDITIONS:
        started = time.perf_counter()
        configure(model, condition)
        losses[condition] = ld.chunk_bits(model, x, y) if condition == 'chunk512' else ld.bits(model, x, y, batch)
        log(f'  {condition:<11} mean {losses[condition].mean():.4f} bits/byte on {x.shape[0]} x {steps} bytes  ({time.perf_counter() - started:.0f} s)')
    first = slice(0, min(ld.TRAIN_LENGTH, steps))
    control = max(float(np.abs(losses[c][:, first] - losses['chunk512'][:, first]).max()) for c in CONDITIONS)
    per_doc = {c: {f'{b[0]}-{b[1]}': v.tolist() for b, v in ld.bin_means(losses[c], steps).items()} for c in CONDITIONS}
    bands = {b: v.tolist() for b, v in band_means(losses['winreset512'] - losses['win512'], steps).items()}
    profile = state_profile(model, x[:profile_docs]) if profile_docs and steps > ld.TRAIN_LENGTH else None
    ld.configure(model, 'full')
    result = dict(snapshot=str(path), step=step, n=int(x.shape[0]), length=int(steps), control=control, per_doc=per_doc, s_bands=bands, state_profile=profile)
    result['reading'] = reading(result)
    return result


def differences(result):
    """(S, G): {bin: (mean, SE)} of winreset512 - win512 and of chunk512 - win512, recomputed from the stored per-article values."""
    ref = result['per_doc']['win512']
    s = {b: ld.paired(np.array(result['per_doc']['winreset512'][b]), np.array(v)) for b, v in ref.items()}
    g = {b: ld.paired(np.array(result['per_doc']['chunk512'][b]), np.array(v)) for b, v in ref.items()}
    return s, g


def reading(result):
    if result['control'] > ld.CONTROL_TOL:
        return dict(verdict='CONTROL FAILED', control=result['control'])
    s, _ = differences(result)
    high = [b for b in s if int(b.split('-')[0]) >= 2 * ld.TRAIN_LENGTH]
    if not high:
        return dict(verdict='INCOMPLETE', reason='no bin at or above 2 x the training length')
    tol = ld.TOL
    if any(s[b][0] >= S_BITS - tol and s[b][0] >= S_SE * s[b][1] - tol for b in high):
        verdict = 'STATE EXTENDS CONTEXT'
    elif any(s[b][0] <= -S_BITS + tol and s[b][0] <= -S_SE * s[b][1] + tol for b in high):
        verdict = 'STATE HURTS'
    elif all(abs(s[b][0]) <= NONE_BITS + tol for b in high):
        verdict = 'NO STATE CONTEXT'
    else:
        verdict = 'SMALL'
    return dict(verdict=verdict, s={b: list(s[b]) for b in high})


def describe(result):
    s, g = differences(result)
    lines = [f'{result["snapshot"]} (step {result["step"]}): {result["n"]} articles x {result["length"]} bytes, control {result["control"]:.1e}']
    lines.append('  S = winreset512 - win512  ' + '  '.join(f'[{b}] {m:+.4f}±{e:.4f}' for b, (m, e) in s.items()))
    lines.append('  G = chunk512 - win512     ' + '  '.join(f'[{b}] {m:+.4f}±{e:.4f}' for b, (m, e) in g.items()))
    lines.append('  S by offset band (bins >= 2 x training length): ' + '  '.join(
        f'[{b}] {np.mean(v):+.4f}' for b, v in result['s_bands'].items()))
    if result['state_profile']:
        lines.append('  state difference k steps after the reset: ' + '  '.join(f'{k}: {v:.2e}' for k, v in result['state_profile'].items()))
    lines.append('  reading: ' + json.dumps(result['reading'], default=float))
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', help='snapshots (stepNNNNNN.pt) or the run checkpoint (latest.pt)')
    p.add_argument('--n', type=int, default=ld.DOCS)
    p.add_argument('--length', type=int, default=ld.LENGTH)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--profile-docs', type=int, default=PROFILE_DOCS)
    p.add_argument('--out', default=None, help='directory for stepNNNNNN-state-reach[-length].json')
    a = p.parse_args()
    for path in a.paths:
        print(path, flush=True)
        model, step = load(path)
        r = run(path, n=a.n, length=a.length, batch=a.batch, log=lambda m: print(m, flush=True), model=model, step=step, profile_docs=a.profile_docs)
        print(describe(r), flush=True)
        if a.out:
            name = ('step%06d' % step if step is not None else Path(path).stem) + '-state-reach' + ('' if a.length == ld.LENGTH else '-' + str(a.length))
            target = Path(a.out) / (name + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(json.dumps(r, indent=1, default=float).encode('utf8'))


if __name__ == '__main__':
    main()
