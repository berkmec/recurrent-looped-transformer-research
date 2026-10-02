"""E-L1K (R5-E1 follow-up): how long does ONE wrong feedback state hurt the L1 model? Evaluation only, CPU, on the snapshots the run saves every 1,000 updates.

E-L1P replaced the fed-back state at EVERY step: that measures dependence. Here the state fed into a single decoder step `at` is replaced and nowhere else (the language analogue of the
kick E-B2 applies to the permutation Core): by the state another sequence of the batch has at that step (`other`: a state about a different text) or by zeros (`zero`). The per-position
loss of the SAME sequences is compared with the untouched run, a paired difference in bits per byte, for the steps k = 0 .. KMAX after the kick; the positions before the kick are a built-in
control (the model is causal: their difference must be 0). What the decay of that difference says:

  the state carries information that attention to the encoder memory does not restore   -> the damage persists for tens of steps (the carried state is a memory)
  the state only helps the decoder at the next few steps (previous byte, local features)  -> the damage is gone after the window-4 cache has moved on (the state is a local feature)

Reading rule (written before the first run; mode `other`, observations pooled over the kick positions; D(a-b) = mean paired difference over steps a..b after the kick, SE across sequences):
  INVALID       the control (positions before the kick) differs from 0 by more than 1e-3 bits: the kick leaked into the past.
  NO BITE       D(0) < 0.05 bits per byte: the kick did not hurt, nothing to read.
  LOCAL         D(8-15) <= 0.1 D(0) AND D(32-63) <= 0.05 D(0): the model heals within about eight steps.
  PERSISTENT    D(32-63) >= 0.3 D(0) AND D(32-63) >= 2 SE: one wrong state still costs a third of the immediate damage 32 steps later.
  INTERMEDIATE  otherwise.
`zero` is read with the same rule and reported next to it. Descriptive (no verdict): the log-decay per step between the 4-7 and the 32-63 bins.

  python -B -m mline.feedback_kick results/L1-scratch-w928-s401/snapshots/step003000.pt --out results/L1-feedback-probe
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from core.model import Core
from mline import data as md
from mline import train as mt
from mline.feedback_probe import load_model, per_position_loss

LN2 = math.log(2)
MODES = ('other', 'zero')
ATS = (63, 127, 255)
KMAX = 128
BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 129))
NO_BITE = 0.05
LOCAL_8, LOCAL_32 = 0.1, 0.05
PERSISTENT_32, PERSISTENT_SE = 0.3, 2.0
CONTROL_TOL = 1e-3


class KickedCore(Core):
    """The sealed Core whose fed-back state is replaced at ONE decoder step (`at`, counted in decoder steps = positions; the first step has no fed-back state) and nowhere else."""

    mode = None
    at = None

    def decode(self, e, s, mk, mv, t):
        if s is not None and self.mode and t == self.at:
            s = dict(s)
            s['s'] = torch.zeros_like(s['s']) if self.mode == 'zero' else s['s'].roll(1, dims=0)
        return super().decode(e, s, mk, mv, t)


def configure(model, mode=None, at=None):
    """Set the kick in place (class swap); mode None restores the trained behaviour at every step."""
    if mode is not None and mode not in MODES:
        raise ValueError('unknown mode: ' + str(mode))
    if mode is not None and (at is None or at < 1):
        raise ValueError('the kick step must be >= 1 (the first step has no fed-back state)')
    model.__class__ = KickedCore
    model.mode, model.at = mode, at
    return model


def bits(model, x, y):
    return per_position_loss(model, torch.from_numpy(x), torch.from_numpy(y)) / LN2


def kick_differences(model, pairs, at, mode, kmax=KMAX, reference=None):
    """(differences, control): `differences` = (sequences, kmax + 1) paired differences kicked - untouched in bits at the steps at .. at + kmax; `control` = the largest absolute
    difference at the positions before the kick (must be 0). `reference` = the untouched per-position bits of every batch (computed here when None; it may be longer than needed)."""
    need = at + kmax + 1
    if reference is None:
        configure(model)
        reference = [bits(model, x[:, :need], y[:, :need]) for x, y in pairs]
    configure(model, mode, at)
    rows, control = [], 0.0
    for (x, y), ref in zip(pairs, reference):
        diff = bits(model, x[:, :need], y[:, :need]) - ref[:, :need]
        control = max(control, float(diff[:, :at].abs().max()))
        rows.append(diff[:, at:need])
    configure(model)
    return torch.cat(rows).numpy(), control


def binned(differences, bins=BINS):
    """{'a-b': dict(mean, se, n)} over the step bins; one observation per sequence = its mean over the bin; the last bin is clipped to the steps available."""
    out = {}
    for lo, hi in bins:
        if lo >= differences.shape[1]:
            continue
        hi = min(hi, differences.shape[1])
        per_row = differences[:, lo:hi].mean(axis=1)
        n = int(len(per_row))
        out[f'{lo}-{hi - 1}'] = dict(mean=float(per_row.mean()), se=(float(per_row.std(ddof=1) / math.sqrt(n)) if n > 1 else None), n=n)
    return out


def decay_per_step(bins):
    """Descriptive: log-decay per step of the mean damage between the 4-7 and the 32-63 bins (None when either is not positive)."""
    a, b = bins.get('4-7'), bins.get('32-63')
    if a and b and a['mean'] > 0 and b['mean'] > 0:
        return math.log(b['mean'] / a['mean']) / (47.5 - 5.5)
    return None


def reading(bins, control=0.0):
    """The rule written before the first run (module docstring) for one mode's pooled bins."""
    if control > CONTROL_TOL:
        return dict(verdict='INVALID: the positions before the kick changed (control %.2e bits)' % control)
    if not all(k in bins for k in ('0-0', '8-15', '32-63')):
        return dict(verdict='INCOMPLETE: the bins 0-0, 8-15 and 32-63 are needed')
    d0, d8, d32, se32 = bins['0-0']['mean'], bins['8-15']['mean'], bins['32-63']['mean'], bins['32-63']['se']
    if d0 < NO_BITE:
        verdict = 'NO BITE'
    elif d8 <= LOCAL_8 * d0 and d32 <= LOCAL_32 * d0:
        verdict = 'LOCAL'
    elif d32 >= PERSISTENT_32 * d0 and se32 is not None and d32 >= PERSISTENT_SE * se32:
        verdict = 'PERSISTENT'
    else:
        verdict = 'INTERMEDIATE'
    return dict(verdict=verdict, d0=d0, d8_15=d8, d32_63=d32, se32_63=se32, ratio_8_15=(d8 / d0 if d0 > 0 else None), ratio_32_63=(d32 / d0 if d0 > 0 else None),
                decay_per_step=decay_per_step(bins))


def brief(bins):
    """One line of a bin table for the log: the immediate damage, the 8-15 bin and the last bin available (labels depend on kmax)."""
    keys = [k for k in ('0-0', '8-15') if k in bins] + [list(bins)[-1]]
    return '  '.join(f'D({k}) {bins[k]["mean"]:.3f}' for k in dict.fromkeys(keys))


def run(snapshot, directory=None, batches=8, batch=32, length=512, ats=ATS, kmax=KMAX, width=928, heads=8, log=print):
    longest = max(ats) + kmax + 1
    if longest > length:
        raise ValueError('the longest kick window (%d positions) does not fit the %d-byte sequences' % (longest, length))
    protocol = mt.protocol()
    val = md.load(directory or protocol['data'], 'validation')
    pairs = [md.batch(val, 991, step, batch, length)[:2] for step in range(1, batches + 1)]
    model = load_model(snapshot, width=width, heads=heads)
    started = time.perf_counter()
    configure(model)
    reference = [bits(model, x[:, :longest], y[:, :longest]) for x, y in pairs]
    log(f'  untouched run on {longest} positions ({time.perf_counter() - started:.0f} s)')
    out = {}
    for mode in MODES:
        per_at, rows, control = {}, [], 0.0
        for at in ats:
            t0 = time.perf_counter()
            diff, ctl = kick_differences(model, pairs, at, mode, kmax, reference)
            rows.append(diff)
            control = max(control, ctl)
            per_at[str(at)] = dict(bins=binned(diff), mean_by_step=[float(v) for v in diff.mean(axis=0)], control=ctl)
            log(f'  {mode:<6} kick at step {at:<4} ' + brief(per_at[str(at)]['bins']) + f' bits  (control {ctl:.1e}; {time.perf_counter() - t0:.0f} s)')
        pooled = binned(np.concatenate(rows))
        out[mode] = dict(per_at=per_at, pooled=pooled, control=control, reading=reading(pooled, control))
        log(f'  {mode:<6} pooled: ' + '  '.join(f'[{k}] {v["mean"]:.3f}' for k, v in pooled.items()))
        log(f'  {mode:<6} reading: ' + json.dumps(out[mode]['reading'], default=float))
    return dict(snapshot=str(snapshot), batches=batches, batch=batch, length=length, ats=list(ats), kmax=kmax, modes=out)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('snapshots', nargs='+')
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--ats', default=','.join(str(a) for a in ATS))
    p.add_argument('--kmax', type=int, default=KMAX)
    p.add_argument('--out', default=None, help='directory for <snapshot>-kick.json')
    a = p.parse_args()
    for path in a.snapshots:
        print(path, flush=True)
        r = run(path, batches=a.batches, ats=tuple(int(v) for v in a.ats.split(',')), kmax=a.kmax, log=lambda m: print(m, flush=True))
        if a.out:
            target = Path(a.out) / (Path(path).stem + '-kick.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(r, indent=1, default=float), encoding='utf8')


if __name__ == '__main__':
    main()
