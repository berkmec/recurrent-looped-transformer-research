"""E-G: chrono-initialized HOLD register + neutral-interval S4/S5 (R-3 step 1, user-approved).

Gate (M-1: identity path): per-dimension WRITE logits over [e_t, s_{t-1}]:
  s_t = g * cand + (1 - g) * s_{t-1}   (soft; NOT exact)
  hard_st: forward uses the hard decision (exact bitwise copy on HOLD),
  backward passes the sigmoid gradient (straight-through), as horizon.gated.
Chrono init (Tallec & Ollivier 2018, single-gate form, R2 §3.1/§5-E3): WRITE bias
  b_i = -ln(u_i - 1), u_i ~ U[2, T_MAX]  ->  timescales 1..T_MAX-1.
T_MAX = 16 default (longest curriculum / neutral span of interest).
R-2 addenda: gate-SIGN telemetry (E-A10 lesson: a gate can invert) and a
decay-free gate parameter group (F12: AdamW x0.47 pulls the gate closed).
`hold_nc` = same vector head with the WRITE prior (bias 2.0, no chrono):
the M-1 prediction-(3) negative control (a gateless-chrono gate must not close).

  python -B -m horizon.depth_run --task perm4 --arm hold --width 64 --seed 1 --steps 15000 --curriculum 1,2,3,4,6,8,12,16 --neutral-p 0.5 --out results/Z1-pilot/eg-<id>.json
  python -B -m horizon.hold results/Z1-pilot/eg-<id>.pt --out results/Z1-pilot/eg-<id>-reading.json
  python -B -m horizon.hold --reading results/Z1-pilot/eg-<id>-reading.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from horizon import gated as gd

T_MAX_DEFAULT = 16
NEUTRAL_PIDS = (0.0, 0.5, 0.9)


def chrono_biases(n, t_max, rng):
    """Per-unit WRITE biases b_i = -ln(u_i - 1), u_i ~ U[2, t_max] (timescales 1..t_max-1)."""
    if t_max < 3:
        raise ValueError('t_max must be >= 3')
    u = rng.uniform(2.0, float(t_max), size=n)
    return (-np.log(u - 1.0)).astype(np.float64)


class HoldCore(gd.GatedCore):
    """Vector HOLD/WRITE gate over the sealed Core. forward returns LOGITS only
    (depth_run.train compatible); gates are stashed on self.last_gates."""

    def __init__(self, core, mode='hard_st', t_max=T_MAX_DEFAULT, chrono=True, seed=0):
        nn.Module.__init__(self)
        if mode not in ('soft', 'hard_st'):
            raise ValueError('HoldCore mode must be soft or hard_st')
        self.core = core
        self.mode = mode
        self.t_max = int(t_max)
        self.chrono = bool(chrono)
        width = core.initial.shape[-1]
        self.head = nn.Linear(2 * width, width)
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            self.head.weight.normal_(0., 0.02)
            if chrono:
                self.head.bias.copy_(torch.from_numpy(chrono_biases(width, self.t_max, rng)).float())
            else:
                self.head.bias.fill_(2.)  # WRITE prior: starts close to the plain Core
        self.last_gates = None
        self.last_glogits = None

    def gate_parameters(self):
        return self.head.parameters()

    def forward(self, x):
        core = self.core
        e, _, _ = core.encode(x)
        n = core.mn(e)
        mk = core.rope(core.split(core.mk(n)))
        mv = core.split(core.mv(n))
        B, T = x.shape
        committed = core.initial.expand(B, -1)
        state = None
        outs, gates, glogits = [], [], []
        for t in range(T):
            z, cand = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
            gl = self.head(torch.cat([e[:, t], committed], -1))
            gs = torch.sigmoid(gl)
            if self.mode == 'soft':
                s = gs * cand['s'] + (1 - gs) * committed
            else:
                m = (gs.detach() > 0.5).to(gs.dtype)
                s = torch.where(m.bool(), cand['s'], committed) + (gs - gs.detach()) * (cand['s'] - committed)
            state = dict(cand, s=s)
            committed = s
            outs.append(core.read(core.outn(s)))
            gates.append(gs.detach())
            glogits.append(gl.detach())
        self.last_gates = torch.stack(gates, 1)
        self.last_glogits = torch.stack(glogits, 1)
        return torch.stack(outs, 1)


def hold_stats(gates, neutral_mask=None):
    """gates: (B, T, W) WRITE probabilities in [0,1]. hold = 1 - gate."""
    g = gates.detach().float()
    hold = 1 - g
    out = dict(hold_mean=float(hold.mean()), hold_per_dim=hold.mean((0, 1)).tolist(),
               tau_per_dim=(1 / g.mean((0, 1)).clamp_min(1e-6)).tolist())
    if neutral_mask is not None:
        m = neutral_mask.bool()
        out['hold_neutral'] = float(hold[m].mean()) if bool(m.any()) else None
        out['hold_informative'] = float(hold[~m].mean()) if bool((~m).any()) else None
        # sign telemetry (R-2a, E-A10 lesson): the gate must WRITE informative steps
        # and HOLD neutral ones; an inverted gate is a failed manipulation.
        out['sign_ok'] = (out['hold_neutral'] is not None and out['hold_informative'] is not None
                          and out['hold_neutral'] > out['hold_informative'])
    return out


def reading(result):
    """Registered E-G rule (M-1): hold>=0.5 AND neutral-loss<=5 -> DEST EKLENIR...;
    refutation hold<0.1 or loss>20; manipulation check hold<0.5 (A§11-4)."""
    for key in ('hold_neutral', 'neutral_loss_points'):
        if result.get(key) is None:
            return dict(verdict='INCOMPLETE', reason=f'missing {key}')
    hold, loss = result['hold_neutral'], result['neutral_loss_points']
    if hold >= 0.5 and loss <= 5:
        verdict = 'DESTEKLENİR'
    elif hold < 0.1 or loss > 20:
        verdict = 'ÇÜRÜTÜLDÜ'
    elif hold < 0.5:
        verdict = 'MANIPULATION_NOT_APPLIED'
    else:
        verdict = 'KISMEN'
    return dict(verdict=verdict, hold_neutral=hold, neutral_loss_points=loss,
                sign_ok=result.get('sign_ok'))


def describe(r):
    return (f"{r['weights']}: hold_neutral={r['hold_neutral']:.3f} "
            f"neutral_loss={r['neutral_loss_points']:.1f}pts sign_ok={r['sign_ok']} "
            f"-> {r['reading']['verdict']}")


def measure(weights, p_ids=NEUTRAL_PIDS, length=1024, n=200, seed=1, batch=50, log=print):
    from . import depth_run as dr
    from . import depth_tasks as dt
    from . import neutral_runs as nr
    model = dr.load_model(weights)
    if not isinstance(model, HoldCore):
        raise ValueError('E-G measure needs a hold/hold_nc checkpoint')
    meta = torch.load(weights, weights_only=True)['meta']
    task = meta['task']
    base = nr.measure(weights, p_ids=(0.0,), length=length, n=n, seed=seed, batch=batch)
    acc0 = base['results'][0.0]['final']
    heavy = nr.measure(weights, p_ids=(0.9,), length=length, n=n, seed=seed, batch=batch)
    acc9 = heavy['results'][0.9]['final']
    nloss = 100.0 * (acc0 - acc9)
    model.eval()
    x = nr.neutral_tokens(task, n, length, 0.9, seed)
    with torch.no_grad():
        for at in range(0, len(x), batch):
            model(x[at:at + batch])
            break
    g0 = model.last_gates.shape[0]
    stats = hold_stats(model.last_gates, neutral_mask=(x[:g0] == 0))
    result = dict(weights=str(weights), task=task, arm=meta['arm'], width=meta['width'], seed=meta['seed'],
                  hold_mean=stats['hold_mean'], hold_neutral=stats['hold_neutral'],
                  hold_informative=stats['hold_informative'], tau_per_dim=stats['tau_per_dim'],
                  sign_ok=stats['sign_ok'], neutral_loss_points=nloss,
                  hold_tmax=meta.get('hold_tmax', T_MAX_DEFAULT))
    result['reading'] = reading(result)
    log(describe(result))
    return result


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdict')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            r['reading'] = reading(r)
            print(describe(r))
        return
    results = []
    for path in a.weights:
        r = measure(path, log=lambda m: print(m, flush=True))
        results.append(r)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')
    if a.out:
        print(json.dumps([r['reading'] for r in results], indent=1))


if __name__ == '__main__':
    main()
