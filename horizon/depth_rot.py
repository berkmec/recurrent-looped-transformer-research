"""Rotary counter for the Core (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md sections 4.4 and 5.63).

The Core's carried state is RMS-normed, so it can hold a count only as a direction (4.4). This closable channel turns a bank of K
angles by a learned step per token: a push and a pop that take opposite steps cancel and the norm never changes. With `quantize`
the step is rounded to -1, 0 or +1 in the forward pass (straight-through gradient of tanh), so the cancellation is exact by
construction rather than learned. The slowest frequency has a period of 2^(K+1) tokens, so its angle stays monotone over every
tested depth and a threshold read from it can extend past the training depth; the faster ones resolve small counts.

    theta_t = omega * cumsum_{s <= t} step(x_s),   step = tanh(c[x])  (or its rounding),   omega_k = 2 pi / 2^(k + 2)
    e'_t    = e_t + gamma * W [cos theta_t, sin theta_t]        (gamma = 0 closes the channel exactly: K3)

The angles are a prefix sum (parallel over time), the decoder is the Core's own (feedback on: K1), the state added is K angles (K4).
Only new modules; core/model.py and horizon/depth_models.py are used as they are. Train like horizon.depth_run:

    python -B -m horizon.depth_rot --task count8_32 --arm core_rotq --width 64 --seed 1 --steps 12000 --out ...
"""
import math
import sys

import torch
from torch import nn

from mline import rltm
from . import depth_models as dm
from . import depth_run as dr
from . import depth_tasks as dt
from . import readings as rd

ROT_ARMS = ('core_rot', 'core_rotq')
N_FREQ = 10


class RotaryCounterCore(nn.Module):
    def __init__(self, core, vocab, n_freq=N_FREQ, quantize=False, gamma_init=1.0, seed=0):
        super().__init__()
        self.core, self.quantize = core, quantize
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.step = nn.Embedding(vocab, n_freq)
            self.read = nn.Linear(2 * n_freq, core.width)
        self.register_buffer('omega', 2 * math.pi / 2.0 ** torch.arange(2, n_freq + 2, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((core.width,), float(gamma_init)))

    def steps(self, x):
        s = torch.tanh(self.step(x))
        return s + (torch.round(s) - s).detach() if self.quantize else s

    def angles(self, x):
        return torch.cumsum(self.steps(x), dim=1) * self.omega

    def forward(self, x):
        e, _, _ = self.core.encode(x)
        theta = self.angles(x)
        e = e + self.gamma * self.read(torch.cat([theta.cos(), theta.sin()], -1))
        return rltm.decode_windows(self.core, e)


def make_arm(arm, task, width, seed, **kw):
    """horizon.depth_models.make_arm plus the two rotary arms (each wraps the plain `core` of the same seed)."""
    if arm not in ROT_ARMS:
        return dm.make_arm(arm, task, width, seed, **kw)
    core = dm.make_arm('core', task, width, seed, **{k: v for k, v in kw.items() if k in ('window', 'alpha')})
    return RotaryCounterCore(core, dt.vocab(task), quantize=(arm == 'core_rotq'), seed=seed + 11)


def reading(paths, r_max=32):
    """Note 5.63: E-C's measures (horizon.readings.ec_run) for the plain Core and the two rotary arms, and the registered verdict."""
    runs = {}
    for p in paths:
        arm = str(p).replace('\\', '/').split('/')[-1].split('-')[1]
        runs[arm] = rd.ec_run(p, r_max, lstm=False)
    valid = {a: r for a, r in runs.items() if r['valid']}
    base = valid['core']['rho'] if 'core' in valid else 0.0       # a plain Core that never learned the range extrapolates nothing
    rot = {a: valid[a]['rho'] for a in ROT_ARMS if a in valid}
    if not rot:
        verdict = 'VACUOUS (no rotary arm learned the range)' if any(a in runs for a in ROT_ARMS) else 'INCOMPLETE'
    elif 'core' not in runs:
        verdict = 'INCOMPLETE'
    elif max(rot.values()) >= 4 and max(rot.values()) >= 2 * base:
        verdict = 'ROTARY EXTRAPOLATES: ' + ', '.join(a for a, v in rot.items() if v >= 4 and v >= 2 * base)
        if 'core' not in valid:
            verdict += ' (the plain Core did not learn the range)'
    elif max(rot.values()) <= base:
        verdict = 'NO EXTRAPOLATION GAIN'
    else:
        verdict = 'PARTIAL'
    return dict(verdict=verdict, runs=runs)


def main(argv=None):
    dr.make_arm = make_arm            # depth_run builds (and reloads) its arms through this name
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--reading' in argv:
        import json
        print(json.dumps(reading(argv[argv.index('--reading') + 1:]), indent=1, default=str))
        return
    sys.argv = [sys.argv[0]] + argv
    dr.main()


if __name__ == '__main__':
    main()
