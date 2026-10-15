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


HOLD_MIN, LOSS_SUPPORT, LOSS_REFUTE = 0.5, 5.0, 20.0   # registered (319f13c), unchanged by c8c64fd
ARM_SEEDS = {'perm4': 3, 'perm5': 2}                    # Z6: registered seeds per arm
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% binomial (Wilson) interval of k/n, descriptive only (Z10)."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def loss_ci(acc0, acc9, n0, n9, z=Z95):
    """95% CI of 100*(acc0 - acc9), two independent binomials (different token draws per p_id), normal approx (Z10)."""
    if not n0 or not n9:
        return None
    se = math.sqrt(acc0 * (1 - acc0) / n0 + acc9 * (1 - acc9) / n9)
    d = acc0 - acc9
    return [100.0 * (d - z * se), 100.0 * (d + z * se)]


def reading(result):
    """Registered E-G seed rule, c8c64fd addendum (Z4/Z5/Z7/Z10), thresholds of 319f13c unchanged.
    Z4 branch order, first match wins:
      (i)   hold >= 0.5 AND loss <= 5 AND sign_ok          -> DESTEKLENİR   (Z5: an inverted gate is not support)
      (ii)  loss > 20                                       -> ÇÜRÜTÜLDÜ
      (iii) hold < 0.5 (hold < 0.1 included)                -> MANIPULATION_NOT_APPLIED (not tested != refuted)
      (iv)  otherwise                                       -> KISMEN
      a missing measurement (hold_neutral, loss or sign_ok) -> INCOMPLETE.
    Z7 (arm 'hold_nc', the negative control): hold < 0.5 or loss > 20 -> MANIPULATION_NOT_APPLIED (loss descriptive);
      hold >= 0.5 AND loss <= 5 -> KISMEN + CHRONO_UNNECESSARY (never support); otherwise KISMEN.
    Z10: when acc0/acc9 and their n are stored, the 95% CIs are written; a loss CI containing 5 or 20 adds SINIRDA.
      The label is the computed one either way (single reading, no re-read).
    Q3 (Meredith, nloss < 0): loss is 100*(acc(0) - acc(0.9)) as registered, without a floor; a negative loss
      (the model is MORE accurate under heavy neutral padding, fewer real updates to compose) is <= 5 and passes the
      loss conjunct. Clamping would be a rule change; the sign is visible in the stored value and its CI."""
    for key in ('hold_neutral', 'neutral_loss_points', 'sign_ok'):
        if result.get(key) is None:
            return dict(verdict='INCOMPLETE', reason=f'missing {key}', flags=[])
    hold, loss, sign_ok = result['hold_neutral'], result['neutral_loss_points'], bool(result['sign_ok'])
    flags = []
    if result.get('arm') == 'hold_nc':
        if hold < HOLD_MIN or loss > LOSS_REFUTE:
            verdict = 'MANIPULATION_NOT_APPLIED'
        else:
            verdict = 'KISMEN'
            if loss <= LOSS_SUPPORT:
                flags.append('CHRONO_UNNECESSARY')
    elif hold >= HOLD_MIN and loss <= LOSS_SUPPORT and sign_ok:
        verdict = 'DESTEKLENİR'
    elif loss > LOSS_REFUTE:
        verdict = 'ÇÜRÜTÜLDÜ'
    elif hold < HOLD_MIN:
        verdict = 'MANIPULATION_NOT_APPLIED'
    else:
        verdict = 'KISMEN'
    out = dict(verdict=verdict, hold_neutral=hold, neutral_loss_points=loss, sign_ok=sign_ok, flags=flags)
    if all(result.get(k) is not None for k in ('acc0', 'acc9', 'n_loss')):
        n = result['n_loss']
        out['acc0_ci'] = wilson(round(result['acc0'] * n), n)
        out['acc9_ci'] = wilson(round(result['acc9'] * n), n)
        out['loss_ci'] = loss_ci(result['acc0'], result['acc9'], n, n)
        lo, hi = out['loss_ci']
        if any(lo <= t <= hi for t in (LOSS_SUPPORT, LOSS_REFUTE)):
            flags.append('SINIRDA')
    return out


def arm_reading(seed_results, task):
    """Z6 arm verdict from the per-seed `reading` labels (hold arm only; hold_nc is never combined).
    perm4 (3 seeds): >= 2 equal labels -> that label, 1-1-1 -> KISMEN; perm5 (2 seeds): 2/2 -> that label, 1-1 -> KISMEN.
    A ÇÜRÜTÜLDÜ seed does not veto but is named, and blocks an arm DESTEKLENİR (-> KISMEN).
    Fewer/more seeds than registered, or a majority of INCOMPLETE seeds -> INCOMPLETE."""
    if task not in ARM_SEEDS:
        raise ValueError(f'Z6 combines perm4/perm5 arms only, not {task}')
    if any(r.get('arm') == 'hold_nc' for r in seed_results):
        raise ValueError('Z6: hold_nc does not enter the combination (Z7 reads it alone)')
    labels = [reading(r)['verdict'] for r in seed_results]
    refuted = [r.get('seed') for r, v in zip(seed_results, labels) if v == 'ÇÜRÜTÜLDÜ']
    out = dict(task=task, seeds=[r.get('seed') for r in seed_results], seed_verdicts=labels, refuted_seeds=refuted)
    if len(labels) != ARM_SEEDS[task]:
        return dict(out, verdict='INCOMPLETE', reason=f'{len(labels)} seeds, registered {ARM_SEEDS[task]}')
    top = max(set(labels), key=labels.count)
    verdict = top if labels.count(top) * 2 > len(labels) else 'KISMEN'
    if verdict == 'DESTEKLENİR' and refuted:
        verdict = 'KISMEN'
    return dict(out, verdict=verdict)


def eg_reading(arm_verdicts):
    """Z6 E-G overall from the two arm verdicts {'perm4': ..., 'perm5': ...}: both DESTEKLENİR -> DESTEKLENİR;
    both ÇÜRÜTÜLDÜ -> ÇÜRÜTÜLDÜ; exactly one ÇÜRÜTÜLDÜ -> KISMEN (scale split, reported); other cases KISMEN.
    An INCOMPLETE or missing arm -> INCOMPLETE (missing measurement, Z4 last line)."""
    v = [arm_verdicts.get(t) for t in ARM_SEEDS]
    if any(x in (None, 'INCOMPLETE') for x in v):
        return dict(verdict='INCOMPLETE', arms=dict(arm_verdicts))
    if all(x == 'DESTEKLENİR' for x in v):
        verdict = 'DESTEKLENİR'
    elif all(x == 'ÇÜRÜTÜLDÜ' for x in v):
        verdict = 'ÇÜRÜTÜLDÜ'
    else:
        verdict = 'KISMEN'
    return dict(verdict=verdict, arms=dict(arm_verdicts), scale_split=v.count('ÇÜRÜTÜLDÜ') == 1)


def describe(r):
    flags = r['reading'].get('flags') or []
    return (f"{r['weights']}: hold_neutral={r['hold_neutral']:.3f} "
            f"neutral_loss={r['neutral_loss_points']:.1f}pts sign_ok={r['sign_ok']} "
            f"-> {r['reading']['verdict']}" + (f" [{','.join(flags)}]" if flags else ''))


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
    nloss = 100.0 * (acc0 - acc9)   # Q3: no floor, may be negative -> see reading()
    model.eval()
    x = nr.neutral_tokens(task, n, length, 0.9, seed)
    # Q1 (Meredith, the old loop + `break`): gate telemetry is the FIRST batch only, by registration (Z1:
    # hold_neutral/hold_informative/sign_ok on the first 50 sequences; the loss above uses all n). Written as one
    # slice so the scope is explicit and persisted as n_gate; last_gates holds exactly this batch.
    xg = x[:batch]
    with torch.no_grad():
        model(xg)
    # Q2 (Meredith, the `x == 0` mask): token 0 is the identity (depth_tasks: perms[0]); neutral_tokens draws it with
    # probability p_id, and a uniform draw may also land on 0 - that step is an identity update too, so it is
    # neutral by definition, not mislabelled. The mask is (B, T) over the same rows as gates (B, T, W).
    assert model.last_gates.shape[:2] == xg.shape
    stats = hold_stats(model.last_gates, neutral_mask=(xg == 0))
    # Q4 persist rule: acc0/acc9 with their n, the evaluation seed and length are stored with every result, so the
    # Z10 CIs are recomputable by --reading; eval_seed (data) is distinct from seed (model). An existing --out file is
    # never overwritten (single reading, Z10) - see main().
    result = dict(weights=str(weights), task=task, arm=meta['arm'], width=meta['width'], seed=meta['seed'],
                  hold_mean=stats['hold_mean'], hold_neutral=stats['hold_neutral'],
                  hold_informative=stats['hold_informative'], tau_per_dim=stats['tau_per_dim'],
                  sign_ok=stats['sign_ok'], neutral_loss_points=nloss,
                  acc0=acc0, acc9=acc9, n_loss=n, n_gate=int(xg.shape[0]), length=length, eval_seed=seed,
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
    if a.out and Path(a.out).exists():
        raise SystemExit(f'{a.out} exists: a reading is written once (Z10), choose a new --out')
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
    # `-m horizon.hold` runs this file as __main__ while depth_models imports horizon.hold, giving two HoldCore classes
    # and a false isinstance failure in measure (Meredith D-2); run the package copy instead.
    from horizon.hold import main as _main
    _main()
