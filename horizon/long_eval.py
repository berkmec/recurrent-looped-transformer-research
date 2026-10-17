"""E-X: how far does exactness extend past the trained length (16)? Evaluation only, no training.

The Z1 permutation pilots were evaluated up to 1,024 positions (64x the trained length). A per-step first-error hazard of h means a sequence of length T survives with
probability about exp(-h T): 0 errors in 203,200 sequence-steps only bounds h below ~1.5e-5, which still allows ~30 % failures at T = 32,768. This evaluates saved S4 / S5
arms on fresh dev sequences of 2,048 ... 32,768 positions (up to 2,048x the trained length) and reports the first-error hazard per window beyond 1,024, pooled over
the evaluated lengths, plus the exactness horizon (the longest evaluated length at which EVERY sequence is error-free).

  python -B -m horizon.long_eval results/Z1-pilot/perm4-gru-w64-s1-cur.pt --lengths 2048,8192,32768 --n 100 --out results/Z1-pilot/long-eval.json

Reading rule (program note 5.13, written before the first run; the BREAK test was REPLACED on 2026-10-02 after the second blind review, note 5.25 / finding R2-1, before the
parameter-matched E-P5 evaluation ran), applied to the windows beyond 1,024 positions, per model:
  CLEAN   no first error in any window beyond 1,024 and an exposure of at least 1e6 sequence-steps (so the hazard is below ~3e-6 per step, 95 %);
  BREAK   some later window with >= 5 first errors has a hazard >= 10x that of the FIRST window beyond 1,024 that has any exposure (the reference is max(its hazard, 3 / its person-steps):
          a rule-of-three bound replaces a rate of 0 or of one or two events) and an exact conditional-binomial p < 0.01 that it is not higher than the first window's; the hazard is
          events / person-steps (the steps actually spent at risk);
  FLOOR   >= 5 first errors beyond 1,024 and no BREAK: a roughly constant per-step floor;
  SPARSE  1-4 first errors beyond 1,024 (a rate bound only);
  SHORT   no first error but less than 1e6 sequence-steps of exposure (not enough to call it clean).
The original BREAK test compared only the LAST window with events with the first window (events / at-risk-sequences x width, a rate capped at 1 / width), which could read constant
floors as BREAK (a zero first window) and steep breaks as FLOOR (the last window with events held < 5). Results stored before this change carry no `person_steps`: their exposure is the
midpoint approximation (at-risk steps - events x width / 2); on the seven stored E-X results the old and new labels agree.

Z6 (I3-d eval zeyli; seal Rev.2, 00:45Z): `--reset-every m` is an opt-in extension — at the decoder steps t > 0 with t % m == 0 the carried state s['s'] is replaced by the learned
initial state (the E-L1X `reset512` semantics of `mline/long_doc.py`; the decoder's window cache dk/dv and the cross-attention memory are untouched). Off (0) the file behaves exactly
as before, bit for bit; the result JSON carries `reset_every` only when it is on. Defined for the recurrent Core arms (feedback=True); evidence: tests/test_long_eval_reset.py +
results/faz-z6/ (off-path bit-for-bit runs against the pre-zeyil file).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from core.model import Core
from . import depth_run as dr
from . import depth_tasks as dt
from .readings import rate_ratio_p

EDGES = [0, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BEYOND = 1024
CLEAN_EXPOSURE = 1_000_000
BREAK_RATIO = 10
BREAK_MIN_EVENTS = 5
BREAK_P = 0.01


def hazard_rows(pred, target, edges):
    """Exactly `depth_run.first_error_hazard` (same windows, `at_risk`, `first_errors`, `rate`) plus `person_steps`: the steps actually spent at risk in the window, i.e. for every sequence
    still error-free at `start`, (first error - start + 1) when it fails inside the window and the full width otherwise. events / person_steps is the maximum-likelihood per-step hazard."""
    rows = dr.first_error_hazard(pred, target, edges)
    wrong = pred != target
    length = wrong.shape[1]
    first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((wrong.shape[0],), length))
    for row in rows:
        alive = first >= row['start']
        row['person_steps'] = int((torch.clamp(first[alive], max=row['stop'] - 1) - row['start'] + 1).sum())
    return rows


class ResetCore(Core):
    """Z6 (I3-d): the Core with the E-L1X periodic reset of the carried state (`mline/long_doc.py`'s `reset512` semantics): at every decoder step t > 0 with t % reset_every == 0 the
    carried state s['s'] is replaced by the learned initial state; the decoder's window cache (dk / dv) and the cross-attention memory are untouched. Only `configure_reset` turns it
    on; a freshly loaded Core never has `reset_every` set."""

    reset_every = None

    def decode(self, e, s, mk, mv, t):
        if self.reset_every and s is not None and t > 0 and t % self.reset_every == 0:
            s = dict(s)
            s['s'] = self.initial.expand_as(s['s'])
        return super().decode(e, s, mk, mv, t)


def configure_reset(model, reset_every):
    """Turn the Z6 reset on in place (class swap, the house pattern: `erosion_source.configure` / `snap_long.configure`). `reset_every >= 1`; the recurrent Core arms only
    (feedback=True, `depth_models.py`). Returns the model."""
    if not isinstance(reset_every, int) or reset_every < 1:
        raise ValueError('--reset-every needs a positive interval')
    if not isinstance(model, Core) or not getattr(model, 'feedback', False):
        raise ValueError('--reset-every is defined for the recurrent Core arms')
    model.__class__ = ResetCore
    model.reset_every = int(reset_every)
    return model


def measure(weights, lengths, n, batch=20, reset_every=0):
    raw = torch.load(weights, weights_only=True)
    meta = raw['meta']
    if dt.is_count(meta['task']):
        raise ValueError('long evaluation is defined for the permutation tasks')
    model = dr.load_model(weights)
    if reset_every:
        if meta['arm'] not in ('core', 'core_clean'):
            raise ValueError('--reset-every is defined for the Core arms')
        configure_reset(model, reset_every)
    curve = {}
    for length in lengths:
        started = time.perf_counter()
        x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
        pred, target = dr._accuracy(model, x, y, batch)
        ok = pred == target
        curve[str(length)] = dict(length=length, n=n, final=float(ok[:, -1].float().mean()), prefix=float(ok.float().mean()), error_free=int(ok.all(1).sum()),
                                  hazard=hazard_rows(pred, target, EDGES), seconds=time.perf_counter() - started)
    result = dict(weights=str(weights), arm=meta['arm'], task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, lengths=list(lengths), curve=curve,
                  train=dict(parameters=dr.parameter_count(model)))
    if reset_every:
        result['reset_every'] = int(reset_every)
    return result


def exactness_horizon(result):
    """The longest evaluated length L such that EVERY evaluated length <= L has every sequence error-free (None when even the shortest has an error)."""
    best = None
    for length in sorted(int(k) for k in result['curve']):
        v = result['curve'][str(length)]
        if v['error_free'] != v['n']:
            break
        best = length
    return best


def pooled_exposure(result):
    """{(start, stop): [first errors, at-risk steps (sequences at risk x width), person-steps]} summed over the evaluation points; person-steps come from the stored `person_steps` when the
    result has them, else from the midpoint approximation at-risk steps - first errors x width / 2 (exact only when the failures are spread evenly over the window)."""
    pooled = {}
    for point in result['curve'].values():
        for w in point.get('hazard') or []:
            width = w['stop'] - w['start']
            cell = pooled.setdefault((w['start'], w['stop']), [0, 0, 0.0])
            cell[0] += w['first_errors']
            cell[1] += w['at_risk'] * width
            cell[2] += w['person_steps'] if 'person_steps' in w else w['at_risk'] * width - w['first_errors'] * width / 2.0
    return dict(sorted(pooled.items()))


def reading(result):
    cells = {w: c for w, c in pooled_exposure(result).items() if w[0] >= BEYOND and c[1] > 0}
    events, exposure = sum(c[0] for c in cells.values()), sum(c[1] for c in cells.values())
    if events == 0:
        verdict = 'CLEAN' if exposure >= CLEAN_EXPOSURE else 'SHORT'
    elif events < 5:
        verdict = 'SPARSE'
    else:
        verdict = 'FLOOR'
        keys = sorted(cells)
        e_first, _, ps_first = cells[keys[0]]
        reference = max(e_first / ps_first, 3.0 / ps_first)         # the first window's hazard; the rule-of-three bound when it holds fewer than three first errors
        for key in keys[1:]:
            e_k, _, ps_k = cells[key]
            # P(window k holds this many first errors or more | one common hazard): the exact conditional binomial of rate_ratio_p with the roles swapped
            if e_k >= BREAK_MIN_EVENTS and e_k / ps_k >= BREAK_RATIO * reference and rate_ratio_p(e_k, ps_k, e_first, ps_first) < BREAK_P:
                verdict = 'BREAK'
                break
    return dict(verdict=verdict, events=events, exposure=exposure, hazard=(events / exposure if exposure else None),
                person_hazard=(events / sum(c[2] for c in cells.values()) if cells else None),
                windows={f'{a}-{b}': dict(events=c[0], exposure=c[1], hazard=c[0] / c[1], person_steps=c[2], person_hazard=(c[0] / c[2] if c[2] else None)) for (a, b), c in cells.items()},
                exactness_horizon=exactness_horizon(result))


def describe(result):
    r = reading(result)
    w = ' | '.join(f"[{k.replace('-', ',')}) {v['events']}/{v['exposure']}" for k, v in r['windows'].items())
    h = '' if r['hazard'] is None else f" = {r['hazard']:.2e} (per person-step {r['person_hazard']:.2e})"
    return (f"{result['arm']} {result['task']} w{result['width']} ({result['train']['parameters']} parameters), n = {result['n']}, lengths {result['lengths']}: {r['verdict']}  "
            f"first errors beyond {BEYOND}: {r['events']}/{r['exposure']}{h}; exactness horizon {r['exactness_horizon']}; windows {w}")


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='+')
    p.add_argument('--lengths', default='2048,8192,32768')
    p.add_argument('--core-lengths', default=None, help='lengths for the Core arm (its decoder loop is sequential: use fewer / shorter)')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--core-n', type=int, default=40)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--reset-every', type=int, default=0, help='Z6: replace the carried state with the learned initial at every step t > 0 where t is a multiple of m (0 = off, the previous behaviour)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    results = []
    for path in a.weights:
        arm = torch.load(path, weights_only=True)['meta']['arm']
        lengths = [int(v) for v in (a.core_lengths if (arm == 'core' and a.core_lengths) else a.lengths).split(',')]
        r = measure(path, lengths, a.core_n if arm == 'core' else a.n, a.batch, a.reset_every)
        results.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
