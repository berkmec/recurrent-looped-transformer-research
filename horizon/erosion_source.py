"""E-A5: where does the S5 Core's long-position erosion live? Evaluation only (no training), CPU, dev split.

The S5 Core (width 128) is exact to about 64 positions and then fails at a roughly constant per-step rate (~2e-3) while its logit margin erodes with position (E-M5), its class
geometry R / M grows (E-B2), a kick of the carried state neither grows nor decays, and state-noise training did not help (E-D5, sigma 0.2). The S4 Core, with the same architecture
and the same training length (16), is exact and has a flat margin. Two sources of a position-dependent deviation are separable at test time, without training anything:

  A  the NON-recurrent context paths. The encoder is one causal self-attention layer over the whole prefix (RoPE) and the decoder cross-attends to the encoder memory of every position
     seen so far. At 16 positions (the trained length) they see at most 16 keys at relative offsets 0..15; at position 500 they see 500 keys at offsets up to 499. `ProbeCore` restricts
     the encoder self-attention (`enc`), the cross-attention (`cross`) or both to the last K positions. RoPE scores depend only on relative offsets and the first token carries no marker, so
     the last K tokens are attended exactly as a trained-range prefix of K tokens would be (tested: the windowed encoder at position t equals the unwindowed encoder on the last K tokens);
     the carried recurrent state keeps its whole history. K = 16 is the trained horizon.
  B  the carried recurrent state. Before the decoder step `at` the carried state (the previous step's final decoder output `z`, which the feedback reads, and optionally the three cached
     decoder keys / values) is replaced by the centroid of the TRUE class of the previous position, measured on in-range (positions 8..15) dev sequences. This is an oracle used for diagnosis
     only. The first-error hazard of the next `span` = 64 positions is measured on all sequences and compared with the hazard of the same window in the untouched run among the sequences that
     are still error-free at `at`. The encoder memory and the current input are NOT reset: only the history carried in the state is.

A restricts what the context paths see and leaves the state alone; B resets the state and leaves the context paths alone. Rescue by A and not B: the erosion lives in the non-recurrent
paths; by B and not A: in the history the recurrent state carries; both or neither are readable too (program note 5.26).

  python -B -m horizon.erosion_source results/Z1-pilot/perm5-core-w128-s1-cur.pt ... --out results/Z1-pilot/erosion-source.json
  python -B -m horizon.erosion_source --reading results/Z1-pilot/erosion-source.json

Reading rule (written before the first run, program note 5.26). Per model and per condition the first-error hazard is events / person-steps (`long_eval.hazard_rows`'s definition);
condition against the untouched run, R = hazard(untouched) / max(hazard(condition), 3 / person-steps(condition)) (a rule-of-three floor for zero events), exact conditional-binomial
tests of `readings.rate_ratio_p` (lower tail: the condition's hazard is lower; swapped roles: higher):
  RESCUES  R >= 5 and p(lower) < 0.01;   PARTIAL  2 <= R < 5 and p(lower) < 0.01;   HARMS  R <= 0.5, p(higher) < 0.01 and >= 5 events in the condition;   NEUTRAL  otherwise.
A: primary condition `both16` (encoder and cross-attention windows of 16), window [16, 1024), the three S5 Core models; the S4 Core models are the control.
B: primary condition `state+cache` (state and cache reset), the first-error hazard of [t0, t0 + 128) pooled over t0 in {128, 256, 512}.
Verdict per part: >= 2 of the 3 S5 models RESCUES -> CARRIES THE EROSION; no S5 model RESCUES or PARTIAL and at least two NEUTRAL (and the S4 control not HARMS) -> NOT THE SOURCE;
no S5 model RESCUES or PARTIAL otherwise -> INCONCLUSIVE (the intervention itself harms); anything else -> MIXED. CONTROL FAILED when the untouched run does not reproduce the stored
first-error windows of the model, when a window changes a prediction before its first K positions, or (B) when the in-range injection at position 8 leaves a model below 0.99 accuracy on
positions 8..15.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from core.model import Core

from . import depth_run as dr
from . import depth_tasks as dt
from .readings import rate_ratio_p
from .state_growth import core_trajectory

WINDOW_FROM = 16                        # a window of 16 cannot change anything before position 16
SPAN = 128                              # positions followed after a state injection
SINCE = ((0, 16), (16, 32), (32, 64), (64, 128))       # descriptive: the first-error hazard by steps since the injection
INJECT_AT = (32, 64, 128, 256, 512)
POOLED_AT = (128, 256, 512)
CONTROL_AT = 8                          # in-range injection control: positions 8..15
BOTH_K = (16, 32, 64, 128)
SINGLE_K = (16, 64)
RANGE_K = (1, 2, 4, 8)                  # in-range reliance probe (descriptive)
RESCUE_RATIO, PARTIAL_RATIO, HARM_RATIO = 5.0, 2.0, 0.5
SIGNIFICANT = 0.01
TOL = 1e-9                              # the exact limits count (R = 5 is a rescue even when the quotient reads 4.999999999999999)
MIN_HARM_EVENTS = 5
FLOOR_EVENTS = 3.0
CONTROL_ACCURACY = 0.99
S4_CONTROL_EVENTS = 2
WINDOW_EDGES = (16, 64, 128, 256, 512, 1024)


class ProbeCore(Core):
    """The sealed Core with (a) the context paths restricted to the last K positions and (b) the carried state replaced before one decoder step. Nothing changes unless it is configured."""

    cross = None            # keep only the last `cross` memory positions in the decoder's cross-attention
    enc = None              # each encoder position attends to itself and the `enc` - 1 previous ones
    at = None               # decoder step before which the carried state is replaced
    state = None            # (batch, width) replacement for the fed-back state at step `at`
    reset_cache = False     # also empty the three cached decoder keys / values at step `at`

    def encode(self, x):
        if self.enc is None or self.enc >= x.shape[1]:
            return super().encode(x)
        steps = x.shape[1]
        e = self.embedding(x)
        n = self.en(e)
        q, k, v = self.rope(self.split(self.eq(n))), self.rope(self.split(self.ek(n))), self.split(self.ev(n))
        idx = torch.arange(steps, device=x.device)
        delta = idx[:, None] - idx[None, :]
        mask = (delta >= 0) & (delta < self.enc)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(q.shape[0], q.shape[2], self.width)
        e = e + self.eo(a)
        e = e + self.eff(self.efn(e))
        return e, k, v

    def decode(self, e, s, mk, mv, t):
        if self.cross is not None and mk.shape[2] > self.cross:
            mk, mv = mk[:, :, -self.cross:], mv[:, :, -self.cross:]
        if s is not None and self.at is not None and t == self.at:
            s = dict(s)
            s['s'] = self.state
            if self.reset_cache:
                s['dk'], s['dv'] = s['dk'][:, :, :0], s['dv'][:, :, :0]
        return super().decode(e, s, mk, mv, t)


def configure(model, cross=None, enc=None, at=None, state=None, reset_cache=False):
    """Set the probe in place (class swap); with no argument the trained behaviour at every step is restored."""
    if not isinstance(model, Core):
        raise ValueError('the attention probe is defined for the Core arm')
    for name, value in (('cross', cross), ('enc', enc)):
        if value is not None and value < 1:
            raise ValueError(f'the {name} window must be >= 1')
    if at is not None and (at < 1 or state is None):
        raise ValueError('a state injection needs a step >= 1 (the first step has no fed-back state) and a replacement state')
    model.__class__ = ProbeCore
    model.cross, model.enc, model.at, model.state, model.reset_cache = cross, enc, at, state, reset_cache
    return model


def predictions(model, x, batch=40):
    model.eval()
    out = []
    with torch.no_grad():
        for at in range(0, len(x), batch):
            out.append(model(x[at:at + batch]).argmax(-1))
    return torch.cat(out)


def first_errors(pred, target):
    wrong = pred != target
    length = wrong.shape[1]
    return torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((wrong.shape[0],), length))


def window_counts(first, start, stop):
    """(first errors, person-steps, at-risk sequences) of the first-error hazard in [start, stop): sequences still error-free at `start`; a sequence that fails inside the window spends
    (first - start + 1) steps at risk, one that does not spends the whole window (`long_eval.hazard_rows`'s definition)."""
    alive = first >= start
    events = int((alive & (first < stop)).sum())
    person = int((torch.clamp(first[alive], max=stop - 1) - start + 1).sum())
    return events, person, int(alive.sum())


def class_centroids(model, task, seed, n=2000, length=16, positions=(8, 16)):
    """{class: state} the mean carried state (the final decoder output after the step) of the sequences whose running product is that class at positions [a, b) of in-range dev sequences."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    configure(model)
    with torch.no_grad():
        states, _ = core_trajectory(model.eval(), x)
    a, b = positions
    s, lab = states[:, a:b].reshape(-1, states.shape[-1]), y[:, a:b].reshape(-1)
    count = dt.classes(task)
    centroids = torch.zeros(count, states.shape[-1])
    for c in range(count):
        members = s[lab == c]
        if len(members) < 5:
            raise ValueError(f'class {c} has only {len(members)} states at positions {positions}: draw more sequences')
        centroids[c] = members.mean(0)
    return centroids


def stored_control(weights, first, n_stored=None):
    """Does the untouched run reproduce the first-error windows stored with the model (same dev draw)? None when no stored result is found."""
    stored_path = Path(weights).with_suffix('.json')
    if not stored_path.exists():
        return None
    stored = json.loads(stored_path.read_text(encoding='utf8'))['curve'].get('1024')
    if stored is None:
        return None
    n_stored = stored['hazard'][0]['at_risk'] if n_stored is None else n_stored        # every sequence is at risk in the first window: that count is the size of the stored draw
    if len(first) < n_stored:
        return None
    edges = [r['start'] for r in stored['hazard']] + [stored['hazard'][-1]['stop']]
    mine = [dict(start=a, stop=b, **dict(zip(('first_errors', 'person', 'at_risk'), window_counts(first[:n_stored], a, b)))) for a, b in zip(edges[:-1], edges[1:])]
    return all(m['first_errors'] == s['first_errors'] and m['at_risk'] == s['at_risk'] for m, s in zip(mine, stored['hazard']))


def condition_label(cross, enc):
    if cross is None and enc is None:
        return 'base'
    if cross is not None and enc is not None:
        return f'both{cross}' if cross == enc else f'cross{cross}+enc{enc}'
    return f'cross{cross}' if cross is not None else f'enc{enc}'


def window_part(model, x, y, batch=40, log=print):
    """Part A on one model: the untouched run and the window conditions, all on the same sequences."""
    conditions = [(None, None)] + [(k, k) for k in BOTH_K] + [(k, None) for k in SINGLE_K] + [(None, k) for k in SINGLE_K]
    out, base_pred = {}, None
    for cross, enc in conditions:
        started = time.perf_counter()
        configure(model, cross=cross, enc=enc)
        pred = predictions(model, x, batch)
        configure(model)
        label = condition_label(cross, enc)
        first = first_errors(pred, y)
        events, person, at_risk = window_counts(first, WINDOW_FROM, x.shape[1])
        row = dict(cross=cross, enc=enc, events=events, person_steps=person, at_risk=at_risk, final=float((pred[:, -1] == y[:, -1]).float().mean()),
                   windows={f'{a}-{b}': dict(zip(('events', 'person_steps', 'at_risk'), window_counts(first, a, min(b, x.shape[1])))) for a, b in zip(WINDOW_EDGES[:-1], WINDOW_EDGES[1:]) if a < x.shape[1]})
        if label == 'base':
            base_pred = pred
        else:
            k = min(v for v in (cross, enc) if v is not None)
            row['prefix_identical'] = bool((pred[:, :k] == base_pred[:, :k]).all())          # the window cannot change a prediction before its first K positions
        out[label] = row
        log(f'  {label:<9} first errors {events:>3}/{person:>6} person-steps = {events / max(person, 1):.2e}  final {row["final"]:.2f}  ({time.perf_counter() - started:.0f} s)')
    return out, base_pred


def range_part(model, task, seed, n=200, length=16, batch=50):
    """Descriptive: in-range (length 16) accuracy when BOTH context paths see only the last K positions (K = 1, 2, 4, 8): how far back the model's attention has to look inside the trained range."""
    x, y, _ = dt.heldout(dr.NAMESPACE, task, seed, 'dev', n, length)
    out = {}
    for k in (None,) + RANGE_K:
        configure(model, cross=k, enc=k)
        pred = predictions(model, x, batch)
        configure(model)
        out['full' if k is None else f'K{k}'] = float((pred == y).float().mean())
    return out


def injection_part(model, x, y, centroids, batch=40, log=print):
    """Part B on one model: for every injection step the untouched hazard of [t0, t0 + SPAN) among the sequences error-free at t0, and the hazard after replacing the carried state with the
    true class's centroid (variants `state`: z only; `state+cache`: z and the three cached keys / values), measured on all sequences from t0 on."""
    out = {}
    for t0 in (CONTROL_AT,) + INJECT_AT:
        span = SPAN if t0 != CONTROL_AT else 8
        if t0 + span > x.shape[1]:
            continue                                            # the sequences are too short for this injection step
        started = time.perf_counter()
        xs = x[:, :t0 + span]
        configure(model)
        base = predictions(model, xs, batch)
        base_first = first_errors(base, y[:, :t0 + span])
        events, person, at_risk = window_counts(base_first, t0, t0 + span)
        row = dict(span=span, untouched=dict(events=events, person_steps=person, at_risk=at_risk,
                                             since={f'{a0}-{b0}': dict(zip(('events', 'person_steps', 'at_risk'), window_counts(base_first, t0 + a0, t0 + b0))) for a0, b0 in SINCE if b0 <= span}))
        for variant, reset in (('state', False), ('state+cache', True)):
            parts = []
            with torch.no_grad():
                for at in range(0, len(xs), batch):
                    configure(model, at=t0, state=centroids[y[at:at + batch, t0 - 1]], reset_cache=reset)
                    parts.append(model(xs[at:at + batch]).argmax(-1))
            configure(model)
            pred = torch.cat(parts)
            wrong = pred[:, t0:] != y[:, t0:t0 + span]
            first = torch.where(wrong.any(1), wrong.int().argmax(1), torch.full((wrong.shape[0],), span)) + t0
            e, p, a = window_counts(first, t0, t0 + span)
            row[variant] = dict(events=e, person_steps=p, at_risk=a, accuracy=float((~wrong).float().mean()),
                                since={f'{a0}-{b0}': dict(zip(('events', 'person_steps', 'at_risk'), window_counts(first, t0 + a0, t0 + b0))) for a0, b0 in SINCE if b0 <= span})
        out[str(t0)] = row
        log(f'  t0 {t0:>3}: untouched {events:>3}/{person:>5}  state {row["state"]["events"]:>3}/{row["state"]["person_steps"]:>5}  '
            f'state+cache {row["state+cache"]["events"]:>3}/{row["state+cache"]["person_steps"]:>5}  ({time.perf_counter() - started:.0f} s)')
    return out


def measure(weights, n=200, length=1024, batch=40, log=print):
    meta = torch.load(weights, weights_only=True)['meta']
    if meta['arm'] != 'core' or dt.is_count(meta['task']):
        raise ValueError('E-A5 is defined for the Core on the permutation tasks')
    model = dr.load_model(weights).eval()
    x, y, _ = dt.heldout(dr.NAMESPACE, meta['task'], meta['seed'], 'dev', n, length)
    log(f'{weights}: {meta["task"]} width {meta["width"]}, {n} dev sequences of {length} positions')
    windows, base_pred = window_part(model, x, y, batch, log)
    centroids = class_centroids(model, meta['task'], meta['seed'])
    injection = injection_part(model, x, y, centroids, batch, log)
    return dict(weights=str(weights), task=meta['task'], width=meta['width'], seed=meta['seed'], n=n, length=length, parameters=dr.parameter_count(model),
                stored_control=stored_control(weights, first_errors(base_pred, y)), windows=windows, injection=injection, in_range=range_part(model, meta['task'], meta['seed']))


def classify(untouched, condition):
    """(label, details) of one condition against the untouched run; both are (events, person_steps)."""
    (e_b, p_b), (e_c, p_c) = untouched, condition
    if p_b <= 0 or p_c <= 0:
        return 'NO DATA', dict(ratio=None)
    base_rate = e_b / p_b
    cond_rate = max(e_c / p_c, FLOOR_EVENTS / p_c)
    ratio = base_rate / cond_rate
    lower, higher = rate_ratio_p(e_b, p_b, e_c, p_c), rate_ratio_p(e_c, p_c, e_b, p_b)
    if ratio >= RESCUE_RATIO - TOL and lower < SIGNIFICANT:
        label = 'RESCUES'
    elif ratio >= PARTIAL_RATIO - TOL and lower < SIGNIFICANT:
        label = 'PARTIAL'
    elif ratio <= HARM_RATIO + TOL and higher < SIGNIFICANT and e_c >= MIN_HARM_EVENTS:
        label = 'HARMS'
    else:
        label = 'NEUTRAL'
    return label, dict(ratio=ratio, base_rate=base_rate, condition_rate=e_c / p_c, p_lower=lower, p_higher=higher)


def window_labels(result, condition='both16'):
    base, cond = result['windows']['base'], result['windows'][condition]
    return classify((base['events'], base['person_steps']), (cond['events'], cond['person_steps']))


def injection_labels(result, variant='state+cache', at=POOLED_AT):
    rows = [result['injection'][str(t0)] for t0 in at if str(t0) in result['injection']]
    untouched = (sum(r['untouched']['events'] for r in rows), sum(r['untouched']['person_steps'] for r in rows))
    injected = (sum(r[variant]['events'] for r in rows), sum(r[variant]['person_steps'] for r in rows))
    return classify(untouched, injected)


def _verdict(labels, s4_labels, carries, source):
    """The part verdict from the S5 labels and the S4 control labels (module docstring)."""
    rescues = sum(1 for v in labels if v == 'RESCUES')
    helped = sum(1 for v in labels if v in ('RESCUES', 'PARTIAL'))
    neutral = sum(1 for v in labels if v == 'NEUTRAL')
    if rescues >= 2:
        return carries
    if helped == 0:
        if 'HARMS' in s4_labels:
            return 'INCONCLUSIVE (the intervention also harms the S4 control)'
        return source if neutral >= 2 else 'INCONCLUSIVE (the intervention harms the S5 models)'
    return 'MIXED'


def reading(results):
    """The registered verdicts for a list of `measure` results (S5 Core models carry the verdict, S4 Core models are the control)."""
    s5 = [r for r in results if r['task'] == 'perm5']
    s4 = [r for r in results if r['task'] == 'perm4']
    if not s5:
        return dict(window='NO S5 MODEL', injection='NO S5 MODEL')
    failed = []
    for r in s5 + s4:
        name = f'{r["task"]} w{r["width"]}'
        if r.get('stored_control') is False:
            failed.append(f'{name}: the untouched run does not reproduce the stored first-error windows')
        for label, row in r['windows'].items():
            if label != 'base' and row.get('prefix_identical') is False:
                failed.append(f'{name}: the window {label} changed a prediction before its first K positions')
        for variant in ('state', 'state+cache'):
            control = r['injection'].get(str(CONTROL_AT), {}).get(variant)
            if control is not None and control['accuracy'] < CONTROL_ACCURACY - 1e-9:
                failed.append(f'{name}: the in-range injection ({variant}) leaves accuracy {control["accuracy"]:.3f} on positions 8..15')
    out = dict(controls=failed, models={})
    for r in s5 + s4:
        out['models'][f'{r["task"]} w{r["width"]} s{r["seed"]} {Path(r["weights"]).stem}'] = dict(
            window={label: window_labels(r, label)[0] for label in r['windows'] if label != 'base'},
            injection={variant: injection_labels(r, variant)[0] for variant in ('state', 'state+cache')})
    if failed:
        out['window'] = out['injection'] = 'CONTROL FAILED: ' + '; '.join(failed)
        return out
    s4_window = [window_labels(r)[0] for r in s4]
    s4_inject = [injection_labels(r)[0] for r in s4]
    # the S4 control for the injection must also stay (almost) error-free: it has no hazard to rescue, so an injection that creates errors there harms
    s4_inject = ['HARMS' if label == 'HARMS' or sum(r['injection'][str(t0)]['state+cache']['events'] for t0 in POOLED_AT if str(t0) in r['injection']) > S4_CONTROL_EVENTS else label
                 for r, label in zip(s4, s4_inject)]
    out['window'] = _verdict([window_labels(r)[0] for r in s5], s4_window, 'ATTENTION PATHS CARRY THE EROSION', 'NOT THE ATTENTION PATHS')
    out['injection'] = _verdict([injection_labels(r)[0] for r in s5], s4_inject, 'THE CARRIED STATE HISTORY CARRIES THE EROSION', 'NOT THE CARRIED STATE')
    return out


def describe(r):
    w = r['windows']
    line = [f'{r["task"]} w{r["width"]} ({r["parameters"]} parameters), n = {r["n"]}, stored-window control {r["stored_control"]}']
    base = w['base']
    line.append(f'  windows [16,{r["length"]}): ' + '  '.join(f'{k} {v["events"]}/{v["person_steps"]}' for k, v in w.items()) + f'   (final accuracy {base["final"]:.2f} -> both16 {w["both16"]["final"]:.2f})')
    line.append('  in-range accuracy by window K (both paths): ' + '  '.join(f'{k} {v:.3f}' for k, v in r['in_range'].items()))
    for t0, row in r['injection'].items():
        line.append(f'  t0 {t0:>3} span {row["span"]:>2}: untouched {row["untouched"]["events"]}/{row["untouched"]["person_steps"]}  state {row["state"]["events"]}/{row["state"]["person_steps"]} (acc {row["state"]["accuracy"]:.3f})  '
                    f'state+cache {row["state+cache"]["events"]}/{row["state+cache"]["person_steps"]} (acc {row["state+cache"]["accuracy"]:.3f})')
    label_a, det_a = window_labels(r)
    label_b, det_b = injection_labels(r)
    line.append(f'  both16: {label_a} (R {det_a.get("ratio")}); state+cache pooled at {POOLED_AT}: {label_b} (R {det_b.get("ratio")})')
    return '\n'.join(line)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--batch', type=int, default=40)
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None, help='a result file written by --out: print the registered verdicts')
    a = p.parse_args()
    if a.reading:
        results = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for r in results:
            print(describe(r))
        print(json.dumps(reading(results), indent=1, default=float))
        return
    results = []
    for path in a.weights:
        r = measure(path, a.n, a.length, a.batch, log=lambda m: print(m, flush=True))
        results.append(r)
        print(describe(r), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1, default=float), encoding='utf8')
    if a.out:
        print(json.dumps(reading(results), indent=1, default=float))


if __name__ == '__main__':
    main()
