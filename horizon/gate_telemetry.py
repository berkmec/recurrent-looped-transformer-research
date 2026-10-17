"""A3: gate-loop telemetry — the opt-in per-step record of the HOLD gate head's gradient (K1-K4 labels).

Train side (off by default; `--gate-telemetry <path.jsonl>` on horizon.depth_run, hold/hold_nc arms only): one JSON line per
update, read AFTER `guarded_step` (the POST-clip gradients are the direction the optimizer actually saw; guards.py zeroes
them at the start of the next step, so they are still live here). Row schema (exactly these keys):
  step, stage_len, loss, applied, clipped, G, P, A, H
    G = ||concat(head.weight.grad, head.bias.grad)||_2        (gate-head gradient norm)
    P = -mean_i(head.bias.grad_i)                             (opening pressure: P>0 => descent OPENS the gate)
    A = mean_i |head.bias.grad_i|                             (pressure amplitude)
    H = mean of the step's forward gates (model.last_gates.mean(); no extra forward pass)
`applied=False` rows carry G/P/A = null (no gradient ever existed); H is still read when the forward produced gates.
The read never consumes RNG, adds no probe, changes no forward/backward numerics and never writes to the gradients.

Reader (notes/_hive_A3_ONKAYIT.md K1-K4 + K5 validity): windows of 250 steps, live / PRESSURE-DEAD windows, G_ratio,
H_final, F (sign-flip rate), the label assignment, the K4 +-20% band flag, and the K5 block. The summary is written ONCE
(Z10 spirit); the raw JSONL is archived and never overwritten.

  python -B -m horizon.gate_telemetry <tel.jsonl> [<tel2.jsonl> ...]                      # dry check (windows + label)
  python -B -m horizon.gate_telemetry --reading <tel.jsonl> --ref <run.json> --out <reading.json> \
      [--r0-equal] [--r2-live | --r2-live-auto] [--r3-loss <x>]
  python -B -m horizon.gate_telemetry --compare <run-a.json> <run-b.json>                # R0 bit-identity acceptance
  python -B -m horizon.gate_telemetry --print <reading.json>
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch

ROW_KEYS = ('step', 'stage_len', 'loss', 'applied', 'clipped', 'G', 'P', 'A', 'H')
WINDOW = 250
G_START_STEPS = 100
G_TAIL_WINDOWS = 2
LIVE_C = 0.05
G_RATIO_MIN = 0.10
H_CLOSED = 0.08
H_OPEN = 0.50
F_MIN = 0.30
DEAD_MAX = 0.50
MISSING_MAX = 0.10
APPLIED_FALSE_MAX = 0.01
R3_LOSS_MAX = 0.5
BAND_G_RATIO = (0.08, 0.12)                  # K4: +-20 % around 0.10
BAND_H_FINAL = ((0.064, 0.096), (0.40, 0.60))  # K4: around H_CLOSED and H_OPEN
BAND_F = (0.24, 0.36)                        # K4: around F_MIN
LABEL_SONUM, LABEL_ACILMA, LABEL_KISIR, LABEL_YONLU, LABEL_KARISIK, LABEL_INC = (
    'SÖNÜM', 'AÇILMA', 'KISIR-DÖNGÜ', 'YÖNLÜ-KİLİT', 'KARIŞIK', 'INCOMPLETE')


def pressure_from_bias(bias_grad):
    """P = -mean_i b_i (opening pressure), A = mean_i |b_i|. Sign convention (acceptance test 3): the loss wants the
    bias LARGER (dL/db < 0), the sigmoid opens further with b => P > 0 means the descent direction OPENS the gate."""
    b = bias_grad.detach().double()
    return -float(b.mean()), float(b.abs().mean())


def read_gate_row(model, applied):
    """The four telemetry numbers of one update. Call right after `guarded_step` (post-clip, pre-zero_grad).
    Never writes to the gradients; grads of `head` only; H reuses the forward's own gates."""
    gates = getattr(model, 'last_gates', None)
    h = None
    if gates is not None and bool(torch.isfinite(gates).all()):
        h = float(gates.detach().mean())
    row = dict(G=None, P=None, A=None, H=h)
    if not applied:
        return row
    head = getattr(model, 'head', None)
    gw = None if head is None or head.weight.grad is None else head.weight.grad
    gb = None if head is None or head.bias.grad is None else head.bias.grad
    if gw is None or gb is None:
        return row
    g2 = float(gw.detach().double().square().sum()) + float(gb.detach().double().square().sum())
    row['G'] = math.sqrt(g2)
    row['P'], row['A'] = pressure_from_bias(gb)
    return row


def append_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    handle.flush()


def read_jsonl(path):
    """(rows, malformed): rows are dicts; unparseable lines are counted, never guessed at."""
    rows, malformed = [], 0
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def _median(values):
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _buckets(rows, width=WINDOW):
    buckets = {}
    for r in rows:
        buckets.setdefault((int(r['step']) - 1) // width + 1, []).append(r)
    return buckets


def windows(rows, width=WINDOW):
    """K1: 250-step windows; medians per window; C_w = P_w/A_w; live = A_w > 0 and |C_w| >= 0.05."""
    out = []
    for w, rs in sorted(_buckets(rows, width).items()):
        p_w = _median([r.get('P') for r in rs])
        a_w = _median([r.get('A') for r in rs])
        c_w = (p_w / a_w) if (p_w is not None and a_w is not None and a_w > 0) else None
        live = bool(a_w is not None and a_w > 0 and c_w is not None and abs(c_w) >= LIVE_C)
        out.append(dict(w=w, start=min(int(r['step']) for r in rs), stop=max(int(r['step']) for r in rs), n=len(rs),
                        L_w=_median([r.get('loss') for r in rs]), G_w=_median([r.get('G') for r in rs]),
                        P_w=p_w, A_w=a_w, C_w=c_w, H_w=_median([r.get('H') for r in rs]), live=live))
    return out


def summarize(rows, malformed=0, r0_equal=None, r2_live=None, r3_loss=None):
    """K1-K5 summary of one telemetry file. Unknown K5 factors stay None (`pending`); a known False forces INCOMPLETE."""
    n_rows = len(rows) + malformed
    missing = malformed + sum(1 for r in rows if r.get('applied') and (r.get('G') is None or r.get('P') is None or r.get('A') is None))
    missing_frac = (missing / n_rows) if n_rows else 1.0
    applied_false_frac = (sum(1 for r in rows if not r.get('applied')) / n_rows) if n_rows else 0.0
    wins = windows(rows)
    buckets = _buckets(rows)
    g0 = _median([r.get('G') for r in rows if int(r['step']) <= G_START_STEPS])
    step1 = next((r for r in rows if int(r['step']) == 1), None)
    h_init = step1.get('H') if step1 else None
    h_final = wins[-1]['H_w'] if wins else None
    tail_windows = [w['w'] for w in wins[-G_TAIL_WINDOWS:]]
    g_end = _median([r.get('G') for w in tail_windows for r in buckets[w]])
    g_ratio = (g_end / g0) if (g0 is not None and g_end is not None and g0 > 0) else None
    live = [w for w in wins if w['live']]
    dead_ratio = (1 - len(live) / len(wins)) if wins else None
    pressure_dead = bool(dead_ratio is not None and dead_ratio > DEAD_MAX)
    f_rate = None
    if len(live) >= 2:
        signs = [1 if w['P_w'] > 0 else (-1 if w['P_w'] < 0 else 0) for w in live]
        f_rate = sum(1 for a, b in zip(signs, signs[1:]) if a != b) / (len(signs) - 1)
    bands = []
    if g_ratio is not None and BAND_G_RATIO[0] <= g_ratio <= BAND_G_RATIO[1]:
        bands.append('G_ratio')
    if h_final is not None and any(lo <= h_final <= hi for lo, hi in BAND_H_FINAL):
        bands.append('H_final')
    if f_rate is not None and BAND_F[0] <= f_rate <= BAND_F[1]:
        bands.append('F')
    k5 = dict(r0_bit_equal=r0_equal, r2_live=r2_live, r3_loss=r3_loss,
              r3_loss_ok=(None if r3_loss is None else bool(r3_loss <= R3_LOSS_MAX)),
              applied_false_frac=applied_false_frac, applied_false_ok=bool(applied_false_frac < APPLIED_FALSE_MAX))
    known = [k5['r0_bit_equal'], k5['r2_live'], k5['r3_loss_ok'], k5['applied_false_ok']]
    k5['passed'] = False if any(v is False for v in known) else (None if any(v is None for v in known) else True)
    out = dict(schema='a3-gate-reading/v1', n_steps=n_rows, n_windows=len(wins), windows=wins,
               G0=g0, H_init=h_init, H_final=h_final, G_end=g_end, G_ratio=g_ratio, F=f_rate,
               live_windows=len(live), dead_ratio=dead_ratio, PRESSURE_DEAD=pressure_dead,
               missing_frac=missing_frac, applied_false_frac=applied_false_frac, bands=bands, needs_seed2=bool(bands), k5=k5)
    out['label'], out['label_reason'] = assign_label(out)
    return out


def assign_label(s):
    """K3, first match wins. K3.1 INCOMPLETE: >10 % rows missing, G0 < 1e-12, or the K5 validity did NOT hold
    (a pending K5 does not force INCOMPLETE: the label stays provisional, `k5.passed` shows it)."""
    if s['n_steps'] == 0:
        return LABEL_INC, 'no telemetry rows'
    if s['missing_frac'] > MISSING_MAX:
        return LABEL_INC, f"missing rows {s['missing_frac']:.1%} > 10 %"
    if s['G0'] is None or s['G0'] < 1e-12:
        return LABEL_INC, 'G0 missing or ~0 (no live gate signal at the start)'
    if s['G_end'] is None or s['G_ratio'] is None:
        return LABEL_INC, 'G_end / G_ratio missing'
    if s['H_final'] is None:
        return LABEL_INC, 'H_final missing'
    if s['k5']['passed'] is False:
        bad = ','.join(k for k in ('r0_bit_equal', 'r2_live', 'r3_loss_ok', 'applied_false_ok') if s['k5'][k] is False)
        return LABEL_INC, f'K5 validity failed: {bad}'
    if s['G_ratio'] < G_RATIO_MIN:
        return LABEL_SONUM, f'G_ratio {s["G_ratio"]:.3f} < 0.10 (gradient decays)'
    if s['H_final'] >= H_OPEN:
        return LABEL_ACILMA, f'G_ratio {s["G_ratio"]:.3f} >= 0.10 and H_final {s["H_final"]:.3f} >= 0.50 (gate opens)'
    if s['H_final'] <= H_CLOSED and s['F'] is not None and s['F'] >= F_MIN and not s['PRESSURE_DEAD']:
        return LABEL_KISIR, f'H_final {s["H_final"]:.3f} <= 0.08, F {s["F"]:.2f} >= 0.30, pressure live (oscillating lock)'
    if s['H_final'] <= H_CLOSED and not s['PRESSURE_DEAD']:
        return LABEL_YONLU, f'H_final {s["H_final"]:.3f} <= 0.08, F {s["F"]} (one-way lock)'
    if s['PRESSURE_DEAD']:
        return LABEL_KARISIK, f'PRESSURE-DEAD (dead ratio {s["dead_ratio"]:.2f} > 0.50); H_final {s["H_final"]:.3f}'
    return LABEL_KARISIK, f'0.08 < H_final {s["H_final"]:.3f} < 0.50 (live, no extreme)'


def compare_runs(path_a, path_b):
    """R0 acceptance: history lists (loss/grad_norm) bit-identical + guard summaries equal."""
    a = json.loads(Path(path_a).read_text(encoding='utf8'))
    b = json.loads(Path(path_b).read_text(encoding='utf8'))
    ta, tb = a.get('train', {}), b.get('train', {})
    hist_equal = ta.get('history') == tb.get('history')
    guard_equal = ta.get('guard') == tb.get('guard')
    return dict(path_a=str(path_a), path_b=str(path_b), steps_a=ta.get('steps'), steps_b=tb.get('steps'),
                history_equal=hist_equal, guard_equal=guard_equal, equal=bool(hist_equal and guard_equal))


def describe(s):
    f = lambda v: 'n/a' if v is None else f'{v:.3f}'
    k5 = s['k5']
    return (f"A3 reading: {s['label']} ({s['label_reason']}) | G0={f(s['G0'])} G_end={f(s['G_end'])} G_ratio={f(s['G_ratio'])} "
            f"H_init={f(s['H_init'])} H_final={f(s['H_final'])} F={f(s['F'])} live={s['live_windows']}/{s['n_windows']} "
            f"miss={s['missing_frac']:.2%} bands={s['bands'] or '-'} k5={k5['passed']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='*', help='telemetry JSONL files (dry check)')
    p.add_argument('--reading', default=None, help='telemetry JSONL: write the K1-K5 summary')
    p.add_argument('--ref', default=None, help='the run .json the telemetry belongs to (recorded next to the summary)')
    p.add_argument('--out', default=None, help='summary path; written once, never overwritten')
    p.add_argument('--r0-equal', action='store_true', help='K5(a): the R0 bit-identity acceptance passed')
    p.add_argument('--r2-live', action='store_true', help='K5(b): the R2 contrast cell is live (G_ratio >= 0.10)')
    p.add_argument('--r2-dead', action='store_true', help='K5(b): the R2 contrast cell is NOT live (instrumentation suspect)')
    p.add_argument('--r2-live-auto', action='store_true', help='K5(b) from this same file: r2_live = (G_ratio >= 0.10)')
    p.add_argument('--r3-loss', type=float, default=None, help='K5(c): the R3 (plain Core) final training loss')
    p.add_argument('--compare', nargs=2, metavar=('A.json', 'B.json'), default=None, help='R0: compare history + guard of two runs')
    p.add_argument('--print', dest='show', default=None, help='an existing summary file: print it')
    a = p.parse_args()

    if a.compare:
        print(json.dumps(compare_runs(a.compare[0], a.compare[1]), ensure_ascii=False))
        return
    if a.show:
        s = json.loads(Path(a.show).read_text(encoding='utf8'))
        print(describe(s))
        return
    if a.reading:
        rows, malformed = read_jsonl(a.reading)
        r2_live = True if a.r2_live else (False if a.r2_dead else None)
        if r2_live is None and a.r2_live_auto:
            probe = summarize(rows, malformed=malformed)
            r2_live = bool(probe['G_ratio'] is not None and probe['G_ratio'] >= G_RATIO_MIN)
        s = summarize(rows, malformed=malformed, r0_equal=(True if a.r0_equal else None), r2_live=r2_live, r3_loss=a.r3_loss)
        s['source_jsonl'] = str(a.reading)
        if a.ref:
            ref = json.loads(Path(a.ref).read_text(encoding='utf8'))
            tr = ref.get('train', {})
            s['ref'] = dict(path=str(a.ref), task=ref.get('task'), arm=ref.get('arm'), width=ref.get('width'), seed=ref.get('seed'),
                            steps=tr.get('steps'), learned_in_range=ref.get('learned_in_range'), wall_seconds=ref.get('wall_seconds'),
                            guard=tr.get('guard'))
            if tr.get('steps') is not None and int(tr['steps']) != s['n_steps']:
                s['ref']['steps_mismatch'] = True
        if a.out:
            path = Path(a.out)
            if path.exists():
                raise SystemExit(f'{a.out} exists: the summary is written once (Z10), choose a new --out')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf8')
        print(describe(s))
        return
    for path in a.paths:
        rows, malformed = read_jsonl(path)
        s = summarize(rows, malformed=malformed)
        print(path + ': ' + describe(s))


if __name__ == '__main__':
    main()
