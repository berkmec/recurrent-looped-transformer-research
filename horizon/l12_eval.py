"""L12: kayıp-1.2 — fixed info count (k = 102), variable neutral-gap evaluation (Stage-E, eval-only).

Rule: notes/_hive_L12_KURAL.md (registered protocol; proposal owner/hüküm: Jim). k = 102 fixed info steps (ONE content
draw), the nested profiles P0 ⊂ P50 ⊂ P90 ⊂ P97 differ only in the neutral gap mass inserted between the same info
steps; the eval machine is the Z1-Z5 one (hold_neutral / hold_informative / sign_ok from hold.hold_stats on the first
50 sequences; acc and loss on 200; CI + SINIRDA per Z10). L12-kayip(P) = 100*(acc(P0) - acc(P)); primary cell is P90.

  python -B -m horizon.l12_eval results/Z1-pilot/perm4-hold-w64-s*-cur-egnp05.pt ... \
      --out results/Z1-pilot/l12-e-session-e1.json --n 200 --seed 1
  python -B -m horizon.l12_eval --reading results/Z1-pilot/l12-e-session-e1.json
  python -B -m horizon.l12_eval <ckpts> --blocks-p90 --out results/Z1-pilot/l12-e-session-e1-blocks.json   # secondary, descriptive

Single write: an existing --out is never overwritten (EG-4/Z10 spirit). Draws are paired: one info sequence per task
(content and order) is shared by every profile and every checkpoint; `base_seed` is recorded in the session (EG-4).
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import depth_run as dr
from . import depth_tasks as dt
from . import hold

PROFILES = (('P0', 0, 102), ('P50', 102, 204), ('P90', 922, 1024), ('P97', 3072, 3174))   # (name, neutral n, length)
K_DEFAULT = 102
DRAW_TAG = 1200            # step tag of the L12 draw in source_seed (kayıp-1.2 mnemonic); recorded in the session
GATE_SEQS = 50             # gate telemetry: first 50 sequences (Z1-Z5 convention)
FINE_LENGTH = 3174         # the P97 grid the gap layouts are drawn on


def contract(pos, l_dst, l_src=FINE_LENGTH):
    """The info positions of one sequence contracted to a coarser profile: proportional scaling into a strictly
    increasing list, clipped to the feasibility band [i, l_dst - k + i]. P0 (l_dst = k) contracts to exactly 0..k-1."""
    pos = np.rint(np.asarray(pos, dtype=np.float64) * (l_dst / l_src)).astype(np.int64)
    k = len(pos)
    lo, hi = np.arange(k), (l_dst - k) + np.arange(k)
    pos = np.clip(pos, lo, hi)
    for i in range(1, k):
        if pos[i] <= pos[i - 1]:
            pos[i] = pos[i - 1] + 1
    return pos


def draw(task, seed, n, k=K_DEFAULT, layout='gaps'):
    """One L12 draw for a task: info content (n, k) — uniform NON-identity tokens — plus per-sequence neutral-gap
    layouts on the fine grid, contracted to every profile. layout='blocks' replaces the P90 gaps with ~8 long neutral
    blocks (descriptive secondary contrast; identical info content and order, single layout for all sequences)."""
    base = dt.source_seed(dr.NAMESPACE, task, seed, 'dev', step=DRAW_TAG, length=0, depth=0)
    rng = np.random.default_rng(base)
    info = rng.integers(1, dt.vocab(task), size=(n, k)).astype(np.int64)
    pos97 = np.sort(rng.random((n, FINE_LENGTH)).argsort(axis=1)[:, :k], axis=1)
    blocks = None
    if layout == 'blocks':
        sizes = np.full(8, (1024 - k) // 8, dtype=np.int64)      # 922 neutral tokens in 8 long blocks
        sizes[:(1024 - k) % 8] += 1
        rng.shuffle(sizes)
        runs = np.full(9, k // 9, dtype=np.int64)                # the 102 info steps in 9 ordered runs between them
        runs[:k % 9] += 1
        pos, cur = [], 0
        for r in range(9):
            pos.extend(range(cur, cur + int(runs[r])))
            cur += int(runs[r])
            if r < 8:
                cur += int(sizes[r])
        if cur != 1024:
            raise AssertionError('block layout does not tile the P90 length')
        blocks = np.tile(np.asarray(pos, dtype=np.int64), (n, 1))
    profiles = {}
    for name, n_neutral, length in PROFILES:
        if name == 'P90' and blocks is not None:
            positions = blocks
        else:
            positions = np.stack([contract(p, length) for p in pos97])
        x = np.zeros((n, length), dtype=np.int64)
        x[np.arange(n)[:, None], positions] = info
        profiles[name] = dict(length=length, n_neutral=n_neutral, density=n_neutral / length, x=torch.from_numpy(x))
    for prof in profiles.values():
        prof['y'] = dt.oracle(task, prof['x'])
    return dict(task=task, seed=seed, base_seed=base, k=k, n=n, layout=layout, profiles=profiles)


def cell_validity(model, meta, task, weights, batch=50):
    """A2 gate (§3a): valid <=> learned_in_range. Taken from the checkpoint's sibling run JSON when it carries the
    registered flag; otherwise computed (dev draw, TRAIN_LENGTH, n = 100) with dr.meets. Returns (valid, acc, source)."""
    sibling = Path(weights).with_suffix('.json')
    lir, acc, src = None, None, None
    if sibling.exists():
        try:
            j = json.loads(sibling.read_text(encoding='utf8'))
            lir, acc, src = j.get('learned_in_range'), j.get('in_range_accuracy'), 'sibling-json'
        except (json.JSONDecodeError, OSError):
            lir = None
    if lir is None:
        x, y, _ = dt.heldout(dr.NAMESPACE, task, int(meta.get('eval_seed', meta['seed'])), 'dev', 100, dt.train_length(task))
        pred, target = dr._accuracy(model, x, y, batch)
        acc = float((pred == target)[:, -1].float().mean())
        lir = dr.meets(acc)
        src = 'computed-dev'
    return bool(lir), acc, src


def measure_cell(weights, drawn, n, batch=50, profiles=('P0', 'P50', 'P90', 'P97')):
    model = dr.load_model(weights)
    if not isinstance(model, hold.HoldCore):
        raise ValueError(f'{weights} is not a hold/hold_nc checkpoint (L12 Stage-E needs the gate)')
    meta = torch.load(weights, weights_only=True)['meta']
    task = meta['task']
    d = drawn[task]
    rows, ok_final = {}, {}
    for name in profiles:
        prof = d['profiles'][name]
        x, y = prof['x'], prof['y']
        pred, target = dr._accuracy(model, x, y, batch)
        ok = pred == target
        okf = ok[:, -1].float()
        acc = float(okf.mean())
        ok_final[name] = okf
        row = dict(length=prof['length'], n_neutral=prof['n_neutral'], n=int(x.shape[0]), acc=acc,
                   acc_ci=hold.wilson(int(round(acc * x.shape[0])), int(x.shape[0])))
        info_acc = np.zeros((x.shape[0], d['k']))     # positions-window profile (descriptive drift signature)
        for s in range(x.shape[0]):
            idx = (x[s] != 0).nonzero(as_tuple=True)[0]
            info_acc[s] = ok[s, idx].float().numpy()
        row['info_step_acc'] = [round(float(v), 4) for v in info_acc.mean(0)]
        row['info_head_tail'] = dict(head25=float(info_acc[:, :25].mean()), tail25=float(info_acc[:, -25:].mean()))
        if name == 'P0':                              # no neutral position: hold_neutral is UNDEFINED there (rule §2)
            row.update(hold_neutral=None, hold_informative=None, sign_ok=None, n_gate=None,
                       note='no neutral position: hold_neutral undefined (rule 2)')
        else:
            xg = x[:min(GATE_SEQS, x.shape[0])]
            model.eval()
            with torch.no_grad():
                model(xg)
            stats = hold.hold_stats(model.last_gates, neutral_mask=(xg == 0))
            row.update(n_gate=int(xg.shape[0]), hold_neutral=stats['hold_neutral'], hold_informative=stats['hold_informative'],
                       hold_mean=stats['hold_mean'], sign_ok=stats['sign_ok'])
        rows[name] = row
    p0 = rows.get('P0')
    for name in profiles:                              # L12-kayip vs P0, paired; CI + SINIRDA per Z10
        if name == 'P0' or p0 is None:
            continue
        dif = (ok_final['P0'] - ok_final[name]).double()
        mean = float(dif.mean())
        se = float(dif.std(unbiased=True)) / math.sqrt(len(dif)) if len(dif) > 1 else None
        ci = None if se is None else [100 * (mean - hold.Z95 * se), 100 * (mean + hold.Z95 * se)]
        rows[name]['l12_loss_points'] = 100 * mean
        rows[name]['loss_ci'] = ci
        rows[name]['loss_sinirda'] = bool(ci and any(ci[0] <= t <= ci[1] for t in (hold.LOSS_SUPPORT, hold.LOSS_REFUTE)))
    valid, valid_acc, valid_source = cell_validity(model, meta, task, weights, batch=batch)
    cell = dict(id=Path(weights).stem, weights=str(weights), task=task, arm=meta['arm'], width=meta['width'],
                seed=meta['seed'], parameters=meta.get('parameters'), valid=valid, validity_accuracy=valid_acc,
                validity_source=valid_source, profiles=rows)
    cell['reading'] = cell_reading(cell)
    if cell['arm'] == 'hold_nc':
        cell['reading']['descriptive'] = True          # Z7: hold_nc never enters the combination
    return cell


def cell_reading(cell):
    """Rule §3b, first match wins; the label is a separate L12 line (the E-G seal is untouched, §3b last line)."""
    p0, p90 = cell['profiles'].get('P0'), cell['profiles'].get('P90')
    if not p0 or not p90:
        return dict(verdict='INCOMPLETE', reason='P0 or P90 missing', flags=[])
    hold_n, loss, sign = p90.get('hold_neutral'), p90.get('l12_loss_points'), p90.get('sign_ok')
    missing = [k for k, v in (('hold_neutral(P90)', hold_n), ('L12-kayip(P90)', loss), ('sign_ok(P90)', sign)) if v is None]
    if missing:
        return dict(verdict='INCOMPLETE', reason='missing ' + ', '.join(missing), flags=[])
    flags = ['SINIRDA'] if p90.get('loss_sinirda') else []
    if hold_n >= hold.HOLD_MIN and loss <= hold.LOSS_SUPPORT and bool(sign):
        verdict, why = 'DESTEKLENİR', 'hold >= 0.5, sign_ok and L12-kayip <= 5: the gate carries the neutral load'
    elif loss > hold.LOSS_REFUTE:
        verdict, why = 'ÇÜRÜTÜLDÜ', 'L12-kayip > 20 even at matched depth'
    elif hold_n < hold.HOLD_MIN:
        verdict, why = 'MANIPULATION_NOT_APPLIED', 'hold_neutral(P90) < 0.5: the gate is off at this profile'
    else:
        verdict, why = 'KISMEN', '(5, 20] loss band or other (rule §3b-iv)'
    return dict(verdict=verdict, reason=why, hold_neutral_p90=hold_n, l12_loss_p90=loss, sign_ok=bool(sign), flags=flags)


def combine(cells):
    """Rule §3c, literally: valid subsets only; perm4 needs >= 2 valid cells; a refuted valid cell blocks arm support;
    only both arms DESTEKLENİR yields the overall DESTEKLENİR; hold_nc is excluded (Z7)."""
    arms = {}
    for task in ('perm4', 'perm5'):
        cs = [c for c in cells if c.get('task') == task and c.get('arm') == 'hold']
        valids = [c for c in cs if c.get('valid')]
        labels = [c['reading']['verdict'] for c in valids]
        if len(valids) < 2:
            arms[task] = dict(verdict='INCOMPLETE', reason=f'{len(valids)} valid cells (>= 2 required)',
                              valid_cells=[c['id'] for c in valids])
            continue
        verdict = labels[0] if all(l == labels[0] for l in labels) else 'KISMEN'      # 2/2 same -> that, split -> KISMEN
        if verdict == 'DESTEKLENİR' and 'ÇÜRÜTÜLDÜ' in labels:
            verdict = 'KISMEN'
        arms[task] = dict(verdict=verdict, labels=dict(zip([c['id'] for c in valids], labels)),
                          valid_cells=[c['id'] for c in valids],
                          refuted=[c['id'] for c, l in zip(valids, labels) if l == 'ÇÜRÜTÜLDÜ'])
    v4, v5 = arms['perm4']['verdict'], arms['perm5']['verdict']
    overall = 'INCOMPLETE' if 'INCOMPLETE' in (v4, v5) else ('DESTEKLENİR' if v4 == v5 == 'DESTEKLENİR' else 'KISMEN')
    return dict(perm4=v4, perm5=v5, overall=overall, arms=arms,
                note='section 3c literal: overall DESTEKLENİR only when both arms are; a refuted cell is named and blocks its arm')


def run_session(weights, n=200, seed=1, batch=50, k=K_DEFAULT, layout='gaps',
                profiles=('P0', 'P50', 'P90', 'P97'), log=print):
    drawn, cells = {}, []
    for w in weights:
        task = torch.load(w, weights_only=True)['meta']['task']
        if task not in drawn:
            drawn[task] = draw(task, seed, n, k=k, layout=layout)
            log(f'L12 draw {task}: base_seed={drawn[task]["base_seed"]} (n={n}, k={k}, layout={layout})')
        cell = measure_cell(w, drawn, n=n, batch=batch, profiles=profiles)
        cells.append(cell)
        p90 = cell['profiles']['P90']
        log(f"{cell['id']}: {cell['reading']['verdict']} | valid={cell['valid']} | L12-kayip(P90)={p90.get('l12_loss_points')} "
            f"hold(P90)={p90.get('hold_neutral')} sign_ok={p90.get('sign_ok')}")
    combo = combine(cells)
    return dict(schema='l12-e/v1', created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                 rule='notes/_hive_L12_KURAL.md', k=k, layout=layout, draw_tag=DRAW_TAG,
                 profile_table=[dict(name=p, n_neutral=nz, length=L, density=nz / L) for p, nz, L in PROFILES],
                 draws={t: dict(task=t, base_seed=d['base_seed']) for t, d in drawn.items()},
                 n=n, batch=batch, gate_seqs=GATE_SEQS, eval_seed=seed, profiles_measured=list(profiles),
                 cells=cells, combination=combo)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('weights', nargs='*')
    p.add_argument('--out', default=None)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batch', type=int, default=50)
    p.add_argument('--k', type=int, default=K_DEFAULT)
    p.add_argument('--blocks-p90', action='store_true', help='secondary descriptive contrast: P90 block layout (measures P0 + P90 only)')
    p.add_argument('--reading', default=None, help='an existing session JSON: print the cell labels and the combination again')
    a = p.parse_args()
    if a.reading:
        session = json.loads(Path(a.reading).read_text(encoding='utf8'))
        for c in session['cells']:
            r = cell_reading(c)
            print(f"{c['id']}: {r['verdict']} ({r.get('reason')}) flags={r.get('flags') or '-'} valid={c.get('valid')}")
        print(json.dumps(combine(session['cells']), ensure_ascii=False, indent=1))
        return
    if not a.weights:
        raise SystemExit('give at least one checkpoint (or --reading <session.json>)')
    if a.out and Path(a.out).exists():
        raise SystemExit(f'{a.out} exists: a session is written once (EG-4/Z10 spirit)')
    profiles = ('P0', 'P90') if a.blocks_p90 else ('P0', 'P50', 'P90', 'P97')
    session = run_session(a.weights, n=a.n, seed=a.seed, batch=a.batch, k=a.k,
                          layout='blocks' if a.blocks_p90 else 'gaps', profiles=profiles)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding='utf8')
    print(json.dumps(session['combination'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
