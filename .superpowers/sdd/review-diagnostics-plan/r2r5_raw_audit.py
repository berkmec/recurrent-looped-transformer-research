"""R2-R5 independent raw audit. READ-ONLY, CPU only, one torch thread.

Recomputes every number of the sealed campaign under results/training from the
raw files (frozen.json, train.json, complete.json, updates.jsonl, timing.jsonl,
*.pt, *.npz, *-evaluation.json, selection.json, test-gate.json, workers,
processes) and only then compares against the producer's evaluation.json,
summary.json and scientific-analysis.json. Does NOT import review_cycle.* or
core.*; does NOT run any model forward pass; does NOT launch anything.

Writes only:
  .superpowers/sdd/review-diagnostics-plan/r2r5-independent-audit.json
"""
import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path('E:/rlt-research/2026-09-17-diagnostics')
OUT = ROOT / 'results' / 'training'
DEST = ROOT / '.superpowers' / 'sdd' / 'review-diagnostics-plan' / 'r2r5-independent-audit.json'
PLAN_PIN = '1949d8d7ceda69a4f1a3bf8bf6b9b7ff9460e0f45d1ff6c3f7a7acc7ce421030'
PROTOCOL_PIN_PREFIX = 'eed6fb30'
T95 = 4.302652729
SOURCES = ('core/model.py', 'review_cycle/training.py', 'review_cycle/training_data.py')
STEPS, BATCH, LEN, N_TEST, N_VAL = 2048, 32, 32, 512, 256
ENDPOINTS = ['L32', 'L128', 'L1024', 'neutral-0-base'] + [
    f'neutral-{s}-{p}' for s in (8, 32, 128) for p in ('prefix', 'middle', 'suffix')]
IDENT = re.compile(r'^(small|development|confirm)-(parity|five)-w(\d+)-(rlt|nofeedback|tbptt8)-'
                   r'(random|blocked)-(dense|sparse|final)-lr([0-9.]+)-s(\d+)$')

CHECKS = {}
ANOMALIES = []


def check(name, ok, **detail):
    CHECKS[name] = dict(status='PASS' if ok else 'FAIL', **detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return ok


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def np_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(('torch.' + str(a.dtype), tuple(a.shape))).encode() + a.tobytes()).hexdigest()


def state_hash(value):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(tensor_hash(x).encode())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def parse(cid):
    m = IDENT.match(cid)
    if not m:
        return None
    stage, task, width, mode, arrangement, supervision, lr, seed = m.groups()
    return dict(stage=stage, task=task, width=int(width), heads=2 if int(width) == 32 else 4, mode=mode,
                arrangement=arrangement, supervision=supervision, lr=float(lr), seed=int(seed))


def ident(c):
    return f"{c['stage']}-{c['task']}-w{c['width']}-{c['mode']}-{c['arrangement']}-{c['supervision']}-lr{c['lr']:g}-s{c['seed']}"


# ---------------------------------------------------------------- data sources (independent re-implementation)
def prefix_targets(x, task):
    x = np.asarray(x)
    s = np.zeros(x.shape[0], dtype=np.int64)
    out = np.empty(x.shape, dtype=np.int64)
    for t in range(x.shape[1]):
        tok = x[:, t]
        if task == 'parity':
            s = s ^ (tok == 1).astype(np.int64)
        else:  # five: token0 -> s+1 mod 5; token1 -> swap states 0<->1; token2 -> identity
            s = np.where(tok == 0, (s + 1) % 5, np.where(tok == 1, np.where(s == 0, 1, np.where(s == 1, 0, s)), s))
        out[:, t] = s
    return out


def generate(task, n, length, seed):
    x = np.random.default_rng(seed).choice(3, (n, length), p=[.4375, .4375, .125]).astype(np.int64)
    return x, prefix_targets(x, task)


def source_seed(bases, split, task, seed, step=0, length=LEN):
    value = f"{bases[split]}|{task}|{seed}|{step}|{length}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def block_neutrals(x, start_index):
    rows = []
    for i, row in enumerate(x):
        active, neutral = row[row != 2], row[row == 2]
        at = (0, len(active) // 2, len(active))[(start_index + i) % 3]
        rows.append(np.concatenate((active[:at], neutral, active[at:])))
    return np.stack(rows)


def insert_neutrals(x, slots, placement):
    at = {'prefix': 0, 'middle': x.shape[1] // 2, 'suffix': x.shape[1], 'base': x.shape[1]}[placement]
    return np.concatenate((x[:, :at], np.full((len(x), slots), 2, dtype=x.dtype), x[:, at:]), 1)


def pack_rows(x):
    w = 3 ** np.arange(x.shape[1], dtype=np.int64)
    return set((x.astype(np.int64) * w).sum(1).tolist())


# ---------------------------------------------------------------- 1. integrity
t0 = time.time()
frozen = json.loads((OUT / 'frozen.json').read_text())
P = frozen['protocol']
BASES = P['source_seeds']
check('1.frozen_sha256', sha_file(OUT / 'frozen.json') == (OUT / 'frozen.sha256').read_text().strip(),
      recomputed=sha_file(OUT / 'frozen.json'), pinned=(OUT / 'frozen.sha256').read_text().strip())
check('1.protocol_hash_canonical_json', digest(P) == frozen['protocol_hash'] and frozen['protocol_hash'].startswith(PROTOCOL_PIN_PREFIX),
      recomputed=digest(P), declared=frozen['protocol_hash'])
src_now = {s: sha_file(ROOT / s) for s in SOURCES}
check('1.source_pins_match_current_files', src_now == frozen['sources'], current=src_now, frozen=frozen['sources'])
plan_sha = sha_file(ROOT / 'docs/training-analysis-plan.md')
ledger = (ROOT / '.superpowers/sdd/review-diagnostics-plan/progress.md').read_text(encoding='utf8', errors='replace')
ledger_line = next((i + 1 for i, l in enumerate(ledger.splitlines()) if 'Analysisplan SHA' + PLAN_PIN in l), None)
check('1.analysis_plan_pin', plan_sha == PLAN_PIN and ledger_line is not None, current_sha256=plan_sha, pin=PLAN_PIN, ledger_line=ledger_line)
expected_protocol = dict(steps=2048, batch=32, train_length=32, small_width=32, large_width=256, small_lr=.003,
                         lr_grid=[.0003, .001, .003], primary_seeds=[117, 143, 191], confirm_seeds=[217, 243, 291],
                         validation_n=256, validation_every=128, checkpoint_every=64, equal_budget_step=64, test_n=512,
                         test_lengths=[32, 128, 1024], neutral_slots=[0, 8, 32, 128], eval_batch_short=32, eval_batch_long=8,
                         eval_long_threshold=512, threads=1, precision='float32')
check('1.protocol_values_as_stated', all(P[k] == v for k, v in expected_protocol.items()),
      mismatches={k: P.get(k) for k, v in expected_protocol.items() if P.get(k) != v})

# workers / processes
worker_summary, failed, cond_complete = {}, [], {}
for ev in sorted((OUT / 'workers').glob('*/events.jsonl')):
    rows = [json.loads(l) for l in ev.read_text(encoding='utf8').splitlines()]
    states = [r['state'] for r in rows]
    failed += [(ev.parent.name, r) for r in rows if r['state'] == 'FAILED']
    for r in rows:
        if r['state'] == 'CONDITION_COMPLETE':
            cond_complete[r['condition_id']] = r['updates']
    worker_summary[ev.parent.name] = dict(events=len(rows), first=states[0], last=states[-1],
                                          protocol_hash_ok=rows[0].get('protocol_hash') == frozen['protocol_hash'])
check('1.workers_no_failed_all_complete', not failed and all(w['last'] == 'COMPLETE' and w['first'] == 'STARTED' and w['protocol_hash_ok'] for w in worker_summary.values()),
      workers=worker_summary, failed=failed, condition_complete_events=len(cond_complete),
      all_condition_complete_updates_2048=all(v == 2048 for v in cond_complete.values()))
stderr = {p.name: p.stat().st_size for p in (OUT / 'processes').glob('*.stderr.log')}
launch_logs = {p.name: p.read_text().strip().splitlines() for p in (OUT / 'processes').glob('launch-*.log')}
check('1.stderr_logs_empty', len(stderr) == 5 and all(v == 0 for v in stderr.values()), sizes=stderr, launch_logs=launch_logs,
      exit_codes_zero=all('code=0' in l for lines in launch_logs.values() for l in lines if 'exited' in l))

# ---------------------------------------------------------------- 2. census
dirs = sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name not in ('workers', 'processes'))
parsed = {d: parse(d) for d in dirs}
check('2.all_directories_parse_as_conditions', all(v is not None and ident(v) == d for d, v in parsed.items()), unparsed=[d for d, v in parsed.items() if v is None])
conds = {d: v for d, v in parsed.items() if v}

# selection rule recomputed from the nine development receipts (before building the expected census)
dev_receipts = {d: json.loads((OUT / d / 'train.json').read_text()) for d, c in conds.items() if c['stage'] == 'development'}
lr_scores = {}
for d, r in dev_receipts.items():
    lr_scores.setdefault(str(r['condition']['lr']), []).append(r['selected_validation']['final_accuracy'])
lr_mean = {k: float(np.mean(v)) for k, v in lr_scores.items()}
my_selected_lr = min(P['lr_grid'], key=lambda lr: (-lr_mean[str(lr)], lr))
selection = json.loads((OUT / 'selection.json').read_text())
sel_body = {k: v for k, v in selection.items() if k != 'selection_hash'}
check('1.selection_json', digest(sel_body) == selection['selection_hash'] and selection['selected_lr'] == my_selected_lr
      and selection['mean_validation_accuracy'] == lr_mean and len(dev_receipts) == 9
      and all(len(v) == 3 for v in lr_scores.values())
      and selection['development_selected_hashes'] == {d: r['selected_state_hash'] for d, r in dev_receipts.items()}
      and sorted(selection['development_conditions']) == sorted(dev_receipts)
      and selection['protocol_hash'] == frozen['protocol_hash'] and selection['sources'] == frozen['sources']
      and selection['final_data_opened'] is False,
      recomputed_selection_hash=digest(sel_body), declared_selection_hash=selection['selection_hash'],
      recomputed_mean_validation_accuracy=lr_mean, per_lr_dev_accuracies=lr_scores,
      recomputed_selected_lr=my_selected_lr, declared_selected_lr=selection['selected_lr'])

expected = set()
for seed in (117, 143, 191):
    for task in ('parity', 'five'):
        for arr in ('random', 'blocked'):
            expected.add(f'small-{task}-w32-rlt-{arr}-dense-lr0.003-s{seed}')
    for mode in ('rlt', 'nofeedback'):
        for sup in ('dense', 'sparse', 'final'):
            expected.add(f'small-parity-w32-{mode}-random-{sup}-lr0.003-s{seed}')
    for mode in ('rlt', 'tbptt8'):
        for sup in ('dense', 'final'):
            expected.add(f'small-five-w32-{mode}-random-{sup}-lr0.003-s{seed}')
    for lr in (0.0003, 0.001, 0.003):
        expected.add(f'development-parity-w256-rlt-random-dense-lr{lr:g}-s{seed}')
for seed in (217, 243, 291):
    expected.add(f'confirm-parity-w256-rlt-random-dense-lr{my_selected_lr:g}-s{seed}')
stage_counts = {s: sum(1 for c in conds.values() if c['stage'] == s) for s in ('small', 'development', 'confirm')}
check('2.condition_census', set(conds) == expected and stage_counts == dict(small=36, development=9, confirm=3) and len(dirs) == 48,
      counts=stage_counts, directories=len(dirs), missing=sorted(expected - set(conds)), unexpected=sorted(set(conds) - expected))

# ---------------------------------------------------------------- 3. real training, per-condition receipts/seals/journals
base_cache = {}


def base_batch(task, seed, step):
    key = (task, seed, step)
    if key not in base_cache:
        s = source_seed(BASES, 'train', task, seed, step)
        x, y = generate(task, BATCH, LEN, s)
        base_cache[key] = (s, x, y, np_hash(x))
    return base_cache[key]


COND = {}
journal_base, journal_input, journal_sparse = {}, {}, {}
train_rows = {}
file_sha = {}
for cid in sorted(conds):
    c = conds[cid]
    d = OUT / cid
    R = json.loads((d / 'train.json').read_text())
    S = json.loads((d / 'complete.json').read_text())
    info = dict(problems=[])
    P_here = dict(P); P_here['device'] = P['devices'][c['stage']]
    exp_files = {'initial.pt', 'resume.pt', 'selected.pt', 'last.pt', 'train.json', 'timing.jsonl', 'updates.jsonl'}
    if c['supervision'] == 'dense':
        exp_files.add('equal-label.pt')
    shas = {f: sha_file(d / f) for f in S['files']}
    file_sha[cid] = shas
    if shas != S['files']:
        info['problems'].append('complete.json file sha mismatch: ' + str([f for f in shas if shas[f] != S['files'][f]]))
    if set(S['files']) != exp_files:
        info['problems'].append('complete.json file set unexpected')
    if S['seal_hash'] != digest({k: v for k, v in S.items() if k != 'seal_hash'}):
        info['problems'].append('seal_hash does not recompute')
    if S['protocol_hash'] != frozen['protocol_hash'] or S['condition'] != c:
        info['problems'].append('seal protocol/condition mismatch')
    if R['protocol_hash'] != frozen['protocol_hash'] or R['sources'] != frozen['sources'] or R['condition'] != c:
        info['problems'].append('receipt protocol/sources/condition mismatch')
    if any(S['files'].get(f) != v for f, v in R['checkpoint_hashes'].items()) or set(R['checkpoint_hashes']) != {p.name for p in d.glob('*.pt')}:
        info['problems'].append('checkpoint_hashes vs seal/directory mismatch')
    if R['status'] != 'COMPLETE' or R['updates'] != STEPS or R['finite_updates'] != STEPS or R['timed_updates'] != STEPS:
        info['problems'].append('receipt update counters not 2048')
    exp_labels = STEPS * BATCH * (LEN if c['supervision'] == 'dense' else 1)
    if R['labels'] != exp_labels or R['input_tokens'] != STEPS * BATCH * LEN or R['input_sequences'] != STEPS * BATCH:
        info['problems'].append(f"receipt labels {R['labels']} != {exp_labels}")
    if R['execution_device'] != P['devices'][c['stage']]:
        info['problems'].append('execution_device differs from frozen devices map')
    stray = [p.name for p in d.iterdir() if p.name in ('active.lock', 'inflight.json', 'uncheckpointed.json', 'finalizing.json')
             or p.suffix == '.tmp' or p.name.startswith('failure-step')]
    if stray:
        info['problems'].append('stray files: ' + str(stray))
    if c['stage'] == 'development':
        leaks = [p.name for p in d.iterdir() if p.name.endswith('-evaluation.json') or p.suffix == '.npz']
        if leaks:
            info['problems'].append('development condition has evaluation files: ' + str(leaks))
    # journals
    rows = [json.loads(l) for l in (d / 'updates.jsonl').read_text(encoding='utf8').splitlines()]
    tim = [json.loads(l) for l in (d / 'timing.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != STEPS or [r['step'] for r in rows] != list(range(1, STEPS + 1)) or [r['real_update'] for r in rows] != list(range(1, STEPS + 1)):
        info['problems'].append('updates.jsonl step/real_update sequence not 1..2048')
    if not all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows):
        info['problems'].append('nonfinite loss/grad_norm in journal')
    if sum(r['labels'] for r in rows) != exp_labels:
        info['problems'].append('per-row label sum != expected')
    if len(tim) != STEPS or [r['step'] for r in tim] != list(range(1, STEPS + 1)) or not all(math.isfinite(r['wall_seconds']) and r['wall_seconds'] >= 0 for r in tim):
        info['problems'].append('timing.jsonl invalid')
    if abs(sum(r['wall_seconds'] for r in tim) - R['wall_seconds']) > 1e-6 or shas['timing.jsonl'] != R['timing_sha256']:
        info['problems'].append('wall_seconds / timing sha mismatch')
    bad_seed = bad_base = bad_input = bad_sparse = 0
    blocked_changed = 0
    for r in rows:
        s, x, y, bh = base_batch(c['task'], c['seed'], r['step'])
        if r['source_seed'] != s:
            bad_seed += 1
        if r['base_input_hash'] != bh:
            bad_base += 1
        if c['arrangement'] == 'blocked':
            xb = block_neutrals(x, (r['step'] - 1) * BATCH)
            if np_hash(xb) != r['input_hash']:
                bad_input += 1
            if not np.array_equal(prefix_targets(xb, c['task'])[:, -1], y[:, -1]):
                bad_input += 1
            blocked_changed += r['input_hash'] != r['base_input_hash']
        elif r['input_hash'] != bh:
            bad_input += 1
        if c['supervision'] == 'sparse':
            pos = np.random.default_rng(source_seed(BASES, 'sparse', c['task'], c['seed'], r['step'])).integers(0, LEN, size=BATCH)
            if r.get('sparse_position_hash') != np_hash(pos):
                bad_sparse += 1
    if bad_seed or bad_base or bad_input or bad_sparse:
        info['problems'].append(f'journal regeneration mismatch seed={bad_seed} base={bad_base} input={bad_input} sparse={bad_sparse}')
    info['blocked_steps_with_changed_input_hash'] = blocked_changed if c['arrangement'] == 'blocked' else None
    val = [(r['step'], r['validation']) for r in rows if 'validation' in r]
    exp_val_steps = list(range(P['validation_every'], STEPS + 1, P['validation_every']))
    if [s for s, _ in val] != exp_val_steps:
        info['problems'].append('validation rows not at every 128 steps')
    best = min(val, key=lambda sv: (sv[1]['final_loss'], sv[0]))
    if best[0] != R['selected_step'] or best[1] != R['selected_validation']:
        info['problems'].append(f"selection rule: journal argmin step {best[0]} vs receipt {R['selected_step']}")
    journal_base[cid] = [r['base_input_hash'] for r in rows]
    journal_input[cid] = [r['input_hash'] for r in rows]
    journal_sparse[cid] = [r.get('sparse_position_hash') for r in rows]
    train_rows[cid] = rows
    # checkpoints, one at a time
    last = load_pt(d / 'last.pt')
    opt_steps = sorted({float(v['step']) for v in last['optimizer']['state'].values()})
    opt_entries = len(last['optimizer']['state'])
    if last['updates'] != STEPS or opt_steps != [float(STEPS)] or opt_entries == 0:
        info['problems'].append(f'last.pt updates/optimizer steps {last["updates"]} {opt_steps}')
    if state_hash(last['model']) != last['model_hash'] or last['model_hash'] != R['last_state_hash']:
        info['problems'].append('last.pt model hash mismatch')
    if state_hash(last['optimizer']) != last['optimizer_hash'] or last['optimizer_hash'] != R['optimizer_hash']:
        info['problems'].append('last.pt optimizer hash mismatch')
    if last['history'] != rows or digest(rows) != last['history_hash']:
        info['problems'].append('last.pt history != updates.jsonl')
    if state_hash(last['selected_model']) != last['selected_hash'] or last['selected_hash'] != R['selected_state_hash']:
        info['problems'].append('last.pt selected_model hash mismatch')
    if last['selected_step'] != R['selected_step'] or last['checkpoint_steps'] != R['checkpoint_steps'] or last['seconds'] != R['train_seconds'] \
            or last['initial_state_hash'] != R['initial_state_hash'] or last['selected_validation'] != R['selected_validation'] \
            or last['condition'] != c or last['protocol_hash'] != frozen['protocol_hash'] or last['sources'] != frozen['sources']:
        info['problems'].append('last.pt receipt fields mismatch')
    last_model = last['model']; selected_model = last['selected_model']; last_opt_state = last['optimizer']['state']; del last
    sel = load_pt(d / 'selected.pt')
    if state_hash(sel) != R['selected_state_hash'] or set(sel) != set(selected_model) or not all(torch.equal(sel[k], selected_model[k]) for k in sel):
        info['problems'].append('selected.pt != last.pt selected_model / receipt')
    ini = load_pt(d / 'initial.pt')
    if state_hash(ini) != R['initial_state_hash']:
        info['problems'].append('initial.pt hash != receipt initial_state_hash')
    n_params = len(ini)
    unchanged_sel = sorted(k for k in ini if torch.equal(ini[k], sel[k]))
    unchanged_last = sorted(k for k in ini if torch.equal(ini[k], last_model[k]))
    changed_sel, changed_last = n_params - len(unchanged_sel), n_params - len(unchanged_last)
    sel_eq_last = all(torch.equal(sel[k], last_model[k]) for k in ini)
    # nofeedback leaves the feedback route (initial, pn, gate, merge) out of the graph: grad None, AdamW skips it (no state entry).
    # tbptt8 + final supervision: the last-position loss has no path to the t=0 state through the truncation, but autograd still
    # delivers an exactly-zero gradient to `initial` (via torch.stack of all positions); zero-initialised + decoupled weight decay
    # keeps it exactly zero, while an AdamW state entry with zero moments exists.
    if c['mode'] == 'nofeedback':
        inactive, exp_entries = ['gate.bias', 'gate.weight', 'initial', 'merge.weight', 'pn.weight'], n_params - 5
    elif c['mode'] == 'tbptt8' and c['supervision'] == 'final':
        inactive, exp_entries = ['initial'], n_params
    else:
        inactive, exp_entries = [], n_params
    info.update(parameter_tensors=n_params, tensors_changed_initial_vs_selected=changed_sel, tensors_changed_initial_vs_last=changed_last,
                unchanged_tensors=unchanged_last, expected_unchanged=inactive, optimizer_state_entries=opt_entries,
                selected_equals_last=sel_eq_last, numel=int(sum(v.numel() for v in ini.values())))
    if unchanged_last != inactive or unchanged_sel != inactive or opt_entries != exp_entries:
        info['problems'].append(f'unchanged tensors {unchanged_last} != expected inactive {inactive} (optimizer entries {opt_entries})')
    if inactive == ['initial']:
        idx = list(last_model).index('initial')  # optimizer param index follows named_parameters order (no buffers in this model)
        st = last_opt_state[idx]
        zero_moments = bool(torch.count_nonzero(st['exp_avg']) == 0 and torch.count_nonzero(st['exp_avg_sq']) == 0)
        info['initial_adamw_moments_exactly_zero'] = zero_moments
        info['initial_is_all_zero'] = bool(torch.count_nonzero(ini['initial']) == 0)
        if not zero_moments or not info['initial_is_all_zero']:
            info['problems'].append('tbptt8-final: `initial` unchanged but AdamW moments / value are not exactly zero')
    if sel_eq_last != (R['selected_step'] == STEPS) or (R['selected_state_hash'] == R['last_state_hash']) != sel_eq_last:
        info['problems'].append('selected/last equality inconsistent with selected_step')
    del ini, sel, last_model, selected_model, last_opt_state
    res = load_pt(d / 'resume.pt')
    r_steps = sorted({float(v['step']) for v in res['optimizer']['state'].values()})
    if res['updates'] != STEPS or r_steps != [float(STEPS)] or res['model_hash'] != R['last_state_hash'] or res['optimizer_hash'] != R['optimizer_hash'] \
            or res['history_hash'] != digest(rows) or res['selected_hash'] != R['selected_state_hash'] or state_hash(res['model']) != res['model_hash']:
        info['problems'].append('resume.pt disagrees with last.pt/receipt')
    del res
    if c['supervision'] == 'dense':
        eq = load_pt(d / 'equal-label.pt')
        e_steps = sorted({float(v['step']) for v in eq['optimizer']['state'].values()})
        if eq['updates'] != 64 or e_steps != [64.0] or eq['checkpoint_steps'] != [0, 64] or eq['history'] != rows[:64] \
                or state_hash(eq['model']) != eq['model_hash'] or eq['initial_state_hash'] != R['initial_state_hash'] or R['equal_label_companion_step'] != 64:
            info['problems'].append('equal-label.pt is not the exact step-64 snapshot')
        info['equal_label_model_hash'] = eq['model_hash']
        del eq
    info.update(selected_step=R['selected_step'], selected_validation_final_accuracy=R['selected_validation']['final_accuracy'],
                selected_validation_final_loss=R['selected_validation']['final_loss'], initial_state_hash=R['initial_state_hash'],
                selected_state_hash=R['selected_state_hash'], last_state_hash=R['last_state_hash'], execution_device=R['execution_device'],
                wall_seconds=R['wall_seconds'], train_seconds=R['train_seconds'], labels=R['labels'],
                loss_first=rows[0]['loss'], loss_last=rows[-1]['loss'], grad_norm_max=max(r['grad_norm'] for r in rows))
    COND[cid] = info
    print(f"  cond {cid}: {'ok' if not info['problems'] else info['problems']}", flush=True)

check('1.receipts_seals_checkpoint_hashes_48', all(not v['problems'] for v in COND.values()), problems={k: v['problems'] for k, v in COND.items() if v['problems']})
check('3.real_training_2048_updates_optimizer_step_params_changed',
      all(not v['problems'] and v['tensors_changed_initial_vs_last'] == v['parameter_tensors'] - len(v['expected_unchanged']) for v in COND.values()),
      conditions=len(COND), min_tensors_changed=min(v['tensors_changed_initial_vs_selected'] for v in COND.values()),
      changed_by_mode={m: sorted({v['tensors_changed_initial_vs_last'] for k, v in COND.items() if conds[k]['mode'] == m}) for m in ('rlt', 'nofeedback', 'tbptt8')},
      optimizer_entries_by_mode={m: sorted({v['optimizer_state_entries'] for k, v in COND.items() if conds[k]['mode'] == m}) for m in ('rlt', 'nofeedback', 'tbptt8')},
      labels_dense=sorted({v['labels'] for k, v in COND.items() if conds[k]['supervision'] == 'dense'}),
      labels_sparse_final=sorted({v['labels'] for k, v in COND.items() if conds[k]['supervision'] != 'dense'}))
check('1.no_development_evaluation_files', not any('evaluation files' in p for v in COND.values() for p in v['problems']))
check('3.selected_step_and_selection_rule', all(not any('selection rule' in p for p in v['problems']) for v in COND.values()),
      selected_steps={k: v['selected_step'] for k, v in COND.items()},
      not_2048=sorted(k for k, v in COND.items() if v['selected_step'] != STEPS))

summary = json.loads((OUT / 'summary.json').read_text())
srows = {r['id']: {k: v for k, v in r.items() if k != 'id'} for r in summary['rows']}
check('1.summary_json_matches_receipts', len(srows) == 48 and summary['complete'] == 48 and summary['final_evaluation'] is True
      and all(srows[cid] == json.loads((OUT / cid / 'train.json').read_text()) for cid in conds), rows=len(srows))

gate = json.loads((OUT / 'test-gate.json').read_text())
evaluated = sorted(k for k, c in conds.items() if c['stage'] != 'development')
check('1.test_gate_selections', gate['status'] == 'COMPLETE_SCOPE' and gate['protocol_hash'] == frozen['protocol_hash']
      and sorted(gate['selections']) == evaluated and all(gate['selections'][k] == COND[k]['selected_state_hash'] for k in evaluated),
      evaluated_conditions=len(evaluated), gate_entries=len(gate['selections']))

# ---------------------------------------------------------------- 4. pairing
pairing = {}
for seed in (117, 143, 191):
    groups = {
        f'parity-s{seed}': [f'small-parity-w32-{m}-random-{s}-lr0.003-s{seed}' for m in ('rlt', 'nofeedback') for s in ('dense', 'sparse', 'final')] + [f'small-parity-w32-rlt-blocked-dense-lr0.003-s{seed}'],
        f'five-s{seed}': [f'small-five-w32-{m}-random-{s}-lr0.003-s{seed}' for m in ('rlt', 'tbptt8') for s in ('dense', 'final')] + [f'small-five-w32-rlt-blocked-dense-lr0.003-s{seed}'],
        f'development-s{seed}': [f'development-parity-w256-rlt-random-dense-lr{lr:g}-s{seed}' for lr in (0.0003, 0.001, 0.003)],
    }
    for g, arms in groups.items():
        random_arms = [a for a in arms if conds[a]['arrangement'] == 'random']
        blocked_arms = [a for a in arms if conds[a]['arrangement'] == 'blocked']
        sparse_arms = [a for a in arms if conds[a]['supervision'] == 'sparse']
        same_init = len({COND[a]['initial_state_hash'] for a in arms}) == 1
        same_base = all(journal_base[a] == journal_base[arms[0]] for a in arms)
        same_input_random = all(journal_input[a] == journal_input[random_arms[0]] for a in random_arms)
        random_input_is_base = all(journal_input[a] == journal_base[a] for a in random_arms)
        blocked_diff = {a: sum(i != b for i, b in zip(journal_input[a], journal_base[a])) for a in blocked_arms}
        same_sparse = all(journal_sparse[a] == journal_sparse[sparse_arms[0]] for a in sparse_arms) if sparse_arms else None
        pairing[g] = dict(arms=arms, same_initial_state=same_init, same_base_input_hash_all_2048=same_base,
                          same_effective_input_hash_random_arms=same_input_random, random_effective_equals_base=random_input_is_base,
                          blocked_steps_effective_differs_from_base=blocked_diff, same_sparse_position_hashes=same_sparse,
                          initial_state_hash=COND[arms[0]]['initial_state_hash'])
    pairing[f'parity-s{seed}']['development_shares_training_inputs_with_small'] = journal_base[f'development-parity-w256-rlt-random-dense-lr0.0003-s{seed}'] == journal_base[f'small-parity-w32-rlt-random-dense-lr0.003-s{seed}']
    pairing[f'parity-s{seed}']['development_initial_state_differs_from_small'] = COND[f'development-parity-w256-rlt-random-dense-lr0.0003-s{seed}']['initial_state_hash'] != COND[f'small-parity-w32-rlt-random-dense-lr0.003-s{seed}']['initial_state_hash']
check('4.pairing_initial_state_and_inputs', all(v['same_initial_state'] and v['same_base_input_hash_all_2048'] and v['same_effective_input_hash_random_arms']
      and v['random_effective_equals_base'] and all(n == STEPS for n in v['blocked_steps_effective_differs_from_base'].values())
      and v['same_sparse_position_hashes'] in (True, None) for v in pairing.values()), groups=pairing)
check('4.distinct_seeds_distinct_initial_and_inputs',
      len({COND[f'small-parity-w32-rlt-random-dense-lr0.003-s{s}']['initial_state_hash'] for s in (117, 143, 191)}) == 3
      and len({journal_base[f'small-parity-w32-rlt-random-dense-lr0.003-s{s}'][0] for s in (117, 143, 191)}) == 3
      and len({COND[f'confirm-parity-w256-rlt-random-dense-lr{my_selected_lr:g}-s{s}']['initial_state_hash'] for s in (217, 243, 291)}) == 3)

# ---------------------------------------------------------------- 5. evaluation recomputation
evaluation = json.loads((OUT / 'evaluation.json').read_text())
expected_source = {}


def expected_endpoint(task, seed, endpoint):
    key = (task, seed, endpoint)
    if key in expected_source:
        return expected_source[key]
    if endpoint.startswith('L'):
        L = int(endpoint[1:])
        s = source_seed(BASES, 'test', task, seed, 0, L)
        x, y = generate(task, N_TEST, L, s)
    else:
        _, slots, place = endpoint.split('-')
        s = source_seed(BASES, 'neutral', task, seed, 0, LEN)
        xb, yb = generate(task, N_TEST, LEN, s)
        x = insert_neutrals(xb, int(slots), place)
        y = prefix_targets(x, task)
        assert np.array_equal(y[:, -1], yb[:, -1])
    ids = np.asarray([hashlib.sha256(f"{s}:{i}".encode()).hexdigest() for i in range(N_TEST)])
    expected_source[key] = dict(seed=s, x_sha=np_hash(x), y_sha=np_hash(y), ids_sha=hashlib.sha256(ids.tobytes()).hexdigest(), shape=x.shape)
    return expected_source[key]


RAW = {}  # (variant_key, endpoint) -> dict(final_pred, final_target, x_sha, y_sha, ids_sha)
EVAL = {}
max_float_diff = 0.0
endpoints_checked = 0
for cid in evaluated:
    c = conds[cid]
    variants = ['selected'] + (['equal-label'] if c['supervision'] == 'dense' else [])
    for variant in variants:
        key = cid if variant == 'selected' else cid + '-equal-label'
        row = json.loads((OUT / cid / f'{variant}-evaluation.json').read_text())
        problems = []
        if evaluation['conditions'].get(key) != row:
            problems.append('evaluation.json aggregate differs from per-condition file')
        if row['metadata_hash'] != digest({k: v for k, v in row.items() if k != 'metadata_hash'}):
            problems.append('metadata_hash does not recompute')
        if row['checkpoint_sha256'] != file_sha[cid][variant + '.pt'] or row['condition'] != c or row['variant'] != variant or row['protocol_hash'] != frozen['protocol_hash']:
            problems.append('checkpoint/condition/variant/protocol identity mismatch')
        records = [(f"L{r['length']}", r) for r in row['lengths']] + [(f"neutral-{r['slots']}-{r['placement']}", r) for r in row['neutral']]
        if [e for e, _ in records] != ENDPOINTS:
            problems.append('endpoint inventory differs')
        base_pred = None
        ep_summary = {}
        for ep, r in records:
            fname = f"{cid}/{variant}-{ep if ep.startswith('L') else 'neutral' + ep.split('-')[1] + '-' + ep.split('-')[2]}.npz"
            path = OUT / fname
            ep_problems = []
            if r['raw_file'] != fname:
                ep_problems.append('raw_file name')
            if sha_file(path) != r['sha256']:
                ep_problems.append('raw file sha256')
            with np.load(path, allow_pickle=False) as z:
                x, pred, y, ids = z['inputs'], z['predictions'], z['targets'], z['input_ids']
                src_seed, cond_name = str(z['source_seed'].item()), str(z['condition'].item())
            exp = expected_endpoint(c['task'], c['seed'], ep)
            if np_hash(x) != exp['x_sha']:
                ep_problems.append('inputs != regenerated declared source')
            if np_hash(y) != exp['y_sha'] or not np.array_equal(prefix_targets(x, c['task']), y):
                ep_problems.append('targets != independent oracle')
            if hashlib.sha256(ids.tobytes()).hexdigest() != exp['ids_sha'] or src_seed != str(exp['seed']) or r['source_seed'] != exp['seed'] or cond_name != cid:
                ep_problems.append('input_ids/source_seed/condition identity')
            classes = 2 if c['task'] == 'parity' else 5
            if pred.shape != y.shape or not np.issubdtype(pred.dtype, np.integer) or pred.min() < 0 or pred.max() >= classes:
                ep_problems.append('prediction shape/dtype/range')
            if ep.startswith('neutral') and ep != 'neutral-0-base':
                slots, place = int(ep.split('-')[1]), ep.split('-')[2]
                bx = RAW[(key, 'neutral-0-base')]['x']
                if not np.array_equal(x, insert_neutrals(bx, slots, place)) or not np.array_equal(y[:, -1], RAW[(key, 'neutral-0-base')]['final_target']):
                    ep_problems.append('neutral construction from base')
            fin = float(np.mean(pred[:, -1] == y[:, -1])); pre = float(np.mean(pred == y))
            th = {str(i): int(np.count_nonzero(y[:, -1] == i)) for i in range(classes)}
            ph = {str(i): int(np.count_nonzero(pred[:, -1] == i)) for i in range(classes)}
            derived = dict(n=len(x), consumed_tokens=x.shape[1], input_hash=np_hash(x), target_histogram=th, prediction_histogram=ph,
                           batch=8 if x.shape[1] >= 512 else 32, source_seed=exp['seed'])
            if ep.startswith('neutral') and ep != 'neutral-0-base':
                was, now = base_pred == y[:, -1], pred[:, -1] == y[:, -1]
                derived.update(correct_to_wrong=int(np.count_nonzero(was & ~now)), wrong_to_correct=int(np.count_nonzero(~was & now)),
                               prediction_changes=int(np.count_nonzero(base_pred != pred[:, -1])))
            for k, v in derived.items():
                if r.get(k) != v:
                    ep_problems.append(f'{k}: file {r.get(k)} vs raw {v}')
            dfin, dpre = abs(r['final_accuracy'] - fin), abs(r['prefix_accuracy'] - pre)
            max_float_diff = max(max_float_diff, dfin, dpre)
            if dfin > 1e-12 or dpre > 1e-12:
                ep_problems.append(f'accuracy: file {r["final_accuracy"]}/{r["prefix_accuracy"]} vs raw {fin}/{pre}')
            if any(not math.isfinite(r[k]) or r[k] < 0 for k in ('seconds', 'prefix_loss', 'final_loss')):
                ep_problems.append('nonfinite/negative loss or seconds')
            if ep == 'neutral-0-base':
                base_pred = pred[:, -1].copy()
            RAW[(key, ep)] = dict(final_pred=pred[:, -1].copy(), final_target=y[:, -1].copy(), x_sha=np_hash(x), y_sha=np_hash(y),
                                  ids_sha=hashlib.sha256(ids.tobytes()).hexdigest(), x=x if ep == 'neutral-0-base' else None,
                                  rows32=pack_rows(x) if x.shape[1] == 32 else None)
            constant = int(np.count_nonzero(list(ph.values()))) == 1
            ep_summary[ep] = dict(final_accuracy=fin, prefix_accuracy=pre, prediction_histogram=ph, target_histogram=th, constant_prediction=constant,
                                  **({k: derived[k] for k in ('correct_to_wrong', 'wrong_to_correct', 'prediction_changes')} if 'correct_to_wrong' in derived else {}))
            if constant:
                ANOMALIES.append(f'constant-class prediction: {key} {ep} histogram {ph}')
            if ep_problems:
                problems.append(f'{ep}: ' + '; '.join(ep_problems))
            endpoints_checked += 1
        EVAL[key] = dict(problems=problems, endpoints=ep_summary)
        print(f"  eval {key}: {'ok' if not problems else problems}", flush=True)
for v in RAW.values():
    v['x'] = None
check('5.evaluation_recomputed_780_endpoints', all(not v['problems'] for v in EVAL.values()) and len(EVAL) == 60 and endpoints_checked == 780
      and set(evaluation['conditions']) == set(EVAL) and evaluation['status'] == 'COMPLETE' and evaluation['protocol_hash'] == frozen['protocol_hash'],
      variants=len(EVAL), endpoints=endpoints_checked, max_abs_accuracy_diff=max_float_diff, problems={k: v['problems'] for k, v in EVAL.items() if v['problems']})

# shared source identity across conditions
by_task_seed_ep, by_task_ep = {}, {}
for (key, ep), v in RAW.items():
    c = conds[key.replace('-equal-label', '')]
    by_task_seed_ep.setdefault((c['task'], c['seed'], ep), set()).add((v['x_sha'], v['y_sha'], v['ids_sha']))
    by_task_ep.setdefault((c['task'], ep), set()).add(v['x_sha'])
check('5.inputs_identical_within_task_seed_endpoint', all(len(s) == 1 for s in by_task_seed_ep.values()),
      groups=len(by_task_seed_ep), variants_per_group_example=sorted({k[0] for k in RAW})[:3],
      identical_across_seeds_of_same_task=all(len(s) == 1 for s in by_task_ep.values()),
      note='source seed is derived from (split, task, training seed, length); test inputs are shared by all arms of one task+seed, and differ across seeds by design')

# heldout vs training disjointness (length-32 rows: test L32, neutral base, validation)
disjoint = {}
for task, seed in sorted({(c['task'], c['seed']) for k, c in conds.items() if c['stage'] != 'development'}):
    train_set = set()
    for step in range(1, STEPS + 1):
        train_set |= pack_rows(base_batch(task, seed, step)[1])
    vx, _ = generate(task, N_VAL, LEN, source_seed(BASES, 'validation', task, seed, 0, LEN))
    val_set = pack_rows(vx)
    any_key = next(k for k in RAW if conds[k[0].replace('-equal-label', '')]['task'] == task and conds[k[0].replace('-equal-label', '')]['seed'] == seed and k[1] == 'L32')
    test32 = RAW[any_key]['rows32']
    base32 = RAW[(any_key[0], 'neutral-0-base')]['rows32']
    disjoint[f'{task}-s{seed}'] = dict(train_rows=len(train_set), train_unique_rows=len(train_set), test32_rows=len(test32), base32_rows=len(base32), validation_rows=len(val_set),
                                       test32_in_train=len(test32 & train_set), base32_in_train=len(base32 & train_set), validation_in_train=len(val_set & train_set),
                                       test32_in_validation=len(test32 & val_set), base32_in_validation=len(base32 & val_set), test32_in_base32=len(test32 & base32))
check('5.heldout_disjoint_from_training_L32', all(v['test32_in_train'] == 0 and v['base32_in_train'] == 0 and v['validation_in_train'] == 0
      and v['test32_in_validation'] == 0 and v['base32_in_validation'] == 0 for v in disjoint.values()), streams=disjoint)

# ---------------------------------------------------------------- 6. contrasts
analysis = json.loads((OUT / 'scientific-analysis.json').read_text())
spec = []
for task in ('parity', 'five'):
    for ep in ENDPOINTS:
        spec.append(('R2', 'blocked-minus-random', task, ep, ('rlt', 'blocked', 'dense'), ('rlt', 'random', 'dense'), 'selected', 'selected', ep == 'neutral-128-suffix'))
for mode in ('rlt', 'nofeedback'):
    for L in (32, 128, 1024):
        spec.append(('R3', f'{mode}-final-minus-sparse', 'parity', f'L{L}', (mode, 'random', 'final'), (mode, 'random', 'sparse'), 'selected', 'selected', L == 128))
    for sup in ('sparse', 'final'):
        for L in (32, 128, 1024):
            spec.append(('R3', f'{mode}-{sup}-minus-dense', 'parity', f'L{L}', (mode, 'random', sup), (mode, 'random', 'dense'), 'selected', 'selected', L == 128))
            spec.append(('R3', f'{mode}-{sup}2048-minus-dense64', 'parity', f'L{L}', (mode, 'random', sup), (mode, 'random', 'dense'), 'selected', 'equal-label', False))
for sup in ('dense', 'sparse', 'final'):
    for L in (32, 128, 1024):
        spec.append(('R3', f'rlt-minus-nofeedback-{sup}', 'parity', f'L{L}', ('rlt', 'random', sup), ('nofeedback', 'random', sup), 'selected', 'selected', L == 128))
for sup in ('dense', 'final'):
    for L in (32, 128, 1024):
        spec.append(('R4', f'full-minus-tbptt8-{sup}', 'five', f'L{L}', ('rlt', 'random', sup), ('tbptt8', 'random', sup), 'selected', 'selected', L == 128))
spec_by_key = {(s[0], s[1], s[2], s[3]): s for s in spec}
file_keys = [(x['research_id'], x['contrast'], x['task'], x['endpoint']) for x in analysis['contrasts']]
contrast_results, contrast_problems, max_contrast_diff = [], [], 0.0
for x in analysis['contrasts']:
    k = (x['research_id'], x['contrast'], x['task'], x['endpoint'])
    s = spec_by_key.get(k)
    probs = []
    if s is None:
        contrast_problems.append(f'{k}: not in prespecified list'); continue
    rid, name, task, ep, cand, base, cv, bv, primary = s
    if x['primary'] != primary or x['candidate_variant'] != cv or x['baseline_variant'] != bv or x['seeds'] != 3 or len(x['per_seed']) != 3:
        probs.append('primary/variant/seed-count flags')
    diffs, per_seed = [], []
    for seed, ps in zip((117, 143, 191), x['per_seed']):
        ca = f"small-{task}-w32-{cand[0]}-{cand[1]}-{cand[2]}-lr0.003-s{seed}"
        cb = f"small-{task}-w32-{base[0]}-{base[1]}-{base[2]}-lr0.003-s{seed}"
        ka, kb = (ca if cv == 'selected' else ca + '-equal-label'), (cb if bv == 'selected' else cb + '-equal-label')
        A, B = RAW[(ka, ep)], RAW[(kb, ep)]
        if ps['seed'] != seed or ps['candidate'] != ca or ps['baseline'] != cb:
            probs.append(f'seed {seed}: candidate/baseline identity')
        if (A['x_sha'], A['y_sha'], A['ids_sha']) != (B['x_sha'], B['y_sha'], B['ids_sha']):
            probs.append(f'seed {seed}: UNPAIRED inputs/targets/ids'); continue
        a, b = A['final_pred'] == A['final_target'], B['final_pred'] == B['final_target']
        mine = dict(n=len(a), candidate_correct=int(a.sum()), baseline_correct=int(b.sum()), gained=int((a & ~b).sum()), lost=int((~a & b).sum()),
                    difference=float(a.mean() - b.mean()))
        for kk in ('n', 'candidate_correct', 'baseline_correct', 'gained', 'lost'):
            if ps[kk] != mine[kk]:
                probs.append(f'seed {seed}: {kk} {ps[kk]} vs {mine[kk]}')
        d = abs(ps['difference'] - mine['difference']); max_contrast_diff = max(max_contrast_diff, d)
        if d > 1e-12:
            probs.append(f'seed {seed}: difference {ps["difference"]} vs {mine["difference"]}')
        diffs.append(mine['difference']); per_seed.append(mine)
    if len(diffs) == 3:
        mean = statistics.mean(diffs); half = T95 * statistics.stdev(diffs) / math.sqrt(3)
        half_np = T95 * float(np.std(diffs, ddof=1)) / math.sqrt(3)
        lo, hi = mean - half, mean + half
        direction = 'POSITIVE_INTERVAL' if lo > 0 else 'NEGATIVE_INTERVAL' if hi < 0 else 'INCLUDES_ZERO'
        for mine_v, file_v, label in ((mean, x['mean'], 'mean'), (lo, x['approximate_seed_t95'][0], 'lo'), (hi, x['approximate_seed_t95'][1], 'hi')):
            d = abs(mine_v - file_v); max_contrast_diff = max(max_contrast_diff, d)
            if d > 1e-12:
                probs.append(f'{label} {file_v} vs {mine_v}')
        if abs(half - half_np) > 1e-12:
            probs.append('statistics.stdev vs numpy ddof=1 disagree')
        if x['direction'] != direction or x['seed_effects'] != [p['difference'] for p in x['per_seed']]:
            probs.append(f'direction {x["direction"]} vs {direction}')
        contrast_results.append(dict(research_id=rid, contrast=name, task=task, endpoint=ep, primary=primary, seed_effects_pp=[100 * d for d in diffs],
                                     mean_pp=100 * mean, t95_pp=[100 * lo, 100 * hi], direction=direction, per_seed=per_seed))
    if probs:
        contrast_problems.append(f'{k}: ' + '; '.join(probs))
check('6.contrasts_71_recomputed', not contrast_problems and len(analysis['contrasts']) == 71 and len(spec) == 71 and set(file_keys) == set(spec_by_key)
      and len(set(file_keys)) == 71 and analysis['analysis_plan_sha256'] == PLAN_PIN and analysis['protocol_hash'] == frozen['protocol_hash']
      and analysis['selected_lr'] == my_selected_lr and analysis['status'] == 'MEASURED' and analysis['complete_training_conditions'] == 48 and analysis['evaluated_models'] == 39,
      contrasts=len(analysis['contrasts']), prespecified=len(spec), order_identical=file_keys == [(s[0], s[1], s[2], s[3]) for s in spec],
      max_abs_diff=max_contrast_diff, problems=contrast_problems, primary_count=sum(1 for s in spec if s[8]))

r5_rows, r5_problems = [], []
for seed, fx in zip((217, 243, 291), analysis['R5_confirmation']):
    cid = f'confirm-parity-w256-rlt-random-dense-lr{my_selected_lr:g}-s{seed}'
    if fx['seed'] != seed or fx['condition'] != cid or fx['selected_step'] != COND[cid]['selected_step'] or fx['selected_lr'] != my_selected_lr or len(fx['endpoints']) != 13:
        r5_problems.append(f'{cid}: header fields')
    mine_eps = []
    for ep, fe in zip(ENDPOINTS, fx['endpoints']):
        A = RAW[(cid, ep)]
        pr, tg = A['final_pred'], A['final_target']
        counts = np.bincount(pr, minlength=2)
        mine = dict(endpoint=ep, n=len(tg), correct=int((pr == tg).sum()), accuracy=float((pr == tg).mean()), prediction_histogram=counts.tolist(),
                    target_histogram=np.bincount(tg, minlength=2).tolist(), observed_constant_prediction=bool(np.count_nonzero(counts) == 1))
        if fe != mine:
            r5_problems.append(f'{cid} {ep}: file {fe} vs raw {mine}')
        mine_eps.append(mine)
    r5_rows.append(dict(seed=seed, condition=cid, selected_step=COND[cid]['selected_step'], selected_validation_final_accuracy=COND[cid]['selected_validation_final_accuracy'],
                        endpoints=mine_eps))
check('6.R5_confirmation_recomputed', not r5_problems and len(analysis['R5_confirmation']) == 3, problems=r5_problems)

# ---------------------------------------------------------------- 7. statistical scope
t_df2 = math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2))  # closed-form df=2 quantile: F(t)=1/2+t/(2*sqrt(t^2+2)), F=0.975
pooled = [x['contrast'] for x in analysis['contrasts'] if any(p['n'] != 512 for p in x['per_seed']) or x['seeds'] != 3 or len(x['seed_effects']) != 3]
check('7.statistical_scope_df2_no_pooling', not pooled and abs(t_df2 - T95) < 1e-9
      and all(x['interval_scope'] == 'three-seed t approximation; not multiplicity adjusted' for x in analysis['contrasts']),
      t975_df2_closed_form=t_df2, coefficient_used=T95, pooled_contrasts=pooled,
      long_batch_rows_L1024_batch8=all(v['problems'] == [] for v in EVAL.values()),
      note='L1024 rows declare batch 8 (eval_long_threshold 512), short rows batch 32; verified as declared fields only; forward-pass invariance to batching NOT re-verified by this audit (no model was run)')

# ---------------------------------------------------------------- 8. anomalies
for k, v in COND.items():
    if v['selected_step'] != STEPS:
        ANOMALIES.append(f"selected_step {v['selected_step']} != 2048: {k} (validation final acc {v['selected_validation_final_accuracy']}, loss {v['selected_validation_final_loss']:.4g})")
    if v['selected_validation_final_accuracy'] < 0.9:
        ANOMALIES.append(f"low selected validation final accuracy {v['selected_validation_final_accuracy']:.4f}: {k}")
for k, v in EVAL.items():
    for ep, e in v['endpoints'].items():
        chance = 0.5 if 'parity' in k else 0.2
        if e['final_accuracy'] <= chance + 0.05 and not e['constant_prediction']:
            ANOMALIES.append(f"near-chance final accuracy {e['final_accuracy']:.4f}: {k} {ep}")

n_pass = sum(1 for v in CHECKS.values() if v['status'] == 'PASS'); n_fail = len(CHECKS) - n_pass
result = dict(
    auditor=dict(script='r2r5_raw_audit.py', mode='read-only, CPU, torch threads 1, no producer imports, no model forward pass',
                 python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, seconds=round(time.time() - t0, 1),
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
    verdict='PASS' if n_fail == 0 else 'FAIL', checks_pass=n_pass, checks_fail=n_fail, checks=CHECKS,
    pins=dict(protocol_hash=frozen['protocol_hash'], frozen_sha256=(OUT / 'frozen.sha256').read_text().strip(), analysis_plan_sha256=plan_sha,
              sources=src_now, selection_hash=selection['selection_hash'], selected_lr=my_selected_lr,
              file_sha256={f: sha_file(OUT / f) for f in ('evaluation.json', 'test-gate.json', 'selection.json', 'summary.json', 'scientific-analysis.json')}),
    census=dict(stage_counts=stage_counts, evaluated=evaluated, development=sorted(dev_receipts)),
    conditions=COND, pairing=pairing, evaluation={k: v['endpoints'] for k, v in EVAL.items()}, source_identity=disjoint,
    contrasts=dict(all=contrast_results, primary=[c for c in contrast_results if c['primary']]), R5_confirmation=r5_rows, anomalies=ANOMALIES)
DEST.write_text(json.dumps(result, indent=1, allow_nan=False), encoding='utf8')
print(f"\nVERDICT {result['verdict']}  PASS {n_pass}  FAIL {n_fail}  ({result['auditor']['seconds']} s)")
for c in result['contrasts']['primary']:
    print(f"  PRIMARY {c['research_id']} {c['task']} {c['contrast']} {c['endpoint']}: mean {c['mean_pp']:+.3f} pp, t95 [{c['t95_pp'][0]:+.3f}, {c['t95_pp'][1]:+.3f}] {c['direction']} seeds {[round(v, 3) for v in c['seed_effects_pp']]}")
for r in r5_rows:
    print(f"  R5 s{r['seed']} step {r['selected_step']}: " + ' '.join(f"{e['endpoint']}={e['accuracy']:.4f}" for e in r['endpoints'] if e['endpoint'] in ('L32', 'L1024', 'neutral-128-prefix', 'neutral-128-middle', 'neutral-128-suffix')))
for a in ANOMALIES:
    print('  ANOMALY', a)
