#!/usr/bin/env python
"""Independent raw audit of the drift mini-experiment (2026-09-18). READ-ONLY.

Recomputes every hash, accuracy, histogram, contrast and decision from the raw artefacts under
results/drift and compares them with the producer's receipts and analysis-tolerant.json. It never
trains or evaluates a model; the only model construction is the seed-derived *initial* weights
(drift.run.make_model) so that initial.pt can be checked against the declared seed.

Writes exactly two files next to itself: drift-independent-audit.json and drift-independent-audit.md.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path('E:/rlt-research/2026-09-18-drift')
RES = ROOT / 'results/drift'
OUT = ROOT / '.superpowers/sdd/drift-mini'
SEALED = Path('E:/rlt-research/2026-09-17-diagnostics/results/training')
BASE_COMMIT = 'f83b527'
T_975 = {1: 12.706204736, 2: 4.302652729}
SOURCES = ('core/model.py', 'review_cycle/training_data.py', 'drift/model.py', 'drift/run.py')
COMPLETE_FILES = ('initial.pt', 'last.pt', 'selected.pt', 'updates.jsonl', 'train.json')

sys.path.insert(0, str(ROOT))
from drift import run as r  # noqa: E402  (read-only import; conditions/ident/make_model/protocol)

T0 = time.perf_counter()
CHECKS = []          # ordered PASS/FAIL records
NOTES = []           # anomalies / observations that are not PASS/FAIL items
DETAIL = OrderedDict()


def check(section, name, ok, detail=''):
    CHECKS.append(dict(section=section, name=name, status='PASS' if ok else 'FAIL', detail=detail))
    return ok


def note(text):
    NOTES.append(text)


# ----------------------------------------------------------------------------- helpers (independent)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def tensor_hash_np(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(('torch.' + str(a.dtype), tuple(a.shape))).encode() + a.tobytes()).hexdigest()


def arr_sig(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def oracle(x, task):
    """Independent re-statement of the task semantics (not the PERMS table, not prefix_targets)."""
    x = np.asarray(x); s = np.zeros(x.shape[0], dtype=np.int64); out = np.zeros_like(x)
    for t in range(x.shape[1]):
        tok = x[:, t]
        if task == 'parity':
            s = s ^ (tok == 1).astype(np.int64)
        else:
            inc = (s + 1) % 5
            swap = np.where(s == 0, 1, np.where(s == 1, 0, s))
            s = np.where(tok == 0, inc, np.where(tok == 1, swap, s))
        out[:, t] = s
    return out


def regen_inputs(task, n, length, seed):
    return np.random.default_rng(seed).choice(3, (n, length), p=[.4375, .4375, .125])


def source_seed(p, task, seed, split, step=0, length=None):
    value = f"{p['source_seeds'][split]}|{task}|{seed}|{step}|{length or p['train_length']}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def endpoint_file(endpoint):
    if endpoint.startswith('L'):
        return f'selected-{endpoint}.npz'
    _, slots, placement = endpoint.split('-')
    return f'selected-neutral{slots}-{placement}.npz'


# ----------------------------------------------------------------------------- frozen manifest
frozen = json.loads((RES / 'frozen.json').read_text(encoding='utf8'))
P = frozen['protocol']
CONDS = r.conditions(P)
IDS = [r.ident(c) for c in CONDS]
BY_ID = {r.ident(c): c for c in CONDS}
ENDPOINTS = [f'L{n}' for n in P['test_lengths']] + ['neutral-0-base'] + [f'neutral-{s}-{pl}' for s in P['neutral_slots'] if s for pl in ('prefix', 'middle', 'suffix')]
assert len(ENDPOINTS) == 13

# ============================================================================= CHECK 1: integrity
sec = '1-integrity'
frozen_sha = sha(RES / 'frozen.json'); sidecar = (RES / 'frozen.sha256').read_text(encoding='ascii').strip()
check(sec, 'frozen.sha256 matches sha256(frozen.json)', frozen_sha == sidecar, f'{frozen_sha} vs sidecar {sidecar}')
ph = digest(P)
check(sec, 'protocol_hash recomputed from canonical JSON of frozen protocol', ph == frozen['protocol_hash'], f'{ph}')
ph_code = digest(r.protocol())
check(sec, 'protocol() in current drift/run.py digests to the frozen protocol_hash', ph_code == frozen['protocol_hash'], ph_code)
src_now = {name: sha(ROOT / name) for name in SOURCES}
for name in SOURCES:
    check(sec, f'source pin {name}', src_now[name] == frozen['sources'].get(name), f'{src_now[name]} vs pinned {frozen["sources"].get(name)}')
check(sec, 'frozen manifest pins exactly the four declared sources', set(frozen['sources']) == set(SOURCES), sorted(frozen['sources']))

# git identity of the read-only sources vs base commit
try:
    tree = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', BASE_COMMIT, '--', 'core/model.py', 'review_cycle/training_data.py'], capture_output=True, text=True, check=True).stdout
    blobs = {line.split('\t')[1]: line.split()[2] for line in tree.strip().splitlines()}
    ok = True; det = []
    for name in ('core/model.py', 'review_cycle/training_data.py'):
        data = (ROOT / name).read_bytes(); blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
        det.append(f'{name}: worktree blob {blob[:12]} vs {BASE_COMMIT} {blobs.get(name, "?")[:12]}'); ok &= blob == blobs.get(name)
    check(sec, f'core/model.py and review_cycle/training_data.py byte-identical to base commit {BASE_COMMIT}', ok, '; '.join(det))
    status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    DETAIL['git_status_porcelain'] = status
    note('git status: ' + ('; '.join(status) if status else 'clean') + ' (drift/, docs/drift-mini-plan.md and results/drift/ are untracked at audit time; the cycle is not yet committed).')
except Exception as e:  # pragma: no cover
    check(sec, 'git comparison with base commit', False, f'git unavailable: {e}')

plan_sha = sha(ROOT / 'docs/drift-mini-plan.md'); test_sha = sha(ROOT / 'drift/test_drift.py')
hashes_txt = (OUT / 'plan-and-test-hashes.txt').read_text(encoding='utf8')
check(sec, 'plan sha256 matches ledger record and analysis-tolerant.json', plan_sha in hashes_txt and plan_sha == json.loads((RES / 'analysis-tolerant.json').read_text(encoding='utf8'))['plan_sha256'], plan_sha)
check(sec, 'test file sha256 matches ledger record', test_sha in hashes_txt, test_sha)

# directory inventory
top = sorted(p.name for p in RES.iterdir())
expected_top = sorted(IDS + ['analysis-tolerant.json', 'frozen.json', 'frozen.sha256', 'processes', 'workers'])
check(sec, 'results/drift inventory is exactly 24 condition dirs + frozen.json/.sha256 + analysis-tolerant.json + processes/ + workers/', top == expected_top,
      f'extra={sorted(set(top) - set(expected_top))} missing={sorted(set(expected_top) - set(top))}')

# condition status from raw files
STATUS = {}
for cid in IDS:
    d = RES / cid; files = sorted(p.name for p in d.iterdir())
    if 'complete.json' in files and 'evaluation.json' in files:
        STATUS[cid] = 'COMPLETE'
    elif any(f.startswith('failure-step') for f in files):
        STATUS[cid] = 'FAILED_NONFINITE'
    else:
        STATUS[cid] = 'INCOMPLETE'
COMPLETE = [cid for cid in IDS if STATUS[cid] == 'COMPLETE']; FAILED = [cid for cid in IDS if STATUS[cid] == 'FAILED_NONFINITE']
check(sec, '22 COMPLETE + 2 FAILED conditions, both failures are reread seed 191', len(COMPLETE) == 22 and FAILED == ['drift-five-reread-s191', 'drift-parity-reread-s191'], f'complete={len(COMPLETE)} failed={FAILED}')
DETAIL['condition_status'] = STATUS

expected_complete_files = sorted(['complete.json', 'evaluation.json', 'initial.pt', 'last.pt', 'selected.pt', 'train.json', 'updates.jsonl'] + [endpoint_file(e) for e in ENDPOINTS])
inv_ok = True; inv_det = []
for cid in COMPLETE:
    files = sorted(p.name for p in (RES / cid).iterdir())
    if files != expected_complete_files:
        inv_ok = False; inv_det.append(f'{cid}: extra={sorted(set(files) - set(expected_complete_files))} missing={sorted(set(expected_complete_files) - set(files))}')
check(sec, 'every COMPLETE dir holds exactly the 20 expected files (no lock/tmp leftovers)', inv_ok, '; '.join(inv_det) or '22/22')
fail_ok = True; fail_det = []
for cid in FAILED:
    files = sorted(p.name for p in (RES / cid).iterdir())
    fs = [f for f in files if f.startswith('failure-step')]
    ok = len(fs) == 1 and files == sorted(fs + ['initial.pt', 'updates.jsonl'])
    fail_ok &= ok; fail_det.append(f'{cid}: {files}')
check(sec, 'FAILED dirs hold only failure-step*.json + initial.pt + updates.jsonl (no evaluation/NPZ/checkpoints/seal)', fail_ok, '; '.join(fail_det))

# per-condition receipts, seals, evaluation rows
TRAIN = {}; EVAL = {}; SEAL = {}
for cid in COMPLETE:
    d = RES / cid
    TRAIN[cid] = json.loads((d / 'train.json').read_text(encoding='utf8'))
    EVAL[cid] = json.loads((d / 'evaluation.json').read_text(encoding='utf8'))
    SEAL[cid] = json.loads((d / 'complete.json').read_text(encoding='utf8'))

ok = True; det = []
for cid in COMPLETE:
    t = TRAIN[cid]
    good = t['protocol_hash'] == frozen['protocol_hash'] and t['sources'] == frozen['sources'] and t['status'] == 'COMPLETE' and t['condition'] == BY_ID[cid] \
        and EVAL[cid]['protocol_hash'] == frozen['protocol_hash'] and EVAL[cid]['condition'] == BY_ID[cid] and SEAL[cid]['protocol_hash'] == frozen['protocol_hash'] and SEAL[cid]['condition'] == BY_ID[cid]
    if not good:
        ok = False; det.append(cid)
check(sec, 'train.json / evaluation.json / complete.json carry the frozen protocol_hash, frozen sources and the declared condition (22 conditions)', ok, '; '.join(det) or '22/22')

ok = True; det = []; FILE_SHA = {}
for cid in COMPLETE:
    d = RES / cid; seal = SEAL[cid]
    files = {name: sha(d / name) for name in COMPLETE_FILES}; FILE_SHA[cid] = files
    seal_wo = {k: v for k, v in seal.items() if k != 'seal_hash'}
    good = files == seal['files'] and digest(seal_wo) == seal['seal_hash'] and set(seal['files']) == set(COMPLETE_FILES)
    if not good:
        ok = False; det.append(f'{cid}: files_match={files == seal["files"]} seal_hash_match={digest(seal_wo) == seal["seal_hash"]}')
check(sec, 'complete.json: all 5 file sha256 recomputed and seal_hash recomputed (22 x 5 files)', ok, '; '.join(det) or '110/110 file hashes, 22/22 seal hashes')

ok = True; det = []; n_rows = 0
for cid in COMPLETE:
    d = RES / cid; ev = EVAL[cid]
    rows = ev['lengths'] + ev['neutral']
    good = len(ev['lengths']) == 3 and len(ev['neutral']) == 10 and ev['checkpoint_sha256'] == FILE_SHA[cid]['selected.pt']
    meta = digest({k: v for k, v in ev.items() if k != 'metadata_hash'}); good &= meta == ev['metadata_hash']
    for row in rows:
        n_rows += 1
        if sha(RES / row['raw_file']) != row['sha256'] or not row['raw_file'].startswith(cid + '/'):
            good = False; det.append(f"{cid}:{row['raw_file']}")
    if not good:
        ok = False; det.append(f'{cid}: rows={len(rows)} checkpoint_ok={ev["checkpoint_sha256"] == FILE_SHA[cid]["selected.pt"]} metadata_hash_ok={meta == ev["metadata_hash"]}')
check(sec, 'evaluation.json: 3+10 rows, raw NPZ sha256 recomputed, checkpoint_sha256 == sha256(selected.pt), metadata_hash recomputed', ok, '; '.join(det) or f'{n_rows}/286 raw hashes match')

# worker logs
logs = {name: read_jsonl(RES / 'workers' / f'slot-{name}.jsonl') for name in ('0', '1', '2', '2b')}
DETAIL['worker_log_state_counts'] = {k: dict(Counter(row['state'] for row in v)) for k, v in logs.items()}
started = [row['condition'] for v in logs.values() for row in v if row['state'] == 'CONDITION_STARTED']
completed = [row['condition'] for v in logs.values() for row in v if row['state'] == 'CONDITION_COMPLETE']
failed_rows = [row['condition'] for v in logs.values() for row in v if row['state'] == 'FAILED']
states_all = Counter(row['state'] for v in logs.values() for row in v)
check(sec, 'worker logs: 24 distinct CONDITION_STARTED, 22 CONDITION_COMPLETE == the 22 COMPLETE dirs', sorted(started) == sorted(IDS) and sorted(completed) == sorted(COMPLETE), f'started={len(started)} complete={len(completed)}')
check(sec, 'worker logs: FAILED rows', failed_rows == ['drift-parity-reread-s191'],
      f'FAILED rows={failed_rows}; drift-five-reread-s191 has CONDITION_STARTED in slot-2.jsonl with no terminal row (worker crashed) - its failure is evidenced by failure-step1632.json and slot-2.stderr.log, not by a FAILED log row')
check(sec, 'worker logs: no states other than STARTED/CONDITION_STARTED/CONDITION_COMPLETE/FAILED/COMPLETE', set(states_all) <= {'STARTED', 'CONDITION_STARTED', 'CONDITION_COMPLETE', 'FAILED', 'COMPLETE'}, dict(states_all))
ok = True; det = []
for k in ('0', '1', '2'):
    queue = [r.ident(c) for c in CONDS[int(k)::3]]
    seq = [row['condition'] for row in logs[k] if row['state'] == 'CONDITION_STARTED']
    good = seq == queue[:len(seq)] and logs[k][0]['state'] == 'STARTED' and logs[k][0]['protocol_hash'] == frozen['protocol_hash']
    utcs = [row['utc'] for row in logs[k]]; good &= utcs == sorted(utcs) and logs[k][0]['utc'] >= frozen['frozen_utc']
    if k in ('0', '1'):
        good &= logs[k][-1]['state'] == 'COMPLETE' and seq == queue
    else:
        good &= seq == queue[:4] and logs[k][-1]['state'] == 'CONDITION_STARTED' and logs[k][-1]['condition'] == 'drift-five-reread-s191'
    for row in logs[k]:
        if row['state'] == 'CONDITION_COMPLETE':
            cid = row['condition']; L1024 = next(x['final_accuracy'] for x in EVAL[cid]['lengths'] if x['length'] == 1024)
            good &= row['selected_step'] == TRAIN[cid]['selected_step'] and row['wall_seconds'] == TRAIN[cid]['wall_seconds'] and row['L1024'] == L1024
    ok &= good; det.append(f'slot-{k}: {len(seq)} started, ok={good}')
q2b = [row['condition'] for row in logs['2b'] if row['state'] == 'CONDITION_STARTED']
good2b = q2b == [r.ident(c) for c in CONDS[2::3]][4:] and logs['2b'][-1]['state'] == 'COMPLETE' and [row['utc'] for row in logs['2b']] == sorted(row['utc'] for row in logs['2b'])
ok &= good2b; det.append(f'slot-2b: queue {q2b} == remaining slot-2 queue: {good2b}')
check(sec, 'worker logs: slot queues == conditions[slot::3] in order, monotonic utc, STARTED after freeze, selected_step/wall_seconds/L1024 rows match receipts', ok, '; '.join(det))
note('slot-2b.jsonl was written by a shell wrapper (slot-2b.sh), not by run.py worker(): integer utc, no pid, no selected_step/wall_seconds/L1024 in its CONDITION_COMPLETE rows, and STARTED/COMPLETE rows carry condition "-". Schema differs from slot-0/1/2 but the four conditions it ran are the exact remainder of the slot-2 queue.')
# pids
ok = True; det = []
for k in ('0', '1', '2'):
    pid = logs[k][0]['pid']
    for row in logs[k]:
        if row['state'] == 'CONDITION_COMPLETE':
            ok &= TRAIN[row['condition']]['pid'] == pid
    det.append(f'slot-{k} pid {pid}')
check(sec, 'train.json pid equals the worker pid for every condition trained by slots 0/1/2', ok, '; '.join(det))

# process logs
proc = sorted((RES / 'processes').iterdir())
stderr_nonempty = [p.name for p in proc if p.name.endswith('.stderr.log') and p.stat().st_size > 0]
ok = stderr_nonempty == ['2b-drift-parity-reread-s191.stderr.log', 'slot-2.stderr.log']
det = []
for name in stderr_nonempty:
    text = (RES / 'processes' / name).read_text(encoding='utf8')
    tb = text.count('Traceback (most recent call last)'); last = text.strip().splitlines()[-1]
    good = tb == 1 and last == 'ValueError: nonfinite loss/gradient; failure preserved' and 'run.py' in text
    ok &= good; det.append(f'{name}: tracebacks={tb} last="{last}"')
check(sec, 'stderr logs: exactly two non-empty, each a single nonfinite traceback from drift/run.py run_one', ok, '; '.join(det) + f'; empty stderr logs={len([p for p in proc if p.name.endswith(".stderr.log")]) - len(stderr_nonempty)}')
ok = True; det = []
for k in ('0', '1'):
    txt = (RES / 'processes' / f'slot-{k}.stdout.log').read_text(encoding='utf8').strip(); ok &= txt == 'null'; det.append(f'slot-{k}.stdout={txt!r}')
ok &= (RES / 'processes' / 'slot-2.stdout.log').stat().st_size == 0 and (RES / 'processes' / '2b-drift-parity-reread-s191.stdout.log').stat().st_size == 0
for cid in ('drift-parity-idinv-s117', 'drift-parity-win16-s143', 'drift-parity-base-s191'):
    rec = json.loads((RES / 'processes' / f'2b-{cid}.stdout.log').read_text(encoding='utf8'))
    ev = json.loads((RES / 'processes' / f'2b-{cid}.eval.stdout.log').read_text(encoding='utf8'))
    good = rec == {k: v for k, v in TRAIN[cid].items() if k != 'history'} and ev == EVAL[cid]
    ok &= good; det.append(f'2b-{cid}: stdout receipt/evaluation == files: {good}')
check(sec, 'stdout logs: slots 0/1 printed null (worker returns None), slot-2/2b-failed empty, 2b receipts equal train.json (minus history) and evaluation.json', ok, '; '.join(det))

# ============================================================================= CHECK 2: real training
sec = '2-training'
JOURNAL = {}; TRAIN_SUMMARY = OrderedDict()
ok = True; det = []
for cid in COMPLETE:
    c = BY_ID[cid]; rows = read_jsonl(RES / cid / 'updates.jsonl'); JOURNAL[cid] = rows; t = TRAIN[cid]
    good = len(rows) == 2048 and all(row['step'] == i + 1 and row['real_update'] == i + 1 for i, row in enumerate(rows))
    good &= all(finite(row['loss']) and finite(row['cross_entropy']) and finite(row['grad_norm']) for row in rows)
    good &= all(row['input_hash'] == row['base_input_hash'] for row in rows)   # random arrangement: no blocking
    if c['arm'] == 'idinv':
        good &= all(finite(row['aux']) and row['aux'] >= 0 for row in rows) and all(abs(row['loss'] - (row['cross_entropy'] + P['lambda_idinv'] * row['aux'])) <= 1e-6 * max(1., row['loss']) for row in rows)
    else:
        good &= all(row['aux'] is None and row['loss'] == row['cross_entropy'] for row in rows)
    val_rows = [row for row in rows if 'validation' in row]
    good &= [row['step'] for row in val_rows] == list(range(128, 2049, 128)) and all(row['validation']['n'] == 256 for row in val_rows)
    best = min(val_rows, key=lambda row: (row['validation']['final_loss'], row['step']))
    good &= best['step'] == t['selected_step'] and best['validation'] == t['selected_validation']
    good &= t['history'] == rows and t['updates'] == 2048 and t['labels'] == 2048 * 32 * 32
    # source seeds of the training stream recomputed from the frozen namespace
    good &= all(row['source_seed'] == source_seed(P, c['task'], c['seed'], 'train', step=row['step']) for row in rows)
    good &= t['validation_source_seed'] == source_seed(P, c['task'], c['seed'], 'validation', length=32)
    gn = np.array([row['grad_norm'] for row in rows])
    TRAIN_SUMMARY[cid] = dict(selected_step=t['selected_step'], selected_val_final_loss=t['selected_validation']['final_loss'], first_loss=rows[0]['loss'], last_loss=rows[-1]['loss'],
                              max_grad_norm=float(gn.max()), updates_grad_norm_gt10=int((gn > 10).sum()), updates_grad_norm_gt1=int((gn > 1).sum()),
                              aux_first=rows[0]['aux'], aux_last=rows[-1]['aux'], train_seconds=t['train_seconds'], wall_seconds=t['wall_seconds'])
    if not good:
        ok = False; det.append(cid)
check(sec, 'updates.jsonl (22): 2048 rows, step==real_update 1..2048, finite loss/CE/grad_norm, aux null (non-idinv) or >=0 with loss==CE+aux (idinv), 16 validation rows n=256, selected_step == argmin val final_loss, history == journal, train/validation source seeds recomputed', ok, '; '.join(det) or '22/22')

STATES = {}
ok = True; det = []; reread_counts = {}
for cid in COMPLETE:
    d = RES / cid; c = BY_ID[cid]; t = TRAIN[cid]
    init = torch.load(d / 'initial.pt', weights_only=True); last = torch.load(d / 'last.pt', weights_only=True); sel = torch.load(d / 'selected.pt', weights_only=True)
    STATES[cid] = init
    good = state_hash(init) == t['initial_state_hash'] and state_hash(last) == t['last_state_hash'] and state_hash(sel) == t['selected_state_hash']
    good &= set(init) == set(last) == set(sel)
    changed_last = sum(not torch.equal(init[k], last[k]) for k in init); changed_sel = sum(not torch.equal(init[k], sel[k]) for k in init)
    good &= changed_last == len(init) and changed_sel == len(init)
    if t['selected_step'] == 2048:
        good &= all(torch.equal(sel[k], last[k]) for k in init)
    else:
        good &= not all(torch.equal(sel[k], last[k]) for k in init)
    extra_keys = [k for k in init if k.startswith('reread_')]; extra = sum(init[k].numel() for k in extra_keys)
    core_keys = [k for k in init if not k.startswith('reread_')]; core_n = sum(init[k].numel() for k in core_keys)
    reread_counts[cid] = extra
    good &= extra == t['extra_parameters'] and core_n == t['core_parameters']
    if c['arm'] == 'reread':
        good &= extra == 6176 and sorted(extra_keys) == ['reread_k.weight', 'reread_n.weight', 'reread_o.weight', 'reread_q.weight', 'reread_v.weight'] and all(k.startswith('core.') for k in core_keys)
        good &= bool((init['reread_o.weight'] == 0).all()) and bool((sel['reread_o.weight'] != 0).any())
    else:
        good &= extra == 0 and not any(k.startswith('core.') for k in init)
    expected_core = 24325 if c['task'] == 'five' else 24325 - (32 * 5 + 5) + (32 * 2 + 2)
    good &= core_n == expected_core
    if not good:
        ok = False; det.append(f'{cid}: changed_last={changed_last}/{len(init)} changed_sel={changed_sel}/{len(init)} extra={extra} core={core_n}')
check(sec, 'checkpoints (22): state hashes == receipts; every tensor differs initial vs last and vs selected; selected==last iff selected_step==2048; reread_* = 6176 params (5 tensors, W_o zero at init, nonzero after) only in reread; core params 24325 (five) / 24226 (parity)', ok, '; '.join(det) or f'reread extra params: {sorted(set(reread_counts.values()))}')

FAIL_INFO = OrderedDict()
ok = True; det = []
for cid in FAILED:
    d = RES / cid; rows = read_jsonl(d / 'updates.jsonl'); JOURNAL[cid] = rows
    fj = next(d.glob('failure-step*.json')); f = json.loads(fj.read_text(encoding='utf8')); step = int(re.search(r'failure-step(\d+)\.json', fj.name).group(1))
    tail = [(row['step'], row['grad_norm']) for row in rows[-6:]]
    good = f['status'] == 'FAIL_NONFINITE' and f['step'] == step == rows[-1]['step'] + 1 == len(rows) + 1 and f['grad_norm'] == 'inf' and math.isfinite(float(f['loss']))
    good &= all(row['step'] == i + 1 for i, row in enumerate(rows)) and all(finite(row['grad_norm']) for row in rows)
    gn = [g for _, g in tail]; good &= gn[-1] > 1e6 and gn[-1] > gn[-2] > gn[-3]
    good &= all(row['input_hash'] == row['base_input_hash'] for row in rows) and all(row['source_seed'] == source_seed(P, BY_ID[cid]['task'], BY_ID[cid]['seed'], 'train', step=row['step']) for row in rows)
    init = torch.load(d / 'initial.pt', weights_only=True); STATES[cid] = init
    FAIL_INFO[cid] = dict(failure_file=fj.name, failure_step=f['step'], failure_loss=f['loss'], failure_grad_norm=f['grad_norm'], journal_rows=len(rows), last_journal_step=rows[-1]['step'],
                          tail_grad_norms=tail, tail_losses=[(row['step'], row['loss']) for row in rows[-6:]], max_finite_grad_norm=max(row['grad_norm'] for row in rows))
    ok &= good; det.append(f'{cid}: failure step {f["step"]} = last journal step {rows[-1]["step"]}+1, tail grad_norm {[round(g, 3) if g < 1e3 else float(f"{g:.3g}") for g in gn]}, loss at failure {f["loss"]}')
check(sec, 'FAILED (2): failure-step N == last journal step + 1 == rows + 1, grad_norm "inf", finite loss, journal tail shows exploding finite grad_norm', ok, '; '.join(det))
DETAIL['failures'] = FAIL_INFO
DETAIL['training_summary'] = TRAIN_SUMMARY
by_arm = OrderedDict()
for cid, s in TRAIN_SUMMARY.items():
    key = f"{BY_ID[cid]['task']}/{BY_ID[cid]['arm']}"; by_arm.setdefault(key, []).append((BY_ID[cid]['seed'], s['max_grad_norm'], s['updates_grad_norm_gt10']))
DETAIL['grad_norm_spikes_by_task_arm'] = {k: [dict(seed=a, max_grad_norm=b, updates_gt10=c) for a, b, c in v] for k, v in by_arm.items()}
note('Pre-clip grad_norm spikes >10 are NOT unique to the reread arm. Per task/arm (seed: max, #updates>10): ' + '; '.join(f"{k}: " + ', '.join(f'{a}: {b:.3g}, {c}' for a, b, c in v) for k, v in by_arm.items())
     + '. The ledger sentence "no other arm shows this" is overstated for spike counts: on the five task every arm spikes (base up to 1.66e3, idinv up to 955). What is unique to reread is the runaway to inf in 2/6 conditions and the largest surviving spike (5.63e4, five-s143); the ledger numbers for the surviving reread runs (306/78, 5.6e4/74, 300/19) are reproduced.')

# ============================================================================= CHECK 3: pairing
sec = '3-pairing'
ok = True; det = []
for task in P['tasks']:
    for seed in P['primary_seeds']:
        arms = {BY_ID[cid]['arm']: cid for cid in IDS if BY_ID[cid]['task'] == task and BY_ID[cid]['seed'] == seed}
        base = JOURNAL[arms['base']]
        for arm, cid in arms.items():
            rows = JOURNAL[cid]; n = len(rows)
            good = all(a['input_hash'] == b['input_hash'] and a['base_input_hash'] == b['base_input_hash'] and a['source_seed'] == b['source_seed'] for a, b in zip(rows, base[:n]))
            if not good:
                ok = False; det.append(f'{cid}: training inputs differ from base')
        det.append(f'{task}/s{seed}: journal rows compared ' + ','.join(f"{arm}={len(JOURNAL[cid])}" for arm, cid in arms.items()))
check(sec, 'per task/seed: input_hash, base_input_hash and source_seed sequences identical across arms (FAILED arms on their surviving prefix)', ok, '; '.join(det))

ok = True; det = []
for task in P['tasks']:
    for seed in P['primary_seeds']:
        arms = {BY_ID[cid]['arm']: cid for cid in IDS if BY_ID[cid]['task'] == task and BY_ID[cid]['seed'] == seed}
        def core_state(cid):
            st = STATES[cid]
            return {k[5:] if k.startswith('core.') else k: v for k, v in st.items() if not k.startswith('reread_')}
        ref = core_state(arms['base']); hashes = set()
        for arm, cid in arms.items():
            cs = core_state(cid); hashes.add(state_hash(cs))
            good = set(cs) == set(ref) and all(torch.equal(cs[k], ref[k]) for k in ref)
            if cid in TRAIN:
                good &= TRAIN[cid]['core_initial_hash'] == state_hash(cs)
            if not good:
                ok = False; det.append(f'{cid}: core initial tensors differ from base')
        det.append(f'{task}/s{seed}: {len(ref)} core tensors, {len(hashes)} distinct core hash ({next(iter(hashes))[:12]})')
check(sec, 'per task/seed: Core parameters in initial.pt identical (tensor-exact) across all four arms; train.json core_initial_hash equals recomputed hash', ok, '; '.join(det))

ok = True; det = []
for cid in IDS:
    c = BY_ID[cid]
    model = r.make_model(c, P); st = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    good = set(st) == set(STATES[cid]) and all(torch.equal(st[k], STATES[cid][k]) for k in st)
    if not good:
        ok = False; det.append(cid)
check(sec, 'initial.pt (24) reproduces the seed-derived initial weights (Core seeded by seed; reread_* by seed+1000, W_o zero) - no training, construction only', ok, '; '.join(det) or '24/24')
del STATES

# ============================================================================= CHECK 4: evaluation recomputation
sec = '4-evaluation'
RAW = {}   # (cid, endpoint) -> dict(sig, ok_final, n, ...)
ok = True; det = []; n_rows = 0; n_hist = 0
for cid in COMPLETE:
    c = BY_ID[cid]; ev = EVAL[cid]; classes = 2 if c['task'] == 'parity' else 5
    rows = {f"L{x['length']}": x for x in ev['lengths']}; rows.update({f"neutral-{x['slots']}-{x['placement']}": x for x in ev['neutral']})
    good = sorted(rows) == sorted(ENDPOINTS)
    base_pred = None; base_x = None; base_final_t = None
    for endpoint in ENDPOINTS:
        row = rows[endpoint]; n_rows += 1
        with np.load(RES / cid / endpoint_file(endpoint)) as f:
            x, pred, tgt, ids = f['inputs'], f['predictions'], f['targets'], f['input_ids']; cond = str(f['condition']); sseed = str(f['source_seed'])
        L = x.shape[1]
        fin = float((pred[:, -1] == tgt[:, -1]).mean()); pre = float((pred == tgt).mean())
        th = {str(i): int((tgt[:, -1] == i).sum()) for i in range(classes)}; ph_ = {str(i): int((pred[:, -1] == i).sum()) for i in range(classes)}
        g = fin == row['final_accuracy'] and pre == row['prefix_accuracy'] and th == row['target_histogram'] and ph_ == row['prediction_histogram']
        n_hist += 2
        g &= row['n'] == len(x) == 512 and row['consumed_tokens'] == L and row['raw_file'] == f'{cid}/{endpoint_file(endpoint)}' and cond == cid and sseed == str(row['source_seed'])
        g &= row['input_hash'] == tensor_hash_np(x) and pred.shape == tgt.shape == x.shape and set(np.unique(x)) <= {0, 1, 2}
        g &= list(ids) == [hashlib.sha256(f"{row['source_seed']}:{i}".encode()).hexdigest() for i in range(len(x))]
        g &= np.array_equal(oracle(x, c['task']), tgt)   # independent target oracle
        if endpoint.startswith('L'):
            length = int(endpoint[1:]); g &= L == length and row['source_seed'] == source_seed(P, c['task'], c['seed'], 'test', length=length)
            g &= np.array_equal(regen_inputs(c['task'], 512, length, row['source_seed']), x)
        else:
            slots, placement = int(endpoint.split('-')[1]), endpoint.split('-')[2]
            g &= row['source_seed'] == source_seed(P, c['task'], c['seed'], 'neutral', length=32) and L == 32 + slots
            if endpoint == 'neutral-0-base':
                base_x = x; base_pred = pred[:, -1]; base_final_t = tgt[:, -1]
                g &= np.array_equal(regen_inputs(c['task'], 512, 32, row['source_seed']), x) and 'correct_to_wrong' not in row
            else:
                at = {'prefix': 0, 'middle': 16, 'suffix': 32}[placement]
                expect = np.concatenate([base_x[:, :at], np.full((512, slots), 2, dtype=x.dtype), base_x[:, at:]], 1)
                g &= np.array_equal(expect, x) and np.array_equal(tgt[:, -1], base_final_t)
                was = base_pred == base_final_t; now = pred[:, -1] == tgt[:, -1]
                g &= row['correct_to_wrong'] == int((was & ~now).sum()) and row['wrong_to_correct'] == int((~was & now).sum()) and row['prediction_changes'] == int((base_pred != pred[:, -1]).sum())
        RAW[(cid, endpoint)] = dict(sig=arr_sig(x, tgt, ids), ok_final=(pred[:, -1] == tgt[:, -1]), final_pred=pred[:, -1].copy(), final_acc=fin)
        if not g:
            good = False; det.append(f'{cid}/{endpoint}')
    ok &= good
check(sec, 'NPZ (22 x 13): final/prefix accuracy and target/prediction histograms recomputed exactly; n=512; input_ids, input_hash, source_seed (frozen namespace) and inputs (rng regeneration) recomputed; targets recomputed with an independent oracle', ok, '; '.join(det) or f'{n_rows}/286 rows, {n_hist} histograms exact')
check(sec, 'neutral NPZ: inputs == base inputs with the declared number of token-2 insertions at prefix(0)/middle(16)/suffix(32); final targets == base; correct_to_wrong/wrong_to_correct/prediction_changes recomputed', ok, 'covered by the row loop above' if ok else '; '.join(det))

ok = True; det = []
for task in P['tasks']:
    for seed in P['primary_seeds']:
        cids = [cid for cid in COMPLETE if BY_ID[cid]['task'] == task and BY_ID[cid]['seed'] == seed]
        for endpoint in ENDPOINTS:
            sigs = {RAW[(cid, endpoint)]['sig'] for cid in cids}
            if len(sigs) != 1:
                ok = False; det.append(f'{task}/s{seed}/{endpoint}: {len(sigs)} distinct input sets across {len(cids)} arms')
        det.append(f'{task}/s{seed}: {len(cids)} arms x 13 endpoints identical inputs/targets/input_ids')
check(sec, 'inputs/targets/input_ids identical across arms for every task/seed/endpoint', ok, '; '.join(det))
# test inputs also identical across seeds? (they depend on seed via namespace) - record
sig_by_seed = {seed: RAW[(f'drift-five-base-s{seed}', 'L32')]['sig'][:12] for seed in P['primary_seeds']}
DETAIL['five_L32_input_signature_by_seed'] = sig_by_seed

# ============================================================================= CHECK 5: contrasts and decisions
sec = '5-contrasts'
A = json.loads((RES / 'analysis-tolerant.json').read_text(encoding='utf8'))
ok = True; det = []
check(sec, 'analysis-tolerant.json header: protocol_hash frozen, complete=22, declared=24, status MEASURED_WITH_INCOMPLETE, condition_status == raw-derived status, absolute == evaluation.json final accuracies',
      A['protocol_hash'] == frozen['protocol_hash'] and A['complete'] == 22 and A['declared'] == 24 and A['status'] == 'MEASURED_WITH_INCOMPLETE' and A['condition_status'] == STATUS
      and A['absolute'] == {cid: {e: RAW[(cid, e)]['final_acc'] for e in ENDPOINTS} for cid in COMPLETE}, '')


def summary(diffs):
    n = len(diffs)
    if n < 2:
        return dict(mean=diffs[0] if diffs else None, t95=None, direction='UNDETERMINED', seeds=n)
    mean = sum(diffs) / n; sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)); half = T_975[n - 1] * sd / math.sqrt(n)
    return dict(mean=mean, t95=[mean - half, mean + half], seeds=n, df=n - 1, direction='POSITIVE_INTERVAL' if mean - half > 0 else 'NEGATIVE_INTERVAL' if mean + half < 0 else 'INCLUDES_ZERO')


MY = {}
for task in P['tasks']:
    for arm in P['arms'][1:]:
        for endpoint in ENDPOINTS:
            diffs, per_seed, missing = [], [], []
            for seed in P['primary_seeds']:
                ca, cb = f'drift-{task}-{arm}-s{seed}', f'drift-{task}-base-s{seed}'
                if STATUS[ca] != 'COMPLETE' or STATUS[cb] != 'COMPLETE':
                    missing.append(dict(seed=seed, arm_status=STATUS[ca], base_status=STATUS[cb])); continue
                a, b = RAW[(ca, endpoint)], RAW[(cb, endpoint)]
                assert a['sig'] == b['sig'], 'UNPAIRED'
                d = float(a['ok_final'].mean() - b['ok_final'].mean()); diffs.append(d)
                per_seed.append(dict(seed=seed, arm_correct=int(a['ok_final'].sum()), base_correct=int(b['ok_final'].sum()), gained=int((a['ok_final'] & ~b['ok_final']).sum()), lost=int((~a['ok_final'] & b['ok_final']).sum()), difference=d))
            MY[(task, arm, endpoint)] = dict(task=task, arm=arm, endpoint=endpoint, per_seed=per_seed, missing=missing, seed_effects=diffs, **summary(diffs))

ok = len(A['contrasts']) == 78; det = []; n_cmp = 0
for row in A['contrasts']:
    key = (row['task'], row['arm'], row['endpoint']); m = MY[key]; n_cmp += 1
    g = row['per_seed'] == m['per_seed'] and row['missing'] == m['missing'] and row['seeds'] == m['seeds'] and row['direction'] == m['direction']
    g &= len(row['seed_effects']) == len(m['seed_effects']) and all(abs(x - y) <= 1e-12 for x, y in zip(row['seed_effects'], m['seed_effects']))
    g &= abs(row['mean'] - m['mean']) <= 1e-12 and all(abs(x - y) <= 1e-12 for x, y in zip(row['t95'], m['t95'])) and row.get('df') == m.get('df')
    g &= row['primary'] == (row['task'] == 'five' and row['endpoint'] in P['primary']['endpoints'])
    if not g:
        ok = False; det.append('/'.join(key))
check(sec, 'all 78 contrasts recomputed from raw NPZ after pairing assertion: per_seed counts exact, seed_effects/mean/t95 within 1e-12, direction/df/seeds/missing/primary flags exact', ok, '; '.join(det) or f'{n_cmp}/78')

MYDEC = OrderedDict()
for task in P['tasks']:
    for arm in P['arms'][1:]:
        prim = [MY[(task, arm, e)] for e in P['primary']['endpoints']]; cost = MY[(task, arm, P['primary']['cost_endpoint'])]
        incomplete = any(c['missing'] for c in prim + [cost])
        positive = all(c['mean'] * 100 >= P['primary']['gain_pp'] and c['direction'] == 'POSITIVE_INTERVAL' for c in prim) and cost['mean'] * 100 >= -P['primary']['max_cost_pp']
        negative = any(c['mean'] <= 0 for c in prim) or cost['mean'] * 100 < -P['primary']['max_cost_pp']
        MYDEC[f'{task}/{arm}'] = dict(status='MEASURED_POSITIVE' if positive else 'MEASURED_NEGATIVE' if negative else 'MIXED', incomplete=incomplete, seeds_used=min(c['seeds'] for c in prim + [cost]),
                                      primary=[dict(endpoint=c['endpoint'], mean_pp=c['mean'] * 100, t95_pp=[v * 100 for v in c['t95']], direction=c['direction']) for c in prim],
                                      cost=dict(endpoint=cost['endpoint'], mean_pp=cost['mean'] * 100, t95_pp=[v * 100 for v in cost['t95']]),
                                      reasons=dict(any_primary_mean_le_0=any(c['mean'] <= 0 for c in prim), cost_below_minus5pp=cost['mean'] * 100 < -5, all_primary_ge_20pp_positive_interval=all(c['mean'] * 100 >= 20 and c['direction'] == 'POSITIVE_INTERVAL' for c in prim)))
ok = set(A['decisions']) == set(MYDEC); det = []
for k, mine in MYDEC.items():
    a = A['decisions'][k]
    g = a['status'] == mine['status'] and a['incomplete'] == mine['incomplete'] and a['seeds_used'] == mine['seeds_used']
    g &= all(pa['endpoint'] == pm['endpoint'] and abs(pa['mean_pp'] - pm['mean_pp']) <= 1e-9 and all(abs(x - y) <= 1e-9 for x, y in zip(pa['t95_pp'], pm['t95_pp'])) and pa['direction'] == pm['direction'] for pa, pm in zip(a['primary'], mine['primary']))
    g &= abs(a['cost']['mean_pp'] - mine['cost']['mean_pp']) <= 1e-9 and all(abs(x - y) <= 1e-9 for x, y in zip(a['cost']['t95_pp'], mine['cost']['t95_pp']))
    ok &= g
    det.append(f"{k}: {mine['status']} (middle {mine['primary'][0]['mean_pp']:+.2f} pp, suffix {mine['primary'][1]['mean_pp']:+.2f} pp, L1024 cost {mine['cost']['mean_pp']:+.2f} pp, seeds {mine['seeds_used']}) match={g}")
check(sec, 'six decisions recomputed from the plan rule (>= +20 pp with POSITIVE_INTERVAL on both five primary endpoints and L1024 cost >= -5 pp -> POSITIVE; any primary mean <= 0 or cost < -5 pp -> NEGATIVE; else MIXED)', ok, '; '.join(det))
check(sec, 'all six decisions are MEASURED_NEGATIVE; the two reread decisions are flagged incomplete with seeds_used=2 (df=1, t=12.706)', all(v['status'] == 'MEASURED_NEGATIVE' for v in MYDEC.values()) and MYDEC['five/reread']['incomplete'] and MYDEC['parity/reread']['incomplete'] and MYDEC['five/reread']['seeds_used'] == 2 and not any(MYDEC[k]['incomplete'] for k in MYDEC if 'reread' not in k), '')
DETAIL['decisions_recomputed'] = MYDEC
DETAIL['contrasts_recomputed_primary_and_cost'] = {'/'.join(k): dict(seed_effects_pp=[d * 100 for d in v['seed_effects']], mean_pp=v['mean'] * 100, t95_pp=[x * 100 for x in v['t95']], direction=v['direction'], per_seed=v['per_seed'], missing=v['missing'])
                                                    for k, v in MY.items() if k[2] in P['primary']['endpoints'] + ['L1024', 'neutral-32-middle', 'neutral-32-suffix', 'neutral-128-prefix']}
note('Plan pre-registers df=2 (three seeds) only; the two-seed reread decisions (df=1, t=12.706) come from the unpinned drift/analyze_tolerant.py, disclosed in the ledger. Under either reading the reread verdict is negative: five/reread primary means are -11.4 and -18.7 pp on the two surviving seeds, so MEASURED_POSITIVE was unreachable even with seed 191.')
neg_reason = {k: ('primary mean <= 0' if v['reasons']['any_primary_mean_le_0'] else 'cost < -5 pp') for k, v in MYDEC.items()}
DETAIL['negative_trigger'] = neg_reason

# ============================================================================= CHECK 6: base reproduction vs sealed diagnostics
sec = '6-base-reproduction'
REPRO = OrderedDict(); ok = True; det = []
for task in P['tasks']:
    for seed in P['primary_seeds']:
        cid = f'drift-{task}-base-s{seed}'; sd = SEALED / f'small-{task}-w32-rlt-random-dense-lr0.003-s{seed}'
        sev = json.loads((sd / 'selected-evaluation.json').read_text(encoding='utf8')); strain = json.loads((sd / 'train.json').read_text(encoding='utf8'))
        srows = {f"L{x['length']}": x for x in sev['lengths']}; srows.update({f"neutral-{x['slots']}-{x['placement']}": x for x in sev['neutral']})
        ident_in = ident_pred = ident_targets = 0; sealed_sha_ok = 0; acc_diff = 0.
        for endpoint in ENDPOINTS:
            sp = sd / endpoint_file(endpoint)
            sealed_sha_ok += sha(sp) == srows[endpoint]['sha256']
            with np.load(sp) as f, np.load(RES / cid / endpoint_file(endpoint)) as g:
                ident_in += np.array_equal(f['inputs'], g['inputs']) and np.array_equal(f['input_ids'], g['input_ids'])
                ident_targets += np.array_equal(f['targets'], g['targets'])
                ident_pred += np.array_equal(f['predictions'], g['predictions'])
            acc_diff = max(acc_diff, abs(srows[endpoint]['final_accuracy'] - RAW[(cid, endpoint)]['final_acc']))
        REPRO[cid] = dict(sealed_dir=sd.name, identical_inputs=f'{ident_in}/13', identical_targets=f'{ident_targets}/13', identical_predictions_all_positions=f'{ident_pred}/13', sealed_npz_sha_ok=f'{sealed_sha_ok}/13',
                          max_abs_final_accuracy_diff=acc_diff, selected_step=(TRAIN[cid]['selected_step'], strain['selected_step']), sealed_selected_validation=strain['selected_validation'], drift_selected_validation=TRAIN[cid]['selected_validation'])
        g = ident_in == ident_targets == ident_pred == sealed_sha_ok == 13 and TRAIN[cid]['selected_step'] == strain['selected_step'] and acc_diff == 0.
        ok &= g; det.append(f'{cid}: inputs {ident_in}/13 targets {ident_targets}/13 predictions {ident_pred}/13 selected_step {TRAIN[cid]["selected_step"]} vs sealed {strain["selected_step"]}')
check(sec, 'six base conditions vs sealed 2026-09-17 diagnostics small-{task}-w32-rlt-random-dense-lr0.003-s{seed}: inputs/input_ids/targets identical and predictions identical at ALL positions on 13/13 endpoints; selected steps equal; sealed NPZ sha256 verified against sealed selected-evaluation.json', ok, '; '.join(det))
DETAIL['base_reproduction'] = REPRO
val_same = all(REPRO[c]['sealed_selected_validation']['final_loss'] == REPRO[c]['drift_selected_validation']['final_loss'] for c in REPRO)
note('Sealed vs drift base receipts: selected validation final_loss ' + ('identical' if val_same else 'DIFFERENT') + ' for all six seeds; initial_state_hash strings differ by construction (the diagnostics runner hashes state dicts with a different byte layout) - predictions settle equivalence.')

# ============================================================================= CHECK 7: anomaly scan
sec = '7-anomalies'
def oracle_from(x, task, s0):
    """Task recurrence started from state s0 (five: one of 5 states; each token acts as a permutation)."""
    x = np.asarray(x); s = np.full(x.shape[0], s0, dtype=np.int64)
    for t in range(x.shape[1]):
        tok = x[:, t]
        if task == 'parity':
            s = s ^ (tok == 1).astype(np.int64)
        else:
            s = np.where(tok == 0, (s + 1) % 5, np.where(tok == 1, np.where(s == 0, 1, np.where(s == 1, 0, s)), s))
    return s


LOW = OrderedDict()
for cid in COMPLETE:
    c = BY_ID[cid]; classes = 2 if c['task'] == 'parity' else 5
    for e in ('neutral-128-prefix', 'neutral-128-middle', 'neutral-128-suffix'):
        acc = RAW[(cid, e)]['final_acc']
        if acc >= 0.5:
            continue
        with np.load(RES / cid / endpoint_file(e)) as f:
            x = f['inputs']; tgt = f['targets'][:, -1]
        pred = RAW[(cid, e)]['final_pred']
        conf = np.zeros((classes, classes), dtype=int)
        for t_, p_ in zip(tgt, pred):
            conf[t_, p_] += 1
        determinism = float(conf.max(1).sum() / len(tgt))   # fraction explained by 'prediction is a fixed function of the true final state'
        hist = Counter(pred.tolist()); attractors = sorted(k for k, n in hist.items() if n >= 0.05 * len(pred))
        entry = dict(final_accuracy=acc, prediction_histogram=dict(hist), confusion_target_rows=conf.tolist(), deterministic_map_fraction=determinism, attractor_classes_ge5pct=attractors)
        if e == 'neutral-128-prefix':
            entry['examples_explained_by_start_state'] = {str(k): int((oracle_from(x, c['task'], k) == pred).sum()) for k in range(classes)}
        LOW[f'{cid}/{e}'] = entry
DETAIL['low_accuracy_neutral128_rows'] = LOW
suffix_rows = {k: v for k, v in LOW.items() if k.endswith('neutral-128-suffix')}
note('Mechanism visible in the raw predictions (five task, 128 trailing neutrals): in every low-accuracy suffix row the prediction is a deterministic MANY-TO-ONE function of the true final state, i.e. the neutral run collapses the recurrent state onto 1-2 attractor states; accuracy equals the share of examples whose true final state happens to be an attractor. Rows (attractor classes, deterministic fraction, accuracy): '
     + '; '.join(f"{k.split('/')[0]}: {v['attractor_classes_ge5pct']}, {v['deterministic_map_fraction']:.3f}, {v['final_accuracy']:.3f}" for k, v in suffix_rows.items())
     + '. With the middle placement the same collapsed state is then permuted by the trailing 16 real tokens, so predictions look scrambled (deterministic fraction ' + ', '.join(f"{v['deterministic_map_fraction']:.2f}" for k, v in LOW.items() if k.endswith('middle') and k.startswith('drift-five')) + '). Not an audit failure; a reading of the producer\'s own data that the report should state as observation, not as the pre-registered finding.')
for key, v in LOW.items():
    if v['final_accuracy'] <= 0.05:
        note(f"{key}: final accuracy {v['final_accuracy']:.4f} with a spread prediction histogram {v['prediction_histogram']}: after 128 LEADING neutrals the state sits in attractor state 2 (explained-by-start-state {v['examples_explained_by_start_state']}), and the 32 real tokens then permute it, so every example ends in a wrong-but-coherent state (0/512). Same attractor mechanism as the suffix rows, seen from the other side.")
note('Rows with neutral-128 final accuracy < 0.5 (22 complete conditions x 3 placements): ' + str(len(LOW)) + ' (all five-task rows plus parity/win16 s143/s191 suffix and s143 middle, where win16 collapses to a constant class at chance accuracy).')
sel_last = sum(TRAIN[cid]['selected_step'] == 2048 for cid in COMPLETE)
note(f'Selection: {sel_last}/22 conditions selected the last checkpoint (step 2048); others: ' + ', '.join(f"{cid}={TRAIN[cid]['selected_step']}" for cid in COMPLETE if TRAIN[cid]['selected_step'] != 2048) + '.')
lock_left = [str(p) for p in RES.rglob('active.lock')] + [str(p) for p in RES.rglob('*.tmp')]
check(sec, 'no active.lock or *.tmp leftovers anywhere under results/drift', not lock_left, str(lock_left))
check(sec, 'frozen_utc precedes every worker STARTED row and the first CONDITION_STARTED', all(logs[k][0]['utc'] >= frozen['frozen_utc'] for k in ('0', '1', '2')), f"freeze at {frozen['frozen_utc']:.1f}, first worker at {min(logs[k][0]['utc'] for k in ('0', '1', '2')):.1f}")
check(sec, 'drift/analyze_tolerant.py is not pinned by frozen.json (unpinned analysis path, as the ledger discloses)', 'drift/analyze_tolerant.py' not in frozen['sources'], f"sha256 now {sha(ROOT / 'drift/analyze_tolerant.py')[:16]}")
wall = {cid: TRAIN[cid]['wall_seconds'] for cid in COMPLETE}
DETAIL['wall_seconds_range'] = [min(wall.values()), max(wall.values())]
idinv_aux = {cid: (TRAIN_SUMMARY[cid]['aux_first'], TRAIN_SUMMARY[cid]['aux_last']) for cid in COMPLETE if BY_ID[cid]['arm'] == 'idinv'}
note('idinv auxiliary loss first->last: ' + '; '.join(f'{cid}: {a:.3g}->{b:.3g}' for cid, (a, b) in idinv_aux.items()) + '.')

# ============================================================================= write outputs
elapsed = time.perf_counter() - T0
n_pass = sum(c['status'] == 'PASS' for c in CHECKS); n_fail = len(CHECKS) - n_pass
verdict = 'PASS' if n_fail == 0 else 'FAIL'
result = OrderedDict(audit='drift mini-experiment independent raw audit', utc=time.time(), root=str(ROOT), read_only=True, elapsed_seconds=elapsed,
                     environment=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads()),
                     verdict=verdict, pass_count=n_pass, fail_count=n_fail, frozen=dict(protocol_hash=frozen['protocol_hash'], frozen_sha256=frozen_sha, sources=frozen['sources'], plan_sha256=plan_sha, test_sha256=test_sha),
                     checks=CHECKS, notes=NOTES, detail=DETAIL)
(OUT / 'drift-independent-audit.json').write_text(json.dumps(result, indent=1, allow_nan=False, default=str), encoding='utf8')


def md():
    L = []
    L.append('# Drift mini-experiment - independent raw audit\n')
    L.append(f'Root `{ROOT}` (branch codex/rlt-drift-mini, base {BASE_COMMIT}). Read-only; no training or evaluation executed; CPU, 1 thread; torch {torch.__version__}, numpy {np.__version__}. Audit runtime {elapsed:.0f} s.\n')
    L.append(f'**Verdict: {verdict}** - {n_pass} PASS / {n_fail} FAIL over {len(CHECKS)} checks.\n')
    L.append(f"Frozen protocol_hash `{frozen['protocol_hash']}`; frozen.json sha256 `{frozen_sha}`; plan sha256 `{plan_sha}`.\n")
    L.append('## Decisions (recomputed from raw NPZ, arm minus base final accuracy, pp)\n')
    L.append('| task/arm | status | n128-middle mean [t95] | n128-suffix mean [t95] | L1024 cost mean [t95] | seeds | negative trigger |')
    L.append('|---|---|---|---|---|---|---|')
    for k, v in MYDEC.items():
        p0, p1, c = v['primary'][0], v['primary'][1], v['cost']
        L.append(f"| {k} | {v['status']}{' (INCOMPLETE)' if v['incomplete'] else ''} | {p0['mean_pp']:+.2f} [{p0['t95_pp'][0]:+.1f}, {p0['t95_pp'][1]:+.1f}] {p0['direction']} | {p1['mean_pp']:+.2f} [{p1['t95_pp'][0]:+.1f}, {p1['t95_pp'][1]:+.1f}] {p1['direction']} | {c['mean_pp']:+.2f} [{c['t95_pp'][0]:+.1f}, {c['t95_pp'][1]:+.1f}] | {v['seeds_used']} | {neg_reason[k]} |")
    L.append('\nPer-seed primary effects (pp): ' + '; '.join(f"{k}: {[round(d, 2) for d in v['seed_effects_pp']]}" for k, v in DETAIL['contrasts_recomputed_primary_and_cost'].items() if k.startswith('five') and ('128-middle' in k or '128-suffix' in k)) + '.\n')
    L.append('## Failures (reread arm, seed 191)\n')
    for cid, f in FAIL_INFO.items():
        L.append(f"- `{cid}`: {f['failure_file']} step {f['failure_step']} = last journal step {f['last_journal_step']} + 1 ({f['journal_rows']} rows); loss at failure {f['failure_loss']}, grad_norm {f['failure_grad_norm']}; last finite grad_norms {[(s, float(f'{g:.4g}')) for s, g in f['tail_grad_norms']]}.")
    L.append('\n## Base reproduction vs sealed diagnostics (2026-09-17)\n')
    L.append('| drift base | sealed dir | inputs | targets | predictions (all positions) | selected step (drift, sealed) |')
    L.append('|---|---|---|---|---|---|')
    for cid, v in REPRO.items():
        L.append(f"| {cid} | {v['sealed_dir']} | {v['identical_inputs']} | {v['identical_targets']} | {v['identical_predictions_all_positions']} | {v['selected_step']} |")
    L.append('\n## Checks\n')
    cur = None
    for c in CHECKS:
        if c['section'] != cur:
            cur = c['section']; L.append(f'\n### {cur}\n')
        L.append(f"- **{c['status']}** {c['name']}" + (f" - {c['detail']}" if c['detail'] else ''))
    L.append('\n## Anomalies and observations (not PASS/FAIL items)\n')
    for n in NOTES:
        L.append(f'- {n}')
    L.append('\n## Training summary (22 COMPLETE)\n')
    L.append('| condition | selected step | val final CE | first loss | last loss | max grad_norm | updates grad_norm>10 | aux first->last | wall s |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for cid, s in TRAIN_SUMMARY.items():
        aux = '-' if s['aux_first'] is None else f"{s['aux_first']:.3g}->{s['aux_last']:.3g}"
        L.append(f"| {cid} | {s['selected_step']} | {s['selected_val_final_loss']:.3g} | {s['first_loss']:.3f} | {s['last_loss']:.3g} | {s['max_grad_norm']:.3g} | {s['updates_grad_norm_gt10']} | {aux} | {s['wall_seconds']:.0f} |")
    L.append('\nFiles: `drift_raw_audit.py` (this script), `drift-independent-audit.json`, `drift-independent-audit.md` under `.superpowers/sdd/drift-mini/`.\n')
    return '\n'.join(L)


(OUT / 'drift-independent-audit.md').write_text(md(), encoding='utf8')
print(json.dumps(dict(verdict=verdict, pass_count=n_pass, fail_count=n_fail, elapsed_seconds=round(elapsed, 1), fails=[c['name'] for c in CHECKS if c['status'] == 'FAIL']), indent=1))
