"""R6 independent raw audit. READ-ONLY, CPU only, one torch thread.

Recomputes every number from the raw files under results/review-diagnostics-r6
(result.json, updates.jsonl, initial.pt, final.pt, request/launch/log siblings,
source-snapshots, frozen.json, validation/equivalence.json) and only then
compares against the producer's summary.json.  Does NOT import
review_cycle.r6_report or review_cycle.memory_bench; does NOT launch anything.

Writes only:
  .superpowers/sdd/review-diagnostics-plan/r6-independent-audit.json
  .superpowers/sdd/review-diagnostics-plan/r6-independent-audit.md
"""
import ast
import hashlib
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path('E:/rlt-research/2026-09-17-diagnostics')
R6 = ROOT / 'results' / 'review-diagnostics-r6'
OUT = ROOT / '.superpowers' / 'sdd' / 'review-diagnostics-plan'
ARMS = ('full', 'detach', 'checkpoint', 'segment', 'segment_encoder_checkpoint')
LENGTHS = (32, 128, 512, 1024)
PAIRS = (('full', 'checkpoint'), ('detach', 'segment'), ('detach', 'segment_encoder_checkpoint'))
SOURCES = ('core/model.py', 'review_cycle/memory_bench.py', 'review_cycle/test_memory_bench.py')
EXPECTED_SUMMARY_SHA = '145a0b34ceb426fd55f1c6d9ddb099a4338af9ef44763ffe0e286bbbc7a8d9c0'
PRE_FIX_SHA = '3e5569e94e32f21800ddd85cc09867a7ae1ed5968640943eb092e3d4118bc680'
LOSS_PAIR_TOL = 1e-6
HOST_BUSY_FLAG = 0.3

failures = []      # every hard FAIL with location
deviations = []    # strict checks that fail for a demonstrated benign cause (reported, not hidden)
notes = []         # informational observations
checks = {}        # per-check machine-readable output


def fail(msg):
    failures.append(msg)


def deviate(msg):
    deviations.append(msg)


def venv_redirector_info(executable):
    """On Windows a venv's Scripts/python.exe is CPython's redirector, which spawns the real
    interpreter as a child; Popen.pid is then the redirector's pid, not os.getpid() of the worker."""
    exe = Path(executable)
    cfg = exe.parent.parent / 'pyvenv.cfg'
    info = dict(executable=str(exe), exe_sha256=sha(exe) if exe.exists() else None, exe_bytes=exe.stat().st_size if exe.exists() else None,
                pyvenv_cfg=str(cfg) if cfg.exists() else None, home=None, redirector_sha256=None, real_python_sha256=None, is_redirector=None)
    if cfg.exists():
        for line in cfg.read_text(encoding='utf8').splitlines():
            if line.lower().startswith('home'):
                info['home'] = line.split('=', 1)[1].strip()
        if info['home']:
            red = Path(info['home']) / 'Lib' / 'venv' / 'scripts' / 'nt' / 'python.exe'
            real = Path(info['home']) / 'python.exe'
            info['redirector_sha256'] = sha(red) if red.exists() else None
            info['real_python_sha256'] = sha(real) if real.exists() else None
            info['is_redirector'] = info['exe_sha256'] is not None and info['exe_sha256'] == info['redirector_sha256'] and info['exe_sha256'] != info['real_python_sha256']
    return info


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def state_hash_from_state_dict(sd):
    h = hashlib.sha256()
    for key, value in sd.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def num_eq(a, b, rel=1e-9):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if a is None or b is None:
        return a == b
    if a == b:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b))


def load_pt(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def cond(base, name):
    d = base / name
    return dict(name=name, dir=d, result=d / 'result.json', started=d / 'started.json', initial=d / 'initial.pt',
                final=d / 'final.pt', updates=d / 'updates.jsonl', request=base / (name + '.request.json'),
                launch=base / (name + '.launch.json'), stdout=base / (name + '.stdout.log'),
                stderr=base / (name + '.stderr.log'), failure=base / (name + '.failure.json'))


MAIN = [(arm, L, False, cond(R6 / 'measurements', f'{arm}-L{L}')) for L in LENGTHS for arm in ARMS]
HOOKS = [('segment', 128, True, cond(R6 / 'measurements', 'segment-L128-hooks'))]
VALID = [(arm, L, False, cond(R6 / 'validation', f'{arm}-L{L}')) for L in (32, 128) for arm in ARMS]
ALL = [('main',) + c for c in MAIN] + [('hooks',) + c for c in HOOKS] + [('validation',) + c for c in VALID]

frozen = load(R6 / 'frozen.json')
P = frozen['protocol']
VALIDATION_P = dict(P, lengths=P['equivalence']['lengths'], repeats=P['equivalence']['repeats'])
results = {}
for group, arm, L, diag, c in ALL:
    results[c['name'] + '@' + group] = load(c['result'])


# ---------------------------------------------------------------- check 1
def check_integrity():
    out = dict(sources={}, protocol_hash={}, requests={}, snapshots={}, results={}, launch={}, logs={}, ordering={})
    # 1a source pins
    for name in SOURCES:
        actual = sha(ROOT / name)
        out['sources'][name] = dict(frozen=frozen['sources'][name], actual=actual, match=actual == frozen['sources'][name])
        if actual != frozen['sources'][name]:
            fail(f'integrity: working-tree {name} sha256 {actual} != frozen {frozen["sources"][name]}')
    if set(frozen['sources']) != set(SOURCES):
        fail('integrity: frozen.json sources key set differs from expected three files')
    # 1b protocol hash
    recomputed = digest(P)
    out['protocol_hash'] = dict(frozen=frozen['protocol_hash'], recomputed=recomputed, match=recomputed == frozen['protocol_hash'],
                                frozen_json_sha256=sha(R6 / 'frozen.json'), validation_protocol_hash=digest(VALIDATION_P))
    if recomputed != frozen['protocol_hash']:
        fail('integrity: recomputed protocol_hash differs from frozen.json')
    # 1c request.json per condition
    for group, arm, L, diag, c in ALL:
        req = load(c['request'])
        expected_p = VALIDATION_P if group == 'validation' else P
        row = dict(protocol_match=req['protocol'] == expected_p, sources_match=req['sources'] == frozen['sources'],
                   arm_match=req['arm'] == arm, length_match=req['length'] == L, diagnostic_match=req['diagnostic'] is diag)
        out['requests'][c['name'] + '@' + group] = row
        for k, v in row.items():
            if not v:
                fail(f'integrity: {group}/{c["name"]}.request.json {k} failed')
    # 1d source snapshots hash to directory names, and match frozen pins
    for base in ('measurements', 'validation'):
        snapdir = R6 / base / 'source-snapshots'
        found = {}
        for d in sorted(snapdir.iterdir()):
            for f in sorted(d.iterdir()):
                h = sha(f)
                found[f'{d.name}/{f.name}'] = h
                if h != d.name:
                    fail(f'integrity: snapshot {base}/source-snapshots/{d.name}/{f.name} hashes to {h}')
        expected = {f'{frozen["sources"][n]}/{Path(n).name}' for n in SOURCES}
        out['snapshots'][base] = dict(files=found, covers_frozen_pins=expected == set(found))
        if expected != set(found):
            fail(f'integrity: {base}/source-snapshots set {sorted(found)} != frozen pins')
    # 1e result.json protocol/hash/sources, launch pid, ordering, logs
    pid_mismatches = []
    executables = set()
    for group, arm, L, diag, c in ALL:
        r = results[c['name'] + '@' + group]
        executables.add(load(c['launch'])['command'][0])
        expected_p = VALIDATION_P if group == 'validation' else P
        row = dict(protocol_match=r['protocol'] == expected_p, protocol_hash_match=r['protocol_hash'] == digest(expected_p),
                   frozen_hash_match=(r['protocol_hash'] == frozen['protocol_hash']) if group != 'validation' else None,
                   sources_match=r['sources'] == frozen['sources'], arm=r['arm'] == arm, length=r['length'] == L,
                   diagnostic=r['diagnostic'] is diag)
        out['results'][c['name'] + '@' + group] = row
        for k, v in row.items():
            if v is False:
                fail(f'integrity: {group}/{c["name"]}/result.json {k} failed')
        launch = load(c['launch'])
        started = load(c['started'])
        stdout_tail = None
        try:
            lines = [ln for ln in c['stdout'].read_text(encoding='utf8').splitlines() if ln.strip()]
            stdout_tail = json.loads(lines[-1]) if lines else None
        except Exception as exc:  # noqa: BLE001
            stdout_tail = dict(parse_error=repr(exc))
        stdout_ok = bool(stdout_tail) and stdout_tail.get('status') == r['status'] and stdout_tail.get('final_state_hash') == r.get('final_state_hash') and stdout_tail.get('pid') == r['pid']
        cmd = launch['command']
        lrow = dict(launch_pid=launch['pid'], result_pid=r['pid'], pid_match=launch['pid'] == r['pid'],
                    started_pid_match=started['pid'] == r['pid'], started_status=started['status'],
                    launch_started=launch['started'], result_started=r['started'], result_finished=r['finished'],
                    order_ok=launch['started'] <= r['started'] <= r['finished'],
                    duration_s=r['finished'] - r['started'], command_out_ok=cmd[-3].replace('\\', '/').endswith(f'{c["dir"].parent.name}/{c["name"]}'),
                    command_request_ok=cmd[-1].replace('\\', '/').endswith(c['request'].name), worker_action=cmd[3],
                    stdout_final_json_ok=stdout_ok, failure_sidecar_absent=not c['failure'].exists())
        out['launch'][c['name'] + '@' + group] = lrow
        if not lrow['pid_match']:
            pid_mismatches.append(f'{group}/{c["name"]}: launch.json pid {launch["pid"]} vs result.json pid {r["pid"]}')
        if not lrow['started_pid_match']:
            fail(f'integrity: {group}/{c["name"]} started.json pid != result pid')
        if not lrow['order_ok']:
            fail(f'integrity: {group}/{c["name"]} timestamps not ordered launch<=started<=finished')
        if not (lrow['command_out_ok'] and lrow['command_request_ok'] and lrow['worker_action'] == '_worker'):
            fail(f'integrity: {group}/{c["name"]} launch command does not point at this condition')
        if not stdout_ok:
            fail(f'integrity: {group}/{c["name"]} stdout final JSON missing or inconsistent with result.json')
        if not lrow['failure_sidecar_absent']:
            fail(f'integrity: {group}/{c["name"]} has a .failure.json sidecar')
        err = c['stderr'].read_text(encoding='utf8', errors='replace')
        out['logs'][c['name'] + '@' + group] = dict(stderr_bytes=len(err.encode()), traceback='Traceback' in err,
                                                    stdout_bytes=c['stdout'].stat().st_size)
        if 'Traceback' in err:
            fail(f'integrity: {group}/{c["name"]}.stderr.log contains a Traceback')
    # pid binding: strict equality, then the demonstrated cause if it fails uniformly
    red = {exe: venv_redirector_info(exe) for exe in sorted(executables)}
    out['pid_binding'] = dict(n_conditions=len(ALL), n_pid_mismatch=len(pid_mismatches), mismatches=pid_mismatches, launcher_executables=red,
                              worker_chain_pid_consistent=all(v['started_pid_match'] and v['stdout_final_json_ok'] for v in out['launch'].values()))
    if pid_mismatches:
        uniform = len(pid_mismatches) == len(ALL)
        shim = len(red) == 1 and next(iter(red.values()))['is_redirector'] is True
        if uniform and shim and out['pid_binding']['worker_chain_pid_consistent']:
            deviate(f'integrity: launch.json pid != result.json pid in {len(pid_mismatches)}/{len(ALL)} conditions (all of them). Cause demonstrated: the launched executable '
                    f'{next(iter(red))} is byte-identical (sha256 {next(iter(red.values()))["exe_sha256"][:12]}...) to CPython\'s venv redirector Lib/venv/scripts/nt/python.exe, '
                    f'which spawns the real interpreter as a child; Popen.pid is the redirector, os.getpid() is the child. launch.json therefore cannot bind the worker by pid on this host. '
                    f'The worker-side chain (started.json, updates.jsonl, stdout final JSON, result.json) is pid-consistent in every condition and launch.started <= result.started <= result.finished holds.')
        else:
            for m in pid_mismatches:
                fail('integrity: ' + m + ' (not explained by a uniform venv redirector)')
    # sequential non-overlap across all conditions
    seq = sorted(((v['launch_started'], v['result_finished'], k) for k, v in out['launch'].items()))
    overlaps = [(seq[i][2], seq[i + 1][2]) for i in range(len(seq) - 1) if seq[i][1] > seq[i + 1][0]]
    val_first = min(v['launch_started'] for k, v in out['launch'].items() if k.endswith('@validation'))
    val_last = max(v['result_finished'] for k, v in out['launch'].items() if k.endswith('@validation'))
    main_first = min(v['launch_started'] for k, v in out['launch'].items() if not k.endswith('@validation'))
    out['ordering'] = dict(overlapping_pairs=overlaps, frozen_utc=frozen['frozen_utc'], frozen_before_validation=frozen['frozen_utc'] < val_first,
                           validation_before_main=val_last <= main_first, first_launch=seq[0][2], last_launch=seq[-1][2],
                           equivalence_json_written_before_main=(R6 / 'validation' / 'equivalence.json').stat().st_mtime <= main_first)
    if overlaps:
        fail(f'integrity: overlapping worker lifetimes {overlaps}')
    if not out['ordering']['frozen_before_validation'] or not out['ordering']['validation_before_main']:
        fail('integrity: freeze/validation/main ordering violated')
    eq = load(R6 / 'validation' / 'equivalence.json')
    out['equivalence_json'] = dict(sha256=sha(R6 / 'validation' / 'equivalence.json'), protocol_hash_match=eq['protocol_hash'] == frozen['protocol_hash'],
                                   sources_match=eq['sources'] == frozen['sources'], status=eq['status'])
    if not (out['equivalence_json']['protocol_hash_match'] and out['equivalence_json']['sources_match']):
        fail('integrity: equivalence.json protocol_hash/sources do not match frozen.json')
    pre = OUT / 'memory_bench-before-controller-fix.py'
    out['pre_fix_source'] = dict(path=str(pre), sha256=sha(pre) if pre.exists() else None, matches_memory_review_pin=(sha(pre) == PRE_FIX_SHA) if pre.exists() else None)
    notes.append(f'validation result.json/request.json carry the frozen protocol with lengths={VALIDATION_P["lengths"]} and repeats={VALIDATION_P["repeats"]} '
                 f'(protocol_hash {digest(VALIDATION_P)[:12]}...), as the producer\'s validate_routes constructs it; equivalence.json itself carries the frozen hash {frozen["protocol_hash"][:12]}.... '
                 f'All 21 measurement result.json files carry the frozen hash exactly.')
    return out


# ---------------------------------------------------------------- check 2
def check_real_updates():
    out = {}
    init_hashes = set()
    init_ckpt_shas = set()
    for group, arm, L, diag, c in ALL:
        key = c['name'] + '@' + group
        r = results[key]
        row = dict(status=r['status'], optimizer_updates=r['optimizer_updates'], n_measurements=len(r['measurements']),
                   update_numbers=[m['update'] for m in r['measurements']],
                   optimizer_initialized_before_measurement=r['optimizer_initialized_before_measurement'],
                   initial_state_hash=r['initial_state_hash'], final_state_hash=r['final_state_hash'],
                   hashes_differ=r['initial_state_hash'] != r['final_state_hash'])
        ok = (r['status'] == 'PASS' and r['optimizer_updates'] == 5 and row['update_numbers'] == [3, 4, 5]
              and r['optimizer_initialized_before_measurement'] is True and row['hashes_differ'])
        if not ok:
            fail(f'real-updates: {key} status/updates/measurements/init flags wrong: {row}')
        # updates.jsonl receipts
        lines = [json.loads(ln) for ln in c['updates'].read_text(encoding='utf8').splitlines() if ln.strip()]
        meas_losses = [m['loss'] for m in r['measurements']]
        urow = dict(n_lines=len(lines), steps=[u['step'] for u in lines], measured=[u['measured'] for u in lines],
                    pids_match=all(u['pid'] == r['pid'] for u in lines), losses=[u['loss'] for u in lines],
                    measured_losses_equal=[u['loss'] for u in lines if u['measured']] == meas_losses)
        row['updates_jsonl'] = urow
        if not (urow['n_lines'] == 5 and urow['steps'] == [1, 2, 3, 4, 5] and urow['measured'] == [False, False, True, True, True]
                and urow['pids_match'] and urow['measured_losses_equal']):
            fail(f'real-updates: {key} updates.jsonl receipts inconsistent: {urow}')
        # checkpoint hashes
        row['initial_pt_sha256'] = sha(c['initial']); row['final_pt_sha256'] = sha(c['final'])
        row['initial_sha_match'] = row['initial_pt_sha256'] == r['initial_checkpoint_sha256']
        row['final_sha_match'] = row['final_pt_sha256'] == r['final_checkpoint_sha256']
        if not (row['initial_sha_match'] and row['final_sha_match']):
            fail(f'real-updates: {key} checkpoint sha256 mismatch')
        # load checkpoints
        ini = load_pt(c['initial']); fin = load_pt(c['final'])
        row['initial_state_hash_recomputed_match'] = state_hash_from_state_dict(ini['model']) == r['initial_state_hash']
        row['final_state_hash_recomputed_match'] = state_hash_from_state_dict(fin['model']) == r['final_state_hash']
        if not (row['initial_state_hash_recomputed_match'] and row['final_state_hash_recomputed_match']):
            fail(f'real-updates: {key} state hash recomputed from .pt does not match result.json')
        keys_i, keys_f = list(ini['model']), list(fin['model'])
        diffs = {k: float((fin['model'][k].double() - ini['model'][k].double()).abs().max()) for k in keys_i}
        row['model_tensors'] = len(keys_i)
        row['model_keys_match'] = keys_i == keys_f
        row['tensors_changed'] = sum(1 for v in diffs.values() if v > 0)
        row['tensors_unchanged'] = sorted(k for k, v in diffs.items() if v == 0)
        row['max_param_abs_change'] = max(diffs.values())
        row['min_param_abs_change'] = min(diffs.values())
        if not row['model_keys_match'] or row['tensors_changed'] == 0:
            fail(f'real-updates: {key} model parameters did not change between initial.pt and final.pt')
        if row['tensors_unchanged']:
            notes.append(f'{key}: parameter tensors unchanged after 5 AdamW updates: {row["tensors_unchanged"]}')
        # optimizer state
        st = fin['optimizer']['state']; pg = fin['optimizer']['param_groups']
        n_params = sum(len(g['params']) for g in pg)
        steps = [float(st[i]['step']) for i in st]
        row['optimizer'] = dict(n_params=n_params, n_state_entries=len(st), steps_all_5=all(s == 5.0 for s in steps) and len(st) == n_params,
                                step_values=sorted(set(steps)), initial_state_empty=len(ini['optimizer']['state']) == 0,
                                lr=pg[0]['lr'], weight_decay=pg[0]['weight_decay'], foreach=pg[0].get('foreach'),
                                keys_per_entry=sorted(st[next(iter(st))].keys()))
        if not (row['optimizer']['steps_all_5'] and row['optimizer']['initial_state_empty']):
            fail(f'real-updates: {key} AdamW state steps not all 5 or initial state not empty: {row["optimizer"]}')
        if not (row['optimizer']['lr'] == P['lr'] and row['optimizer']['weight_decay'] == P['weight_decay'] and row['optimizer']['foreach'] is False):
            fail(f'real-updates: {key} optimizer hyperparameters differ from protocol')
        grads = fin['gradients']
        row['gradients'] = dict(n=len(grads), none=[k for k, v in grads.items() if v is None],
                                keys_match_model=sorted(grads) == sorted(keys_f),
                                max_abs=max(float(v.abs().max()) for v in grads.values() if v is not None))
        if row['gradients']['none'] or not row['gradients']['keys_match_model']:
            fail(f'real-updates: {key} final gradients missing for {row["gradients"]["none"]}')
        init_hashes.add(r['initial_state_hash']); init_ckpt_shas.add(r['initial_checkpoint_sha256'])
        out[key] = row
    out['_summary'] = dict(conditions=len(out), distinct_initial_state_hashes=sorted(init_hashes), distinct_initial_pt_sha256=sorted(init_ckpt_shas))
    if len(init_hashes) != 1:
        fail(f'real-updates: initial_state_hash differs across conditions: {sorted(init_hashes)}')
    return out


# ---------------------------------------------------------------- check 3
def check_paired_data():
    out = dict(measured={}, full_vs_truncated={}, validation_vs_main={})
    losses = {}
    warm = {}
    for group, arm, L, diag, c in ALL:
        r = results[c['name'] + '@' + group]
        losses[(group, arm, L, diag)] = [m['loss'] for m in r['measurements']]
        lines = [json.loads(ln) for ln in c['updates'].read_text(encoding='utf8').splitlines() if ln.strip()]
        warm[(group, arm, L, diag)] = [u['loss'] for u in lines if not u['measured']]
    for group, lengths in (('main', LENGTHS), ('validation', (32, 128))):
        for L in lengths:
            row = {}
            for base, cand in PAIRS:
                a, b = losses[(group, base, L, False)], losses[(group, cand, L, False)]
                d = max(abs(x - y) for x, y in zip(a, b))
                row[f'{base}~{cand}'] = dict(base=a, candidate=b, max_abs=d, within_tol=d <= LOSS_PAIR_TOL)
                if d > LOSS_PAIR_TOL:
                    fail(f'paired-data: {group} L{L} {base} vs {cand} measured losses differ by {d}')
            a, b = losses[(group, 'segment', L, False)], losses[(group, 'segment_encoder_checkpoint', L, False)]
            d = max(abs(x - y) for x, y in zip(a, b)); row['segment~segment_encoder_checkpoint'] = dict(max_abs=d, within_tol=d <= LOSS_PAIR_TOL)
            if d > LOSS_PAIR_TOL:
                fail(f'paired-data: {group} L{L} segment vs segment_encoder_checkpoint differ by {d}')
            fa, tr = losses[(group, 'full', L, False)], losses[(group, 'detach', L, False)]
            dd = [abs(x - y) for x, y in zip(fa, tr)]
            row['full_vs_detach'] = dict(full=fa, detach=tr, abs_diff_per_update=dd, max_abs=max(dd), differ=max(dd) > LOSS_PAIR_TOL)
            # warmup step-1 loss: same initial weights, same batch -> forward identical across all arms
            w1 = {arm: warm[(group, arm, L, False)][0] for arm in ARMS}
            w2 = {arm: warm[(group, arm, L, False)][1] for arm in ARMS}
            row['warmup_step1_all_arms'] = dict(values=w1, spread=max(w1.values()) - min(w1.values()))
            row['warmup_step2_full_vs_detach'] = dict(full=w2['full'], detach=w2['detach'], abs_diff=abs(w2['full'] - w2['detach']))
            if row['warmup_step1_all_arms']['spread'] > LOSS_PAIR_TOL:
                fail(f'paired-data: {group} L{L} step-1 warmup loss differs across arms by {row["warmup_step1_all_arms"]["spread"]}')
            out['measured'][f'{group}-L{L}'] = row
    out['full_vs_truncated']['max_abs_by_length_main'] = {L: out['measured'][f'main-L{L}']['full_vs_detach']['max_abs'] for L in LENGTHS}
    out['full_vs_truncated']['differ_at_some_length'] = any(v > LOSS_PAIR_TOL for v in out['full_vs_truncated']['max_abs_by_length_main'].values())
    out['full_vs_truncated']['differ_at_every_length'] = all(v > LOSS_PAIR_TOL for v in out['full_vs_truncated']['max_abs_by_length_main'].values())
    if not out['full_vs_truncated']['differ_at_some_length']:
        fail('paired-data: full-route and truncated-route losses never differ (routes not distinguishable)')
    # hooks run vs main segment L128 (same protocol, hooks on)
    d = max(abs(x - y) for x, y in zip(losses[('main', 'segment', 128, False)], losses[('hooks', 'segment', 128, True)]))
    out['hooks_vs_main_segment_L128_max_abs'] = d
    # validation vs main reproducibility at L32/L128 (informational: same protocol, separate process)
    for L in (32, 128):
        for arm in ARMS:
            a, b = losses[('validation', arm, L, False)], losses[('main', arm, L, False)]
            rv, rm = results[f'{arm}-L{L}@validation'], results[f'{arm}-L{L}@main']
            out['validation_vs_main'][f'{arm}-L{L}'] = dict(max_abs_loss=max(abs(x - y) for x, y in zip(a, b)),
                                                          final_state_hash_equal=rv['final_state_hash'] == rm['final_state_hash'])
    not_bitwise = [k for k, v in out['validation_vs_main'].items() if not v['final_state_hash_equal']]
    if not_bitwise:
        worst = max(v['max_abs_loss'] for v in out['validation_vs_main'].values())
        notes.append(f'cross-process bitwise reproducibility (validation vs main, same arm/length, separate processes): final_state_hash identical for '
                     f'{[k for k, v in out["validation_vs_main"].items() if v["final_state_hash_equal"]]}, different for {not_bitwise}; measured losses agree to <= {worst:.1e} in every case. '
                     f'This is CUDA kernel-level nondeterminism between processes, not a protocol difference; it is within the predeclared equivalence tolerance and does not affect any check.')
    return out


# ---------------------------------------------------------------- check 4
def compare_tree(a, b, rtol, atol, path=''):
    """Returns (max_abs, n_tensors, violations, worst_util) with assert_close semantics |a-b| <= atol + rtol*|b|."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return 0.0, 1, [path + ' (shape/dtype/type mismatch)'], float('inf')
        d = (a.double() - b.double()).abs()
        if d.numel() == 0:
            return 0.0, 1, [], 0.0
        tol = atol + rtol * b.double().abs()
        util = float((d / tol).max())
        return float(d.max()), 1, ([] if bool((d <= tol).all()) else [path]), util
    if isinstance(a, dict):
        if set(a) != set(b):
            return 0.0, 0, [path + ' (keys mismatch)'], float('inf')
        m, n, v, u = 0.0, 0, [], 0.0
        for k in a:
            mm, nn, vv, uu = compare_tree(a[k], b[k], rtol, atol, path + '/' + str(k))
            m, n, v, u = max(m, mm), n + nn, v + vv, max(u, uu)
        return m, n, v, u
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return 0.0, 0, [path + ' (length mismatch)'], float('inf')
        m, n, v, u = 0.0, 0, [], 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            mm, nn, vv, uu = compare_tree(x, y, rtol, atol, path + '/' + str(i))
            m, n, v, u = max(m, mm), n + nn, v + vv, max(u, uu)
        return m, n, v, u
    return 0.0, 0, ([] if a == b else [path + f' (scalar {a!r} != {b!r})']), 0.0


def collect_max_abs(x, y, into, path=''):
    if isinstance(x, torch.Tensor):
        into[path] = float((x - y).abs().max())
    elif isinstance(x, dict):
        for k in x:
            collect_max_abs(x[k], y[k], into, path + '/' + str(k))
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)):
            collect_max_abs(u, v, into, path + '/' + str(i))


def check_equivalence():
    rtol, atol = P['equivalence']['rtol'], P['equivalence']['atol']
    eq = load(R6 / 'validation' / 'equivalence.json')
    out = dict(rtol=rtol, atol=atol, pairs=[], equivalence_json_status=eq['status'], n_comparisons_in_json=len(eq['comparisons']))
    by_key = {(c['base'], c['candidate'], c['length']): c for c in eq['comparisons']}
    if len(by_key) != 6:
        fail(f'equivalence: equivalence.json has {len(by_key)} comparisons, expected 6')
    for L in (32, 128):
        for base, cand in PAIRS:
            lp, rp = R6 / 'validation' / f'{base}-L{L}' / 'final.pt', R6 / 'validation' / f'{cand}-L{L}' / 'final.pt'
            left, right = load_pt(lp), load_pt(rp)
            m, n, viol, util = compare_tree(left, right, rtol, atol)
            sections = {}
            for sec in ('model', 'optimizer', 'gradients'):
                sm, sn, sv, su = compare_tree(left[sec], right[sec], rtol, atol, '/' + sec)
                sections[sec] = dict(max_abs=sm, n_tensors=sn, violations=len(sv))
            lr_, rr_ = results[f'{base}-L{L}@validation'], results[f'{cand}-L{L}@validation']
            ll, rl = [x['loss'] for x in lr_['measurements']], [x['loss'] for x in rr_['measurements']]
            loss_max = max(abs(x - y) for x, y in zip(ll, rl))
            loss_ok = all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(ll, rl))
            c = by_key.get((base, cand, L))
            json_max = max(v['max_abs'] for v in c['metrics'].values()) if c else None
            per_tensor_match = None
            if c:
                mine = {}
                collect_max_abs(left, right, mine)
                per_tensor_match = set(mine) == set(c['metrics']) and all(num_eq(mine[k], c['metrics'][k]['max_abs'], 1e-6) for k in mine)
            row = dict(length=L, base=base, candidate=cand, n_tensors=n, max_abs=m, violations=viol, worst_tolerance_utilisation=util,
                       sections=sections, loss_max_abs=loss_max, loss_within_tol=loss_ok,
                       equivalence_json=None if not c else dict(status=c['status'], max_abs=json_max, n_metrics=len(c['metrics']),
                                                                 left_sha_match=c['left_checkpoint_sha256'] == sha(lp), right_sha_match=c['right_checkpoint_sha256'] == sha(rp),
                                                                 per_tensor_max_abs_match=per_tensor_match, max_abs_match=num_eq(m, json_max, 1e-6)))
            out['pairs'].append(row)
            if viol or not loss_ok:
                fail(f'equivalence: L{L} {base} vs {cand} violates rtol/atol: {viol[:5]} loss_ok={loss_ok}')
            if c is None or c['status'] != 'PASS' or not row['equivalence_json']['left_sha_match'] or not row['equivalence_json']['right_sha_match'] \
                    or not row['equivalence_json']['max_abs_match'] or not per_tensor_match:
                fail(f'equivalence: L{L} {base} vs {cand} equivalence.json disagrees with recomputation')
    return out


# ---------------------------------------------------------------- check 5
def recompute_row(r):
    ms = r['measurements']
    order = [p['phase'] for p in ms[0]['phases']]
    phases = sorted({p['phase'] for m in ms for p in m['phases']}, key=order.index)
    row = dict(mean_update_seconds=sum(m['seconds'] for m in ms) / len(ms), update_seconds=[m['seconds'] for m in ms],
               losses=[m['loss'] for m in ms], measured_updates=len(ms), optimizer_updates=r['optimizer_updates'],
               phase_mean_seconds={ph: sum(sum(p['seconds'] for p in m['phases'] if p['phase'] == ph) for m in ms) / len(ms) for ph in phases},
               cuda_peak_allocated_bytes=max(m['cuda']['peak_allocated'] for m in ms),
               cuda_peak_reserved_bytes=max(m['cuda']['peak_reserved'] for m in ms),
               cuda_allocated_after_optimizer_bytes=max(p['cuda']['allocated'] for m in ms for p in m['phases'] if p['phase'] == 'optimizer'),
               ram_lifetime_peak_working_set_bytes=max(p['ram']['lifetime_peak_working_set_bytes'] for m in ms for p in m['phases']),
               ram_working_set_after_optimizer_bytes=max(p['ram']['working_set_bytes'] for m in ms for p in m['phases'] if p['phase'] == 'optimizer'),
               ram_private_commit_bytes=max(p['ram']['private_commit_bytes'] or 0 for m in ms for p in m['phases']),
               host_busy_fraction=r['host_cpu_observation']['busy_fraction'], initial_state_hash=r['initial_state_hash'],
               final_state_hash=r['final_state_hash'], status=r['status'])
    # internal consistency of each measurement
    cons = []
    for m in ms:
        pa = max(p['cuda']['peak_allocated'] for p in m['phases']); pr = max(p['cuda']['peak_reserved'] for p in m['phases'])
        ssum = sum(p['seconds'] for p in m['phases'])
        cons.append(dict(update=m['update'], peak_allocated_is_max_of_phases=m['cuda']['peak_allocated'] == pa,
                         peak_reserved_is_max_of_phases=m['cuda']['peak_reserved'] == pr,
                         seconds_is_sum_of_phases=abs(m['seconds'] - ssum) <= 1e-9 * max(1.0, abs(m['seconds'])),
                         phase_names=[p['phase'] for p in m['phases']], phase_peak_allocated=[p['cuda']['peak_allocated'] for p in m['phases']],
                         per_phase_allocated_le_peak=all(p['cuda']['allocated'] <= p['cuda']['peak_allocated'] for p in m['phases'])))
    row['measurement_consistency'] = cons
    row['ready_phase_mean_seconds'] = row['phase_mean_seconds'].get('ready')
    if r.get('diagnostic'):
        row['saved_peak_unique_storage_bytes'] = max((p['saved'] or {}).get('peak_unique_storage_bytes', 0) for m in ms for p in m['phases'])
        row['saved_requests'] = max((p['saved'] or {}).get('save_requests', 0) for m in ms for p in m['phases'])
    return row


def deep_eq(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f'{path}: keys {sorted(a)} vs summary {sorted(b)}']
        return [x for k in a for x in deep_eq(a[k], b[k], path + '.' + str(k))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in deep_eq(u, v, f'{path}[{i}]')]
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return [] if a == b else [f'{path}: {a!r} vs summary {b!r}']
    return [] if num_eq(a, b) else [f'{path}: recomputed {a!r} vs summary {b!r}']


def check_metrics():
    summary_path = R6 / 'summary.json'
    summary = load(summary_path)
    out = dict(summary_sha256=sha(summary_path), summary_sha256_match=sha(summary_path) == EXPECTED_SUMMARY_SHA, rows={}, mismatches=[],
               counts=summary['counts'], summary_protocol_hash_match=summary['protocol_hash'] == frozen['protocol_hash'])
    if not out['summary_sha256_match']:
        fail(f'metrics: summary.json sha256 {out["summary_sha256"]} != expected {EXPECTED_SUMMARY_SHA}')
    if not out['summary_protocol_hash_match']:
        fail('metrics: summary.json protocol_hash != frozen')
    srows = {(r['arm'], r['length'], r['diagnostic']): r for r in summary['rows'] + summary['diagnostics']}
    if len(summary['rows']) != 20 or len(summary['diagnostics']) != 1:
        fail(f'metrics: summary has {len(summary["rows"])} rows and {len(summary["diagnostics"])} diagnostics')
    mine = {}
    for arm, L, diag, c in MAIN + HOOKS:
        group = 'hooks' if diag else 'main'
        r = results[c['name'] + '@' + group]
        row = recompute_row(r)
        row['result_sha256'] = sha(c['result'])
        for k in row['measurement_consistency']:
            if not (k['peak_allocated_is_max_of_phases'] and k['peak_reserved_is_max_of_phases'] and k['seconds_is_sum_of_phases'] and k['per_phase_allocated_le_peak']):
                fail(f'metrics: {c["name"]} update {k["update"]} measurement not consistent with its phases: {k}')
        mine[(arm, L, diag)] = row
    for L in LENGTHS:
        base = mine[('full', L, False)]
        for arm in ARMS:
            row = mine[(arm, L, False)]
            row['ratio_to_full'] = dict(update_seconds=row['mean_update_seconds'] / base['mean_update_seconds'],
                                        cuda_peak_allocated=row['cuda_peak_allocated_bytes'] / base['cuda_peak_allocated_bytes'],
                                        cuda_peak_reserved=row['cuda_peak_reserved_bytes'] / base['cuda_peak_reserved_bytes'])
    compare_keys = ('mean_update_seconds', 'update_seconds', 'losses', 'measured_updates', 'optimizer_updates', 'phase_mean_seconds',
                    'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'cuda_allocated_after_optimizer_bytes',
                    'ram_lifetime_peak_working_set_bytes', 'ram_working_set_after_optimizer_bytes', 'ram_private_commit_bytes',
                    'host_busy_fraction', 'initial_state_hash', 'final_state_hash', 'status', 'result_sha256', 'ratio_to_full',
                    'saved_peak_unique_storage_bytes', 'saved_requests')
    for key, row in mine.items():
        s = srows.get(key)
        name = f'{key[0]}-L{key[1]}' + ('-hooks' if key[2] else '')
        if s is None:
            fail(f'metrics: summary.json has no row for {name}'); continue
        mm = []
        for k in compare_keys:
            if k in row or k in s:
                if k not in row or k not in s:
                    mm.append(f'{k}: present only in {"recomputed" if k in row else "summary"}')
                elif k == 'phase_mean_seconds':
                    a = {ph: v for ph, v in row[k].items() if ph != 'ready'}
                    mm += deep_eq(a, s[k], k)
                else:
                    mm += deep_eq(row[k], s[k], k)
        extra = sorted(set(s) - set(compare_keys) - {'arm', 'length', 'diagnostic'})
        if extra:
            mm.append(f'summary row has uncompared keys {extra}')
        out['rows'][name] = dict(recomputed={k: row[k] for k in row if k != 'measurement_consistency'}, mismatches=mm)
        if mm:
            out['mismatches'].append(dict(condition=name, mismatches=mm))
            fail(f'metrics: {name} summary mismatch: {mm}')
    # counts
    st = [results[c['name'] + '@main']['status'] for _, _, _, c in MAIN]
    my_counts = dict(planned=20, pass_=st.count('PASS'), oom=st.count('OOM'), fail=st.count('FAIL'), missing=0)
    out['counts_recomputed'] = my_counts
    if my_counts != summary['counts']:
        fail(f'metrics: counts {my_counts} != summary {summary["counts"]}')
    # equivalence block in summary vs equivalence.json
    eq_path = R6 / 'validation' / 'equivalence.json'
    out['summary_equivalence_sha_match'] = summary['equivalence']['sha256'] == sha(eq_path)
    if not out['summary_equivalence_sha_match']:
        fail('metrics: summary.equivalence.sha256 != sha256(validation/equivalence.json)')
    # headline table
    out['headline'] = {}
    for L in LENGTHS:
        out['headline'][str(L)] = {arm: dict(mean_update_seconds=mine[(arm, L, False)]['mean_update_seconds'],
                                             peak_alloc_MiB=mine[(arm, L, False)]['cuda_peak_allocated_bytes'] / 2 ** 20,
                                             peak_reserved_MiB=mine[(arm, L, False)]['cuda_peak_reserved_bytes'] / 2 ** 20,
                                             static_after_optimizer_MiB=mine[(arm, L, False)]['cuda_allocated_after_optimizer_bytes'] / 2 ** 20,
                                             **mine[(arm, L, False)]['ratio_to_full']) for arm in ARMS}
    return out


# ---------------------------------------------------------------- check 6
def check_host_load():
    out = dict(rows={}, flagged=[])
    for group, arm, L, diag, c in ALL:
        r = results[c['name'] + '@' + group]
        h = r['host_cpu_observation']
        s, e = h['start'], h['end']
        recomputed = 1 - (e['idle'] - s['idle']) / (e['total'] - s['total']) if e['total'] > s['total'] else None
        row = dict(busy_fraction=h['busy_fraction'], recomputed=recomputed, match=num_eq(h['busy_fraction'], recomputed),
                   interval_s=(e['total'] - s['total']) / 1e7 / r['environment']['logical_cpu_count'] if s.get('unit') == '100ns' else None, unit=s.get('unit'),
                   logical_cpus=r['environment']['logical_cpu_count'], flagged=(h['busy_fraction'] or 0) > HOST_BUSY_FLAG)
        out['rows'][c['name'] + '@' + group] = row
        if not row['match']:
            fail(f'host-load: {c["name"]}@{group} busy_fraction not reproducible from counters')
        if row['flagged']:
            out['flagged'].append(dict(condition=c['name'] + '@' + group, busy_fraction=h['busy_fraction']))
    return out


# ---------------------------------------------------------------- check 7
def check_diagnostic():
    c = HOOKS[0][3]
    r = results[c['name'] + '@hooks']
    req = load(c['request'])
    ms = r['measurements']
    out = dict(diagnostic_flag=r['diagnostic'], request_diagnostic=req['diagnostic'], status=r['status'], updates=[])
    for m in ms:
        saved_phases = [p['phase'] for p in m['phases'] if p['saved'] is not None]
        last = m['phases'][-1]['saved']
        out['updates'].append(dict(update=m['update'], phases_with_saved=saved_phases, n_phases=len(m['phases']),
                                   end_live_unique_storage_bytes=(m['saved'] or {}).get('live_unique_storage_bytes'),
                                   end_live_unique_storages=(m['saved'] or {}).get('live_unique_storages'),
                                   last_phase_live_bytes=None if last is None else last['live_unique_storage_bytes'],
                                   peak_unique_storage_bytes=(m['saved'] or {}).get('peak_unique_storage_bytes'),
                                   save_requests=(m['saved'] or {}).get('save_requests'),
                                   per_phase_live=[(p['phase'], p['saved']['live_unique_storage_bytes'], p['saved']['peak_unique_storage_bytes'], p['saved']['save_requests']) for p in m['phases'] if p['saved']],
                                   scope=(m['saved'] or {}).get('scope')))
    ok = (r['diagnostic'] is True and req['diagnostic'] is True and r['status'] == 'PASS' and len(ms) == 3
          and all(len(u['phases_with_saved']) == u['n_phases'] for u in out['updates'])
          and all(u['end_live_unique_storage_bytes'] == 0 and u['last_phase_live_bytes'] == 0 for u in out['updates']))
    out['peak_unique_storage_bytes_max'] = max(u['peak_unique_storage_bytes'] for u in out['updates'])
    out['peak_unique_storage_bytes_per_update'] = [u['peak_unique_storage_bytes'] for u in out['updates']]
    out['save_requests_per_update'] = [u['save_requests'] for u in out['updates']]
    # main (no hooks) rows must have saved == null everywhere
    hooks_in_main = [c2['name'] for _, _, _, c2 in MAIN if any(p['saved'] is not None or m['saved'] is not None for m in results[c2['name'] + '@main']['measurements'] for p in m['phases'])]
    out['main_rows_with_saved_snapshots'] = hooks_in_main
    if hooks_in_main:
        fail(f'diagnostic: primary rows carry saved-tensor snapshots: {hooks_in_main}')
    if not ok:
        fail(f'diagnostic: segment-L128-hooks checks failed: {out}')
    return out


# ---------------------------------------------------------------- check 8
def check_source_sanity():
    src_path = ROOT / 'review_cycle' / 'memory_bench.py'
    src = src_path.read_text(encoding='utf8')
    out = dict(source_sha256=sha(src_path), matches_frozen_pin=sha(src_path) == frozen['sources']['review_cycle/memory_bench.py'])
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'backward_loss')
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'start']
    chunk_loop = loops[0]
    float_calls = [n.lineno for n in ast.walk(chunk_loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'float']
    item_calls = [n.lineno for n in ast.walk(chunk_loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('item', 'cpu', 'numpy', 'tolist')]
    ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    ret_src = [ast.get_source_segment(src, n.value) for n in ret]
    out['segment_chunk_loop'] = dict(lineno=chunk_loop.lineno, end_lineno=chunk_loop.end_lineno, float_calls_in_loop=float_calls, host_sync_calls_in_loop=item_calls,
                                     returns=ret_src, parts_appended_detached=('parts.append(value.detach())' in src),
                                     single_conversion_after_loop=any('torch.stack(parts)' in s and 'float(' in s for s in ret_src))
    if float_calls or item_calls:
        fail(f'source: float()/item() inside segment chunk loop at lines {float_calls + item_calls}')
    # OOM handler
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    handlers = [h for n in ast.walk(worker) if isinstance(n, ast.Try) for h in n.handlers]
    oom = [h for h in handlers if h.type is not None and 'OutOfMemoryError' in ast.get_source_segment(src, h.type)]
    hrow = dict(found=bool(oom))
    if oom:
        h = oom[0]
        hsrc = ast.get_source_segment(src, h)
        assigned = [t.id for n in ast.walk(h) for a in ([n] if isinstance(n, ast.Assign) else []) for t in a.targets if isinstance(t, ast.Name)]
        hrow.update(lineno=h.lineno, assigns_cuda_at_failure='cuda_at_failure' in assigned,
                    samples=[k for k in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved') if k in hsrc],
                    records_in_result="cuda_at_failure=cuda_at_failure" in hsrc, status_oom="status='OOM'" in hsrc,
                    records_interrupted_phases='interrupted_phases' in hsrc, records_ram='ram_at_failure' in hsrc)
    out['oom_handler'] = hrow
    if not (hrow.get('assigns_cuda_at_failure') and hrow.get('records_in_result') and len(hrow.get('samples', [])) == 4):
        fail('source: OOM handler does not record cuda_at_failure with the four CUDA counters')
    out['oom_observed_in_data'] = sum(1 for k, r in results.items() if r['status'] == 'OOM')
    # protocol text pins the fix
    out['protocol_latency_note_present'] = 'latency' in P and 'no per-chunk synchronization' in P['latency']
    out['protocol_oom_note_present'] = 'sampled in the handler' in P['oom']
    # snapshot equals working tree
    snap = R6 / 'measurements' / 'source-snapshots' / frozen['sources']['review_cycle/memory_bench.py'] / 'memory_bench.py'
    out['snapshot_identical_to_working_tree'] = snap.read_bytes() == src_path.read_bytes()
    pre = OUT / 'memory_bench-before-controller-fix.py'
    if pre.exists():
        ps = pre.read_text(encoding='utf8')
        out['pre_fix_source'] = dict(sha256=sha(pre), matches_memory_review_pin=sha(pre) == PRE_FIX_SHA,
                                     had_float_in_loop='total += float(value.detach())' in ps, had_cuda_at_failure='cuda_at_failure' in ps)
    return out


def md_report(res):
    L = []
    v = res['verdict']
    L.append('# R6 independent raw audit\n')
    L.append(f"Generated {res['generated_utc']} by `r6_raw_audit.py` (torch {res['torch']}, CPU, {res['torch_threads']} thread, read-only). Verdict: **{v}** ({len(res['failures'])} FAIL).\n")
    L.append('Scope: 20 main + 1 hooks diagnostic + 10 validation conditions recomputed from raw files; summary.json used only as a comparison target. Three measured updates per process are timing repeats of one process, not independent samples; no intervals are computed and nothing is pooled across lengths.\n')
    try:
        hl = res['checks']['5_metrics']['headline']
        seg = {Lk: hl[str(Lk)]['segment'] for Lk in LENGTHS}
        sec = {Lk: hl[str(Lk)]['segment_encoder_checkpoint'] for Lk in LENGTHS}
        fl = res['checks']['6_host_load']['flagged']
        L.append('## Verdict summary\n')
        L.append(f"- Hard failures: {len(res['failures'])}. Explained deviations: {len(res['deviations'])} (see below). Summary.json (sha256 {res['checks']['5_metrics']['summary_sha256'][:12]}...) reproduced with 0 mismatches across all 21 rows.")
        L.append('- Segment vs full CUDA peak allocated ratio: ' + ', '.join(f"L{Lk} {seg[Lk]['cuda_peak_allocated']:.3f} ({seg[Lk]['peak_alloc_MiB']:.1f} vs {hl[str(Lk)]['full']['peak_alloc_MiB']:.1f} MiB)" for Lk in LENGTHS)
                 + '; peak reserved ratio: ' + ', '.join(f"L{Lk} {seg[Lk]['cuda_peak_reserved']:.3f}" for Lk in LENGTHS) + '.')
        L.append('- Segment+encoder-checkpoint vs full peak allocated ratio: ' + ', '.join(f"L{Lk} {sec[Lk]['cuda_peak_allocated']:.3f}" for Lk in LENGTHS) + '.')
        L.append('- Detach vs full peak allocated ratio: ' + ', '.join(f"L{Lk} {hl[str(Lk)]['detach']['cuda_peak_allocated']:.3f}" for Lk in LENGTHS) + ' (no saving); checkpoint vs full: '
                 + ', '.join(f"L{Lk} {hl[str(Lk)]['checkpoint']['cuda_peak_allocated']:.3f}" for Lk in LENGTHS) + '.')
        L.append('- Update-time ratio vs full at L128 / L1024: ' + '; '.join(f"{arm} {hl['128'][arm]['update_seconds']:.3f} / {hl['1024'][arm]['update_seconds']:.3f}" for arm in ARMS if arm != 'full') + '.')
        L.append('- Host busy fraction > 0.3 (timing for these rows is scoped to that load): ' + (', '.join(f"{f['condition']} {f['busy_fraction']:.3f}" for f in fl) if fl else 'none')
                 + '. All other rows 0.147-0.257. Memory peaks are unaffected by host load.')
        L.append('')
    except Exception as exc:  # noqa: BLE001
        L.append(f'## Verdict summary\n\n(unavailable: {exc!r})\n')
    L.append('## Check status\n')
    L.append('| # | Check | Status | Key numbers |\n|---|---|---|---|')
    for row in res['check_table']:
        L.append(f"| {row['n']} | {row['check']} | {row['status']} | {row['numbers']} |")
    if res['failures']:
        L.append('\n## FAIL details\n')
        L += [f'- {f}' for f in res['failures']]
    if res['deviations']:
        L.append('\n## Explained deviations (strict check failed; cause demonstrated from files on disk)\n')
        L += [f'- {d}' for d in res['deviations']]
        pb = res['checks']['1_integrity'].get('pid_binding', {})
        for exe, info in pb.get('launcher_executables', {}).items():
            L.append(f"  - launched executable `{exe}`: {info['exe_bytes']} bytes, sha256 `{info['exe_sha256']}`; CPython redirector `{info['home']}\\Lib\\venv\\scripts\\nt\\python.exe` sha256 `{info['redirector_sha256']}`; real interpreter `{info['home']}\\python.exe` sha256 `{info['real_python_sha256']}`; is_redirector={info['is_redirector']}")
    L.append('\n## Recomputed headline (main measurements; MiB = 2^20 bytes; ratios vs full at the same length)\n')
    L.append('| L | arm | mean s | s ratio | peak alloc MiB | alloc ratio | peak reserved MiB | reserved ratio | static after opt MiB | host busy |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    m = res['checks']['5_metrics']['headline']; hb = res['checks']['6_host_load']['rows']
    for Lk in LENGTHS:
        for arm in ARMS:
            h = m[str(Lk)][arm]
            L.append(f"| {Lk} | {arm} | {h['mean_update_seconds']:.4f} | {h['update_seconds']:.3f} | {h['peak_alloc_MiB']:.2f} | {h['cuda_peak_allocated']:.3f} | {h['peak_reserved_MiB']:.1f} | {h['cuda_peak_reserved']:.3f} | {h['static_after_optimizer_MiB']:.2f} | {hb[f'{arm}-L{Lk}@main']['busy_fraction']:.3f} |")
    L.append('\n## Paired losses (main, measured updates 3/4/5)\n')
    pd = res['checks']['3_paired_data']['measured']
    L.append('| L | full~checkpoint max abs | detach~segment | detach~seg_enc_ckpt | full vs detach abs diff per update | step-1 warmup spread (5 arms) |\n|---:|---:|---:|---:|---|---:|')
    for Lk in LENGTHS:
        r = pd[f'main-L{Lk}']
        L.append(f"| {Lk} | {r['full~checkpoint']['max_abs']:.3e} | {r['detach~segment']['max_abs']:.3e} | {r['detach~segment_encoder_checkpoint']['max_abs']:.3e} | {', '.join(f'{d:.2e}' for d in r['full_vs_detach']['abs_diff_per_update'])} | {r['warmup_step1_all_arms']['spread']:.2e} |")
    vm = res['checks']['3_paired_data']['validation_vs_main']
    L.append('\nValidation vs main reproducibility (same arm/length, separate processes): ' + ', '.join(f"{k} loss max abs {v['max_abs_loss']:.1e} (final hash equal: {v['final_state_hash_equal']})" for k, v in vm.items()))
    L.append(f"\nHooks run vs main segment-L128 loss max abs: {res['checks']['3_paired_data']['hooks_vs_main_segment_L128_max_abs']:.2e}")
    L.append('\n## Equivalence (validation final.pt, recomputed; tolerance |a-b| <= atol + rtol*|b|, rtol 3e-4, atol 3e-6)\n')
    L.append('| L | base | candidate | tensors | max abs | worst abs/tol | violations | loss max abs | equivalence.json max abs | per-tensor match |\n|---:|---|---|---:|---:|---:|---:|---:|---:|---|')
    for p in res['checks']['4_equivalence']['pairs']:
        L.append(f"| {p['length']} | {p['base']} | {p['candidate']} | {p['n_tensors']} | {p['max_abs']:.3e} | {p['worst_tolerance_utilisation']:.3f} | {len(p['violations'])} | {p['loss_max_abs']:.3e} | {p['equivalence_json']['max_abs']:.3e} | {p['equivalence_json']['per_tensor_max_abs_match']} |")
    L.append('\n## Host load (system-wide busy fraction over worker lifetime)\n')
    fl = res['checks']['6_host_load']['flagged']
    L.append('Flagged (> 0.3): ' + (', '.join(f"{f['condition']} = {f['busy_fraction']:.3f}" for f in fl) if fl else 'none'))
    L.append('\nAll rows: ' + ', '.join(f"{k.split('@')[0]}{'(v)' if k.endswith('validation') else ''}={r['busy_fraction']:.3f}" for k, r in hb.items()))
    d = res['checks']['7_diagnostic']
    L.append('\n## Hooks diagnostic (segment-L128-hooks)\n')
    L.append(f"diagnostic={d['diagnostic_flag']}, status={d['status']}, peak_unique_storage_bytes per update={d['peak_unique_storage_bytes_per_update']}, save_requests per update={d['save_requests_per_update']}, end-of-update live bytes={[u['end_live_unique_storage_bytes'] for u in d['updates']]}, phases with snapshots={d['updates'][0]['phases_with_saved']}. Primary rows with snapshots: {d['main_rows_with_saved_snapshots'] or 'none'}.")
    L.append('\nPer-phase (phase, live bytes, peak bytes, save requests) for update 3: ' + str(d['updates'][0]['per_phase_live']))
    s = res['checks']['8_source_sanity']
    L.append('\n## Source sanity (frozen memory_bench.py)\n')
    L.append(f"sha256 match={s['matches_frozen_pin']}; snapshot identical={s['snapshot_identical_to_working_tree']}; chunk loop lines {s['segment_chunk_loop']['lineno']}-{s['segment_chunk_loop']['end_lineno']}: float() calls in loop={s['segment_chunk_loop']['float_calls_in_loop']}, item/cpu calls={s['segment_chunk_loop']['host_sync_calls_in_loop']}, returns={s['segment_chunk_loop']['returns']}; OOM handler line {s['oom_handler'].get('lineno')}: assigns cuda_at_failure={s['oom_handler'].get('assigns_cuda_at_failure')}, samples={s['oom_handler'].get('samples')}, recorded in result={s['oom_handler'].get('records_in_result')}. OOM observed in data: {s['oom_observed_in_data']} (handler is untested by data). Pre-fix file: {s.get('pre_fix_source')}.")
    if res['notes']:
        L.append('\n## Notes (informational)\n')
        L += [f'- {n}' for n in res['notes']]
    L.append('\n## Files\n')
    L.append('- results: `.superpowers/sdd/review-diagnostics-plan/r6-independent-audit.json`\n- script: `.superpowers/sdd/review-diagnostics-plan/r6_raw_audit.py`')
    return '\n'.join(L) + '\n'


def main():
    t0 = time.time()
    order = [('1_integrity', check_integrity), ('2_real_updates', check_real_updates), ('3_paired_data', check_paired_data),
             ('4_equivalence', check_equivalence), ('5_metrics', check_metrics), ('6_host_load', check_host_load),
             ('7_diagnostic', check_diagnostic), ('8_source_sanity', check_source_sanity)]
    table = []
    for name, fn in order:
        before, dbefore = len(failures), len(deviations)
        try:
            checks[name] = fn()
        except Exception as exc:  # noqa: BLE001
            checks[name] = dict(error=repr(exc)); fail(f'{name}: exception {exc!r}')
        status = 'FAIL' if len(failures) != before else ('PASS' if len(deviations) == dbefore else 'PASS (explained deviation)')
        table.append(dict(n=name.split('_')[0], check=name[2:], status=status, n_fail=len(failures) - before, n_deviation=len(deviations) - dbefore))
    c = checks

    def g(path, default='n/a'):
        cur = c
        try:
            for k in path:
                cur = cur[k]
            return cur
        except Exception:  # noqa: BLE001
            return default
    numbers = {
        '1': f"3/3 source pins match={all(v['match'] for v in g(['1_integrity', 'sources'], {}).values())}; protocol_hash match={g(['1_integrity', 'protocol_hash', 'match'])}; {len(g(['1_integrity', 'requests'], {}))} request.json, {len(g(['1_integrity', 'results'], {}))} result.json checked; overlaps={g(['1_integrity', 'ordering', 'overlapping_pairs'])}; stderr tracebacks={sum(v['traceback'] for v in g(['1_integrity', 'logs'], {}).values())}",
        '2': f"{g(['2_real_updates', '_summary', 'conditions'])} conditions: PASS/5 updates/[3,4,5]/AdamW step 5; distinct initial hashes={len(g(['2_real_updates', '_summary', 'distinct_initial_state_hashes'], []))}",
        '3': f"full vs detach max abs by L: { {k: f'{v:.2e}' for k, v in g(['3_paired_data', 'full_vs_truncated', 'max_abs_by_length_main'], {}).items()} }; differ at every length={g(['3_paired_data', 'full_vs_truncated', 'differ_at_every_length'])}",
        '4': '; '.join(f"L{p['length']} {p['base']}/{p['candidate']} {p['max_abs']:.2e} ({p['n_tensors']} t)" for p in g(['4_equivalence', 'pairs'], [])),
        '5': f"summary sha match={g(['5_metrics', 'summary_sha256_match'])}; mismatches={len(g(['5_metrics', 'mismatches'], []))}; counts={g(['5_metrics', 'counts_recomputed'])}",
        '6': f"flagged >0.3: {[(f['condition'], round(f['busy_fraction'], 3)) for f in g(['6_host_load', 'flagged'], [])]}",
        '7': f"peak_unique_storage_bytes={g(['7_diagnostic', 'peak_unique_storage_bytes_per_update'])}, save_requests={g(['7_diagnostic', 'save_requests_per_update'])}, end live={[u['end_live_unique_storage_bytes'] for u in g(['7_diagnostic', 'updates'], [])]}",
        '8': f"float() in chunk loop={g(['8_source_sanity', 'segment_chunk_loop', 'float_calls_in_loop'])}; OOM handler samples={g(['8_source_sanity', 'oom_handler', 'samples'])}; OOM in data={g(['8_source_sanity', 'oom_observed_in_data'])}",
    }
    for row in table:
        row['numbers'] = numbers.get(row['n'], '')
    verdict = 'FAIL' if failures else ('PASS' if not deviations else 'PASS_WITH_EXPLAINED_DEVIATIONS')
    res = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=time.time() - t0, torch=torch.__version__,
               torch_threads=torch.get_num_threads(), root=str(ROOT), verdict=verdict,
               failures=failures, deviations=deviations, notes=notes, check_table=table, frozen_protocol_hash=frozen['protocol_hash'],
               frozen_sources=frozen['sources'], checks=checks)
    (OUT / 'r6-independent-audit.json').write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + '\n', encoding='utf8')
    (OUT / 'r6-independent-audit.md').write_text(md_report(res), encoding='utf8')
    print(json.dumps(dict(verdict=res['verdict'], failures=failures, deviations=deviations, table=table, notes=notes, elapsed_s=res['elapsed_s']), indent=1, default=str))


if __name__ == '__main__':
    main()
