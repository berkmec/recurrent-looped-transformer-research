"""Frozen, paired R2--R5 diagnostics; development selection precedes test access.

Authoritative resume snapshots are atomic at declared boundaries; every update
has a durable journal. Interrupted uncheckpointed segments fail closed. Completed
bundles are immutable and never repaired implicitly.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from core.model import Core
from . import training_data as data

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('core/model.py', 'review_cycle/training.py', 'review_cycle/training_data.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(data.tensor_hash(x).encode())
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def protocol(device=None):
    default_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return dict(version=1, purpose='review-R2-R5-fresh-diagnostics',
                steps=2048, batch=32, train_length=32, small_width=32, small_heads=2,
                large_width=256, large_heads=4, small_lr=.003, lr_grid=[.0003, .001, .003],
                primary_seeds=[117, 143, 191], confirm_seeds=[217, 243, 291],
                weight_decay=.01, clip=1., window=4, alpha=.1,
                validation_n=256, validation_every=128, checkpoint_every=64,
                equal_budget_step=64, test_n=512, test_lengths=[32, 128, 1024],
                neutral_slots=[0, 8, 32, 128], eval_batch_short=32, eval_batch_long=8,
                eval_long_threshold=512, source_seeds=dict(train=81010000, sparse=81020000,
                    validation=82010000, test=83010000, neutral=83020000),
                device=default_device, threads=1,
                devices=dict(small=device or 'cpu', development=default_device, confirm=default_device),
                precision='float32', selection='minimum validation final cross entropy; earliest step on ties',
                lr_selection='mean selected validation final accuracy across all 3 development seeds; lowest LR on ties',
                tbptt_route='detach s, dk, dv every8; retain global encoder and projected memory routes',
                blocked_placement='global example index mod3: prefix,middle,suffix; middle splits non-2 sequence floor(n/2)',
                neutral_placement='insert token2 at base input prefix, floor(length/2), suffix; same cases',
                supervision='dense all prefix labels; sparse one independent uniform position; final last label',
                checkpoint_policy='every64, validation-selected, equal-label, orderly stop and finish; interrupted uncheckpointed segment fails closed',
                final_gate='all small and R5 confirmation selections fixed; explicit subset is partial diagnostic')


def conditions(p):
    found = {}
    def add(task, seed, mode='rlt', supervision='dense', arrangement='random'):
        c = dict(task=task, seed=seed, width=p['small_width'], heads=p['small_heads'],
                 mode=mode, supervision=supervision, arrangement=arrangement, lr=p['small_lr'], stage='small')
        found[ident(c)] = c
    for seed in p['primary_seeds']:
        for task in ('parity', 'five'):
            for arrangement in ('random', 'blocked'):
                add(task, seed, arrangement=arrangement)
        for mode in ('rlt', 'nofeedback'):
            for supervision in ('dense', 'sparse', 'final'):
                add('parity', seed, mode, supervision)
        for mode in ('rlt', 'tbptt8'):
            for supervision in ('dense', 'final'):
                add('five', seed, mode, supervision)
    return list(found.values())


def large_conditions(p, lr=None, confirm=False):
    if confirm and lr not in p['lr_grid']:
        raise ValueError('confirmation requires predeclared selected LR')
    return [dict(task='parity', seed=seed, width=p['large_width'], heads=p['large_heads'],
                 mode='rlt', supervision='dense', arrangement='random', lr=rate,
                 stage='confirm' if confirm else 'development')
            for rate in ([lr] if confirm else p['lr_grid'])
            for seed in (p['confirm_seeds'] if confirm else p['primary_seeds'])]


def ident(c):
    return f"{c['stage']}-{c['task']}-w{c['width']}-{c['mode']}-{c['arrangement']}-{c['supervision']}-lr{c['lr']:g}-s{c['seed']}"


def configure(p):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if p['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('frozen CUDA device unavailable')
    if p['device'] not in ('cpu', 'cuda') or not isinstance(p['checkpoint_every'], int) or p['checkpoint_every'] < 1:
        raise ValueError('device must be cpu/cuda; checkpoint cadence must be a positive integer')
    if p['threads'] != 1:
        raise ValueError('one CPU thread required')


def execution_protocol(p, c):
    runtime = dict(p)
    runtime['device'] = p['devices'][c['stage']]
    return runtime


def make_model(c, p):
    # Explicit seed is the same across ordering, supervision, LR and ablation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        model = Core(width=c['width'], heads=c['heads'], window=p['window'],
                     classes=2 if c['task'] == 'parity' else 5,
                     feedback=c['mode'] != 'nofeedback', alpha=p['alpha'])
    return model


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def freeze(out, p=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    configure(p)
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(),
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     numpy=np.__version__, cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory; old receipts cannot be imported')
    write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    try:
        if sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text():
            raise ValueError('frozen manifest hash mismatch')
        manifest = json.loads((out / 'frozen.json').read_text())
    except OSError as e:
        raise ValueError('missing freeze') from e
    if manifest['protocol_hash'] != digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    configure(manifest['protocol'])
    return manifest


@contextmanager
def lock(directory):
    path = Path(directory) / 'active.lock'
    try:
        handle = path.open('x', encoding='utf8')
    except FileExistsError as e:
        raise ValueError(f'active/stale lock must be audited before continuing: {path}') from e
    handle.write(json.dumps(dict(pid=os.getpid(), time=time.time()))); handle.close()
    try:
        yield
    finally:
        path.unlink()


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def supervised_loss(logits, y, c, p, step):
    if c['supervision'] == 'dense':
        return F.cross_entropy(logits.flatten(0, 1), y.flatten())
    if c['supervision'] == 'final':
        return F.cross_entropy(logits[:, -1], y[:, -1])
    idx = data.sparse_positions(c, p, step).to(y.device)
    rows = torch.arange(len(y), device=y.device)
    return F.cross_entropy(logits[rows, idx], y[rows, idx])


@torch.no_grad()
def eval_data(model, x, y, p):
    model.eval()
    device = p['device']
    batch = p['eval_batch_long'] if x.shape[1] >= p['eval_long_threshold'] else p['eval_batch_short']
    predictions, losses, final_losses = [], 0., 0.
    synchronize(device); start = time.perf_counter()
    for at in range(0, len(x), batch):
        xx, yy = x[at:at+batch].to(device), y[at:at+batch].to(device)
        z = model(xx)
        predictions.append(z.argmax(-1).cpu().numpy())
        losses += float(F.cross_entropy(z.flatten(0, 1), yy.flatten(), reduction='sum'))
        final_losses += float(F.cross_entropy(z[:, -1], yy[:, -1], reduction='sum'))
    synchronize(device)
    pred = np.concatenate(predictions)
    target = y.numpy()
    return dict(final_accuracy=float((pred[:, -1] == target[:, -1]).mean()),
                prefix_accuracy=float((pred == target).mean()), prefix_loss=losses/y.numel(),
                final_loss=final_losses/len(x), seconds=time.perf_counter()-start,
                batch=batch, n=len(x)), pred


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _check_snapshot(s, c, manifest):
    p = manifest['protocol']
    required = {'condition', 'protocol_hash', 'sources', 'model', 'optimizer', 'updates',
                'history', 'initial_state_hash', 'selected_model', 'selected_step',
                'best_loss', 'optimizer_hash', 'model_hash', 'seconds', 'history_hash', 'selected_hash', 'checkpoint_steps'}
    if not required <= set(s):
        raise ValueError('incomplete checkpoint')
    if s['condition'] != c or s['protocol_hash'] != manifest['protocol_hash'] or s['sources'] != manifest['sources']:
        raise ValueError('checkpoint provenance mismatch')
    if s['model_hash'] != state_hash(s['model']) or s['optimizer_hash'] != state_hash(s['optimizer']):
        raise ValueError('checkpoint state hash mismatch')
    if s['history_hash'] != digest(s['history']) or s['selected_hash'] != state_hash(s['selected_model']):
        raise ValueError('checkpoint history/selection mismatch')
    if (s['checkpoint_steps'] != sorted(set(s['checkpoint_steps'])) or
            s['checkpoint_steps'][0] != 0 or s['checkpoint_steps'][-1] != s['updates']):
        raise ValueError('checkpoint boundary mismatch')
    if not 0 <= s['updates'] <= p['steps'] or len(s['history']) != s['updates']:
        raise ValueError('checkpoint update/history mismatch')
    if [r['step'] for r in s['history']] != list(range(1, s['updates']+1)):
        raise ValueError('noncontiguous update receipt')
    if any(not math.isfinite(r['loss']) or not math.isfinite(r['grad_norm']) for r in s['history']):
        raise ValueError('nonfinite update receipt')
    optimizer_steps = [int(v['step']) for v in s['optimizer']['state'].values()]
    if optimizer_steps and any(n != s['updates'] for n in optimizer_steps):
        raise ValueError('actual optimizer update count mismatch')
    if s['updates'] and not optimizer_steps:
        raise ValueError('missing optimizer state')


def _known_condition(out, manifest, name):
    p = manifest['protocol']
    cs = conditions(p) + large_conditions(p)
    if (Path(out) / 'selection.json').exists():
        selection = _read_selection(out, manifest)
        cs += large_conditions(p, selection['selected_lr'], confirm=True)
    for c in cs:
        if ident(c) == name:
            return c
    raise ValueError('condition not predeclared or confirmatory LR not selected')


def _receipt(c, p, s, status):
    selected = s['selected_model']
    return dict(status=status, condition=c, updates=s['updates'], execution_device=p['device'],
                labels=data.label_count(c['supervision'], s['updates'], p['batch'], p['train_length']),
                input_sequences=s['updates']*p['batch'], input_tokens=s['updates']*p['batch']*p['train_length'],
                initial_state_hash=s['initial_state_hash'], last_state_hash=s['model_hash'],
                selected_state_hash=state_hash(selected) if selected is not None else None,
                selected_step=s['selected_step'], selected_validation=s.get('selected_validation'),
                protocol_hash=s['protocol_hash'], sources=s['sources'], train_seconds=s['seconds'],
                optimizer_hash=s['optimizer_hash'], finite_updates=len(s['history']),
                checkpoint_steps=s['checkpoint_steps'],
                equal_label_companion_step=p['equal_budget_step'] if c['supervision'] == 'dense' else None)


def _read_lines(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines()]
    except (OSError, ValueError) as e:
        raise ValueError(f'missing or incomplete journal: {path}') from e


def _check_journals(directory, s):
    timings = _read_lines(directory / 'timing.jsonl') if s['updates'] else []
    updates = _read_lines(directory / 'updates.jsonl') if s['updates'] else []
    if ([row.get('step') for row in timings] != list(range(1, s['updates']+1)) or
            digest(updates) != s['history_hash']):
        raise ValueError('journal disagrees with checkpoint; uncheckpointed updates cannot be replayed')
    if any(not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0 for row in timings):
        raise ValueError('invalid timing journal')
    # Even a zero-update snapshot must not conceal a later durable journal.
    if not s['updates'] and any((directory / file).exists() and (directory / file).stat().st_size for file in ('timing.jsonl', 'updates.jsonl')):
        raise ValueError('uncheckpointed journal after zero-update snapshot')
    return timings


def _append_line(path, row):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def run_one(out, condition_id, stop_after=None):
    out = Path(out)
    manifest = verify_frozen(out)
    c = _known_condition(out, manifest, condition_id)
    p = execution_protocol(manifest['protocol'], c)
    configure(p)
    directory = out / condition_id
    directory.mkdir(exist_ok=True)
    existed = list(directory.iterdir())
    if existed and not (directory / 'resume.pt').exists():
        raise ValueError('incomplete condition directory without resumable checkpoint')
    if any((directory / name).exists() for name in ('inflight.json', 'uncheckpointed.json', 'finalizing.json')) or any(directory.glob('*.tmp')):
        raise ValueError('incomplete in-flight update/write; requires audit, never automatic reuse')
    with lock(directory):
        invocation_start = time.perf_counter()
        receipt_path = directory / 'train.json'
        try:
            prior_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        except ValueError as e:
            raise ValueError('invalid existing receipt; no implicit repair') from e
        terminal = any((directory / name).exists() for name in ('complete.json', 'last.pt', 'selected.pt'))
        if terminal or (prior_receipt is not None and prior_receipt.get('status') != 'PAUSED'):
            return _read_complete(out, c, manifest)
        model = make_model(c, p).to(p['device'])
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        if existed:
            s = load_checkpoint(directory / 'resume.pt')
            _check_snapshot(s, c, manifest)
            if s['updates'] == p['steps']:
                raise ValueError('unsealed terminal snapshot requires explicit audit')
            model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
            initial = load_checkpoint(directory / 'initial.pt')
            if state_hash(initial) != s['initial_state_hash']:
                raise ValueError('initial checkpoint drift')
            if c['supervision'] == 'dense' and s['updates'] >= p['equal_budget_step']:
                companion_path = directory / 'equal-label.pt'
                if not companion_path.exists():
                    raise ValueError('missing exact equal-label-budget companion')
                companion = load_checkpoint(companion_path)
                _check_snapshot(companion, c, manifest)
                if companion['updates'] != p['equal_budget_step']:
                    raise ValueError('wrong equal-label-budget update')
        else:
            initial = _cpu_state(model)
            save_checkpoint(directory / 'initial.pt', initial)
            s = dict(condition=c, protocol_hash=manifest['protocol_hash'], sources=manifest['sources'],
                     initial_state_hash=state_hash(initial), model=initial,
                     optimizer=opt.state_dict(), updates=0, history=[], selected_model=None,
                     selected_step=None, best_loss=None, seconds=0., model_hash=state_hash(initial),
                     optimizer_hash=state_hash(opt.state_dict()), history_hash=digest([]), selected_hash=state_hash(None), checkpoint_steps=[0])
            save_checkpoint(directory / 'resume.pt', s)
        timing_path = directory / 'timing.jsonl'
        timings = _check_journals(directory, s)
        vx, vy, _ = data.heldout(c, p, 'validation', p['validation_n'], p['train_length'])
        target_step = p['steps'] if stop_after is None else min(stop_after, p['steps'])
        if target_step < s['updates'] or target_step < 1:
            raise ValueError('stop boundary precedes current progress')
        for step in range(s['updates'] + 1, target_step + 1):
            if not (directory / 'uncheckpointed.json').exists():
                write_json(directory / 'uncheckpointed.json', dict(checkpoint_updates=s['updates'], next_step=step))
            write_json(directory / 'inflight.json', dict(step=step, prior_updates=s['updates'], pid=os.getpid()))
            synchronize(p['device']); start = time.perf_counter()
            batch = data.train_batch(c, p, step)
            x, y = batch['x'].to(p['device']), batch['y'].to(p['device'])
            model.train(); opt.zero_grad(set_to_none=True)
            logits = model(x, tbptt=8 if c['mode'] == 'tbptt8' else 0)
            loss = supervised_loss(logits, y, c, p, step)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip']))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):
                write_json(directory / f'failure-step{step}.json', dict(status='FAIL_NONFINITE', step=step,
                           loss=str(float(loss.detach())), grad_norm=str(grad), accepted_updates=s['updates']))
                raise ValueError('nonfinite loss/gradient; failure preserved')
            opt.step()
            row = dict(step=step, real_update=step, loss=float(loss.detach()), grad_norm=grad,
                       input_hash=batch['input_hash'], base_input_hash=batch['base_input_hash'],
                       source_seed=batch['source_seed'], labels=data.label_count(c['supervision'], 1, p['batch'], p['train_length']))
            if c['supervision'] == 'sparse':
                row['sparse_position_hash'] = data.tensor_hash(data.sparse_positions(c, p, step))
            selected_now = False
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _ = eval_data(model, vx, vy, p)
                row['validation'] = stats
                if s['best_loss'] is None or stats['final_loss'] < s['best_loss']:
                    s.update(best_loss=stats['final_loss'], selected_step=step,
                             selected_model=_cpu_state(model), selected_validation=stats)
                    selected_now = True
            synchronize(p['device'])
            row['compute_seconds'] = time.perf_counter() - start
            s['history'].append(row)
            s['updates'] = step
            s['seconds'] += time.perf_counter() - start
            equal_now = c['supervision'] == 'dense' and step == p['equal_budget_step']
            checkpoint_now = step % p['checkpoint_every'] == 0 or selected_now or equal_now or step == target_step
            if checkpoint_now:
                s.update(model=_cpu_state(model), optimizer=opt.state_dict())
                s.update(model_hash=state_hash(s['model']), optimizer_hash=state_hash(s['optimizer']))
                s.update(history_hash=digest(s['history']), selected_hash=state_hash(s['selected_model']))
                s['checkpoint_steps'].append(step)
                save_checkpoint(directory / 'resume.pt', s)
                if equal_now:
                    save_checkpoint(directory / 'equal-label.pt', s)
            timing = dict(step=step, wall_seconds=time.perf_counter()-start)
            _append_line(directory / 'updates.jsonl', row)
            _append_line(timing_path, timing)
            timings.append(timing)
            (directory / 'inflight.json').unlink()
            if checkpoint_now:
                (directory / 'uncheckpointed.json').unlink()
        status = 'COMPLETE' if s['updates'] == p['steps'] else 'PAUSED'
        _check_snapshot(s, c, manifest)
        if status == 'COMPLETE':
            write_json(directory / 'finalizing.json', dict(updates=s['updates'], protocol_hash=manifest['protocol_hash']))
            save_checkpoint(directory / 'last.pt', s)
            save_checkpoint(directory / 'selected.pt', s['selected_model'])
        receipt = _receipt(c, p, s, status)
        receipt.update(wall_seconds=sum(row['wall_seconds'] for row in timings), timed_updates=len(timings),
                       invocation_seconds=time.perf_counter()-invocation_start, timing_sha256=sha(timing_path))
        receipt['checkpoint_hashes'] = {path.name: sha(path) for path in directory.glob('*.pt')}
        write_json(directory / 'train.json', receipt)
        if status == 'COMPLETE':
            seal = dict(protocol_hash=manifest['protocol_hash'], condition=c,
                        files={file:sha(directory / file) for file in list(receipt['checkpoint_hashes']) + ['train.json', 'timing.jsonl', 'updates.jsonl']})
            seal['seal_hash'] = digest(seal)
            write_json(directory / 'complete.json', seal)
            (directory / 'finalizing.json').unlink()
        return receipt


def _read_complete(out, c, manifest):
    directory = Path(out) / ident(c)
    try:
        seal = json.loads((directory / 'complete.json').read_text())
        expected_files = {'initial.pt', 'resume.pt', 'selected.pt', 'last.pt', 'train.json', 'timing.jsonl', 'updates.jsonl'}
        if c['supervision'] == 'dense' and manifest['protocol']['equal_budget_step'] <= manifest['protocol']['steps']:
            expected_files.add('equal-label.pt')
        if (seal['seal_hash'] != digest({k:v for k,v in seal.items() if k != 'seal_hash'}) or
                seal['protocol_hash'] != manifest['protocol_hash'] or seal['condition'] != c or set(seal['files']) != expected_files):
            raise ValueError('invalid completion seal')
        if any(sha(directory / file) != value for file, value in seal['files'].items()):
            raise ValueError('completed bundle changed; no implicit repair')
        receipt = json.loads((directory / 'train.json').read_text())
        s = load_checkpoint(directory / 'last.pt')
        resumed = load_checkpoint(directory / 'resume.pt')
    except (OSError, EOFError, KeyError) as e:
        raise ValueError(f'training incomplete: {ident(c)}') from e
    if any((directory / file).exists() for file in ('inflight.json', 'uncheckpointed.json', 'finalizing.json')):
        raise ValueError('incomplete terminal transition')
    _check_snapshot(s, c, manifest)
    _check_snapshot(resumed, c, manifest)
    if state_hash(s) != state_hash(resumed):
        raise ValueError('terminal and resume checkpoint disagree')
    _check_journals(directory, s)
    expected = _receipt(c, execution_protocol(manifest['protocol'], c), s, 'COMPLETE')
    if s['updates'] != manifest['protocol']['steps'] or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('completion receipt/checkpoint mismatch')
    for file, value in receipt['checkpoint_hashes'].items():
        if sha(directory / file) != value:
            raise ValueError('checkpoint file drift')
    if sha(directory / 'timing.jsonl') != receipt['timing_sha256']:
        raise ValueError('actual-time ledger drift')
    return receipt


def _read_selection(out, manifest):
    path = Path(out) / 'selection.json'
    value = json.loads(path.read_text())
    if value['protocol_hash'] != manifest['protocol_hash'] or value['sources'] != manifest['sources']:
        raise ValueError('selection provenance mismatch')
    if value['selection_hash'] != digest({k:v for k,v in value.items() if k != 'selection_hash'}):
        raise ValueError('selection hash mismatch')
    if value['selected_lr'] not in manifest['protocol']['lr_grid']:
        raise ValueError('unregistered learning rate')
    return value


def run_selection(out):
    manifest = verify_frozen(out); p = manifest['protocol']
    # Every call is sequential; parent reserves GPU before invoking this action.
    rows = [run_one(out, ident(c)) for c in large_conditions(p)]
    scores = {str(lr): float(np.mean([r['selected_validation']['final_accuracy'] for r in rows if r['condition']['lr'] == lr])) for lr in p['lr_grid']}
    selected = min(p['lr_grid'], key=lambda lr: (-scores[str(lr)], lr))
    value = dict(selected_lr=selected, mean_validation_accuracy=scores,
                 development_conditions=[ident(r['condition']) for r in rows],
                 development_selected_hashes={ident(r['condition']): r['selected_state_hash'] for r in rows},
                 protocol_hash=manifest['protocol_hash'], sources=manifest['sources'],
                 final_data_opened=False, role='development-only; not final scientific evidence')
    value['selection_hash'] = digest(value)
    path = Path(out) / 'selection.json'
    if path.exists() and _read_selection(out, manifest) != value:
        raise ValueError('refusing changed LR selection')
    if not path.exists():
        write_json(path, value)
    confirmations = [run_one(out, ident(c)) for c in large_conditions(p, selected, confirm=True)]
    return dict(selection=value, confirmations=confirmations)


def _raw_evaluation(model, x, y, c, p, out, filename, seed, baseline=None):
    stats, pred = eval_data(model, x, y, p)
    path = Path(out) / filename
    path.parent.mkdir(exist_ok=True)
    ids = np.asarray([hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in range(len(x))])
    np.savez_compressed(path, inputs=x.numpy(), predictions=pred, targets=y.numpy(), input_ids=ids,
                        source_seed=np.asarray(str(seed)), condition=ident(c))
    classes = 2 if c['task'] == 'parity' else 5
    stats.update(raw_file=filename, sha256=sha(path), source_seed=seed,
                 input_hash=data.tensor_hash(x), consumed_tokens=x.shape[1],
                 target_histogram={str(i):int((y[:, -1] == i).sum()) for i in range(classes)},
                 prediction_histogram={str(i):int((pred[:, -1] == i).sum()) for i in range(classes)})
    if baseline is not None:
        was_correct = baseline == y[:, -1].numpy()
        now_correct = pred[:, -1] == y[:, -1].numpy()
        stats.update(correct_to_wrong=int((was_correct & ~now_correct).sum()),
                     wrong_to_correct=int((~was_correct & now_correct).sum()),
                     prediction_changes=int((baseline != pred[:, -1]).sum()))
    return stats, pred[:, -1]


def _validate_evaluation(row, c, p, manifest, out, variant, checkpoint):
    """Recount cached discrete metrics from raw arrays and check full inventory."""
    if (row.get('metadata_hash') != digest({k:v for k,v in row.items() if k != 'metadata_hash'}) or
            row.get('condition') != c or row.get('variant') != variant or
            row.get('protocol_hash') != manifest['protocol_hash'] or row.get('checkpoint_sha256') != sha(checkpoint)):
        raise ValueError('evaluation metadata/identity/checkpoint drift')
    expected_neutral = [(slots, place) for slots in sorted(p['neutral_slots'])
                        for place in (['base'] if slots == 0 else ['prefix', 'middle', 'suffix'])]
    if ([r.get('length') for r in row['lengths']] != p['test_lengths'] or
            [(r.get('slots'), r.get('placement')) for r in row['neutral']] != expected_neutral):
        raise ValueError('evaluation row inventory differs from frozen protocol')
    classes = 2 if c['task'] == 'parity' else 5
    name = ident(c)
    baseline = None
    for kind, records in [('test', row['lengths']), ('neutral', row['neutral'])]:
        for record in records:
            length = record['length'] if kind == 'test' else p['train_length']
            x, y, seed = data.heldout(c, p, kind, p['test_n'], length)
            suffix = f"L{length}" if kind == 'test' else f"neutral{record['slots']}-{record['placement']}"
            filename = f'{name}/{variant}-{suffix}.npz'
            if kind == 'neutral':
                x, y = data.insert_neutrals(x, c['task'], record['slots'], record['placement'])
            if record['raw_file'] != filename or sha(Path(out) / filename) != record['sha256']:
                raise ValueError('raw evaluation identity/hash drift')
            with np.load(Path(out) / filename, allow_pickle=False) as raw:
                pred, targets, inputs = raw['predictions'], raw['targets'], raw['inputs']
                ids = np.asarray([hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest() for i in range(p['test_n'])])
                if (pred.shape != tuple(y.shape) or not np.issubdtype(pred.dtype, np.integer) or
                        np.any((pred < 0) | (pred >= classes)) or not np.array_equal(targets, y.numpy()) or
                        not np.array_equal(inputs, x.numpy()) or not np.array_equal(raw['input_ids'], ids) or
                        str(raw['source_seed'].item()) != str(seed) or str(raw['condition'].item()) != name):
                    raise ValueError('raw evaluation source/shape/labels/IDs mismatch')
            derived = dict(final_accuracy=float(np.mean(pred[:, -1] == targets[:, -1])),
                           prefix_accuracy=float(np.mean(pred == targets)), n=len(x),
                           target_histogram={str(i):int(np.count_nonzero(targets[:, -1] == i)) for i in range(classes)},
                           prediction_histogram={str(i):int(np.count_nonzero(pred[:, -1] == i)) for i in range(classes)},
                           source_seed=seed, input_hash=data.tensor_hash(x), consumed_tokens=x.shape[1],
                           batch=p['eval_batch_long'] if x.shape[1] >= p['eval_long_threshold'] else p['eval_batch_short'])
            if kind == 'neutral':
                if record['slots'] == 0:
                    baseline = pred[:, -1].copy()
                else:
                    if baseline is None:
                        raise ValueError('neutral comparison lacks shared baseline')
                    was_correct, now_correct = baseline == targets[:, -1], pred[:, -1] == targets[:, -1]
                    derived.update(correct_to_wrong=int(np.count_nonzero(was_correct & ~now_correct)),
                                   wrong_to_correct=int(np.count_nonzero(~was_correct & now_correct)),
                                   prediction_changes=int(np.count_nonzero(baseline != pred[:, -1])))
            if any(record.get(k) != v for k,v in derived.items()):
                raise ValueError('cached evaluation metric disagrees with raw predictions')
            if any(not math.isfinite(record[k]) or record[k] < 0 for k in ('seconds', 'prefix_loss', 'final_loss')):
                raise ValueError('invalid cached loss/timing')


def evaluate(out, condition_ids=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    cs = conditions(p)
    if condition_ids is None:
        if not (out / 'selection.json').exists():
            raise ValueError('final test access blocked until R5 LR selection and all training complete')
        selection = _read_selection(out, manifest)
        cs += large_conditions(p, selection['selected_lr'], confirm=True)
    else:
        cs = [_known_condition(out, manifest, name) for name in condition_ids]
        if any(c['stage'] == 'development' for c in cs):
            raise ValueError('development conditions may only use validation')
    receipts = {ident(c): _read_complete(out, c, manifest) for c in cs}
    gate = dict(protocol_hash=manifest['protocol_hash'], selections={k:r['selected_state_hash'] for k,r in receipts.items()},
                status='COMPLETE_SCOPE' if condition_ids is None else 'PARTIAL_SCOPE')
    gate_path = out / ('test-gate.json' if condition_ids is None else 'test-gate-partial.json')
    if gate_path.exists() and json.loads(gate_path.read_text()) != gate:
        raise ValueError('test selection gate changed after test opening')
    write_json(gate_path, gate)
    results = {}
    for c in cs:
        p = execution_protocol(manifest['protocol'], c)
        configure(p)
        name = ident(c)
        variants = [('selected', out / name / 'selected.pt')]
        if c['supervision'] == 'dense':
            variants.append(('equal-label', out / name / 'equal-label.pt'))
        for variant, checkpoint in variants:
            key = name if variant == 'selected' else name + '-equal-label'
            ep = out / name / (variant + '-evaluation.json')
            if ep.exists():
                row = json.loads(ep.read_text())
                _validate_evaluation(row, c, p, manifest, out, variant, checkpoint)
                results[key] = row
                continue
            model = make_model(c, p).to(p['device'])
            saved = load_checkpoint(checkpoint)
            model.load_state_dict(saved if variant == 'selected' else saved['model'])
            row = dict(condition=c, variant=variant, protocol_hash=manifest['protocol_hash'],
                       checkpoint_sha256=sha(checkpoint), lengths=[], neutral=[])
            for length in p['test_lengths']:
                x, y, seed = data.heldout(c, p, 'test', p['test_n'], length)
                stats, _ = _raw_evaluation(model, x, y, c, p, out, f'{name}/{variant}-L{length}.npz', seed)
                row['lengths'].append(dict(length=length, **stats))
            x, y, seed = data.heldout(c, p, 'neutral', p['test_n'], p['train_length'])
            baseline = None
            for slots in sorted(p['neutral_slots']):
                for placement in (['base'] if slots == 0 else ['prefix', 'middle', 'suffix']):
                    xx, yy = data.insert_neutrals(x, c['task'], slots, placement)
                    stats, pred = _raw_evaluation(model, xx, yy, c, p, out,
                            f'{name}/{variant}-neutral{slots}-{placement}.npz', seed, baseline)
                    if slots == 0:
                        baseline = pred
                    row['neutral'].append(dict(slots=slots, placement=placement, **stats))
            row['metadata_hash'] = digest(row)
            _validate_evaluation(row, c, p, manifest, out, variant, checkpoint)
            write_json(ep, row); results[key] = row
    result = dict(status='COMPLETE' if condition_ids is None else 'PARTIAL', protocol_hash=manifest['protocol_hash'], conditions=results)
    write_json(out / ('evaluation.json' if condition_ids is None else 'evaluation-partial.json'), result)
    return result


def summarize(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    cs = conditions(p) + large_conditions(p)
    if (out / 'selection.json').exists():
        cs += large_conditions(p, _read_selection(out, manifest)['selected_lr'], confirm=True)
    rows = []
    for c in cs:
        try:
            receipt = _read_complete(out, c, manifest)
            rows.append(dict(id=ident(c), **receipt))
        except ValueError as e:
            directory = out / ident(c)
            rows.append(dict(id=ident(c), status='MISSING' if not directory.exists() else 'INCOMPLETE_OR_INVALID', reason=str(e)))
    result = dict(protocol_hash=manifest['protocol_hash'], expected_small=36,
                  declared_conditions=len(cs), complete=sum(r['status'] == 'COMPLETE' for r in rows),
                  final_evaluation=(out / 'evaluation.json').exists(), rows=rows)
    write_json(out / 'summary.json', result)
    return result


def calibrate(out, updates=8):
    if not 1 <= updates <= 32:
        raise ValueError('calibration budget must be 1..32 updates')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    target = out / 'calibration.json'
    if target.exists():
        raise ValueError('preserve previous calibration; choose a fresh output directory')
    rows = []
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        for width, heads in [(32, 2), (256, 4)]:
            p = protocol(device); configure(p)
            c = dict(task='parity', seed=9117, width=width, heads=heads, mode='rlt',
                     supervision='dense', arrangement='random', lr=.003, stage='pilot')
            model = make_model(c, p).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
            measured = []
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for step in range(1, updates + 1):
                batch = data.train_batch(c, p, step)
                synchronize(device); start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                z = model(batch['x'].to(device))
                loss = supervised_loss(z, batch['y'].to(device), c, p, step)
                loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip']))
                if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):
                    raise ValueError('pilot nonfinite')
                opt.step(); synchronize(device)
                measured.append(dict(step=step, loss=float(loss.detach()), grad_norm=grad, seconds=time.perf_counter()-start))
            snap = dict(model=_cpu_state(model), optimizer=opt.state_dict())
            start = time.perf_counter()
            save_checkpoint(out / f'pilot-{device}-w{width}.pt', snap)
            snapshot_seconds = time.perf_counter()-start
            steady = float(np.mean([r['seconds'] for r in measured[1:]] or [measured[0]['seconds']]))
            rows.append(dict(device=device, width=width, updates=updates, seed=c['seed'], rows=measured,
                             steady_update_seconds=steady, snapshot_seconds=snapshot_seconds,
                             projected_train_seconds_per_condition=2048*steady+math.ceil(2048/p['checkpoint_every'])*snapshot_seconds,
                             projected_resume_snapshots=math.ceil(2048/p['checkpoint_every']),
                             checkpoint_every=p['checkpoint_every'],
                             peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                             final_data_opened=False))
            del model, opt, snap
            if device == 'cuda':
                torch.cuda.empty_cache()
            write_json(target, dict(status='RUNNING', rows=rows))
    result = dict(status='COMPLETE', rows=rows, small_conditions=36, development_large_conditions=9,
                  confirm_large_conditions=3, caveat='projections exclude validation, hashing, final evaluation, journal fsync, initial/companion/final checkpoint writes; pilot allocation is not steady memory benchmark')
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--calibrate', action='store_true')
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run-one')
    group.add_argument('--run-selection', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    group.add_argument('--summarize', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--pilot-updates', type=int, default=8)
    args = parser.parse_args()
    if args.freeze:
        result = freeze(args.out, protocol(args.device))
    elif args.calibrate:
        result = calibrate(args.out, args.pilot_updates)
    elif args.run_one:
        result = run_one(args.out, args.run_one)
    elif args.run_selection:
        result = run_selection(args.out)
    elif args.evaluate:
        result = evaluate(args.out)
    else:
        result = summarize(args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
