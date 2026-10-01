"""Drift mini-experiment runner: freeze, train, evaluate, analyze.

New code for a separate research root. The immutable Core and the sealed deterministic
source generators (review_cycle.training_data) are imported read-only; nothing under the
sealed campaigns is written. Every condition writes a per-update journal, initial/last/selected
checkpoints and a hashed receipt; evaluation writes raw predictions as NPZ.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from core.model import Core
from review_cycle import training_data as data
from drift.model import ARMS, RereadCore, core_of, run_sequence, neutral_invariance_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'review_cycle/training_data.py', 'drift/model.py', 'drift/run.py')
T95 = 4.302652729


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


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def protocol():
    return dict(version=1, purpose='drift mini-experiment: neutral-run state drift under window/invariance/reread arms',
                steps=2048, batch=32, train_length=32, lr=.003, weight_decay=.01, clip=1.,
                width=32, heads=2, window=4, win16_window=16, alpha=.1, lambda_idinv=1., reread_seed_offset=1000,
                tasks=['five', 'parity'], arms=list(ARMS), primary_seeds=[117, 143, 191],
                validation_every=128, validation_n=256, selection='minimum validation final cross entropy; earliest step on ties',
                test_n=512, test_lengths=[32, 128, 1024], neutral_slots=[0, 8, 32, 128],
                eval_batch_short=32, eval_batch_long=8, eval_long_threshold=512,
                source_seeds=dict(train=81010000, sparse=81020000, validation=82010000, test=83010000, neutral=83020000),
                source_note='same source namespaces as the sealed 2026-09-17 diagnostics campaign; test inputs identical there; never used for selection here',
                device='cpu', threads=1,
                primary=dict(task='five', endpoints=['neutral-128-middle', 'neutral-128-suffix'], gain_pp=20., cost_endpoint='L1024', max_cost_pp=5.))


def configure(p):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if p['threads'] != 1 or p['device'] != 'cpu':
        raise ValueError('one CPU thread required by this protocol')


def conditions(p):
    out = []
    for task in p['tasks']:
        for seed in p['primary_seeds']:
            for arm in p['arms']:
                out.append(dict(task=task, seed=seed, arm=arm, width=p['width'], heads=p['heads'],
                                window=p['win16_window'] if arm == 'win16' else p['window'], lr=p['lr'], stage='drift'))
    return out


def ident(c):
    return f"drift-{c['task']}-{c['arm']}-s{c['seed']}"


def shim(c):
    return dict(task=c['task'], seed=c['seed'], arrangement='random')


def train_batch(c, p, step):
    return data.train_batch(shim(c), p, step)


def heldout(c, p, split, n, length):
    return data.heldout(shim(c), p, split, n, length)


def make_model(c, p):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        core = Core(width=c['width'], heads=c['heads'], window=c['window'],
                    classes=2 if c['task'] == 'parity' else 5, feedback=True, alpha=p['alpha'])
    if c['arm'] == 'reread':
        return RereadCore(core, c['seed'] + p['reread_seed_offset'])
    return core


def core_initial_hash(c, p):
    return state_hash(core_of(make_model(c, p)).state_dict())


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol()); configure(p)
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(),
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                    frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    configure(manifest['protocol'])
    return manifest


def known(manifest, condition_id):
    for c in conditions(manifest['protocol']):
        if ident(c) == condition_id:
            return c
    raise ValueError('condition not predeclared: ' + condition_id)


@torch.no_grad()
def predict(model, x, y, c, p):
    model.eval()
    batch = p['eval_batch_long'] if x.shape[1] >= p['eval_long_threshold'] else p['eval_batch_short']
    preds, prefix_loss, final_loss = [], 0., 0.
    start = time.perf_counter()
    for at in range(0, len(x), batch):
        xx, yy = x[at:at + batch], y[at:at + batch]
        z, _ = run_sequence(model, xx, c['arm'])
        preds.append(z.argmax(-1).numpy())
        prefix_loss += float(F.cross_entropy(z.flatten(0, 1), yy.flatten(), reduction='sum'))
        final_loss += float(F.cross_entropy(z[:, -1], yy[:, -1], reduction='sum'))
    pred = np.concatenate(preds); target = y.numpy()
    return dict(final_accuracy=float((pred[:, -1] == target[:, -1]).mean()), prefix_accuracy=float((pred == target).mean()),
                prefix_loss=prefix_loss / y.numel(), final_loss=final_loss / len(x), seconds=time.perf_counter() - start,
                batch=batch, n=len(x)), pred


def append_line(path, row):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def run_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if ident(c) != ident(known(manifest, ident(c))):
        raise ValueError('unknown condition')
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    lock = directory / 'active.lock'
    lock.write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall_start = time.perf_counter()
        model = make_model(c, p)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        initial = cpu_state(model); torch.save(initial, directory / 'initial.pt')
        vx, vy, vseed = heldout(c, p, 'validation', p['validation_n'], p['train_length'])
        history, best, selected, train_seconds = [], None, None, 0.
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = train_batch(c, p, step)
            x, y = batch['x'], batch['y']
            model.train(); opt.zero_grad(set_to_none=True)
            logits, states = run_sequence(model, x, c['arm'])
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            aux = neutral_invariance_loss(states, x) if c['arm'] == 'idinv' else None
            loss = ce if aux is None else ce + p['lambda_idinv'] * aux
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip']))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):
                write_json(directory / f'failure-step{step}.json', dict(status='FAIL_NONFINITE', step=step, loss=str(float(loss.detach())), grad_norm=str(grad)))
                raise ValueError('nonfinite loss/gradient; failure preserved')
            opt.step()
            train_seconds += time.perf_counter() - start
            row = dict(step=step, real_update=step, loss=float(loss.detach()), cross_entropy=float(ce.detach()),
                       aux=None if aux is None else float(aux.detach()), grad_norm=grad,
                       input_hash=batch['input_hash'], base_input_hash=batch['base_input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _ = predict(model, vx, vy, c, p)
                row['validation'] = stats
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=cpu_state(model), validation=stats)
            history.append(row); append_line(directory / 'updates.jsonl', row)
        last = cpu_state(model); torch.save(last, directory / 'last.pt'); torch.save(selected['model'], directory / 'selected.pt')
        extra = sum(v.numel() for k, v in model.state_dict().items() if k.startswith('reread_'))
        receipt = dict(status='COMPLETE', condition=c, updates=p['steps'], labels=p['steps'] * p['batch'] * p['train_length'],
                       initial_state_hash=state_hash(initial), last_state_hash=state_hash(last),
                       selected_state_hash=state_hash(selected['model']), selected_step=selected['step'], selected_validation=selected['validation'],
                       validation_source_seed=vseed, core_initial_hash=core_initial_hash(c, p),
                       core_parameters=sum(v.numel() for v in core_of(model).state_dict().values()), extra_parameters=extra,
                       protocol_hash=manifest['protocol_hash'], sources=manifest['sources'], history=history,
                       train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall_start, device=p['device'], pid=os.getpid())
        write_json(directory / 'train.json', receipt)
        files = {name: sha(directory / name) for name in ('initial.pt', 'last.pt', 'selected.pt', 'updates.jsonl', 'train.json')}
        seal = dict(condition=c, files=files, protocol_hash=manifest['protocol_hash']); seal['seal_hash'] = digest(seal)
        write_json(directory / 'complete.json', seal)
        return receipt
    finally:
        lock.unlink(missing_ok=True)


def raw_evaluation(model, x, y, c, p, out, filename, seed, baseline=None):
    stats, pred = predict(model, x, y, c, p)
    path = Path(out) / filename; path.parent.mkdir(exist_ok=True)
    ids = np.asarray([hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest() for i in range(len(x))])
    np.savez_compressed(path, inputs=x.numpy(), predictions=pred, targets=y.numpy(), input_ids=ids, source_seed=np.asarray(str(seed)), condition=ident(c))
    classes = 2 if c['task'] == 'parity' else 5
    stats.update(raw_file=filename, sha256=sha(path), source_seed=seed, input_hash=data.tensor_hash(x), consumed_tokens=int(x.shape[1]),
                 target_histogram={str(i): int((y[:, -1] == i).sum()) for i in range(classes)},
                 prediction_histogram={str(i): int((pred[:, -1] == i).sum()) for i in range(classes)})
    if baseline is not None:
        was = baseline == y[:, -1].numpy(); now = pred[:, -1] == y[:, -1].numpy()
        stats.update(correct_to_wrong=int((was & ~now).sum()), wrong_to_correct=int((~was & now).sum()), prediction_changes=int((baseline != pred[:, -1]).sum()))
    return stats, pred[:, -1]


def evaluate_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    directory = out / ident(c)
    path = directory / 'evaluation.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    if not (directory / 'complete.json').exists():
        raise ValueError('evaluation requires a complete condition: ' + ident(c))
    model = make_model(c, p); model.load_state_dict(torch.load(directory / 'selected.pt', weights_only=True))
    row = dict(condition=c, protocol_hash=manifest['protocol_hash'], checkpoint_sha256=sha(directory / 'selected.pt'), lengths=[], neutral=[])
    for length in p['test_lengths']:
        x, y, seed = heldout(c, p, 'test', p['test_n'], length)
        stats, _ = raw_evaluation(model, x, y, c, p, out, f'{ident(c)}/selected-L{length}.npz', seed)
        row['lengths'].append(dict(length=length, **stats))
    x, y, seed = heldout(c, p, 'neutral', p['test_n'], p['train_length'])
    baseline = None
    for slots in sorted(p['neutral_slots']):
        for placement in (['base'] if slots == 0 else ['prefix', 'middle', 'suffix']):
            xx, yy = data.insert_neutrals(x, c['task'], slots, placement)
            stats, pred = raw_evaluation(model, xx, yy, c, p, out, f'{ident(c)}/selected-neutral{slots}-{placement}.npz', seed, baseline)
            if slots == 0:
                baseline = pred
            row['neutral'].append(dict(slots=slots, placement=placement, **stats))
    row['metadata_hash'] = digest(row)
    write_json(path, row)
    return row


def load_raw(out, c, endpoint):
    name = f"{ident(c)}/selected-{'L' + endpoint[1:] if endpoint.startswith('L') else 'neutral' + endpoint.split('-')[1] + '-' + endpoint.split('-')[2]}.npz"
    with np.load(Path(out) / name) as f:
        return {k: f[k] for k in f.files}


def seed_summary(diffs):
    mean = statistics.mean(diffs); half = T95 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return dict(seed_effects=diffs, mean=mean, t95=[mean - half, mean + half],
                direction='POSITIVE_INTERVAL' if mean - half > 0 else 'NEGATIVE_INTERVAL' if mean + half < 0 else 'INCLUDES_ZERO')


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    cs = conditions(p)
    endpoints = [f'L{n}' for n in p['test_lengths']] + ['neutral-0-base'] + [f'neutral-{s}-{pl}' for s in p['neutral_slots'] if s for pl in ('prefix', 'middle', 'suffix')]
    contrasts = []; absolute = {}
    for c in cs:
        ev = evaluate_one(out, c)
        absolute[ident(c)] = {**{f"L{r['length']}": r['final_accuracy'] for r in ev['lengths']},
                              **{f"neutral-{r['slots']}-{r['placement']}": r['final_accuracy'] for r in ev['neutral']}}
    for task in p['tasks']:
        for arm in p['arms']:
            if arm == 'base':
                continue
            for endpoint in endpoints:
                diffs, per_seed = [], []
                for seed in p['primary_seeds']:
                    ca = next(c for c in cs if c['task'] == task and c['seed'] == seed and c['arm'] == arm)
                    cb = next(c for c in cs if c['task'] == task and c['seed'] == seed and c['arm'] == 'base')
                    a, b = load_raw(out, ca, endpoint), load_raw(out, cb, endpoint)
                    for key in ('inputs', 'targets', 'input_ids'):
                        if not np.array_equal(a[key], b[key]):
                            raise ValueError('UNPAIRED_RAW_' + key)
                    ok_a = a['predictions'][:, -1] == a['targets'][:, -1]; ok_b = b['predictions'][:, -1] == b['targets'][:, -1]
                    d = float(ok_a.mean() - ok_b.mean()); diffs.append(d)
                    per_seed.append(dict(seed=seed, arm_correct=int(ok_a.sum()), base_correct=int(ok_b.sum()), gained=int((ok_a & ~ok_b).sum()), lost=int((~ok_a & ok_b).sum()), difference=d))
                contrasts.append(dict(task=task, arm=arm, endpoint=endpoint, per_seed=per_seed, **seed_summary(diffs),
                                      primary=(task == p['primary']['task'] and endpoint in p['primary']['endpoints'])))
    decisions = {}
    for arm in p['arms']:
        if arm == 'base':
            continue
        prim = [c for c in contrasts if c['primary'] and c['arm'] == arm]
        cost = next(c for c in contrasts if c['arm'] == arm and c['task'] == p['primary']['task'] and c['endpoint'] == p['primary']['cost_endpoint'])
        positive = all(c['mean'] * 100 >= p['primary']['gain_pp'] and c['direction'] == 'POSITIVE_INTERVAL' for c in prim) and cost['mean'] * 100 >= -p['primary']['max_cost_pp']
        negative = any(c['mean'] <= 0 for c in prim) or cost['mean'] * 100 < -p['primary']['max_cost_pp']
        decisions[arm] = dict(status='MEASURED_POSITIVE' if positive else 'MEASURED_NEGATIVE' if negative else 'MIXED',
                              primary=[dict(endpoint=c['endpoint'], mean_pp=c['mean'] * 100, t95_pp=[v * 100 for v in c['t95']], direction=c['direction']) for c in prim],
                              cost=dict(endpoint=cost['endpoint'], mean_pp=cost['mean'] * 100, t95_pp=[v * 100 for v in cost['t95']]))
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], plan_sha256=sha(ROOT / 'docs/drift-mini-plan.md'),
                  conditions=len(cs), contrasts=contrasts, absolute=absolute, decisions=decisions,
                  caveat='three seeds; exploratory t intervals without multiplicity adjustment; predeclared decision rule in docs/drift-mini-plan.md')
    write_json(out / 'analysis.json', result)
    return result


def worker(out, slot, slots):
    out = Path(out); manifest = verify_frozen(out)
    log = out / 'workers' / f'slot-{slot}.jsonl'; log.parent.mkdir(exist_ok=True)
    append_line(log, dict(state='STARTED', pid=os.getpid(), utc=time.time(), protocol_hash=manifest['protocol_hash']))
    for c in conditions(manifest['protocol'])[slot::slots]:
        append_line(log, dict(state='CONDITION_STARTED', condition=ident(c), utc=time.time()))
        r = run_one(out, c)
        e = evaluate_one(out, c)
        append_line(log, dict(state='CONDITION_COMPLETE', condition=ident(c), selected_step=r['selected_step'], wall_seconds=r['wall_seconds'],
                              L1024=next(x['final_accuracy'] for x in e['lengths'] if x['length'] == 1024), utc=time.time()))
    append_line(log, dict(state='COMPLETE', utc=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-one'); group.add_argument('--evaluate-one')
    group.add_argument('--worker', type=int); group.add_argument('--analyze', action='store_true'); group.add_argument('--list', action='store_true')
    parser.add_argument('--out', required=True); parser.add_argument('--slots', type=int, default=3)
    args = parser.parse_args()
    if args.freeze:
        result = freeze(args.out)
    elif args.list:
        result = [ident(c) for c in conditions(protocol())]
    elif args.run_one:
        result = run_one(args.out, known(verify_frozen(args.out), args.run_one)); result = {k: v for k, v in result.items() if k != 'history'}
    elif args.evaluate_one:
        result = evaluate_one(args.out, known(verify_frozen(args.out), args.evaluate_one))
    elif args.worker is not None:
        result = worker(args.out, args.worker, args.slots)
    else:
        result = analyze(args.out); result = dict(status=result['status'], decisions=result['decisions'])
    print(json.dumps(result, indent=1, allow_nan=False, default=str))


if __name__ == '__main__':
    main()
