"""U1 t-axis sweep: freeze, train at horizon 128, evaluate 128..4096 and neutral extensions, analyze.

Core is used through its public forward; the stack family needs a fourth input token, so the
embedding table of a freshly built Core is replaced by a seeded nn.Embedding(4, width) — an
input-vocabulary extension recorded in every receipt, not an architectural change.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.model import Core
from horizon import task_families as tf
from horizon import metrics as mt

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('core/model.py', 'review_cycle/training_data.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/metrics.py')


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


def append_line(path, row):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def protocol():
    return dict(version=1, purpose='U1 t-axis: fixed-state (five) versus growing-memory (stack) families trained at horizon 128',
                steps=2048, batch=32, train_length=128, lr=.003, weight_decay=.01, clip=1., width=32, heads=2, window=4, alpha=.1,
                tasks=['five', 'stack'], primary_seeds=[117, 143, 191],
                validation_every=128, validation_n=256, selection='minimum validation final cross entropy; earliest step on ties',
                test_n=1000, test_lengths=[128, 256, 512, 1024, 2048, 4096], neutral_slots=[0, 128, 512],
                neutral_placements=['prefix', 'middle', 'suffix'], eval_batch=32, threshold=.95,
                source_seeds=dict(train=91010000, validation=92010000, test=93010000, neutral=93020000),
                stack_probabilities=list(tf.STACK_PROBABILITIES), vocabulary_extension='stack: embedding replaced by seeded nn.Embedding(4, width)',
                device='cpu', threads=1, storage='raw arrays stored as uint8',
                targets=dict(all_points_accuracy=.95, neutral_loss_pp=2.))


def configure(p):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if p['threads'] != 1 or p['device'] != 'cpu':
        raise ValueError('one CPU thread required by this protocol')


def conditions(p):
    return [dict(task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'], lr=p['lr'], stage='u1')
            for task in p['tasks'] for seed in p['primary_seeds']]


def ident(c):
    return f"u1-{c['task']}-w{c['width']}-s{c['seed']}"


def make_model(c, p):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        model = Core(width=c['width'], heads=c['heads'], window=c['window'], classes=tf.CLASSES[c['task']], feedback=True, alpha=p['alpha'])
        if tf.VOCAB[c['task']] != model.embedding.num_embeddings:
            model.embedding = nn.Embedding(tf.VOCAB[c['task']], c['width'])
    return model


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol()); configure(p)
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(),
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__), frozen_utc=time.time())
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
def predict(model, x, y, p):
    model.eval()
    preds, prefix_loss, final_loss = [], 0., 0.
    start = time.perf_counter()
    for at in range(0, len(x), p['eval_batch']):
        xx, yy = x[at:at + p['eval_batch']], y[at:at + p['eval_batch']]
        z = model(xx)
        preds.append(z.argmax(-1).numpy())
        prefix_loss += float(F.cross_entropy(z.flatten(0, 1), yy.flatten(), reduction='sum'))
        final_loss += float(F.cross_entropy(z[:, -1], yy[:, -1], reduction='sum'))
    pred = np.concatenate(preds); target = y.numpy()
    return dict(final_accuracy=float((pred[:, -1] == target[:, -1]).mean()), prefix_accuracy=float((pred == target).mean()),
                prefix_loss=prefix_loss / y.numel(), final_loss=final_loss / len(x), seconds=time.perf_counter() - start, n=len(x)), pred


def run_one(out, c):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']; known(manifest, ident(c))
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    lock = directory / 'active.lock'; lock.write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall_start = time.perf_counter()
        model = make_model(c, p)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        initial = cpu_state(model); torch.save(initial, directory / 'initial.pt')
        vx, vy, vseed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', p['validation_n'], p['train_length'])
        history, best, selected, train_seconds = [], None, None, 0.
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train(); opt.zero_grad(set_to_none=True)
            logits = model(batch['x'])
            loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip']))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):
                write_json(directory / f'failure-step{step}.json', dict(status='FAIL_NONFINITE', step=step, loss=str(float(loss.detach())), grad_norm=str(grad)))
                raise ValueError('nonfinite loss/gradient; failure preserved')
            opt.step()
            train_seconds += time.perf_counter() - start
            row = dict(step=step, real_update=step, loss=float(loss.detach()), grad_norm=grad, input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _ = predict(model, vx, vy, p)
                row['validation'] = stats
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=cpu_state(model), validation=stats)
            history.append(row); append_line(directory / 'updates.jsonl', row)
        last = cpu_state(model); torch.save(last, directory / 'last.pt'); torch.save(selected['model'], directory / 'selected.pt')
        receipt = dict(status='COMPLETE', condition=c, updates=p['steps'], labels=p['steps'] * p['batch'] * p['train_length'],
                       vocabulary=tf.VOCAB[c['task']], classes=tf.CLASSES[c['task']], parameters=sum(v.numel() for v in last.values()),
                       initial_state_hash=state_hash(initial), last_state_hash=state_hash(last), selected_state_hash=state_hash(selected['model']),
                       selected_step=selected['step'], selected_validation=selected['validation'], validation_source_seed=vseed,
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
    stats, pred = predict(model, x, y, p)
    path = Path(out) / filename; path.parent.mkdir(exist_ok=True)
    ids = np.asarray([hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest() for i in range(len(x))])
    np.savez_compressed(path, inputs=x.numpy().astype(np.uint8), predictions=pred.astype(np.uint8), targets=y.numpy().astype(np.uint8),
                        input_ids=ids, source_seed=np.asarray(str(seed)), condition=ident(c))
    classes = tf.CLASSES[c['task']]
    stats.update(raw_file=filename, sha256=sha(path), source_seed=seed, consumed_tokens=int(x.shape[1]),
                 target_histogram={str(i): int((y[:, -1] == i).sum()) for i in range(classes)},
                 prediction_histogram={str(i): int((pred[:, -1] == i).sum()) for i in range(classes)})
    if baseline is not None:
        stats.update(mt.flip_counts(baseline, pred[:, -1], y[:, -1].numpy()))
    return stats, pred[:, -1]


def evaluate_one(out, c, split='test', n=None, tag='selected'):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    directory = out / ident(c)
    suffix = '' if split == 'test' and n is None else f'-{split}{n or p["test_n"]}'
    path = directory / f'evaluation{suffix}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    if not (directory / 'complete.json').exists():
        raise ValueError('evaluation requires a complete condition: ' + ident(c))
    n = n or p['test_n']
    model = make_model(c, p); model.load_state_dict(torch.load(directory / f'{tag}.pt', weights_only=True))
    row = dict(condition=c, split=split, n=n, protocol_hash=manifest['protocol_hash'], checkpoint_sha256=sha(directory / f'{tag}.pt'), lengths=[], neutral=[])
    for length in p['test_lengths']:
        x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], split, n, length)
        stats, _ = raw_evaluation(model, x, y, c, p, out, f'{ident(c)}/{tag}{suffix}-L{length}.npz', seed)
        row['lengths'].append(dict(length=length, **stats))
    x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral' if split == 'test' else split, n, p['train_length'])
    baseline = None
    for slots in sorted(p['neutral_slots']):
        for placement in (['base'] if slots == 0 else p['neutral_placements']):
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred = raw_evaluation(model, xx, yy, c, p, out, f'{ident(c)}/{tag}{suffix}-neutral{slots}-{placement}.npz', seed, baseline)
            if slots == 0:
                baseline = pred
            row['neutral'].append(dict(slots=slots, placement=placement, **stats))
    row['metadata_hash'] = digest(row)
    write_json(path, row)
    return row


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    cs = conditions(p); per_condition = {}; per_task = {}
    for c in cs:
        ev = evaluate_one(out, c)
        acc = {r['length']: r['final_accuracy'] for r in ev['lengths']}
        base = next(r for r in ev['neutral'] if r['slots'] == 0)['final_accuracy']
        neutral = {f"{r['slots']}-{r['placement']}": dict(final_accuracy=r['final_accuracy'], loss_pp=(base - r['final_accuracy']) * 100,
                                                        correct_to_wrong=r.get('correct_to_wrong'), wrong_to_correct=r.get('wrong_to_correct'))
                   for r in ev['neutral'] if r['slots']}
        per_condition[ident(c)] = dict(task=c['task'], seed=c['seed'], accuracy_by_length=acc,
                                       prefix_accuracy_by_length={r['length']: r['prefix_accuracy'] for r in ev['lengths']},
                                       effective_horizon=mt.effective_horizon(acc, p['threshold']), neutral_base=base, neutral=neutral)
    for task in p['tasks']:
        rows = [v for v in per_condition.values() if v['task'] == task]
        summary = dict(seeds=[r['seed'] for r in rows],
                       accuracy_by_length={str(L): mt.seed_summary([r['accuracy_by_length'][L] for r in rows]) for L in p['test_lengths']},
                       effective_horizon=[r['effective_horizon'] for r in rows],
                       all_points_ge_threshold=[all(a >= p['threshold'] for a in r['accuracy_by_length'].values()) for r in rows],
                       neutral_loss_pp={k: mt.seed_summary([r['neutral'][k]['loss_pp'] for r in rows]) for k in rows[0]['neutral']},
                       neutral_within_target=[all(v['loss_pp'] <= p['targets']['neutral_loss_pp'] for v in r['neutral'].values()) for r in rows])
        summary['status'] = ('MEASURED_POSITIVE' if all(summary['all_points_ge_threshold']) and all(summary['neutral_within_target'])
                             else 'MEASURED_NEGATIVE' if not any(summary['all_points_ge_threshold']) else 'MIXED')
        per_task[task] = summary
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], conditions=per_condition, tasks=per_task,
                  caveat='three seeds per family; t axis only (k, d deferred); 1000 fresh problems per cell; intervals exploratory')
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
                              accuracy={str(x['length']): x['final_accuracy'] for x in e['lengths']}, utc=time.time()))
    append_line(log, dict(state='COMPLETE', utc=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-one'); group.add_argument('--evaluate-one')
    group.add_argument('--pilot-eval', help='development-only evaluation of a completed condition on the validation namespace')
    group.add_argument('--worker', type=int); group.add_argument('--analyze', action='store_true'); group.add_argument('--list', action='store_true')
    parser.add_argument('--out', required=True); parser.add_argument('--slots', type=int, default=3); parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()
    if args.freeze:
        result = freeze(args.out)
    elif args.list:
        result = [ident(c) for c in conditions(protocol())]
    elif args.run_one:
        result = {k: v for k, v in run_one(args.out, known(verify_frozen(args.out), args.run_one)).items() if k != 'history'}
    elif args.evaluate_one:
        result = evaluate_one(args.out, known(verify_frozen(args.out), args.evaluate_one))
    elif args.pilot_eval:
        result = evaluate_one(args.out, known(verify_frozen(args.out), args.pilot_eval), split='validation', n=args.n)
    elif args.worker is not None:
        result = worker(args.out, args.worker, args.slots)
    else:
        result = analyze(args.out); result = dict(status=result['status'], tasks={k: dict(status=v['status'], effective_horizon=v['effective_horizon']) for k, v in result['tasks'].items()})
    print(json.dumps(result, indent=1, allow_nan=False, default=str))


if __name__ == '__main__':
    main()
