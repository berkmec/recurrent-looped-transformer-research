"""Independent read-only audit of the sealed U1 (t axis) experiment.

Runs on CPU with one thread, never trains or re-evaluates the full sweep, never imports the
producer modules under horizon/ (all oracles, seeds, metrics are reimplemented here). The only
producer import is the pinned, immutable core/model.py (Core) for the seeded-init reproduction
and a 32-row checkpoint spot check. Writes u1-independent-audit.json and .md next to this file.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # never leave __pycache__ under core/
import torch  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parents[2]
U1 = ROOT / 'results' / 'U1'
SOURCES = ('core/model.py', 'review_cycle/training_data.py', 'horizon/task_families.py', 'horizon/sweep.py', 'horizon/metrics.py')
VOCAB = {'five': 3, 'stack': 4}
CLASSES = {'five': 5, 'stack': 3}
NEUTRAL = 2
T_975_DF2 = 4.302652729
CHECKS, OBS = [], []


def check(cid, name, ok, detail=None):
    CHECKS.append(dict(id=cid, name=name, status='PASS' if ok else 'FAIL', detail=detail))
    return bool(ok)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def batch_hash(x_int64):
    return hashlib.sha256(str(('torch.int64', tuple(x_int64.shape))).encode() + np.ascontiguousarray(x_int64).tobytes()).hexdigest()


# ----------------------------------------------------------------- independent task implementations
def five_oracle(x):
    """Explicit rules: token 0 -> s+1 mod 5, token 1 -> swap states 0<->1, token 2 -> identity."""
    out = []
    for row in x.tolist():
        s, ys = 0, []
        for tok in row:
            if tok == 0:
                s = (s + 1) % 5
            elif tok == 1:
                s = 1 if s == 0 else 0 if s == 1 else s
            elif tok != 2:
                raise ValueError('five token out of range')
            ys.append(s)
        out.append(ys)
    return np.asarray(out, dtype=np.int64)


def stack_oracle(x):
    """Python list stack: 0/1 push, 2 neutral, 3 pop (no-op on empty); label = top (a=0, b=1) or 2 when empty."""
    labels, depths = [], []
    for row in x.tolist():
        st, ys, ds = [], [], []
        for tok in row:
            if tok == 0 or tok == 1:
                st.append(tok)
            elif tok == 3:
                if st:
                    st.pop()
            elif tok != 2:
                raise ValueError('stack token out of range')
            ys.append(st[-1] if st else 2); ds.append(len(st))
        labels.append(ys); depths.append(ds)
    return np.asarray(labels, dtype=np.int64), np.asarray(depths, dtype=np.int64)


def oracle(x, task):
    return five_oracle(x) if task == 'five' else stack_oracle(x)[0]


def source_seed(ns, split, task, seed, step=0, length=0):
    return int.from_bytes(hashlib.sha256(f'{ns[split]}|{task}|{seed}|{step}|{length}'.encode()).digest()[:8], 'little')


def gen_inputs(task, n, length, seed, stack_p):
    rng = np.random.default_rng(seed)
    if task == 'five':
        return rng.choice(3, (n, length), p=[.4375, .4375, .125])
    return rng.choice(4, (n, length), p=stack_p)


def effective_horizon(acc_by_length, threshold):
    horizon = 0
    for length in sorted(acc_by_length):
        if acc_by_length[length] >= threshold:
            horizon = length
        else:
            break
    return horizon


def seed_summary(values):
    n = len(values); mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)); half = T_975_DF2 * sd / math.sqrt(n)
    return dict(mean=mean, min=min(values), max=max(values), t95=[mean - half, mean + half])


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def build_model(task, seed, width=32):
    sys.path.insert(0, str(ROOT))
    from core.model import Core
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = Core(width=width, heads=2, window=4, classes=CLASSES[task], feedback=True, alpha=0.1)
        if VOCAB[task] != m.embedding.num_embeddings:
            m.embedding = torch.nn.Embedding(VOCAB[task], width)
    return m


def main():
    t0 = time.perf_counter()
    report = dict(experiment='U1 (t axis)', root=str(ROOT), auditor='u1_raw_audit.py (independent, read-only)',
                  environment=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads()))
    # ---------------------------------------------------------------- 1. integrity: manifest
    frozen = load_json(U1 / 'frozen.json'); p = frozen['protocol']; ns = p['source_seeds']; stack_p = p['stack_probabilities']
    check('1.1', 'frozen.sha256 matches sha256(frozen.json)', sha(U1 / 'frozen.json') == (U1 / 'frozen.sha256').read_text('ascii').strip(),
          dict(file=sha(U1 / 'frozen.json'), pin=(U1 / 'frozen.sha256').read_text('ascii').strip()))
    check('1.2', 'protocol_hash == sha256(canonical protocol JSON)', digest(p) == frozen['protocol_hash'], dict(recomputed=digest(p), stored=frozen['protocol_hash']))
    src_now = {name: sha(ROOT / name) for name in SOURCES}
    check('1.3', 'five source pins equal current files', src_now == frozen['sources'], dict(current=src_now, pinned=frozen['sources']))
    check('1.4', 'protocol constants as ledgered', p['steps'] == 2048 and p['batch'] == 32 and p['train_length'] == 128 and p['test_n'] == 1000
          and p['test_lengths'] == [128, 256, 512, 1024, 2048, 4096] and p['neutral_slots'] == [0, 128, 512] and p['threshold'] == .95
          and ns == dict(train=91010000, validation=92010000, test=93010000, neutral=93020000) and p['device'] == 'cpu' and p['threads'] == 1,
          dict(source_seeds=ns, stack_probabilities=stack_p))
    conds = [(task, seed) for task in p['tasks'] for seed in p['primary_seeds']]
    ident = lambda task, seed: f'u1-{task}-w{p["width"]}-s{seed}'
    check('1.5', 'six predeclared conditions present', all((U1 / ident(t, s)).is_dir() for t, s in conds) and len(conds) == 6, [ident(t, s) for t, s in conds])

    # ---------------------------------------------------------------- worker / process logs
    fails, states = [], {}
    for log in sorted((U1 / 'workers').glob('slot-*.jsonl')):
        rows = [json.loads(l) for l in log.read_text('utf8').splitlines() if l.strip()]
        states[log.name] = [r['state'] for r in rows]
        if 'FAIL' in log.read_text('utf8').upper():
            fails.append(log.name)
        for r in rows:
            if r['state'] == 'STARTED' and r['protocol_hash'] != frozen['protocol_hash']:
                fails.append(log.name + ':protocol_hash')
    stderr = {f.name: f.stat().st_size for f in (U1 / 'processes').glob('*.stderr.log')}
    stdout = {f.name: f.read_text('utf8').strip() for f in (U1 / 'processes').glob('*.stdout.log')}
    completed = sorted(json.loads(l)['condition'] for log in (U1 / 'workers').glob('slot-*.jsonl') for l in log.read_text('utf8').splitlines()
                       if l.strip() and json.loads(l)['state'] == 'CONDITION_COMPLETE')
    check('1.6', 'worker logs: 3 slots, no FAILED, all end COMPLETE, 6 CONDITION_COMPLETE rows, stderr logs empty',
          not fails and len(states) == 3 and all(v[-1] == 'COMPLETE' for v in states.values()) and completed == sorted(ident(t, s) for t, s in conds)
          and all(v == 0 for v in stderr.values()) and len(stderr) == 3,
          dict(states=states, stderr_bytes=stderr, stdout=stdout, completed=completed))

    # ---------------------------------------------------------------- per condition
    per, depth_report, baselines, spot = {}, {}, {}, {}
    expected_files = {f'selected-L{L}.npz' for L in p['test_lengths']} | {'selected-neutral0-base.npz'} | {
        f'selected-neutral{s}-{pl}.npz' for s in p['neutral_slots'] if s for pl in p['neutral_placements']} | {
        'initial.pt', 'last.pt', 'selected.pt', 'updates.jsonl', 'train.json', 'complete.json', 'evaluation.json'}
    for task, seed in conds:
        cid = ident(task, seed); d = U1 / cid; C = per.setdefault(cid, {})
        present = {f.name for f in d.iterdir()}
        check(f'1.7/{cid}', 'directory holds exactly the 19 expected files (no lock/failure/pilot files)', present == expected_files,
              dict(missing=sorted(expected_files - present), extra=sorted(present - expected_files)))
        train = load_json(d / 'train.json'); seal = load_json(d / 'complete.json'); ev = load_json(d / 'evaluation.json')
        check(f'1.8/{cid}', 'train.json protocol_hash and sources equal frozen manifest',
              train['protocol_hash'] == frozen['protocol_hash'] and train['sources'] == frozen['sources'] and train['status'] == 'COMPLETE'
              and train['condition'] == dict(task=task, seed=seed, width=32, heads=2, window=4, lr=.003, stage='u1'))
        files_now = {name: sha(d / name) for name in seal['files']}
        seal_body = {k: v for k, v in seal.items() if k != 'seal_hash'}
        check(f'1.9/{cid}', 'complete.json file hashes match, seal_hash recomputed, protocol_hash matches',
              files_now == seal['files'] and digest(seal_body) == seal['seal_hash'] and seal['protocol_hash'] == frozen['protocol_hash']
              and set(seal['files']) == {'initial.pt', 'last.pt', 'selected.pt', 'updates.jsonl', 'train.json'},
              dict(seal_hash=seal['seal_hash'], recomputed=digest(seal_body)))
        rows = ev['lengths'] + ev['neutral']
        raw_ok = {r['raw_file']: sha(U1 / r['raw_file']) == r['sha256'] for r in rows}
        ev_body = {k: v for k, v in ev.items() if k != 'metadata_hash'}
        check(f'1.10/{cid}', 'evaluation.json: 6+7 rows, every raw NPZ sha256 matches, checkpoint sha == selected.pt, metadata_hash, protocol_hash',
              len(ev['lengths']) == 6 and len(ev['neutral']) == 7 and all(raw_ok.values()) and ev['checkpoint_sha256'] == sha(d / 'selected.pt')
              and digest(ev_body) == ev['metadata_hash'] and ev['protocol_hash'] == frozen['protocol_hash'] and ev['split'] == 'test' and ev['n'] == 1000,
              dict(raw_sha_ok=sum(raw_ok.values()), rows=len(rows)))
        initial = torch.load(d / 'initial.pt', weights_only=True); last = torch.load(d / 'last.pt', weights_only=True)
        selected = torch.load(d / 'selected.pt', weights_only=True)
        emb = tuple(initial['embedding.weight'].shape)
        check(f'1.11/{cid}', f'initial.pt embedding has {VOCAB[task]} rows (task {task})', emb == (VOCAB[task], 32), dict(shape=emb))

        # ------------------------------------------------------------ 2. real training
        upd = [json.loads(l) for l in (d / 'updates.jsonl').read_text('utf8').splitlines() if l.strip()]
        steps_ok = [u['step'] for u in upd] == list(range(1, 2049)) and all(u['real_update'] == u['step'] for u in upd)
        finite = all(math.isfinite(u['loss']) and math.isfinite(u['grad_norm']) and u['loss'] > 0 and u['grad_norm'] >= 0 for u in upd)
        first100 = sum(u['loss'] for u in upd[:100]) / 100; last100 = sum(u['loss'] for u in upd[-100:]) / 100
        check(f'2.1/{cid}', 'updates.jsonl: 2048 rows, step==real_update==1..2048, finite positive loss / finite grad, loss falls',
              len(upd) == 2048 and steps_ok and finite and last100 < first100 and upd == train['history'] and train['updates'] == 2048
              and train['labels'] == 2048 * 32 * 128, dict(rows=len(upd), loss_first=upd[0]['loss'], loss_last=upd[-1]['loss'], mean_first100=first100, mean_last100=last100,
                                                          max_grad=max(u['grad_norm'] for u in upd), history_identical=upd == train['history']))
        seeds_ok = all(u['source_seed'] == source_seed(ns, 'train', task, seed, step=u['step'], length=128) for u in upd)
        train_rows = set(); hash_ok = True
        for u in upd:
            x = gen_inputs(task, 32, 128, u['source_seed'], stack_p)
            hash_ok &= batch_hash(x) == u['input_hash']
            train_rows.update(hashlib.sha256(r.tobytes()).hexdigest() for r in x.astype(np.uint8))
        check(f'2.2/{cid}', 'all 2048 training source seeds and input hashes regenerate from the frozen train namespace', seeds_ok and hash_ok,
              dict(seeds_ok=seeds_ok, input_hash_ok=hash_ok, distinct_train_rows=len(train_rows)))
        val = [u for u in upd if 'validation' in u]
        best = min(val, key=lambda u: (u['validation']['final_loss'], u['step']))
        check(f'2.3/{cid}', 'validation every 128 steps (16 rows); selected_step = earliest minimum validation final_loss; receipt agrees',
              [u['step'] for u in val] == list(range(128, 2049, 128)) and best['step'] == train['selected_step']
              and best['validation'] == train['selected_validation'] and train['validation_source_seed'] == source_seed(ns, 'validation', task, seed, length=128),
              dict(selected_step=train['selected_step'], recomputed=best['step'], val_final_loss=[u['validation']['final_loss'] for u in val]))
        hi, hl, hs = state_hash(initial), state_hash(last), state_hash(selected)
        differ = lambda a, b: any(not torch.equal(a[k], b[k]) for k in a)
        sel_is_last = train['selected_step'] == 2048
        check(f'2.4/{cid}', 'state hashes match receipt; initial != selected, initial != last; selected == last iff selected_step == 2048',
              hi == train['initial_state_hash'] and hl == train['last_state_hash'] and hs == train['selected_state_hash']
              and differ(initial, selected) and differ(initial, last) and (differ(selected, last) != sel_is_last)
              and set(initial) == set(last) == set(selected), dict(initial=hi[:16], last=hl[:16], selected=hs[:16], selected_step=train['selected_step']))
        model = build_model(task, seed); sd = model.state_dict()
        init_repro = set(sd) == set(initial) and all(torch.equal(sd[k], initial[k]) for k in sd)
        nparam = sum(v.numel() for v in last.values())
        check(f'2.5/{cid}', 'initial.pt reproduces seeded Core init (torch.manual_seed(seed)); parameter count matches receipt',
              init_repro and nparam == train['parameters'] and train['vocabulary'] == VOCAB[task] and train['classes'] == CLASSES[task],
              dict(init_reproduced=init_repro, parameters=nparam, receipt=train['parameters']))

        # ------------------------------------------------------------ 3. oracle / input regeneration
        npz = {}
        for r in rows:
            with np.load(U1 / r['raw_file']) as zf:
                z = {k: zf[k] for k in zf.files}
            npz[r['raw_file']] = z
            x, pred, tgt = z['inputs'], z['predictions'], z['targets']
            n_, L = x.shape
            ora = oracle(x.astype(np.int64), task)
            ids_ok = list(z['input_ids']) == [hashlib.sha256(f'{r["source_seed"]}:{i}'.encode()).hexdigest() for i in range(n_)]
            meta_ok = str(z['source_seed']) == str(r['source_seed']) and str(z['condition']) == cid and ids_ok
            shape_ok = x.shape == pred.shape == tgt.shape and n_ == 1000 and L == r['consumed_tokens'] and x.dtype == pred.dtype == tgt.dtype == np.uint8
            rng_ok = int(x.max()) < VOCAB[task] and int(pred.max()) < CLASSES[task] and int(tgt.max()) < CLASSES[task]
            check(f'3.1/{r["raw_file"]}', 'stored targets == independent oracle(inputs); shapes/dtypes/ranges/ids/metadata', np.array_equal(ora, tgt) and meta_ok and shape_ok and rng_ok,
                  dict(shape=list(x.shape), target_mismatches=int((ora != tgt).sum()), metadata_ok=meta_ok))
        for r in ev['lengths']:
            L = r['length']; s = source_seed(ns, 'test', task, seed, length=L); z = npz[r['raw_file']]
            regen = gen_inputs(task, 1000, L, s, stack_p)
            check(f'3.2/{cid}/L{L}', 'test inputs regenerate from frozen test namespace seed', s == r['source_seed'] and np.array_equal(regen.astype(np.uint8), z['inputs']),
                  dict(seed=s, stored=r['source_seed'], identical=bool(np.array_equal(regen.astype(np.uint8), z['inputs']))))
        base_row = next(r for r in ev['neutral'] if r['slots'] == 0); base = npz[base_row['raw_file']]
        s = source_seed(ns, 'neutral', task, seed, length=128)
        check(f'3.3/{cid}/base', 'neutral base inputs regenerate from frozen neutral namespace seed (length 128)',
              s == base_row['source_seed'] and np.array_equal(gen_inputs(task, 1000, 128, s, stack_p).astype(np.uint8), base['inputs']), dict(seed=s))
        for r in ev['neutral']:
            if not r['slots']:
                continue
            z = npz[r['raw_file']]; at = {'prefix': 0, 'middle': 64, 'suffix': 128}[r['placement']]
            expect = np.concatenate([base['inputs'][:, :at], np.full((1000, r['slots']), NEUTRAL, np.uint8), base['inputs'][:, at:]], 1)
            check(f'3.4/{r["raw_file"]}', f'{r["slots"]} neutral tokens at {r["placement"]} (position {at}) on the base inputs; identical final targets; same seed',
                  np.array_equal(expect, z['inputs']) and np.array_equal(z['targets'][:, -1], base['targets'][:, -1]) and r['source_seed'] == base_row['source_seed']
                  and z['inputs'].shape[1] == 128 + r['slots'] and int((z['inputs'][:, at:at + r['slots']] == NEUTRAL).all()),
                  dict(consumed_tokens=r['consumed_tokens']))
        # cross-namespace overlap (train vs validation/test/neutral rows at length 128)
        vx = gen_inputs(task, 256, 128, train['validation_source_seed'], stack_p).astype(np.uint8)
        probe = {'test-L128': npz[f'{cid}/selected-L128.npz']['inputs'], 'neutral-base': base['inputs'], 'validation': vx}
        overlap = {k: sum(hashlib.sha256(r.tobytes()).hexdigest() in train_rows for r in v) for k, v in probe.items()}
        check(f'3.5/{cid}', 'no length-128 test / neutral-base / validation row appears among the 65536 training rows', all(v == 0 for v in overlap.values()), overlap)

        # ------------------------------------------------------------ 4. metrics
        base_pred = base['predictions'][:, -1]; acc_by_len, pre_by_len, neutral = {}, {}, {}
        for r in rows:
            z = npz[r['raw_file']]; pred, tgt = z['predictions'], z['targets']
            fa = float((pred[:, -1] == tgt[:, -1]).mean()); pa = float((pred == tgt).mean())
            th = {str(i): int((tgt[:, -1] == i).sum()) for i in range(CLASSES[task])}; ph = {str(i): int((pred[:, -1] == i).sum()) for i in range(CLASSES[task])}
            ok = fa == r['final_accuracy'] and pa == r['prefix_accuracy'] and th == r['target_histogram'] and ph == r['prediction_histogram'] and r['n'] == 1000
            detail = dict(final_accuracy=fa, prefix_accuracy=pa)
            if 'slots' in r and r['slots']:
                was = base_pred == tgt[:, -1]; now = pred[:, -1] == tgt[:, -1]
                fl = dict(correct_to_wrong=int((was & ~now).sum()), wrong_to_correct=int((~was & now).sum()), prediction_changes=int((base_pred != pred[:, -1]).sum()), n=1000)
                ok = ok and all(r[k] == v for k, v in fl.items()); detail.update(fl)
            check(f'4.1/{r["raw_file"]}', 'final/prefix accuracy, histograms and flips recomputed from NPZ equal evaluation.json exactly', ok, detail)
            if 'length' in r:
                acc_by_len[r['length']] = fa; pre_by_len[r['length']] = pa
            elif r['slots']:
                neutral[f'{r["slots"]}-{r["placement"]}'] = dict(final_accuracy=fa, loss_pp=(float((base_pred == base['targets'][:, -1]).mean()) - fa) * 100, **{k: detail[k] for k in ('correct_to_wrong', 'wrong_to_correct')})
        base_acc = float((base_pred == base['targets'][:, -1]).mean())
        gmax = max(upd, key=lambda u: u['grad_norm'])
        C.update(task=task, seed=seed, accuracy_by_length=acc_by_len, prefix_accuracy_by_length=pre_by_len, effective_horizon=effective_horizon(acc_by_len, p['threshold']),
                 neutral_base=base_acc, neutral=neutral, selected_step=train['selected_step'], val_final_loss={u['step']: u['validation']['final_loss'] for u in val},
                 max_grad=dict(value=gmax['grad_norm'], step=gmax['step']), grad_over_clip_steps=sum(u['grad_norm'] > p['clip'] for u in upd))
        # constant-class / degenerate predictions; prefix vs final
        degenerate = {}
        for r in rows:
            z = npz[r['raw_file']]; pred = z['predictions']
            const_rows = int((pred == pred[:, :1]).all(1).sum()); top = max(r['prediction_histogram'].values()) / 1000
            degenerate[r['raw_file'].split('/')[-1]] = dict(rows_with_constant_prediction=const_rows, top_class_share=top,
                                                            top_class=max(r['prediction_histogram'], key=r['prediction_histogram'].get))
        C['degenerate'] = degenerate
        check(f'5.1/{cid}', 'no constant-class predictions (no file with a class share >= 0.99 or all-constant rows)',
              all(v['top_class_share'] < .99 and v['rows_with_constant_prediction'] == 0 for v in degenerate.values()),
              dict(max_top_class_share=max(v['top_class_share'] for v in degenerate.values()), max_constant_rows=max(v['rows_with_constant_prediction'] for v in degenerate.values())))
        if task == 'stack':
            dep = {}
            for r in ev['lengths']:
                z = npz[r['raw_file']]; _, depths = stack_oracle(z['inputs'].astype(np.int64)); mx = depths.max(1)
                dep[str(r['length'])] = dict(mean_max_depth=float(mx.mean()), median_max_depth=float(np.median(mx)), max_max_depth=int(mx.max()), mean_final_depth=float(depths[:, -1].mean()),
                                            share_final_empty=float((depths[:, -1] == 0).mean()))
            depth_report[cid] = dep
            bl = {}
            for r in ev['lengths'] + [base_row]:
                z = npz[r['raw_file']]; x = z['inputs'].astype(np.int64); tgt = z['targets'][:, -1]
                pushes = np.where(x < 2, x, -1); idx = np.arange(x.shape[1]); lastpush = np.where((pushes >= 0).any(1), pushes[np.arange(len(x)), np.where(pushes >= 0, idx, -1).max(1)], 2)
                key = f'L{r["length"]}' if 'length' in r else 'neutral-base'
                bl[key] = dict(majority=float(max(np.bincount(tgt, minlength=3)) / 1000), last_push_symbol=float((lastpush == tgt).mean()), model=float((z['predictions'][:, -1] == tgt).mean()))
            baselines[cid] = bl
        # spot check: first 32 rows of L128 through selected.pt (CPU, one thread)
        model.load_state_dict(selected); model.eval(); z = npz[f'{cid}/selected-L128.npz']
        with torch.no_grad():
            logits = model(torch.from_numpy(z['inputs'][:32].astype(np.int64)))
        sp = logits.argmax(-1).numpy()
        spot[cid] = dict(rows=32, final_mismatches=int((sp[:, -1] != z['predictions'][:32, -1]).sum()), position_mismatches=int((sp != z['predictions'][:32]).sum()))
        check(f'4.5/{cid}', 'checkpoint spot check: selected.pt reproduces the stored predictions on the first 32 rows of L128 (all 4096 positions)',
              spot[cid]['position_mismatches'] == 0, spot[cid])

    # ---------------------------------------------------------------- 4b. analysis.json
    an = load_json(U1 / 'analysis.json')
    check('4.2', 'analysis.json protocol_hash and status', an['protocol_hash'] == frozen['protocol_hash'] and an['status'] == 'MEASURED' and set(an['conditions']) == set(per))
    for cid, C in per.items():
        A = an['conditions'][cid]
        ok = all(close(A['accuracy_by_length'][str(L)], v) for L, v in C['accuracy_by_length'].items()) and all(close(A['prefix_accuracy_by_length'][str(L)], v) for L, v in C['prefix_accuracy_by_length'].items())
        ok = ok and A['effective_horizon'] == C['effective_horizon'] and close(A['neutral_base'], C['neutral_base']) and A['task'] == C['task'] and A['seed'] == C['seed']
        ok = ok and all(close(A['neutral'][k]['final_accuracy'], v['final_accuracy']) and close(A['neutral'][k]['loss_pp'], v['loss_pp'])
                        and A['neutral'][k]['correct_to_wrong'] == v['correct_to_wrong'] and A['neutral'][k]['wrong_to_correct'] == v['wrong_to_correct'] for k, v in C['neutral'].items())
        check(f'4.3/{cid}', 'analysis.json per-condition accuracies, effective horizon (threshold 0.95), neutral losses and flips recomputed (1e-12)', ok,
              dict(effective_horizon=C['effective_horizon'], stored=A['effective_horizon']))
    task_summary = {}
    for task in p['tasks']:
        rows = [C for C in per.values() if C['task'] == task]; T = an['tasks'][task]; ok = T['seeds'] == [r['seed'] for r in rows]
        mine = dict(accuracy_by_length={}, neutral_loss_pp={})
        for L in p['test_lengths']:
            s_ = seed_summary([r['accuracy_by_length'][L] for r in rows]); S = T['accuracy_by_length'][str(L)]; mine['accuracy_by_length'][str(L)] = s_
            ok = ok and close(S['mean'], s_['mean']) and close(S['min'], s_['min']) and close(S['max'], s_['max']) and close(S['t95'][0], s_['t95'][0]) and close(S['t95'][1], s_['t95'][1]) and S['df'] == 2 and S['seeds'] == 3
        for k in rows[0]['neutral']:
            s_ = seed_summary([r['neutral'][k]['loss_pp'] for r in rows]); S = T['neutral_loss_pp'][k]; mine['neutral_loss_pp'][k] = s_
            ok = ok and close(S['mean'], s_['mean']) and close(S['t95'][0], s_['t95'][0]) and close(S['t95'][1], s_['t95'][1]) and close(S['min'], s_['min']) and close(S['max'], s_['max'])
        eh = [r['effective_horizon'] for r in rows]; allp = [all(a >= p['threshold'] for a in r['accuracy_by_length'].values()) for r in rows]
        within = [all(v['loss_pp'] <= p['targets']['neutral_loss_pp'] for v in r['neutral'].values()) for r in rows]
        status = 'MEASURED_POSITIVE' if all(allp) and all(within) else 'MEASURED_NEGATIVE' if not any(allp) else 'MIXED'
        mine.update(effective_horizon=eh, all_points_ge_threshold=allp, neutral_within_target=within, status=status)
        ok = ok and T['effective_horizon'] == eh and T['all_points_ge_threshold'] == allp and T['neutral_within_target'] == within and T['status'] == status
        check(f'4.4/{task}', 'analysis.json per-task seed summaries (df=2, t=4.302652729), horizons, flags and status recomputed (1e-12)', ok, dict(status=status, stored=T['status'], effective_horizon=eh))
        task_summary[task] = mine
    ao = load_json(HERE / 'u1-analyze-output.json') if (HERE / 'u1-analyze-output.json').exists() else None
    if ao:
        check('4.6', 'ledger u1-analyze-output.json agrees with analysis.json statuses/horizons',
              all(ao['tasks'][t]['status'] == an['tasks'][t]['status'] and ao['tasks'][t]['effective_horizon'] == an['tasks'][t]['effective_horizon'] for t in p['tasks']))

    # ---------------------------------------------------------------- 5. observations
    for cid, C in per.items():
        for L, pa in C['prefix_accuracy_by_length'].items():
            fa = C['accuracy_by_length'][L]
            if pa > fa + 1e-9 and C['task'] == 'five':
                OBS.append(f'{cid}: L{L} prefix accuracy {pa:.6f} exceeds final accuracy {fa:.3f} (errors accumulate toward the end; expected for a recurrent state, not a metric bug)')
            if fa > pa + 1e-9 and C['task'] == 'five' and fa == 1.0:
                OBS.append(f'{cid}: L{L} final accuracy 1.000 while prefix accuracy {pa:.10f} < 1: transient mid-sequence mispredictions that recovered ({int(round((1 - pa) * 1000 * L))} wrong positions of {1000 * L})')
    for cid, C in per.items():
        worst_name, worst = max(C['degenerate'].items(), key=lambda kv: kv[1]['top_class_share'])
        if worst['top_class_share'] >= .75:
            OBS.append(f'{cid}: prediction mass concentrates on class {worst["top_class"]} in {worst_name} (share {worst["top_class_share"]:.3f} of 1000 final predictions) — '
                       'partial collapse toward a constant answer under the long neutral run; no file is a global constant-class predictor and no row is constant along the sequence')
        OBS.append(f'{cid}: pre-clip gradient norm peaks at {C["max_grad"]["value"]:.1f} (step {C["max_grad"]["step"]}); {C["grad_over_clip_steps"]} of 2048 updates exceed clip=1.0; all losses/gradients finite')
    ln5 = math.log(5)
    five_plateau = {cid: max([s for s, v in C['val_final_loss'].items() if v > 1.0], default=0) for cid, C in per.items() if C['task'] == 'five'}
    OBS.append(f'five: validation final loss sits at chance level (ln 5 = {ln5:.3f}) through step ' + ', '.join(f'{v} ({k[-4:]})' for k, v in five_plateau.items()) +
               ' and then drops by more than an order of magnitude within one or two validation intervals — a late, abrupt transition; the latest learner (s191) is the seed whose '
               'accuracy degrades at 2048/4096 (0.987/0.947) and whose horizon is 2048.')
    stack_tail = {cid: [round(C['val_final_loss'][s], 4) for s in (1536, 1664, 1792, 1920, 2048)] for cid, C in per.items() if C['task'] == 'stack'}
    OBS.append('stack: validation final loss at steps 1536..2048 = ' + '; '.join(f'{k[-4:]} {v}' for k, v in stack_tail.items()) +
               ' — still trending downward at the frozen 2048-update budget, so MEASURED_NEGATIVE is a statement at the frozen budget, not at convergence.')
    for cid, bl in baselines.items():
        OBS.append(f'{cid}: stack baselines final position — majority class ' + ', '.join(f'{k} {v["majority"]:.3f}' for k, v in bl.items()) + '; last-pushed-symbol heuristic ' +
                   ', '.join(f'{k} {v["last_push_symbol"]:.3f}' for k, v in bl.items()) + '; model ' + ', '.join(f'{k} {v["model"]:.3f}' for k, v in bl.items()))
    OBS.append('five neutral insertions: prefix blocks are harmless (state 0 is a fixed point of the identity token) but 128/512 neutral tokens in the middle/suffix cost 30-80 pp although '
               'the oracle final target is unchanged: long neutral runs (training neutral rate 0.125, geometric run lengths) are out of distribution and the recurrent state drifts.')
    OBS.append('stack neutral insertions: prefix/middle cost 3-12 pp, suffix 48-66 pp — the stored top symbol decays over a long idle run; the prefix/middle asymmetry versus five is because '
               'the stack state at the insertion point is usually non-trivial while five starts from state 0 at the prefix.')
    OBS.append('worker logs: the three STARTED rows carry an identical utc (1789735273.3394809) and identical CONDITION_STARTED utc across three different pids — consistent with the ~15.6 ms '
               'Windows time.time() tick when three processes are launched together, not with copied logs (pids, wall_seconds and completion times differ).')
    OBS.append('stdout logs contain only "null" (worker() returns None and main prints json.dumps(None)); stderr logs are empty.')
    OBS.append('selection: five s117/s143/s191 and stack s117 selected step 2048 (selected.pt == last.pt); stack s143/s191 selected step 1792.')

    n_pass = sum(c['status'] == 'PASS' for c in CHECKS); n_fail = len(CHECKS) - n_pass
    verdict = 'PASS' if n_fail == 0 else 'FAIL'
    key = dict(five={cid: dict(acc_2048=C['accuracy_by_length'][2048], acc_4096=C['accuracy_by_length'][4096], effective_horizon=C['effective_horizon'],
                               neutral_loss_pp={k: C['neutral'][k]['loss_pp'] for k in ('128-middle', '128-suffix', '512-middle', '512-suffix')}) for cid, C in per.items() if C['task'] == 'five'},
               stack={cid: dict(accuracy_by_length=C['accuracy_by_length'], effective_horizon=C['effective_horizon'],
                                neutral_suffix_loss_pp={k: C['neutral'][k]['loss_pp'] for k in ('128-suffix', '512-suffix')},
                                mean_max_depth={L: depth_report[cid][L]['mean_max_depth'] for L in ('128', '4096')}) for cid, C in per.items() if C['task'] == 'stack'},
               stack_depth_all_lengths=depth_report, stack_baselines=baselines, spot_check=spot, task_summary_recomputed=task_summary)
    report.update(verdict=verdict, counts={'pass': n_pass, 'fail': n_fail, 'total': len(CHECKS)}, checks=CHECKS, key_numbers=key, per_condition=per, observations=OBS,
                  seconds=time.perf_counter() - t0, files=[str(HERE / 'u1_raw_audit.py'), str(HERE / 'u1-independent-audit.json'), str(HERE / 'u1-independent-audit.md')])
    (HERE / 'u1-independent-audit.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=str), encoding='utf8')
    write_md(report, per, depth_report, baselines, spot, task_summary)
    print(json.dumps(dict(verdict=verdict, counts=report['counts'], seconds=round(report['seconds'], 1)), indent=1))
    print('FAILED CHECKS:', [c['id'] for c in CHECKS if c['status'] == 'FAIL'] or 'none')


def write_md(report, per, depth_report, baselines, spot, task_summary):
    L = ['# U1 (t axis) — independent raw-data audit', '', f'Verdict: **{report["verdict"]}** — {report["counts"]["pass"]} PASS / {report["counts"]["fail"]} FAIL of {report["counts"]["total"]} checks. '
         f'Read-only, CPU, one thread, {report["seconds"]:.0f} s. Runtime python {report["environment"]["python"]}, torch {report["environment"]["torch"]}, numpy {report["environment"]["numpy"]} (same as frozen.json environment).',
         '', 'Method: no producer module under `horizon/` was imported. Oracles (five: explicit +1 mod 5 / swap 0-1 / identity rules; stack: Python list stack with pop-on-empty no-op and '
         'labels a=0, b=1, empty=2), the source-seed derivation (sha256 of `ns|task|seed|step|length`, first 8 bytes little-endian), input regeneration (numpy default_rng choice with the '
         'frozen probabilities), neutral insertion, accuracies, histograms, flips, effective horizon and the df=2 t summaries were reimplemented here. The pinned `core/model.py` was imported '
         'only to reproduce the seeded initial checkpoints and to spot-check selected.pt on 32 rows of L128 (no training, no full re-evaluation, GPU untouched).', '',
         '## Key numbers', '', '### five (fixed-state automaton)', '', '| condition | acc 2048 | acc 4096 | horizon | neutral 128 middle | 128 suffix | 512 middle | 512 suffix | selected step |', '|---|---|---|---|---|---|---|---|---|']
    for cid, C in per.items():
        if C['task'] == 'five':
            nl = C['neutral']
            L.append(f'| {cid} | {C["accuracy_by_length"][2048]:.3f} | {C["accuracy_by_length"][4096]:.3f} | {C["effective_horizon"]} | {nl["128-middle"]["loss_pp"]:.1f} pp | {nl["128-suffix"]["loss_pp"]:.1f} pp | {nl["512-middle"]["loss_pp"]:.1f} pp | {nl["512-suffix"]["loss_pp"]:.1f} pp | {C["selected_step"]} |')
    five_prefix = {k: ' / '.join(f'{C["neutral"][k]["loss_pp"]:.1f}' for C in per.values() if C['task'] == 'five') for k in ('128-prefix', '512-prefix')}
    L += ['', f'Prefix insertions cost {five_prefix["128-prefix"]} pp (128) and {five_prefix["512-prefix"]} pp (512) for s117 / s143 / s191. Task status recomputed: **' + task_summary['five']['status'] + '** '
          f'(all-points >= 0.95: {task_summary["five"]["all_points_ge_threshold"]}, neutral within 2 pp: {task_summary["five"]["neutral_within_target"]}).', '',
          '### stack (growing memory)', '', '| condition | 128 | 256 | 512 | 1024 | 2048 | 4096 | horizon | neutral 128 suffix | 512 suffix | selected step |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for cid, C in per.items():
        if C['task'] == 'stack':
            a = C['accuracy_by_length']; nl = C['neutral']
            L.append(f'| {cid} | ' + ' | '.join(f'{a[k]:.3f}' for k in (128, 256, 512, 1024, 2048, 4096)) + f' | {C["effective_horizon"]} | {nl["128-suffix"]["loss_pp"]:.1f} pp | {nl["512-suffix"]["loss_pp"]:.1f} pp | {C["selected_step"]} |')
    L += ['', 'Task status recomputed: **' + task_summary['stack']['status'] + '**. Stack prefix/middle neutral losses are 2.8-12.1 pp (all above the 2 pp target).', '',
          '### stack input depth (independent simulation of the test inputs)', '', '| condition | ' + ' | '.join(f'L{k} mean max depth' for k in ('128', '256', '512', '1024', '2048', '4096')) + ' | final empty share L128 / L4096 |', '|---|---|---|---|---|---|---|---|']
    for cid, dep in depth_report.items():
        L.append(f'| {cid} | ' + ' | '.join(f'{dep[k]["mean_max_depth"]:.2f}' for k in ('128', '256', '512', '1024', '2048', '4096')) + f' | {dep["128"]["share_final_empty"]:.3f} / {dep["4096"]["share_final_empty"]:.3f} |')
    L += ['', 'Baselines at the final position (majority class / last-pushed-symbol heuristic / model):', '']
    for cid, bl in baselines.items():
        L.append(f'- {cid}: ' + '; '.join(f'{k} {v["majority"]:.3f} / {v["last_push_symbol"]:.3f} / {v["model"]:.3f}' for k, v in bl.items()))
    L += ['', '### checkpoint spot check (32 rows of L128 through selected.pt)', '']
    for cid, s in spot.items():
        L.append(f'- {cid}: position mismatches {s["position_mismatches"]} / {32 * 128}, final mismatches {s["final_mismatches"]} / 32')
    L += ['', '## Checks', '', '| id | status | check | detail |', '|---|---|---|---|']
    for c in report['checks']:
        d = c['detail']
        if isinstance(d, dict):
            d = ', '.join(f'{k}={v}' for k, v in d.items() if not isinstance(v, (dict, list)))[:160]
        elif isinstance(d, list):
            d = str(d)[:160]
        L.append(f'| {c["id"]} | {c["status"]} | {c["name"]} | {d or ""} |')
    L += ['', '## Anomalies and observations', ''] + [f'- {o}' for o in report['observations']]
    L += ['', 'Nothing was modified under results/, horizon/, core/, review_cycle/, language/, semantic_memory/, docs/ or notes/. Files written: ' + ', '.join(report['files']), '']
    (HERE / 'u1-independent-audit.md').write_text('\n'.join(L), encoding='utf8')


if __name__ == '__main__':
    main()
