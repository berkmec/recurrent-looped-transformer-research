"""E-BLIMP (program note 5.67): the first standard benchmark for the language runs. BLiMP (Warstadt et al. 2020, TACL; 67 paradigms x 1,000
minimal pairs of English sentences, CC BY 4.0; nyu-mll/blimp at a pinned revision) scored the way language models are scored on it: a pair is
right when the grammatical sentence gets the higher total log-probability. L1 (feedback on, 30,518 updates) against A0F (the same Core without
feedback: the same data, order, seed and schedule) measures what the token feedback does to grammatical knowledge. Evaluation only, CPU.

Every sentence is scored as the start of a new document: the model reads the training corpus's separator and then the sentence's UTF-8 bytes;
the score is the summed log-probability of the sentence bytes (the first one given the separator). The models are causal, so right padding
leaves every earlier position unchanged: sentences of different lengths share a batch and the padding is masked out (tested).

Reading rule (written before the first run, program note 5.67), on the mean over the 67 paradigms of the pair accuracy (the usual BLiMP score):
  delta = acc_L1 - acc_A0F in percentage points; 95% interval from a paired bootstrap over the pairs, resampled within every paradigm
  (2,000 draws, seed 0)
  FEEDBACK HELPS GRAMMAR   delta >= +1.0 and the interval excludes 0
  FEEDBACK HURTS GRAMMAR   delta <= -1.0 and the interval excludes 0
  NO DIFFERENCE            the interval lies inside (-1.0, +1.0)
  INCONCLUSIVE             anything else
A tie (equal scores) counts as wrong and is reported. Descriptive: each model against chance (50%), per field and per paradigm.

  python -B -m mline.blimp score results/L1-scratch-w928-s401/latest.pt --data __RLT_ROOT__/datasets/nyu-mll__blimp --out results/BLIMP/L1.json
  python -B -m mline.blimp reading results/BLIMP/L1.json results/BLIMP/A0F.json
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mline import data as md
from mline import lm as ml

LN2 = math.log(2)
PARADIGMS, PAIRS = 67, 1000
EFFECT, DRAWS, CI = 1.0, 2000, 0.95


def load_pairs(directory, paradigms=PARADIGMS, pairs=PAIRS):
    """{paradigm: [{'good', 'bad', 'field', 'term', 'pair_id'}]} from the dataset's one parquet file per paradigm, and their sha256."""
    import pyarrow.parquet as pq
    files = sorted(Path(directory).glob('*/*.parquet'))
    out, digests = {}, {}
    for f in files:
        rows = pq.read_table(f).to_pylist()
        out[f.parent.name] = [dict(good=r['sentence_good'], bad=r['sentence_bad'], field=r['field'], term=r['linguistics_term'],
                                   pair_id=int(r['pair_id'])) for r in rows]
        digests[f.parent.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    if len(out) != paradigms or any(len(v) != pairs for v in out.values()):
        raise ValueError(f'expected {paradigms} paradigms x {pairs} pairs, found {len(out)} with sizes {sorted({len(v) for v in out.values()})}')
    return out, digests


def load(path, seed=401):
    """(model, step, arm) from a run's checkpoint (latest.pt: {'step', 'model', ...}) or snapshot; the arm, width and heads are read from
    the run's probe.json (the run directory, or its parent for a snapshot), window 4 and alpha 0.1 as mline.longrun trains them."""
    path = Path(path)
    run = path.parent if (path.parent / 'probe.json').exists() else path.parent.parent
    cfg = json.loads((run / 'probe.json').read_bytes().decode('utf8'))
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:                                   # the run's own checkpoint carries non-tensor metadata
        obj = torch.load(path, map_location='cpu', weights_only=False)
    state, step = (obj['model'], int(obj['step'])) if isinstance(obj, dict) and 'model' in obj and 'step' in obj else (obj, None)
    model = ml.build(cfg['arm'], width=cfg['width'], heads=cfg['heads'], window=4, alpha=0.1, seed=seed)
    model.load_state_dict(state)
    return model.eval(), step, cfg['arm']


def score(model, sentences, prefix=md.SEPARATOR, batch=64):
    """(n,) summed negative log-probability in bits of each sentence's bytes after `prefix` (the first byte given the prefix)."""
    seqs = [prefix + s.encode('utf8') for s in sentences]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros(len(seqs))
    with torch.no_grad():
        for at in range(0, len(order), batch):
            idx = order[at:at + batch]
            width = max(len(seqs[i]) for i in idx)
            grid = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width - 1), dtype=bool)
            for r, i in enumerate(idx):
                grid[r, :len(seqs[i])] = np.frombuffer(seqs[i], dtype=np.uint8)
                mask[r, len(prefix) - 1:len(seqs[i]) - 1] = True        # the predictions of the sentence bytes, no padding
            x, y = torch.from_numpy(grid[:, :-1]), torch.from_numpy(grid[:, 1:])
            logits = ml.forward_logits(model, x)
            loss = F.cross_entropy(logits.transpose(1, 2), y, reduction='none').numpy() / LN2
            out[idx] = (loss * mask).sum(1)
    return out


def run(weights, data, batch=64, log=print):
    pairs, digests = load_pairs(data)
    model, step, arm = load(weights)
    started = time.perf_counter()
    scores = {}
    for k, (name, rows) in enumerate(pairs.items()):
        good = score(model, [r['good'] for r in rows], batch=batch)
        bad = score(model, [r['bad'] for r in rows], batch=batch)
        scores[name] = dict(good=good.round(5).tolist(), bad=bad.round(5).tolist())
        log(f'  {k + 1:2d}/{len(pairs)} {name:<55} acc {float((good < bad).mean()):.3f}  ({time.perf_counter() - started:.0f} s)')
    fields = {name: rows[0]['field'] for name, rows in pairs.items()}
    return dict(weights=str(weights), step=step, arm=arm, data=str(data), sha256=digests, fields=fields, scores=scores,
                summary=summary(scores, fields))


def correct(scores):
    """{paradigm: (n,) bool}: the grammatical sentence strictly more probable (fewer bits); a tie is wrong."""
    return {p: np.array(v['good']) < np.array(v['bad']) for p, v in scores.items()}


def summary(scores, fields):
    c = correct(scores)
    per = {p: float(v.mean()) for p, v in c.items()}
    by_field = {}
    for p, a in per.items():
        by_field.setdefault(fields[p], []).append(a)
    ties = int(sum((np.array(v['good']) == np.array(v['bad'])).sum() for v in scores.values()))
    return dict(accuracy=100 * float(np.mean(list(per.values()))), by_field={f: 100 * float(np.mean(v)) for f, v in sorted(by_field.items())},
                per_paradigm={p: 100 * a for p, a in per.items()}, ties=ties)


def reading(r_l1, r_a0, draws=DRAWS, seed=0):
    a, b = correct(r_l1['scores']), correct(r_a0['scores'])
    if set(a) != set(b) or any(len(a[p]) != len(b[p]) for p in a):
        raise ValueError('the two results do not score the same pairs')
    names = sorted(a)
    delta = 100 * float(np.mean([a[p].mean() - b[p].mean() for p in names]))
    rng = np.random.default_rng(seed)
    boot = np.zeros(draws)
    for p in names:
        idx = rng.integers(0, len(a[p]), size=(draws, len(a[p])))
        boot += (a[p][idx].mean(1) - b[p][idx].mean(1))
    boot = 100 * boot / len(names)
    lo, hi = (float(np.quantile(boot, q)) for q in ((1 - CI) / 2, (1 + CI) / 2))
    if delta >= EFFECT and lo > 0:
        verdict = 'FEEDBACK HELPS GRAMMAR'
    elif delta <= -EFFECT and hi < 0:
        verdict = 'FEEDBACK HURTS GRAMMAR'
    elif lo > -EFFECT and hi < EFFECT:
        verdict = 'NO DIFFERENCE'
    else:
        verdict = 'INCONCLUSIVE'
    return dict(verdict=verdict, delta=delta, ci=[lo, hi], l1=r_l1['summary']['accuracy'], a0=r_a0['summary']['accuracy'],
                l1_by_field=r_l1['summary']['by_field'], a0_by_field=r_a0['summary']['by_field'])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('weights')
    s.add_argument('--data', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--batch', type=int, default=64)
    s.add_argument('--threads', type=int, default=2)
    r = sub.add_parser('reading')
    r.add_argument('l1')
    r.add_argument('a0')
    a = p.parse_args()
    if a.cmd == 'score':
        torch.set_num_threads(a.threads)
        result = run(a.weights, a.data, batch=a.batch, log=lambda m: print(m, flush=True))
        print(json.dumps(dict(weights=result['weights'], step=result['step'], arm=result['arm'], **result['summary']['by_field'],
                              accuracy=result['summary']['accuracy'], ties=result['summary']['ties'])), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(json.dumps(result).encode('utf8'))
    else:
        load_ = lambda f: json.loads(Path(f).read_bytes().decode('utf8'))
        print(json.dumps(reading(load_(a.l1), load_(a.a0)), indent=1))


if __name__ == '__main__':
    main()
