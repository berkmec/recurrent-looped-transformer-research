"""Bug hunt (note 5.72; user request 2026-10-24 ~10:25: "maybe we did not get the results because of low-quality data; hunt for bugs
too"). The CNN validation split shares documents with train: 323 of its 2,336 documents (14.7 %) occur byte for byte in train.bin, and
L1 / A0F read train about 2.4 times (500M bytes over 210 MB). This script
  1. measures the corpus: exact and 200-byte-prefix duplicates within train and between validation and train, and boilerplate
     markers (Daily Mail bylines and time stamps, the "(CNN) --" dateline);
  2. re-measures the two final models on validation split by that overlap. Rows are drawn exactly as mline.train.validate draws them
     (md.batch, seed 991, 32 rows of 512 bytes, 24 batches) from three streams: the original validation, its documents that are not
     in train ("clean") and those that are ("seen"). Per row: the mean bits of its bytes; per stream: both models' bits per byte, the
     paired difference A0F - L1 and a bootstrap interval over rows;
  3. decomposes the recorded end-of-run number: the first 8 batches of the original stream are the rows the battery used
     (mline.longrun eval_fn -> validate(batches=8)); each is labelled by the share of its bytes that lie in a seen document. The
     recorded L1 value (1.37449) must be reproduced, otherwise the measurement is invalid.
Evaluation only, nothing is trained: the CNN rule (note 5.71) closes training on that corpus, not measurement.

  python -B scripts/diag_cnn_contamination.py --out results/DIAG-cnn-contamination --device cuda
"""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SEP = b'\n\n<|doc|>\n\n'
CNN = Path('__RLT_ROOT__/mline-data/cnn-bytes')
RUNS = dict(L1='results/L1-scratch-w928-s401/latest.pt', A0F='results/L1-a0-w928-s401/latest.pt')
RECORDED = dict(L1=1.3744900967162967)                    # results/L1-scratch-w928-s401/train.log, step 30518
SEED, BATCH, LENGTH, BATCHES, BATTERY = 991, 32, 512, 24, 8
MARKERS = {'daily_mail_published': b'PUBLISHED: .', 'daily_mail_updated': b'UPDATED: .', 'byline_dot': b'By . ',
           'cnn_dateline': b'(CNN) --', 'last_updated_at': b'Last updated at'}


def documents(raw):
    """[(start, end)] byte spans of the documents of a corpus stream written as text + SEP, text + SEP, ..."""
    spans, at = [], 0
    while at < len(raw):
        j = raw.find(SEP, at)
        end = len(raw) if j < 0 else j
        if end > at:
            spans.append((at, end))
        at = end + len(SEP)
    return spans


def starts(n_bytes, seed, step, batch_size, length):
    """The row starts mline.data.batch draws for (seed, step): the same generator, so a row can be traced to its documents."""
    rng = np.random.default_rng((seed * 1_000_003 + step) & 0xFFFFFFFF)
    return rng.integers(0, n_bytes - length - 1, size=batch_size)


def rows(stream, steps):
    """(x, y, starts) for md.batch(stream, SEED, step, BATCH, LENGTH) over `steps`, built from the traced starts."""
    s = np.concatenate([starts(len(stream), SEED, step, BATCH, LENGTH) for step in steps])
    x = np.stack([stream[a:a + LENGTH] for a in s]).astype(np.int64)
    y = np.stack([stream[a + 1:a + LENGTH + 1] for a in s]).astype(np.int64)
    return x, y, s


def corpus_stats(train, val):
    tr, va = documents(train), documents(val)
    digest = lambda raw, s: hashlib.sha1(raw[s[0]:s[1]]).digest()
    tr_count = collections.Counter(digest(train, s) for s in tr)
    va_digest = [digest(val, s) for s in va]
    seen = np.array([d in tr_count for d in va_digest])
    tr_prefix = {train[s[0]:s[0] + 200] for s in tr if s[1] - s[0] >= 200}
    prefix_seen = np.array([val[s[0]:s[0] + 200] in tr_prefix for s in va])
    dup_train = sum(c - 1 for c in tr_count.values() if c > 1)
    markers = {k: dict(train=float(np.mean([m in train[s[0]:s[1]] for s in tr])), validation=float(np.mean([m in val[s[0]:s[1]] for s in va])))
               for k, m in MARKERS.items()}
    stats = dict(train_documents=len(tr), validation_documents=len(va), train_exact_duplicates=dup_train,
                 train_exact_duplicate_share=dup_train / len(tr), validation_seen_in_train=int(seen.sum()),
                 validation_seen_share=float(seen.mean()), validation_prefix200_seen=int(prefix_seen.sum()),
                 validation_seen_byte_share=float(sum(s[1] - s[0] for s, f in zip(va, seen) if f) / sum(s[1] - s[0] for s in va)),
                 mean_document_bytes=dict(train=float(np.mean([s[1] - s[0] for s in tr])), validation=float(np.mean([s[1] - s[0] for s in va]))),
                 markers=markers)
    return stats, va, seen


def interval(d, seed=0, n=10_000):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='2 rows per stream, timings only, no numbers printed')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train = bytes(np.memmap(CNN / 'train.bin', dtype=np.uint8, mode='r'))
    val_raw = bytes(np.memmap(CNN / 'validation.bin', dtype=np.uint8, mode='r'))
    stats, spans, seen = corpus_stats(train, val_raw)
    stats['epochs_of_the_runs'] = 500e6 / len(train)
    del train
    seen_byte = np.zeros(len(val_raw), dtype=np.uint8)
    for (s, e), f in zip(spans, seen):
        if f:
            seen_byte[s:e + len(SEP)] = 1
    streams = dict(original=np.frombuffer(val_raw, dtype=np.uint8),
                   clean=np.frombuffer(b''.join(val_raw[s:e] + SEP for (s, e), f in zip(spans, seen) if not f), dtype=np.uint8),
                   seen=np.frombuffer(b''.join(val_raw[s:e] + SEP for (s, e), f in zip(spans, seen) if f), dtype=np.uint8))
    stats['stream_bytes'] = {k: int(len(v)) for k, v in streams.items()}
    import torch
    import torch.nn.functional as F
    from mline import blimp as bl
    from mline import data as md
    from mline import lm as ml
    check_x, _, _ = md.batch(streams['original'], SEED, 1, BATCH, LENGTH)
    traced_x, _, traced_starts = rows(streams['original'], [1])
    assert np.array_equal(check_x, traced_x), 'traced row starts differ from mline.data.batch'
    models = {}
    for name, path in RUNS.items():
        model, step, arm = bl.load(path)
        models[name] = (model.to(a.device).eval(), step, arm)
    result = dict(corpus=stats, models={k: dict(path=RUNS[k], step=v[1], arm=v[2]) for k, v in models.items()}, streams={})
    for stream_name, stream in streams.items():
        x, y, s = rows(stream, range(1, BATCHES + 1))
        if a.smoke:
            x, y, s = x[:2], y[:2], s[:2]
        bits = {}
        for name, (model, _, _) in models.items():
            started, per_row = time.perf_counter(), []
            with torch.no_grad():
                for at in range(0, len(x), BATCH):
                    xb, yb = torch.from_numpy(x[at:at + BATCH]).to(a.device), torch.from_numpy(y[at:at + BATCH]).to(a.device)
                    logits, _ = ml.forward_with_gates(model, xb)
                    ce = F.cross_entropy(logits.transpose(1, 2).float(), yb, reduction='none') / np.log(2)
                    per_row.append(ce.mean(1).cpu().numpy())
            bits[name] = np.concatenate(per_row).astype(np.float64)
            print(f'{stream_name:8s} {name:3s} rows {len(x)}  {time.perf_counter() - started:.1f} s', flush=True)
        d = bits['A0F'] - bits['L1']
        entry = dict(rows=int(len(x)), bits_per_byte={k: float(v.mean()) for k, v in bits.items()}, delta_a0f_minus_l1=float(d.mean()),
                     delta_interval=interval(d) if len(d) > 2 else None)
        if stream_name == 'original':
            share = np.array([seen_byte[b:b + LENGTH + 1].mean() for b in s])
            battery = slice(0, min(len(x), BATTERY * BATCH))
            entry['battery'] = dict(rows=int(battery.stop), bits_per_byte={k: float(v[battery].mean()) for k, v in bits.items()},
                                    delta=float(d[battery].mean()), rows_mostly_seen=int((share[battery] >= .5).sum()))
            for label, mask in (('rows_mostly_seen', share >= .5), ('rows_mostly_clean', share < .5)):
                entry[label] = dict(rows=int(mask.sum()), bits_per_byte={k: float(v[mask].mean()) for k, v in bits.items()} if mask.any() else None,
                                    delta=float(d[mask].mean()) if mask.any() else None,
                                    delta_interval=interval(d[mask]) if mask.sum() > 2 else None)
        result['streams'][stream_name] = entry
    result['seconds'] = time.perf_counter() - t0
    result['utc'] = time.time()
    if a.smoke:
        print('smoke ok: %.0f s; numbers not printed' % result['seconds'])
        return
    (out / 'diag.json').write_bytes(json.dumps(result, indent=1).encode('utf8'))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
