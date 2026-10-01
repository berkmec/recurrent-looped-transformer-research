"""M-hattı M0: build a byte-level corpus from the locally cached datasets, on E: because C: has 0.6 GB left.

Byte level (vocabulary 256) is deliberate: no tokenizer to train, no extra download, fully deterministic, and it makes
the recurrent state carry information over more steps, which sharpens exactly the property the campaign measures.

The corpus is written once as a uint8 memmap plus a manifest recording the source dataset, the article range, the
byte count and the sha256 of the output, so a training run can be traced back to the exact bytes it saw. Train and
validation come from DISJOINT article ranges, never from a shuffle of the same pool.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

# The cache lives on C:, which is nearly full; nothing here writes to it, we only read.
SOURCES = {
    'cnn_dailymail': dict(path='cnn_dailymail', config='3.0.0', field='article'),
    'plato': dict(path='hugfaceguy0001/stanford_plato', config=None, field='text'),
}
SEPARATOR = b'\n\n<|doc|>\n\n'


def sha_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_texts(spec, split):
    from datasets import load_dataset
    ds = load_dataset(spec['path'], spec['config'], split=split) if spec['config'] else load_dataset(spec['path'], split=split)
    field = spec['field']
    if field not in ds.column_names:
        field = next(c for c in ds.column_names if ds.features[c].dtype == 'string')
    for row in ds:
        text = row[field]
        if text:
            yield text


def build(out, source='cnn_dailymail', target_mb=200, validation_mb=8, split='train'):
    """Write train.bin and validation.bin as raw bytes, from disjoint prefixes of the same stream."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec = SOURCES[source]
    target, val_target = int(target_mb * 1024 * 1024), int(validation_mb * 1024 * 1024)
    train_path, val_path = out / 'train.bin', out / 'validation.bin'
    if (out / 'corpus.json').exists():
        return json.loads((out / 'corpus.json').read_text(encoding='utf8'))
    written_val, written_train, docs_val, docs_train = 0, 0, 0, 0
    t0 = time.perf_counter()
    with open(val_path, 'wb') as fv, open(train_path, 'wb') as ft:
        for text in _iter_texts(spec, split):
            blob = text.encode('utf8', errors='ignore') + SEPARATOR
            if written_val < val_target:                      # the first documents are the held-out split
                fv.write(blob); written_val += len(blob); docs_val += 1
            elif written_train < target:
                ft.write(blob); written_train += len(blob); docs_train += 1
            else:
                break
    manifest = dict(source=source, dataset=spec['path'], config=spec['config'], split=split, encoding='utf8 bytes, vocabulary 256',
                    separator=SEPARATOR.decode('utf8'), train_bytes=written_train, validation_bytes=written_val,
                    train_documents=docs_train, validation_documents=docs_val,
                    train_sha256=sha_file(train_path), validation_sha256=sha_file(val_path),
                    disjoint='validation is the first documents of the stream, train is what follows; no document appears in both',
                    seconds=time.perf_counter() - t0, utc=time.time(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__))
    (out / 'corpus.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf8')
    return manifest


def load(out, split='train'):
    """Memory-mapped bytes; nothing is read into RAM until a batch asks for it."""
    path = Path(out) / f'{split}.bin'
    return np.memmap(path, dtype=np.uint8, mode='r')


def batch(data, seed, step, batch_size, length):
    """A deterministic batch: positions come from a seeded generator, so a step always yields the same window."""
    rng = np.random.default_rng((seed * 1_000_003 + step) & 0xFFFFFFFF)
    top = len(data) - length - 1
    starts = rng.integers(0, top, size=batch_size)
    x = np.stack([np.asarray(data[s:s + length], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1:s + length + 1], dtype=np.int64) for s in starts])
    return x, y, int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=os.environ.get('MLINE_DATA', '__RLT_ROOT__/mline-data/cnn-bytes'))
    parser.add_argument('--source', default='cnn_dailymail', choices=sorted(SOURCES))
    parser.add_argument('--target-mb', type=float, default=200)
    parser.add_argument('--validation-mb', type=float, default=8)
    args = parser.parse_args()
    m = build(args.out, args.source, args.target_mb, args.validation_mb)
    print(json.dumps({k: v for k, v in m.items() if k != 'separator'}, indent=1))


if __name__ == '__main__':
    main()
