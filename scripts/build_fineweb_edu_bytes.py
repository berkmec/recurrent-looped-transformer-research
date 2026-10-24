"""Build the natural-language byte corpus from FineWeb-Edu (program note 5.70; user rule 2026-10-24: no model is trained on the CNN corpus again,
datasets are chosen by purpose). One parquet file of HuggingFaceFW/fineweb-edu sample-10BT (ODC-By 1.0) at a pinned revision becomes uint8 byte
files in mline.data's format (documents joined by the separator), taken as disjoint prefixes of the file's row order:
  validation.bin  the first ~16 MB of documents   (monitoring during training)
  test.bin        the next ~16 MB                 (not opened before the final evaluation)
  train.bin       the rest, up to --train-mb
and corpus.json with the source, revision, parquet sha256, byte and document counts and the sha256 of every output. A document never
straddles two splits. Runs without torch: in this environment pyarrow (user site) crashes once torch is loaded (note 5.67.1).

  python -P scripts/build_fineweb_edu_bytes.py <parquet> __RLT_ROOT__/mline-data/fineweb-edu-bytes --revision <sha> --train-mb 2048
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

SEPARATOR = b'\n\n<|doc|>\n\n'


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('parquet')
    p.add_argument('out')
    p.add_argument('--revision', required=True)
    p.add_argument('--train-mb', type=int, default=2048)
    p.add_argument('--validation-mb', type=int, default=16)
    p.add_argument('--test-mb', type=int, default=16)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [('validation', a.validation_mb << 20), ('test', a.test_mb << 20), ('train', a.train_mb << 20)]
    files = {name: open(out / f'{name}.bin', 'wb') for name, _ in budgets}
    written = {name: 0 for name, _ in budgets}
    docs = {name: 0 for name, _ in budgets}
    stage, started, rows = 0, time.perf_counter(), 0
    pf = pq.ParquetFile(a.parquet)
    try:
        for batch in pf.iter_batches(batch_size=4096, columns=['text']):
            for text in batch.column('text').to_pylist():
                rows += 1
                name, budget = budgets[stage]
                blob = text.encode('utf8', errors='ignore') + SEPARATOR
                files[name].write(blob)
                written[name] += len(blob)
                docs[name] += 1
                if written[name] >= budget:
                    stage += 1
                    if stage == len(budgets):
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        for f in files.values():
            f.close()
    manifest = dict(
        source='HuggingFaceFW/fineweb-edu', config='sample-10BT', file=Path(a.parquet).name, revision=a.revision, license='odc-by',
        parquet_sha256=sha256(a.parquet), parquet_rows=pf.metadata.num_rows, rows_read=rows,
        encoding='utf8 bytes, vocabulary 256', separator=SEPARATOR.decode('utf8'),
        **{f'{n}_bytes': written[n] for n, _ in budgets}, **{f'{n}_documents': docs[n] for n, _ in budgets},
        **{f'{n}_sha256': sha256(out / f'{n}.bin') for n, _ in budgets},
        disjoint='validation is the first documents of the file, test the next, train what follows; no document is in two splits; '
                 'the test split is not read before the final evaluation',
        seconds=time.perf_counter() - started, utc=time.time(), python=sys.version.split()[0])
    (out / 'corpus.json').write_bytes(json.dumps(manifest, indent=1).encode('utf8'))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('sha256')}, indent=1))


if __name__ == '__main__':
    main()
