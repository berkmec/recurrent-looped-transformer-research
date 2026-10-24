"""E-BLIMP data step (program note 5.67.1): convert the downloaded BLiMP parquet files (nyu-mll/blimp, one per paradigm) to one JSONL file
next to them, with the sha256 of every parquet file in a side file. Run without torch: in this environment pyarrow (user site) crashes the
interpreter with a segmentation fault once torch and the project modules are loaded (2026-10-24 02:34), so mline.blimp reads the JSONL instead.

  python -P scripts/blimp_to_jsonl.py __RLT_ROOT__/datasets/nyu-mll__blimp
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

FIELDS = ('sentence_good', 'sentence_bad', 'field', 'linguistics_term', 'UID', 'pair_id')


def main(directory):
    directory = Path(directory)
    files = sorted(directory.glob('*/*.parquet'))
    digests, lines = {}, []
    for f in files:
        digests[f.parent.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        for row in pq.read_table(f).to_pylist():
            lines.append(json.dumps(dict(paradigm=f.parent.name, **{k: row[k] for k in FIELDS}), ensure_ascii=False))
    (directory / 'blimp.jsonl').write_bytes(('\n'.join(lines) + '\n').encode('utf8'))
    (directory / 'blimp.sha256.json').write_bytes(json.dumps(digests, indent=1, sort_keys=True).encode('utf8'))
    print(f'{len(files)} paradigms, {len(lines)} pairs -> {directory / "blimp.jsonl"}')


if __name__ == '__main__':
    main(sys.argv[1])
