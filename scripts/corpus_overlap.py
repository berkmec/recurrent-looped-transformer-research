"""Corpus hygiene check (note 5.71.2; written after bug hunt 1 found 13.9 % of the CNN validation documents in train): for a byte corpus in
mline.data's format, the exact duplicates inside train, the validation documents that occur in train byte for byte or by their first
200 bytes, and the document length quantiles. The test split is never read. Streams train.bin in blocks, so a 2 GB corpus fits.

  python -B scripts/corpus_overlap.py __RLT_ROOT__/mline-data/fineweb-edu-bytes --out results/FWEDU-build/overlap.json
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

SEP = b'\n\n<|doc|>\n\n'


def stream_documents(path, block=1 << 28):
    """Yield the documents of a corpus file (text + SEP, text + SEP, ...) reading it in blocks."""
    rest = b''
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            parts = (rest + chunk).split(SEP)
            rest = parts.pop()
            yield from (p for p in parts if p)
    if rest:
        yield rest


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('corpus')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.corpus)
    digests, prefixes, lengths = collections.Counter(), set(), []
    for doc in stream_documents(root / 'train.bin'):
        digests[hashlib.sha1(doc).digest()] += 1
        if len(doc) >= 200:
            prefixes.add(hashlib.sha1(doc[:200]).digest())
        lengths.append(len(doc))
    val = list(stream_documents(root / 'validation.bin'))
    seen = [hashlib.sha1(d).digest() in digests for d in val]
    prefix_seen = [hashlib.sha1(d[:200]).digest() in prefixes for d in val]
    lengths = np.array(lengths)
    out = dict(corpus=str(root), train_documents=int(len(lengths)), train_exact_duplicates=int(sum(c - 1 for c in digests.values() if c > 1)),
               validation_documents=len(val), validation_seen_in_train=int(sum(seen)), validation_prefix200_seen=int(sum(prefix_seen)),
               validation_seen_byte_share=float(sum(len(d) for d, s in zip(val, seen) if s) / max(1, sum(len(d) for d in val))),
               train_length_quantiles={q: int(np.quantile(lengths, q)) for q in (.1, .25, .5, .75, .9, .99)},
               train_mean_length=float(lengths.mean()), train_share_at_least={n: float((lengths >= n).mean()) for n in (512, 2048, 4096, 8192)},
               test_split='not read')
    Path(a.out).write_bytes(json.dumps(out, indent=1).encode('utf8'))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
