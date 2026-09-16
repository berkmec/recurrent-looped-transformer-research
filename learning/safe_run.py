"""Reviewed successor entrypoint; refuses overwriting any existing run artifacts.

The original run.py remains frozen evidence of protocol-v3 execution. Do not
invoke it directly for a new run. A fresh experiment requires a new output copy
with its own reviewed protocol and empty results/checkpoints directories.
"""
from pathlib import Path
import hashlib,json


def assert_protocol_integrity(output_root):
    root=Path(output_root).resolve()
    actual=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    if actual!=(root/'protocol.sha256').read_text().strip():raise ValueError('protocol differs from frozen SHA256')
    receipts=[json.loads(line) for line in (root/'execution.jsonl').read_text().splitlines()]
    frozen=[r for r in receipts if r.get('event')=='protocol_frozen' and r.get('sha256')==actual]
    if len(frozen)!=1:raise ValueError('missing or ambiguous original protocol receipt')
    sources=json.loads((root/'source-hashes.json').read_text())
    if sources!=frozen[0]['sources']:raise ValueError('source hash index differs from original frozen receipt')
    for name,digest in sources.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('source drift: '+name)
    return actual


def assert_fresh_output(output_root):
    root=Path(output_root).resolve()
    evidence=[]
    for name in ('results','checkpoints'):
        folder=root/name
        if folder.exists():evidence.extend(str(path) for path in folder.iterdir())
    for name in ('training.json','revision-selection.json'):
        if (root/name).exists():evidence.append(str(root/name))
    if evidence:raise FileExistsError('Existing evidence prohibits restart: '+', '.join(evidence[:3]))


def main():
    root=Path(__file__).resolve().parent
    assert_fresh_output(root)
    assert_protocol_integrity(root)
    from .run import main as frozen_runner
    frozen_runner()

if __name__=='__main__':main()
