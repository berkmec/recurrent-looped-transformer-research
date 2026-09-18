"""Validate immutable delivery hashes before rerunning mutable result scripts."""
from pathlib import Path
import hashlib
import json
import sys

if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    failures=[]
    for item in manifest['files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root):
            failures.append(item['path']+': unsafe path');continue
        if not path.is_file():
            failures.append(item['path']+': missing');continue
        blob=path.read_bytes()
        if len(blob)!=item['bytes'] or hashlib.sha256(blob).hexdigest()!=item['sha256']:
            failures.append(item['path']+': changed')
    print(json.dumps({'checked':len(manifest['files']),'success':not failures,
                      'failures':failures},ensure_ascii=False,indent=2))
    sys.exit(1 if failures else 0)
