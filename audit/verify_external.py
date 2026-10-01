"""Verify a vendored external package against its own MANIFEST.json and record the check (our code; the package is not executed)."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, out, extra_sources=None):
    root = Path(root)
    man = json.loads((root / 'MANIFEST.json').read_text(encoding='utf8'))
    listed = {f['path'] for f in man['files']}
    rows = []
    for f in man['files']:
        p = root / f['path']
        rows.append(dict(path=f['path'], bytes=p.stat().st_size, sha256=sha(p), matches_manifest=(sha(p) == f['sha256'] and p.stat().st_size == f['bytes'])))
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in listed)
    rec = dict(vendored_root=root.as_posix(), manifest_delivery=man.get('delivery'), files=rows, all_match=all(r['matches_manifest'] for r in rows),
               files_not_in_manifest=extra, extra_sources={k: dict(path=v, sha256=sha(v)) for k, v in (extra_sources or {}).items()},
               role='external design reference (user-commissioned package); its code is NOT executed in this worktree; its own tests are its own contract tests',
               verified_utc=time.time())
    Path(out).write_text(json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf8')
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--source', action='append', default=[], help='name=path of an original file to hash alongside')
    args = parser.parse_args()
    rec = verify(args.root, args.out, dict(s.split('=', 1) for s in args.source))
    print(json.dumps(dict(files=len(rec['files']), all_match=rec['all_match'], not_in_manifest=rec['files_not_in_manifest']), ensure_ascii=False))


if __name__ == '__main__':
    main()
