"""Scoped cycle manifests for the Phase 2 root: seal the files of one cycle (paths given) so later cycles can add
files without invalidating earlier seals. Reuses the digest/identity helpers of review_cycle.cycle_manifest read-only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from review_cycle import cycle_manifest as cm

ROOT = cm.ROOT
EXCLUDED_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache'}


def inventory(root, paths, exclude_files):
    root = Path(root).resolve(); out = {}
    for rel in paths:
        base = root / rel
        candidates = [base] if base.is_file() else sorted(base.rglob('*'))
        for path in candidates:
            relative = path.relative_to(root)
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                raise cm.IntegrityError('linked evidence path refused: ' + relative.as_posix())
            if any(p in EXCLUDED_DIRECTORIES for p in relative.parts) or not path.is_file() or path.suffix == '.pyc':
                continue
            if relative.as_posix() in exclude_files:
                continue
            out[relative.as_posix()] = path
    return dict(sorted(out.items()))


def names(cycle):
    return (f'audit/phase2-{cycle}-manifest.json', f'audit/phase2-{cycle}-manifest.sha256', f'audit/phase2-{cycle}-manifest-verification.json')


def build(root, cycle, paths):
    root = Path(root).resolve(); manifest, sidecar, verification = (root / n for n in names(cycle))
    if manifest.exists() or sidecar.exists():
        raise FileExistsError('preserve existing cycle manifest: ' + str(manifest))
    files = inventory(root, paths, set(names(cycle)))
    rows, signatures = [], {}
    for name, path in files.items():
        before = path.stat(); h = cm.digest(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise cm.IntegrityError('file changed while sealing: ' + name)
        signatures[name] = (after.st_size, after.st_mtime_ns); rows.append(dict(path=name, bytes=after.st_size, sha256=h))
    for name, path in files.items():
        now = path.stat()
        if (now.st_size, now.st_mtime_ns) != signatures[name]:
            raise cm.IntegrityError('file changed while sealing: ' + name)
    payload = dict(status='SEALED_CYCLE_SCOPE', cycle=cycle, utc=datetime.now(timezone.utc).isoformat(), root=str(root), scope=list(paths),
                   identity=dict(phase1=cm.phase1_identity(root), phase2_plan=cm.phase2_plan_identity()), file_count=len(rows),
                   total_bytes=sum(r['bytes'] for r in rows), files=rows, limit='Internal integrity of the listed scope only; not an attestation.')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf8') as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
    with sidecar.open('x', encoding='ascii') as out:
        out.write(cm.digest(manifest) + '\n')
    return dict(files=len(rows), bytes=payload['total_bytes'], verification=verify(root, cycle))


def verify(root, cycle):
    root = Path(root).resolve(); manifest, sidecar, verification = (root / n for n in names(cycle))
    if cm.digest(manifest) != sidecar.read_text(encoding='ascii').strip():
        raise cm.IntegrityError('manifest changed')
    payload = json.loads(manifest.read_text(encoding='utf8'))
    files = inventory(root, payload['scope'], set(names(cycle)))
    listed = [r['path'] for r in payload['files']]
    if set(listed) != set(files):
        raise cm.IntegrityError(f'scope inventory differs; missing={sorted(set(listed)-set(files))[:10]} extra={sorted(set(files)-set(listed))[:10]}')
    for row in payload['files']:
        path = files[row['path']]
        if path.stat().st_size != row['bytes'] or cm.digest(path) != row['sha256']:
            raise cm.IntegrityError('evidence changed: ' + row['path'])
    result = dict(status='PASS', cycle=cycle, utc=datetime.now(timezone.utc).isoformat(), files_verified=len(listed), bytes_verified=payload['total_bytes'],
                  manifest_sha256=cm.digest(manifest))
    verification.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify']); parser.add_argument('--cycle', required=True); parser.add_argument('--paths', nargs='*')
    args = parser.parse_args()
    result = build(ROOT, args.cycle, args.paths) if args.action == 'build' else verify(ROOT, args.cycle)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
