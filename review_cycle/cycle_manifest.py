"""Seal/verify the review-diagnostics cycle delivery without touching the Phase 1 manifest.

Manifest: audit/review-diagnostics-manifest.json (+ .sha256 sidecar, + verification record).
Includes every file in this worktree except .git, __pycache__, .pytest_cache and the three
sidecars, so raw checkpoints, NPZ arrays, journals and the SDD ledger workspace are covered.
Also rechecks the Phase 1 campaign identity (tracked manifest bytes) and the Phase 2 plan/review
file hashes recorded by their own closure receipts. Internal integrity only; not an attestation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'audit/review-diagnostics-manifest.json'
DIGEST = 'audit/review-diagnostics-manifest.sha256'
VERIFICATION = 'audit/review-diagnostics-manifest-verification.json'
EXCLUDED_FILES = {MANIFEST, DIGEST, VERIFICATION}
EXCLUDED_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache'}
PHASE1_MANIFEST_SHA256 = '146eb693e3303e17b1caa6bba2b1eb7662e20d87bb0ff3e71bbdaeb1e54484af'
PHASE1_HEAD = '3c118c7afd21fb1b04733ad0e12c32fef3d895ba'
PHASE2_PLAN = Path('E:/rlt-research/2026-09-16-phase2-plan')


class IntegrityError(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    out = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise IntegrityError('linked evidence path refused: ' + relative.as_posix())
        if any(p in EXCLUDED_DIRECTORIES for p in relative.parts):
            continue
        if not path.is_file() or relative.as_posix() in EXCLUDED_FILES or path.suffix == '.pyc':
            continue
        out[relative.as_posix()] = path
    return dict(sorted(out.items()))


def phase1_identity(root):
    manifest = root / 'audit/final-manifest.json'
    observed = digest(manifest)
    sidecar = (root / 'audit/final-manifest.sha256').read_text(encoding='ascii').strip()
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(['git', '-C', str(root), 'rev-parse', '--abbrev-ref', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if observed != PHASE1_MANIFEST_SHA256 or sidecar != PHASE1_MANIFEST_SHA256:
        raise IntegrityError('Phase 1 manifest identity changed in this worktree')
    return dict(phase1_manifest_sha256=observed, phase1_sidecar=sidecar, git_head=head, git_branch=branch,
                base_commit_expected=PHASE1_HEAD, base_commit_reachable=subprocess.run(
                    ['git', '-C', str(root), 'merge-base', '--is-ancestor', PHASE1_HEAD, 'HEAD']).returncode == 0)


def phase2_plan_identity():
    receipts = {}
    for receipt in ('plan-kontrol.json', 'reviews/2026-09-16-inceleme-kapanis.json'):
        record = json.loads((PHASE2_PLAN / receipt).read_text(encoding='utf8'))
        base = PHASE2_PLAN / 'reviews' if receipt.startswith('reviews/') else PHASE2_PLAN
        rows = []
        for artifact in record['artifacts']:
            path = base / artifact['path']
            observed = digest(path)
            rows.append(dict(path=artifact['path'], expected=artifact['sha256'], observed=observed, match=observed == artifact['sha256']))
        receipts[receipt] = rows
    if not all(r['match'] for rows in receipts.values() for r in rows):
        raise IntegrityError('Phase 2 plan/review file changed since its closure receipt')
    return receipts


def build(root, dry=False):
    root = Path(root).resolve()
    manifest = root / MANIFEST
    sidecar = root / DIGEST
    if not dry and (manifest.exists() or sidecar.exists()):
        raise FileExistsError('preserve existing cycle manifest')
    identity = dict(phase1=phase1_identity(root), phase2_plan=phase2_plan_identity())
    paths = inventory(root)
    rows = []
    signatures = {}
    for name, path in paths.items():
        before = path.stat()
        h = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IntegrityError('file changed while sealing: ' + name)
        signatures[name] = (after.st_size, after.st_mtime_ns)
        rows.append(dict(path=name, bytes=after.st_size, sha256=h))
    if list(paths) != list(inventory(root)):
        raise IntegrityError('inventory changed while sealing')
    for name, path in paths.items():
        now = path.stat()
        if (now.st_size, now.st_mtime_ns) != signatures[name]:
            raise IntegrityError('file changed while sealing: ' + name)
    payload = dict(status='SEALED_LOCAL_CYCLE_DELIVERY', utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                   cycle='review-diagnostics R1-R6 (2026-09-17)', identity=identity,
                   file_count=len(rows), total_bytes=sum(r['bytes'] for r in rows), files=rows,
                   excluded_files=sorted(EXCLUDED_FILES), excluded_directories=sorted(EXCLUDED_DIRECTORIES),
                   limit='Internal integrity of current bytes, not external attestation or proof of any scientific hypothesis.')
    if payload['total_bytes'] >= 10 * 2**30:
        raise IntegrityError('delivery exceeds predeclared 10GiB bound')
    summary = dict(files=len(rows), bytes=payload['total_bytes'], identity=identity)
    if dry:
        summary['dry_run'] = True
        return summary
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf8') as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
    with sidecar.open('x', encoding='ascii') as out:
        out.write(digest(manifest) + '\n')
    summary['verification'] = verify(root)
    return summary


def verify(root):
    root = Path(root).resolve()
    manifest = root / MANIFEST
    sidecar = root / DIGEST
    if digest(manifest) != sidecar.read_text(encoding='ascii').strip():
        raise IntegrityError('manifest changed')
    payload = json.loads(manifest.read_text(encoding='utf8'))
    rows = payload['files']
    paths = inventory(root)
    names = [r['path'] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(paths):
        missing = sorted(set(names) - set(paths))[:10]
        extra = sorted(set(paths) - set(names))[:10]
        raise IntegrityError(f'evidence inventory differs; missing={missing} extra={extra}')
    if len(rows) != payload['file_count'] or sum(r['bytes'] for r in rows) != payload['total_bytes']:
        raise IntegrityError('manifest counts differ')
    for row in rows:
        path = paths[row['path']]
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise IntegrityError('evidence changed: ' + row['path'])
    result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), files_verified=len(rows),
                  bytes_verified=payload['total_bytes'], manifest_sha256=digest(manifest),
                  phase1=phase1_identity(root), phase2_plan_receipts_match=True if phase2_plan_identity() else False)
    (root / VERIFICATION).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'dry'])
    args = parser.parse_args()
    if args.action == 'build':
        result = build(ROOT)
    elif args.action == 'dry':
        result = build(ROOT, dry=True)
    else:
        result = verify(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
