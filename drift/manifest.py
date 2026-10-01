"""Seal/verify the drift mini-experiment root with its own manifest path.

Reuses the inventory/digest/identity helpers of review_cycle.cycle_manifest read-only; the
review-diagnostics manifest tracked in this worktree is left untouched and re-verified here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from review_cycle import cycle_manifest as cm

ROOT = cm.ROOT
MANIFEST = 'audit/drift-mini-manifest.json'
DIGEST = 'audit/drift-mini-manifest.sha256'
VERIFICATION = 'audit/drift-mini-manifest-verification.json'
EXCLUDED = {MANIFEST, DIGEST, VERIFICATION}


def inventory(root):
    return {k: v for k, v in cm.inventory(root).items() if k not in EXCLUDED}


def prior_cycle_identity(root):
    manifest = root / 'audit/review-diagnostics-manifest.json'
    observed = cm.digest(manifest)
    sidecar = (root / 'audit/review-diagnostics-manifest.sha256').read_text(encoding='ascii').strip()
    if observed != sidecar:
        raise cm.IntegrityError('review-diagnostics manifest identity changed in this worktree')
    return dict(review_diagnostics_manifest_sha256=observed)


def build(root, dry=False):
    root = Path(root).resolve()
    manifest, sidecar = root / MANIFEST, root / DIGEST
    if not dry and (manifest.exists() or sidecar.exists()):
        raise FileExistsError('preserve existing drift manifest')
    identity = dict(phase1=cm.phase1_identity(root), phase2_plan=cm.phase2_plan_identity(), prior_cycle=prior_cycle_identity(root))
    paths = inventory(root)
    rows, signatures = [], {}
    for name, path in paths.items():
        before = path.stat(); h = cm.digest(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise cm.IntegrityError('file changed while sealing: ' + name)
        signatures[name] = (after.st_size, after.st_mtime_ns)
        rows.append(dict(path=name, bytes=after.st_size, sha256=h))
    if list(paths) != list(inventory(root)):
        raise cm.IntegrityError('inventory changed while sealing')
    for name, path in paths.items():
        now = path.stat()
        if (now.st_size, now.st_mtime_ns) != signatures[name]:
            raise cm.IntegrityError('file changed while sealing: ' + name)
    payload = dict(status='SEALED_LOCAL_CYCLE_DELIVERY', utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                   cycle='drift mini-experiment (2026-09-18)', identity=identity, file_count=len(rows),
                   total_bytes=sum(r['bytes'] for r in rows), files=rows, excluded_files=sorted(EXCLUDED),
                   excluded_directories=sorted(cm.EXCLUDED_DIRECTORIES),
                   limit='Internal integrity of current bytes, not external attestation or proof of any scientific hypothesis.')
    if payload['total_bytes'] >= 10 * 2**30:
        raise cm.IntegrityError('delivery exceeds predeclared 10GiB bound')
    summary = dict(files=len(rows), bytes=payload['total_bytes'], identity=identity)
    if dry:
        summary['dry_run'] = True
        return summary
    with manifest.open('x', encoding='utf8') as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
    with sidecar.open('x', encoding='ascii') as out:
        out.write(cm.digest(manifest) + '\n')
    summary['verification'] = verify(root)
    return summary


def verify(root):
    root = Path(root).resolve()
    manifest, sidecar = root / MANIFEST, root / DIGEST
    if cm.digest(manifest) != sidecar.read_text(encoding='ascii').strip():
        raise cm.IntegrityError('manifest changed')
    payload = json.loads(manifest.read_text(encoding='utf8'))
    rows = payload['files']; paths = inventory(root)
    names = [r['path'] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(paths):
        raise cm.IntegrityError(f'evidence inventory differs; missing={sorted(set(names)-set(paths))[:10]} extra={sorted(set(paths)-set(names))[:10]}')
    if len(rows) != payload['file_count'] or sum(r['bytes'] for r in rows) != payload['total_bytes']:
        raise cm.IntegrityError('manifest counts differ')
    for row in rows:
        path = paths[row['path']]
        if path.stat().st_size != row['bytes'] or cm.digest(path) != row['sha256']:
            raise cm.IntegrityError('evidence changed: ' + row['path'])
    result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), files_verified=len(rows), bytes_verified=payload['total_bytes'],
                  manifest_sha256=cm.digest(manifest), phase1=cm.phase1_identity(root), prior_cycle=prior_cycle_identity(root),
                  phase2_plan_receipts_match=bool(cm.phase2_plan_identity()))
    (root / VERIFICATION).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'dry'])
    args = parser.parse_args()
    result = build(ROOT) if args.action == 'build' else build(ROOT, dry=True) if args.action == 'dry' else verify(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
