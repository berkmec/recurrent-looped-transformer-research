"""Read-only verification of frozen upstream deliveries and local resources."""
from __future__ import annotations
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('__WORKSPACE__')


def main():
    records = []
    for folder, name in (
        ('rlt-derin-arastirma', 'artifact_manifest.json'),
        ('rlt-ogrenme-deneyi-01', 'delivery_manifest.json'),
        ('rlt-revizyon-deneyi-02', 'delivery_manifest.json'),
    ):
        base = (UPSTREAM / folder).resolve()
        manifest = base / name
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        verified = []
        for item in payload['files']:
            path = (base / item['path']).resolve()
            if not path.is_relative_to(base):
                raise ValueError('manifest escaped upstream root')
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != item['sha256']:
                raise ValueError(f'upstream changed: {path}')
            if 'bytes' in item and len(data) != item['bytes']:
                raise ValueError(f'upstream length changed: {path}')
            verified.append({'path': item['path'], 'sha256': actual, 'bytes': len(data)})
        records.append({'root': str(base), 'manifest': str(manifest),
                        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        'files_verified': len(verified), 'files': verified})
    out = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
           'python': platform.python_version(), 'platform': platform.platform(),
           'upstream': records, 'total_verified': sum(x['files_verified'] for x in records),
           'note': 'Read-only SHA256 integrity verification, not scientific replication.'}
    destination = ROOT / 'audit' / 'upstream_preflight.json'
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps({'path': str(destination), 'verified': out['total_verified']}))


if __name__ == '__main__':
    main()
