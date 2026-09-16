"""Recheck prior frozen evidence against the original read-only preflight."""
from pathlib import Path
from datetime import datetime,timezone
import json
from audit.delivery_integrity import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    original=json.loads((ROOT/'audit/upstream_preflight.json').read_text(encoding='utf8'))
    groups=[]
    for group in original['upstream']:
        base=Path(group['root']).resolve()
        assert digest(group['manifest'])==group['manifest_sha256'],'prior manifest changed'
        for row in group['files']:
            path=(base/row['path']).resolve()
            assert path.is_relative_to(base),'prior manifest path escaped root'
            assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],f'prior evidence changed: {path}'
        groups.append({'root':str(base),'files_verified':len(group['files']),
                       'manifest_sha256':group['manifest_sha256']})
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
            'total_verified':sum(g['files_verified'] for g in groups),'groups':groups,
            'preflight_sha256':digest(ROOT/'audit/upstream_preflight.json'),
            'scope':'byte preservation of prior evidence, not renewed scientific replication'}
    assert result['total_verified']==original['total_verified']==424
    with (ROOT/'audit/upstream_postflight.json').open('x',encoding='utf8') as out:
        json.dump(result,out,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
