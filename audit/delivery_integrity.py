"""Preserve/verify the complete local delivery, including Git-ignored raw data.

This detects accidental byte changes and missing/extra files. It is not a
signature against an actor able to replace both evidence and its manifest.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='audit/final-manifest.json'
DIGEST='audit/final-manifest.sha256'
VERIFICATION='audit/final-manifest-verification.json'
EXCLUDED_FILES={MANIFEST,DIGEST,VERIFICATION}
EXCLUDED_DIRECTORIES={'.git','__pycache__','.pytest_cache','matplotlib-cache','plot-cache','.superpowers'}


class IntegrityError(RuntimeError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def contained(root,relative):
    path=root/relative
    if not path.resolve().is_relative_to(root):
        raise IntegrityError('path escaped delivery root: '+relative)
    return path


def inventory(root):
    root=Path(root).resolve()
    out={}
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        # Windows junctions are reparse points, even where Path.is_symlink is false.
        if path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&0x400:
            raise IntegrityError('linked evidence path refused: '+relative.as_posix())
        if any(p in EXCLUDED_DIRECTORIES for p in relative.parts):
            continue
        if not path.is_file() or relative.as_posix() in EXCLUDED_FILES or path.suffix=='.pyc':
            continue
        if not path.resolve().is_relative_to(root):
            raise IntegrityError('evidence path escaped delivery root')
        out[relative.as_posix()]=path
    return dict(sorted(out.items()))


def build_manifest(root):
    root=Path(root).resolve()
    manifest=contained(root,MANIFEST);sidecar=contained(root,DIGEST)
    if manifest.exists() or sidecar.exists():
        raise FileExistsError('preserve existing final manifest')
    closure=json.loads(contained(root,'audit/campaign-closure.json').read_text(encoding='utf8'))
    if closure.get('status')!='COMPLETE_BOUNDED_RESEARCH':
        raise IntegrityError('campaign not complete')
    rows=[];signatures={}
    paths=inventory(root)
    for name,path in paths.items():
        before=path.stat()
        h=digest(path)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise IntegrityError('file changed while sealing: '+name)
        signatures[name]=(after.st_size,after.st_mtime_ns)
        rows.append({'path':name,'bytes':after.st_size,'sha256':h})
    if list(paths)!=list(inventory(root)):
        raise IntegrityError('inventory changed while sealing')
    for name,path in paths.items():
        now=path.stat()
        if (now.st_size,now.st_mtime_ns)!=signatures[name]:
            raise IntegrityError('file changed while sealing: '+name)
    payload={'status':'SEALED_LOCAL_DELIVERY','utc':datetime.now(timezone.utc).isoformat(),
             'root':str(root),'file_count':len(rows),'total_bytes':sum(r['bytes'] for r in rows),
             'files':rows,'excluded_files':sorted(EXCLUDED_FILES),
             'excluded_directories':sorted(EXCLUDED_DIRECTORIES),
             'limit':'Internal integrity, not external attestation or proof of scientific success.'}
    if payload['total_bytes']>=10*2**30:
        raise IntegrityError('delivery exceeds predeclared 10GiB bound')
    with manifest.open('x',encoding='utf8') as out:
        json.dump(payload,out,indent=2,ensure_ascii=False)
    with sidecar.open('x',encoding='ascii') as out:
        out.write(digest(manifest)+'\n')
    verify_manifest(root)
    return {'files_sealed':len(rows),'bytes':payload['total_bytes']}


def verify_manifest(root):
    root=Path(root).resolve()
    manifest=contained(root,MANIFEST);sidecar=contained(root,DIGEST)
    if digest(manifest)!=sidecar.read_text(encoding='ascii').strip():
        raise IntegrityError('manifest changed')
    payload=json.loads(manifest.read_text(encoding='utf8'))
    rows=payload['files']
    paths=inventory(root)
    names=[r['path'] for r in rows]
    if len(names)!=len(set(names)) or set(names)!=set(paths):
        raise IntegrityError('evidence inventory differs')
    if len(rows)!=payload['file_count'] or sum(r['bytes'] for r in rows)!=payload['total_bytes']:
        raise IntegrityError('manifest counts differ')
    for row in rows:
        path=paths[row['path']]
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise IntegrityError('evidence changed: '+row['path'])
    return {'status':'PASS','files_verified':len(rows),'bytes_verified':payload['total_bytes'],
            'manifest_sha256':digest(manifest)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','verify'])
    args=parser.parse_args()
    print(json.dumps(build_manifest(ROOT) if args.action=='build' else verify_manifest(ROOT)))
