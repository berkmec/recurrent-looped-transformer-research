"""Pinned small-tier dataset downloads (after models). LongMemEval-M (2.6 GB) is skipped by pattern."""
import hashlib, json, os, time, urllib.request
os.environ['HF_HOME'] = 'E:/rlt-research/hf-cache'
from huggingface_hub import snapshot_download
OUT = 'E:/rlt-research/datasets'
PINS = 'E:/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/dataset-pins.json'
plan = [('dgslibisey/MuSiQue', None), ('xiaowu0162/longmemeval-cleaned', ['*longmemeval_s_cleaned*', '*oracle*', '*.md', '*.json']), ('bigcode/bigcodebench', None)]
pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(2**22), b''): h.update(c)
    return h.hexdigest()
for repo, patterns in plan:
    if pins.get(repo, {}).get('status') == 'VERIFIED': print('skip', repo); continue
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}?blobs=true'))
    rev = info['sha']; local = f"{OUT}/{repo.replace('/', '__')}"
    t0 = time.time(); print(f'{time.strftime("%H:%M:%S")} download {repo}@{rev[:12]} patterns={patterns}', flush=True)
    kw = dict(repo_id=repo, repo_type='dataset', revision=rev, local_dir=local, max_workers=4)
    if patterns: kw['allow_patterns'] = patterns
    if repo.startswith('xiaowu0162'): kw['ignore_patterns'] = ['*longmemeval_m*']
    snapshot_download(**kw)
    files, bad = {}, []
    for s in info['siblings']:
        p = os.path.join(local, s['rfilename'])
        if not os.path.exists(p): continue
        expected = (s.get('lfs') or {}).get('sha256'); observed = sha(p)
        files[s['rfilename']] = dict(bytes=os.path.getsize(p), sha256=observed, lfs_sha256=expected)
        if expected and expected != observed: bad.append(('sha256', s['rfilename']))
    pins[repo] = dict(revision=rev, local_dir=local, license=[t for t in info.get('tags', []) if t.startswith('license:')],
                      files=files, skipped=[s['rfilename'] for s in info['siblings'] if s['rfilename'] not in files], seconds=time.time() - t0,
                      status='VERIFIED' if not bad and files else 'FAILED', problems=bad, downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(pins, open(PINS, 'w'), indent=1)
    print(f'{time.strftime("%H:%M:%S")} {repo} {pins[repo]["status"]} files={len(files)} bytes={sum(f["bytes"] for f in files.values())/2**20:.0f}MiB skipped={pins[repo]["skipped"]}', flush=True)
print('ALL DATASETS DONE')
