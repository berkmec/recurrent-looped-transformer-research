"""Pinned model downloads for Phase 2 G0 (revision = HF commit sha at query time; LFS sha256 verified after download)."""
import hashlib, json, os, sys, time, urllib.request
os.environ['HF_HOME'] = 'E:/rlt-research/hf-cache'
from huggingface_hub import snapshot_download
OUT = 'E:/rlt-research/models'
PINS = 'E:/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/model-pins.json'
repos = ['Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Reranker-0.6B', 'Qwen/Qwen3-4B']
pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(2**22), b''): h.update(c)
    return h.hexdigest()
for repo in repos:
    if pins.get(repo, {}).get('status') == 'VERIFIED': print('skip', repo); continue
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true'))
    rev = info['sha']; local = f"{OUT}/{repo.split('/')[1]}"
    t0 = time.time(); print(f'{time.strftime("%H:%M:%S")} download {repo}@{rev[:12]} -> {local}', flush=True)
    snapshot_download(repo_id=repo, revision=rev, local_dir=local, max_workers=4)
    files = {}
    bad = []
    for s in info['siblings']:
        p = os.path.join(local, s['rfilename'])
        if not os.path.exists(p): bad.append(('missing', s['rfilename'])); continue
        expected = (s.get('lfs') or {}).get('sha256')
        observed = sha(p)
        files[s['rfilename']] = dict(bytes=os.path.getsize(p), sha256=observed, lfs_sha256=expected)
        if expected and expected != observed: bad.append(('sha256', s['rfilename']))
    pins[repo] = dict(revision=rev, local_dir=local, license=[t for t in info.get('tags', []) if t.startswith('license:')],
                      files=files, seconds=time.time() - t0, status='VERIFIED' if not bad else 'FAILED', problems=bad,
                      downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(pins, open(PINS, 'w'), indent=1)
    print(f'{time.strftime("%H:%M:%S")} {repo} {pins[repo]["status"]} files={len(files)} bytes={sum(f["bytes"] for f in files.values())/2**30:.2f}GiB in {time.time()-t0:.0f}s problems={bad}', flush=True)
print('ALL MODELS DONE')
