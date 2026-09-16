"""Finish bounded evidence work after the frozen companion completes."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
while True:
 try:
  if json.loads((r/'results'/'final-only-metrics.json').read_text())['status']=='COMPLETE':break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(10)
jobs=[(sys.executable,'core.benchmark'),('C:/Users/pc/Documents/Codex/2026-09-16/rlt-derin-arastirma/.venv/Scripts/python.exe','core.plots'),(sys.executable,'core.summarize'),(sys.executable,'core.safe_entrypoint')]
for executable,module in jobs:
 with (r/(module.split('.')[-1]+'-finish.log')).open('x',encoding='utf8') as out:
  command=[executable,'-u','-m',module]+(['--bind-completed'] if module=='core.safe_entrypoint' else [])
  subprocess.run(command,cwd=r.parent,stdout=out,stderr=subprocess.STDOUT,check=True)
manifest={str(p.relative_to(r)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(r/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CORE_FINISH_READY_FOR_INDEPENDENT_AUDIT',flush=True)
