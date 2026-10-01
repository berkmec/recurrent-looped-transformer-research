"""Schedule independent diagnostics and delayed-error companion after primary finishes."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
while True:
 p=r/'results'/'metrics.json'
 if p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE':break
 time.sleep(10)
for module in ['core.diagnostics','core.run_final_only']:
 with (r/(module.split('.')[-1]+'-console.log')).open('w',encoding='utf8') as out:
  done=subprocess.run([sys.executable,'-u','-m',module],cwd=r.parent,stdout=out,stderr=subprocess.STDOUT)
 if done.returncode:raise SystemExit(done.returncode)
