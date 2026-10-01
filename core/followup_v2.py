"""Version2: only diagnostics wait for primary; companion runs independently."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
while True:
 try:
  if json.loads((r/'results'/'metrics.json').read_text()).get('status')=='COMPLETE':break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(10)
with (r/'diagnostics-v2-console.log').open('x',encoding='utf8') as out:subprocess.run([sys.executable,'-u','-m','core.diagnostics'],cwd=r.parent,stdout=out,stderr=subprocess.STDOUT,check=True)
