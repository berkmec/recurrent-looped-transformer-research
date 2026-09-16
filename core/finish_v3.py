"""Version3: build reports and EXIT; binding is manual after all writers stop."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
def ready():
 try:return all(json.loads((r/'results'/name).read_text()).get('status')=='COMPLETE' for name in ['metrics.json','final-only-metrics.json']) and len(json.loads((r/'results'/'diagnostics.json').read_text())['rows'])==9 and len((r/'evaluation-seals.jsonl').read_text().splitlines())==21
 except (FileNotFoundError,json.JSONDecodeError):return False
while not ready():time.sleep(10)
jobs=[(sys.executable,'core.benchmark'),('C:/Users/pc/Documents/Codex/2026-09-16/rlt-derin-arastirma/.venv/Scripts/python.exe','core.plots'),(sys.executable,'core.summarize')]
for executable,module in jobs:
 with (r/(module.split('.')[-1]+'-finish.log')).open('x',encoding='utf8') as out:subprocess.run([executable,'-u','-m',module],cwd=r.parent,stdout=out,stderr=subprocess.STDOUT,check=True)
print('CORE_REPORT_READY_UNBOUND_AWAIT_QUIESCENCE',flush=True)
