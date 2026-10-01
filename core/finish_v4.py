"""Wait for independently scheduled diagnostics, then reports; never bind live writers."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
def ready():
 try:return all(json.loads((r/'results'/name).read_text()).get('status')=='COMPLETE' for name in ['metrics.json','final-only-metrics.json']) and len(json.loads((r/'results'/'diagnostics.json').read_text())['rows'])==9 and len(json.loads((r/'results'/'benchmark.json').read_text())['rows'])==18 and len((r/'evaluation-seals.jsonl').read_text().splitlines())==21
 except (FileNotFoundError,json.JSONDecodeError):return False
while not ready():time.sleep(10)
for executable,module in [('__WORKSPACE__/rlt-derin-arastirma/.venv/Scripts/python.exe','core.plots'),(sys.executable,'core.summarize')]:
 with (r/(module.split('.')[-1]+'-v4-finish.log')).open('x',encoding='utf8') as out:subprocess.run([executable,'-u','-m',module],cwd=r.parent,stdout=out,stderr=subprocess.STDOUT,check=True)
print('CORE_REPORT_READY_UNBOUND_AWAIT_QUIESCENCE',flush=True)
