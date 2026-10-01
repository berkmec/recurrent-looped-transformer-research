"""Schedule-only revision: fixed-checkpoint diagnostics during primary evaluation."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).parent
receipt=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='All training completed; released CPU slot. Retired waiting followup_v2 and finish_v3; original primary untouched.',change='Only scheduling; original diagnostics and benchmark modules/protocols/data unchanged. Concurrent timing remains disclosed.',hashes={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['diagnostics.py','benchmark.py','model.py','run.py','protocol.json','benchmark-protocol.json']})
with (r/'advanced-diagnostics-schedule.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
for module in ['core.diagnostics','core.benchmark']:
 with (r/(module.split('.')[-1]+'-advanced.log')).open('x',encoding='utf8') as out:subprocess.run([sys.executable,'-u','-m',module],cwd=r.parent,stdout=out,stderr=subprocess.STDOUT,check=True)
print('ADVANCED_DIAGNOSTICS_COMPLETE',flush=True)
