"""Append-only first-observation seals; never modifies the frozen evaluator."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=ROOT/'evaluation-observer-protocol.json'
if protocol.exists():raise RuntimeError('Observer already initialized; refuse overlapping/restarted sealer')
with protocol.open('x',encoding='utf8') as f:json.dump(dict(expected_primary_conditions=21,poll_seconds=2,purpose='seal each completed evaluation JSON and referenced arrays on first observation, before post-run binding',observer_source_sha256=sha(Path(__file__)),primary_protocol_sha256=sha(ROOT/'protocol.json')),f,indent=2)
(ROOT/'evaluation-observer-protocol.sha256').write_text(sha(protocol));seen=set();output=ROOT/'evaluation-seals.jsonl'
while len(seen)<21:
 for path in sorted((ROOT/'results').glob('*-eval.json')):
  if path.name in seen:continue
  try:r=json.loads(path.read_text())
  except json.JSONDecodeError:continue
  files=[path]+[Path(x['predictions_file']) for x in r['long_horizon']]+[Path(x['file']) for x in r['neutral_slots']]
  if any(not p.resolve().is_relative_to(ROOT.resolve()) for p in files):raise RuntimeError('seal path escapes core root')
  seal=dict(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),condition=r['condition'],evaluation_file=path.relative_to(ROOT).as_posix(),protocol_hash=sha(protocol),files={p.relative_to(ROOT).as_posix():sha(p) for p in files})
  with output.open('a',encoding='utf8') as f:f.write(json.dumps(seal)+'\n')
  seen.add(path.name);print('SEALED',path.name,len(seen),flush=True)
 time.sleep(2)
print('ALL21_PRIMARY_EVALUATIONS_SEALED',flush=True)
