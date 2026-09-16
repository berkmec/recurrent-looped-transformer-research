"""Read-only fail-closed entry guard. The frozen legacy runner is not restart safe.

Use this guard before considering reuse. Partial directories are never resumed.
Post-run binding authenticates internal consistency, not a trusted signature.
"""
import hashlib,json
from pathlib import Path
class Refusal(RuntimeError):pass
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_file_receipts(root,receipts):
 root=Path(root).resolve()
 for rel,expected in receipts.items():
  p=(root/rel).resolve()
  if not p.is_relative_to(root) or not p.is_file() or digest(p)!=expected:raise Refusal('RECEIPT_MISMATCH: '+str(rel))
def verify_manifest_only(root,b):
 required={'results/metrics.json','results/final-only-metrics.json','protocol.json','protocol.sha256','environment.json','model.py','run.py'}
 if not required.issubset(b.get('files',{})):raise Refusal('INCOMPLETE_BINDING')
 verify_file_receipts(root,b['files'])
def validate_completeness(root):
 import numpy as np
 def load(rel):return json.loads((root/rel).read_text())
 def key(c):return (c['task'],c['width'],c['mode'],c['seed'])
 expected={(task,32,mode,seed) for task in ['parity','five'] for seed in [17,43,91] for mode in (['rlt','nofeedback','tbptt'] if task=='parity' else ['rlt','nofeedback'])}|{('parity',48,mode,seed) for seed in [17,43,91] for mode in ['rlt','nofeedback']}
 primary=load('results/metrics.json');companion=load('results/final-only-metrics.json')
 if primary.get('status')!='COMPLETE' or companion.get('status')!='COMPLETE':raise Refusal('PARTIAL_RESULTS')
 for field in ['training','evaluation']:
  rows=primary[field]
  if len(rows)!=21 or {key(r['condition']) for r in rows}!=expected:raise Refusal('INCOMPLETE_PRIMARY_'+field)
 def verify_training(r):
  if r['steps']!=2000 or [h['step'] for h in r['history']]!=list(range(100,2001,100)) or r['selected_step'] not in range(100,2001,100):raise Refusal('INCOMPLETE_TRAINING_BUDGET')
 def verify_array(path,length,stats):
  p=Path(path).resolve()
  if not p.is_relative_to(root):raise Refusal('ARRAY_PATH_ESCAPE')
  with np.load(p) as a:
   if any(a[n].shape!=(512,length) for n in ['inputs','predictions','targets']):raise Refusal('ARRAY_SHAPE')
   if float((a['predictions'][:,-1]==a['targets'][:,-1]).mean())!=stats['final_accuracy'] or float((a['predictions']==a['targets']).mean())!=stats['prefix_accuracy']:raise Refusal('ARRAY_METRIC_MISMATCH')
 for r in primary['training']:
  verify_training(r);c=r['condition'];name=f"{c['task']}-w{c['width']}-{c['mode']}-s{c['seed']}"
  if load('results/'+name+'-train.json')!=r:raise Refusal('TRAINING_RECEIPT_DISAGREEMENT')
  verify_file_receipts(root,{f'checkpoints/{name}-{suffix}.pt':h for suffix,h in r['checkpoint_hashes'].items()})
  if set(r['checkpoint_hashes'])!={'initial','selected','last'}:raise Refusal('CHECKPOINT_RECEIPT_SET')
 seals=[json.loads(line) for line in (root/'evaluation-seals.jsonl').read_text().splitlines()]
 observer=load('evaluation-observer-protocol.json')
 if observer['observer_source_sha256']!=digest(root/'observe_evaluation.py') or observer['primary_protocol_sha256']!=digest(root/'protocol.json') or (root/'evaluation-observer-protocol.sha256').read_text().strip()!=digest(root/'evaluation-observer-protocol.json'):raise Refusal('OBSERVER_SOURCE_OR_PROTOCOL_DRIFT')
 if len(seals)!=21 or {key(s['condition']) for s in seals}!=expected:raise Refusal('INCOMPLETE_FIRST_OBSERVATION_SEALS')
 for seal in seals:
  if seal['protocol_hash']!=digest(root/'evaluation-observer-protocol.json'):raise Refusal('OBSERVER_PROTOCOL_MISMATCH')
  verify_file_receipts(root,seal['files'])
 for ev in primary['evaluation']:
  c=ev['condition'];name=f"{c['task']}-w{c['width']}-{c['mode']}-s{c['seed']}"
  if load('results/'+name+'-eval.json')!=ev:raise Refusal('EVALUATION_RECEIPT_DISAGREEMENT')
  if len(ev['long_horizon'])!=6 or {r['length'] for r in ev['long_horizon']}!={32,64,128,256,512,1024}:raise Refusal('INCOMPLETE_LENGTH_SWEEP')
  for row in ev['long_horizon']:
   verify_file_receipts(root,{row['predictions_file']:row['sha256']});verify_array(row['predictions_file'],row['length'],row)
  if len(ev['neutral_slots'])!=4 or {r['slots'] for r in ev['neutral_slots']}!={0,8,32,128}:raise Refusal('INCOMPLETE_NEUTRAL_SWEEP')
  for row in ev['neutral_slots']:verify_array(row['file'],32+row['slots'],row)
 expected_final={('parity',32,mode,seed) for seed in [17,43,91] for mode in ['rlt','tbptt']}
 if len(companion['records'])!=6 or {key(r['condition']) for r in companion['records']}!=expected_final:raise Refusal('INCOMPLETE_COMPANION')
 for r in companion['records']:
  verify_training(r);c=r['condition'];name=f"finalonly-parity-w32-{c['mode']}-s{c['seed']}"
  if load('results/'+name+'.json')!=r:raise Refusal('COMPANION_RECEIPT_DISAGREEMENT')
  if set(r['hashes'])!={'initial','selected','last'}:raise Refusal('COMPANION_CHECKPOINT_SET')
  verify_file_receipts(root,{f'checkpoints/{name}-{suffix}.pt':h for suffix,h in r['hashes'].items()})
  if len(r['evaluation'])!=2 or {e['length'] for e in r['evaluation']}!={32,128}:raise Refusal('INCOMPLETE_COMPANION_LENGTHS')
  for row in r['evaluation']:
   verify_file_receipts(root,{row['file']:row['sha256']});verify_array(row['file'],row['length'],row)
 diagnostics=load('results/diagnostics.json')['rows']
 if len(diagnostics)!=9 or {(r['mode'],r['seed']) for r in diagnostics}!={(m,s) for m in ['rlt','nofeedback','tbptt'] for s in [17,43,91]}:raise Refusal('INCOMPLETE_DIAGNOSTICS')
 for row in diagnostics:
  if len(row['noise'])!=4 or {n['std'] for n in row['noise']}!={0,.001,.01,.1}:raise Refusal('INCOMPLETE_NOISE_SWEEP')
  for n in row['noise']:
   verify_file_receipts(root,{n['file']:n['sha256']})
   with np.load(n['file']) as a:
    if a['inputs'].shape!=(512,32) or a['predictions'].shape!=(512,) or a['targets'].shape!=(512,) or float((a['predictions']==a['targets']).mean())!=n['accuracy']:raise Refusal('NOISE_ARRAY_MISMATCH')
 benchmark=load('results/benchmark.json')
 if benchmark['protocol_hash']!=digest(root/'benchmark-protocol.json'):raise Refusal('BENCHMARK_PROTOCOL_MISMATCH')
 expected_bench={(loss,mode,rep) for loss in ['prefix','final_only'] for mode in ['full','tbptt8','checkpoint'] for rep in range(3)}
 if len(benchmark['rows'])!=18 or {(r['loss'],r['mode'],r['repeat']) for r in benchmark['rows']}!=expected_bench:raise Refusal('INCOMPLETE_BENCHMARK')
 from collections import Counter
 events=[json.loads(line) for line in (root/'execution.jsonl').read_text().splitlines()];counts=Counter(e['event'] for e in events)
 expected_events={'protocol_frozen':1,'train_start':21,'validation':420,'all_training_complete':1,'heldout':126,'primary_complete':1,'final_only_protocol_frozen':1,'final_only_train_start':6,'final_only_validation':120,'final_only_complete':1}
 if any(counts[name]!=count for name,count in expected_events.items()):raise Refusal('INCOMPLETE_OR_DUPLICATE_EXECUTION_RECEIPTS')
def check_entry(root):
 root=Path(root).resolve();binding=root/'evidence-binding.json'
 evidence=any((root/d).exists() and any((root/d).iterdir()) for d in ['results','checkpoints'])
 if not binding.exists():
  if evidence:raise Refusal('PARTIAL_OR_UNBOUND: evidence exists; legacy rerun could overwrite it. Preserve this directory; do not resume.')
  return 'NEW'
 b=json.loads(binding.read_text());verify_manifest_only(root,b)
 try:validate_completeness(root)
 except (KeyError,ValueError,OSError) as e:raise Refusal('INVALID_OR_PARTIAL_EVIDENCE: '+str(e)) from e
 return 'VERIFIED_COMPLETE'
def bind_completed(root):
 root=Path(root).resolve();target=root/'evidence-binding.json'
 if target.exists():return check_entry(root)
 try:validate_completeness(root)
 except (KeyError,ValueError,OSError) as e:raise Refusal('INVALID_OR_PARTIAL_EVIDENCE: '+str(e)) from e
 for metrics,protocol in [('metrics.json','protocol.json'),('final-only-metrics.json','final-only-protocol.json'),('diagnostics.json','diagnostics-protocol.json')]:
  m=json.loads((root/'results'/metrics).read_text());expected=digest(root/protocol)
  if m['protocol_hash']!=expected:raise Refusal('PROTOCOL_MISMATCH')
  if metrics!='diagnostics.json' and m.get('status')!='COMPLETE':raise Refusal('PARTIAL_RESULTS')
  if (root/protocol.replace('.json','.sha256')).read_text().strip()!=expected:raise Refusal('PROTOCOL_DIGEST_MISMATCH')
 env=json.loads((root/'environment.json').read_text())
 for envname in ['environment.json','final-only-environment.json','diagnostics-environment.json']:
  launch=json.loads((root/envname).read_text())
  for name,expected in launch['source_hashes'].items():
   if digest(root/name)!=expected or digest(root/'source-snapshots'/expected/name)!=expected:raise Refusal('LAUNCH_SOURCE_MISMATCH')
 for record in (root/'results').glob('*-train.json'):
  r=json.loads(record.read_text());prefix=record.name.removesuffix('-train.json')
  for suffix,h in r['checkpoint_hashes'].items():
   if digest(root/'checkpoints'/(prefix+'-'+suffix+'.pt'))!=h:raise Refusal('CHECKPOINT_MISMATCH')
 for record in (root/'results').glob('finalonly-*.json'):
  r=json.loads(record.read_text());prefix=record.stem
  for suffix,h in r['hashes'].items():
   if digest(root/'checkpoints'/(prefix+'-'+suffix+'.pt'))!=h:raise Refusal('FINAL_CHECKPOINT_MISMATCH')
 paths=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'plot-cache' not in p.parts and p.suffix in ['.py','.json','.jsonl','.sha256','.npz','.pt'] and p.name not in ['manifest.json','evidence-binding.json']]
 b=dict(binding_type='post-run consistency binding of an uninterrupted legacy-source run; not an original signed execution receipt',actual_training_source=env['source_hashes'],files={p.relative_to(root).as_posix():digest(p) for p in paths})
 with target.open('x',encoding='utf8') as f:json.dump(b,f,indent=2)
 return check_entry(root)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--bind-completed',action='store_true');args=parser.parse_args();root=Path(__file__).parent
 try:print(bind_completed(root) if args.bind_completed else check_entry(root))
 except Refusal as e:raise SystemExit(str(e))
