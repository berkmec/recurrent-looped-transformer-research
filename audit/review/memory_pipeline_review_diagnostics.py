import sys,json,hashlib,sqlite3
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()));torch.set_num_threads(1)
from learning.api import load_model,events_from_records
from integration.pipeline import run_from_records,run_text_episode
from memory.api import Archive,TitleController,verify_candidate
from memory.headings import HeadingRegistry
from learning.safe_run import assert_fresh_output
from core.safe_entrypoint import check_entry,Refusal
out=Path('audit/review');results={}
model=load_model('learning/checkpoints/feedback-17-step600.pt')
tape=[dict(kind='edge',child='x1',parent='x0',value=.2),dict(kind='root',child='x1',parent=None,value=.7),dict(kind='query',child='x0',parent=None,value=None),dict(kind='revision',child='x1',parent=None,value=.9),dict(kind='query',child='x0',parent=None,value=None),dict(kind='query',child='x1',parent=None,value=None)]
a=run_from_records(model,tape,max_steps=3,prior_rounds=[1,2],adaptive=False)
x,mapping=events_from_records(tape);r=model.new_runtime(x)
with torch.no_grad():
 for n in [1,2,3]:
  assert r.consume_to_query()
  for _ in range(n):c=r.refine_one()
  r.commit_feedback()
 direct=float(c['prediction'][0])
results['replay']={'absolute_difference':abs(a['prediction']-direct),'actual_refinements':a['refinements_executed'],'tokens':a['consumed_tokens']}
assert a['prediction']==direct and a['refinements_executed']==6
# Unrelated request must not affect repeated request.
run_from_records(model,[dict(kind='root',child='x9',parent=None,value=-.8),dict(kind='query',child='x9',parent=None,value=None)],max_steps=2)
b=run_from_records(model,tape,max_steps=3,prior_rounds=[1,2],adaptive=False)
results['reset_isolation_exact']=a['prediction']==b['prediction'];assert results['reset_isolation_exact']
db=out/'inverse-review.sqlite'
assert not db.exists();arc=Archive(db);controller=TitleController(2,arc)
for event in tape:controller.append('fresh-review','inverse',event)
compact=controller.compact_events('fresh-review','inverse');keys=arc.keys('fresh-review','inverse');before=[arc.read(k) for k in keys]
cert=verify_candidate({'x0':.7,'x1':.9},compact,'x0');assert cert['status']=='VERIFIED'
controller.append('fresh-review','inverse',dict(kind='revision',child='x1',parent=None,value=1.1));assert controller.summary_status('fresh-review','inverse')=='COMPACTION_REJECTED'
compact2=controller.compact_events('fresh-review','inverse');assert verify_candidate({'x0':.9,'x1':1.1},compact2,'x0')['status']=='VERIFIED'
assert before==[arc.read(k) for k in keys];arc.close();arc=Archive(db);assert before==[arc.read(k) for k in keys];arc.close()
results['inverse_compaction']={'facets':len(compact2),'old_raw_rows_unchanged':True,'compactions':controller.compactions,'reopens':controller.reopens}
h=HeadingRegistry()
for e in tape:h.consume('fresh-review',e)
serialized=h.serialize();assert HeadingRegistry.from_serialized(serialized).serialize()==serialized
results['heading_roundtrip_exact']=True
# Contradictory adapter fixture: parser marks valid and abstained simultaneously.
from learning.model import EventModel
zero=EventModel();zero.eval()
with torch.no_grad():
 for p in zero.parameters():p.zero_()
class Contradictory:
 def parse_batch(self,texts):return [dict(valid=True,abstained=True,events=[e]) for e in [dict(kind='root',child='x0',parent=None,value=0.),dict(kind='query',child='x0',parent=None,value=None)]]
z=run_text_episode(['fixture-root','fixture-query'],zero,Contradictory(),out/'contradictory-adapter.sqlite',max_steps=2)
results['contradictory_adapter']={'output_status':z['outputs'][0]['status'],'heading_count':z['heading_count'],'rejected':z['rejected_observations'],'scope':'inconsistent injected adapter, not production LearnedParser output'}
# Existing evidence refusal; fresh protocol-only mismatch not checked by learning guard.
fresh=out/'guard-fixture';fresh.mkdir();(fresh/'protocol.json').write_text('{"changed":true}');(fresh/'protocol.sha256').write_text('not-the-file-hash')
assert_fresh_output(fresh);results['learning_guard_protocol_only_mismatch_admitted']=True
(fresh/'results').mkdir();(fresh/'results'/'evidence.txt').write_text('preserve')
try:assert_fresh_output(fresh);raise AssertionError('overwrite admitted')
except FileExistsError:results['learning_guard_existing_evidence_refused']=True
try:check_entry(fresh);raise AssertionError('partial admitted')
except Refusal:results['core_guard_partial_evidence_refused']=True
# Independently recompute already completed memory-v3 reported aggregate metrics.
metrics=json.loads(Path('memory/results/metrics.json').read_text());eng=metrics['engineering'];mem=np.load('memory/results-v3/memory-predictions.npz')
for j,condition in enumerate(['full_raw','boundary_compact','answer_only','mean','forced_stale']):
 err=abs(mem['predictions'][:,j]-mem['targets']);assert abs(float(err.mean())-eng['memory'][condition]['mean_absolute_error'])<1e-12
halting=[json.loads(s) for s in Path('memory/results-v3/halting-examples.jsonl').read_text().splitlines()]
for budget in [1,2,4,8]:
 rows=[r for r in halting if r['budget']==budget];m=eng['halting'][str(budget)]
 assert sum(r['status']=='VERIFIED' for r in rows)/len(rows)==m['accepted_coverage'];assert np.mean([r['executed_steps'] for r in rows])==m['mean_executed_steps']
results['memory_v3_recomputed']={'memory_examples':len(mem['targets']),'halting_records':len(halting),'match':True}
(out/'memory-pipeline-review-diagnostics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
