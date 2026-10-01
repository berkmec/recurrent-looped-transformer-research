"""Frozen bounded engineering experiments; numeric candidates here are synthetic.

The learned model integration is owned by root, not silently substituted here.
"""
import hashlib,json,platform,random,subprocess,sys,time
from pathlib import Path
import numpy as np
from .api import Archive,TitleController,symbolic_values,verify_candidate,halt_candidates,canonical,constraints
ROOT=Path(__file__).parent
def save_jsonl(path,rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    started=time.perf_counter(); protocol_sha=sha(ROOT/'protocol.json'); results=ROOT/'results'; results.mkdir(exist_ok=True)
    with (ROOT/'execution.jsonl').open('a') as f: f.write(json.dumps(dict(event='engineering_evaluation_start',time=time.time(),protocol_sha256=protocol_sha))+'\n')
    db=ROOT/'archive.sqlite'; archive=Archive(db)
    if archive.count(): raise RuntimeError('Preserve prior evidence: nonempty experiment archive exists')
    rng=random.Random(20260916); rows=[]; retrieval_rows=[]; halt_rows=[]; timings=[]
    for i in range(192):
        n=[8,16,32,64][i%4]; capacity=[4,8,16][i%3]; task=f'heldout-task-{i}'; title='affine-path'; c=TitleController(capacity,archive)
        values=[round(rng.uniform(-1,1),5) for _ in range(n)]; events=[dict(kind='root',child='x0',parent=None,value=values[0])]+[dict(kind='edge',child=f'x{j}',parent=f'x{j-1}',value=values[j]) for j in range(1,n)]
        for e in events: c.append(task,title,e)
        old=symbolic_values(events); query=f'x{n-1}'; old_answer=old[query]; old_compact=c.compact_events(task,title)
        # Genuine separate subproblem titles compete for resident record slots.
        for k in range(2):
            for j in range(capacity): c.append(task,f'independent-{k}',dict(kind='root' if j==0 else 'revision',child='x0',parent=None,value=float(j)))
        target_child=f'x{1+i%(n-1)}'; parent=f'x{int(target_child[1:])-1}'; revision=dict(kind='revision',child=target_child,parent=parent,value=values[int(target_child[1:])]+2.0)
        c.append(task,title,revision); events.append(revision); rejected=c.summary_status(task,title)=='COMPACTION_REJECTED'
        t0=time.perf_counter(); restored=c.reconstruct(task,title); compact=c.compact_events(task,title); read_seconds=time.perf_counter()-t0
        truth=symbolic_values(events)[query]; full=symbolic_values(restored)[query]; compressed=symbolic_values(compact)[query]; stale=symbolic_values(old_compact)[query]; mean=float(np.mean(list(old.values())))
        rawbytes=len(canonical(restored).encode()); compactbytes=len(canonical(compact).encode()); summary=c.summaries.get((task,title),{})
        rows.append(dict(id=i,t=n+2*capacity+1,k=0,d=n-1,capacity=capacity,length=n,target=truth,full_raw=full,boundary_compact=compressed,answer_only=old_answer,mean=mean,forced_stale=stale,errors={name:abs(value-truth) for name,value in [('full_raw',full),('boundary_compact',compressed),('answer_only',old_answer),('mean',mean),('forced_stale',stale)]},condition_loss={'boundary_compact':len(constraints(restored))-len(constraints(compact)),'answer_only':n,'mean':n},rejected_stale=rejected,reopens=c.reopens,compactions=c.compactions,premature=c.premature_compactions,max_active=c.max_active,raw_count=len(archive.history(task,title)),raw_bytes=rawbytes,compact_bytes=compactbytes,metadata_bytes=len(canonical(c.summaries).encode()) if False else sum(len(canonical(s).encode()) for s in c.summaries.values()),read_records=c.read_records,reconstruct_seconds=read_seconds,restoration_exact=restored==events,title_count=len(c.titles)))
        # Real sequential generator: one quarter stable-wrong, others converge.
        for budget in [1,2,4,8]:
            executed=[]
            def candidates():
                true_nodes=symbolic_values(events)
                for step in range(budget):
                    error=.5 if i%4==0 else .8*(.5**step)
                    executed.append(step); yield {key:value+error for key,value in true_nodes.items()}
            result=halt_candidates(candidates(),events,query,tolerance=.1,max_steps=budget)
            pred=result['candidate'][query]; fixed_error=.5 if i%4==0 else .8*(.5**(budget-1))
            halt_rows.append(dict(id=i,budget=budget,status=result['status'],steps=result['steps'],executed_steps=len(executed),error=abs(pred-truth),bound=result['bound'],wrong_accepted=result['status']=='VERIFIED' and abs(pred-truth)>.1000000001,fixed_error=fixed_error,fixed_steps=budget,candidate_source='synthetic_shared_offset_trajectory',t=n+1,k=result['steps'],d=n-1))
    # Retrieval creates deliberate stale, similar-entity, contradictory and cross-
    # task records. Query uses only task/title/entity descriptors, not exact key.
    for i in range(192):
        task=f'retrieval-task-{i}'; title='entity-conditions'; child=f'x{i%7}'
        stale=archive.append(task,title,dict(kind='root',child=child,parent=None,value=1))
        correct=archive.append(task,title,dict(kind='revision',child=child,parent=None,value=2))
        archive.append(task,title,dict(kind='root',child=child+'0',parent=None,value=2))
        archive.append(task+'-other',title,dict(kind='root',child=child,parent=None,value=2))
        retrieval_rows.append(dict(id=i,task=task,title=title,child=child,correct=correct,stale=stale))
    for row in retrieval_rows:
        query=f"{row['task']} {row['title']} {row['child']} value condition"; ts=time.perf_counter(); raw=archive.search(query,5); duration=time.perf_counter()-ts
        row['raw']=raw; row['retrieval_seconds']=duration; row['by_k']={}
        for k in [1,3,5]:
            ids=[r['id'] for r in raw[:k]]; filtered=[x for x in ids if archive.applicable(x,row['task'],row['title'],row['child'])]
            # A retrieved scalar used without filtering can corrupt a query;
            # report record inapplicability and actual wrong numeric value apart.
            raw_damage=bool(ids) and archive.read(ids[0])['event'].get('value')!=2
            row['by_k'][str(k)]=dict(recall=float(row['correct'] in ids),filtered=filtered,precision=float(all(x==row['correct'] for x in filtered)) if filtered else None,coverage=bool(filtered),raw_top1_inapplicable=bool(ids) and not archive.applicable(ids[0],row['task'],row['title'],row['child']),raw_numeric_damage=raw_damage,filtered_numeric_damage=any(archive.read(x)['event']['value']!=2 for x in filtered))
        row['oracle_exact_key_applicable']=archive.applicable(row['correct'],row['task'],row['title'],row['child'])
        row['missing_reference_coverage']=any(archive.applicable(r['id'],row['task'],row['title'],'x_missing') for r in raw)
    total_raw=archive.count(); archive.close()
    # Truly fresh interpreter verifies all persisted content hashes and vectors.
    code="from memory.api import Archive; import json; a=Archive(r'%s'); keys=[r[0] for r in a.db.execute('SELECT id FROM records')]; [a.read(k) for k in keys]; print(json.dumps({'hashes_verified':len(keys),'vectors':a.db.execute('SELECT COUNT(*) FROM records WHERE vector IS NOT NULL').fetchone()[0]})); a.close()"%str(db)
    fresh=subprocess.run([sys.executable,'-c',code],cwd=ROOT.parent,capture_output=True,text=True,check=True); (ROOT/'fresh-process-reload.json').write_text(fresh.stdout)
    save_jsonl(results/'memory-examples.jsonl',rows); save_jsonl(results/'retrieval-examples.jsonl',retrieval_rows); save_jsonl(results/'halting-examples.jsonl',halt_rows)
    np.savez_compressed(results/'memory-predictions.npz',targets=np.array([r['target'] for r in rows]),predictions=np.array([[r[k] for k in ['full_raw','boundary_compact','answer_only','mean','forced_stale']] for r in rows]),ids=np.arange(192))
    np.savez_compressed(results/'halting-predictions.npz',errors=np.array([r['error'] for r in halt_rows]),steps=np.array([r['steps'] for r in halt_rows]),accepted=np.array([r['status']=='VERIFIED' for r in halt_rows]),ids=np.array([r['id'] for r in halt_rows]))
    metrics=dict(protocol_sha256=protocol_sha,engineering_examples=len(rows),memory={name:dict(mean_absolute_error=float(np.mean([r['errors'][name] for r in rows])),max_absolute_error=max(r['errors'][name] for r in rows)) for name in ['full_raw','boundary_compact','answer_only','mean','forced_stale']},controller=dict(capacity_violations=sum(r['max_active']>r['capacity'] for r in rows),premature_compactions=sum(r['premature'] for r in rows),compactions=sum(r['compactions'] for r in rows),restoration_exact=sum(r['restoration_exact'] for r in rows),stale_detection=sum(r['rejected_stale'] for r in rows),reopen_repairs=sum(r['reopens']>0 and r['errors']['boundary_compact']==0 for r in rows),mean_compact_to_raw_bytes=float(np.mean([r['compact_bytes']/r['raw_bytes'] for r in rows])),mean_read_records=float(np.mean([r['read_records'] for r in rows])),active_budget_scope='resident raw records only; constraint summaries/metadata and transient reconstruction are separately unbounded'),retrieval={},halting={},persisted_records=total_raw,fresh_process=json.loads(fresh.stdout),wall_seconds=time.perf_counter()-started)
    for k in [1,3,5]:
        rs=[r['by_k'][str(k)] for r in retrieval_rows]; accepted=[r for r in rs if r['coverage']]
        metrics['retrieval'][str(k)]=dict(recall_at_k=float(np.mean([r['recall'] for r in rs])),filtered_precision=float(np.mean([r['precision'] for r in accepted])) if accepted else None,filtered_coverage=len(accepted)/len(rs),unfiltered_top1_inapplicable=float(np.mean([r['raw_top1_inapplicable'] for r in rs])),unfiltered_numeric_damage=float(np.mean([r['raw_numeric_damage'] for r in rs])),filtered_numeric_damage=sum(r['filtered_numeric_damage'] for r in rs),oracle_exact_key_coverage=1.,missing_reference_coverage=sum(r['missing_reference_coverage'] for r in retrieval_rows)/192)
    for budget in [1,2,4,8]:
        rs=[r for r in halt_rows if r['budget']==budget]; accepted=[r for r in rs if r['status']=='VERIFIED']; metrics['halting'][str(budget)]=dict(accepted_coverage=len(accepted)/192,wrong_accepted=sum(r['wrong_accepted'] for r in rs),accepted_error_max=max([r['error'] for r in accepted],default=None),mean_executed_steps=float(np.mean([r['steps'] for r in rs])),fixed_steps=budget,saved_step_fraction=1-float(np.mean([r['steps'] for r in rs]))/budget,candidate_source='synthetic, not learned')
    (results/'engineering-metrics.json').write_text(json.dumps(metrics,indent=2))
    manifest=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,protocol_sha256=protocol_sha,sources={str(p.relative_to(ROOT.parent)):sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT.parent/'tests').glob('test_memory*.py'))})
    (ROOT/'environment-source-hashes.json').write_text(json.dumps(manifest,indent=2))
    with (ROOT/'execution.jsonl').open('a') as f: f.write(json.dumps(dict(event='engineering_complete',time=time.time(),protocol_sha256=protocol_sha,wall_seconds=metrics['wall_seconds']))+'\n')
    print(json.dumps(metrics,indent=2)); return metrics

if __name__=='__main__': run()
