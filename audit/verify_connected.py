"""Independent streaming audit of the completed connected experiment.

No inference/training or evaluator/memory/learning imports. Refuses to read final
datasets unless the complete24-group metrics manifest exists and is coherent.
Source labels use graph propagation, not evaluator least squares. SQLite opens
read-only; results are never modified. One episode archive is resident at once.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,collections,datetime,hashlib,json,math,sqlite3,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'integration'/'connected';RESULT=OUT/'results'
PIN='0b61590618f799ecece783cac7d43c08521f7d2ae901228f293a1df1c5a4e2ac'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text(encoding='utf8'))
def mean(x):return sum(x)/len(x) if x else None
def ensure(ok,message):
    if not ok:raise AssertionError(message)
def hid(task,entity):return 'heading:'+hashlib.sha256(canonical([task,entity]).encode()).hexdigest()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def close(a,b,tol=1e-9):
    if isinstance(a,dict) and isinstance(b,dict):return all(k in b and close(v,b[k],tol) for k,v in a.items())
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(close(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):return abs(a-b)<=tol*max(1,abs(a),abs(b))
    return a==b
def assertion_map(events):
    facets={}
    for e in events:
        if e['kind']=='query':continue
        parent=None if e['kind']=='root' else e.get('parent')
        facets[('absolute:' if parent is None else 'edge:')+e['child']]=[e['child'],parent,float(e['value'])]
    return facets
def propagate(facets):
    graph=collections.defaultdict(list);known={};queue=collections.deque()
    for child,parent,value in facets.values():
        graph[child]
        if parent is None:
            if child in known:ensure(abs(known[child]-value)<1e-8,'conflicting source anchors')
            known[child]=value;queue.append(child)
        else:graph[parent].append((child,value));graph[child].append((parent,-value))
    while queue:
        parent=queue.popleft()
        for child,offset in graph[parent]:
            value=known[parent]+offset
            if child in known:ensure(abs(known[child]-value)<1e-8,'conflicting source edge cycle')
            else:known[child]=value;queue.append(child)
    ensure(set(graph)<=set(known),'source component lacks anchor');return known
def independent_labels(records):
    seen=[];labels=[]
    for i,event in enumerate(records):
        if event['kind']=='query':labels.append(dict(source_index=i,query=event['child'],answer=propagate(assertion_map(seen))[event['child']]))
        else:seen.append(event)
    return labels
def event_exact(a,b):
    if not isinstance(a,dict) or any(a.get(k)!=b.get(k) for k in ['kind','child','parent']):return False
    x,y=a.get('value'),b.get('value')
    return x is None and y is None if x is None or y is None else finite(x) and abs(x-y)<=1e-12
def score_independent(case,result,labels):
    outputs=result['outputs'];ensure(len({x['source_index'] for x in outputs})==len(outputs),'duplicate output source positions')
    bypos={x['source_index']:x for x in outputs};queries=[]
    for label in labels:
        actual=bypos.get(label['source_index'],{});prediction=actual.get('prediction');emitted=finite(prediction);query_matches=actual.get('query')==label['query'];err=abs(prediction-label['answer']) if emitted else None
        correct=bool(emitted and query_matches and err<=.1);accepted=actual.get('status')=='VERIFIED_PARSED_CONSTRAINTS'
        queries.append(dict(**label,prediction=prediction if emitted else None,status=actual.get('status','MISSING_QUERY_OUTPUT'),query_matches=query_matches,absolute_error=err,squared_error=err*err if err is not None else None,source_correct=correct,accepted=accepted,accepted_source_failure=accepted and not correct,steps=actual.get('steps',0),emitted=emitted))
    exact=[]
    for i,event in enumerate(case['records']):
        p=result['parsed'][i] if i<len(result['parsed']) else {};es=p.get('events',[]);extracted=es[0] if p.get('valid') and len(es)==1 else None;exact.append(event_exact(extracted,event))
    positions={x['source_index'] for x in labels};spurious=[o for o in outputs if o['source_index'] not in positions]
    return dict(id=case['id'],affected=case['affected'],inverse=case['inverse'],depth=case['depth'],queries=queries,final=queries[-1],source_event_exact=exact,all_source_events_exact=all(exact),spurious_query_outputs=len(spurious),spurious_accepted_outputs=sum(x.get('status')=='VERIFIED_PARSED_CONSTRAINTS' for x in spurious))
def wilson(fails,n):
    if not n:return None
    z=1.959963984540054;p=fails/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0,c-r),min(1,c+r)]
def summarize(scores):
    def queries(rows,independent):
        n=len(rows);accepted=sum(x['accepted'] for x in rows);fails=sum(x['accepted_source_failure'] for x in rows);errors=[x['squared_error'] for x in rows if x['emitted']]
        return dict(intended_queries=n,emitted_count=len(errors),coverage=accepted/n,accepted_count=accepted,accepted_failures=fails,accepted_error_rate=fails/accepted if accepted else None,accepted_error_rate_ci95=wilson(fails,accepted) if independent else None,accuracy_all_intended=sum(x['source_correct'] for x in rows)/n,mse_emitted=mean(errors),missing_outputs=n-len(errors),mean_refinements_at_query=mean([x['steps'] for x in rows]))
    allq=[q for s in scores for q in s['queries']];all_events=[e for s in scores for e in s['source_event_exact']];spurious=sum(s['spurious_accepted_outputs'] for s in scores);accepted=sum(q['accepted'] for q in allq)+spurious;fails=sum(q['accepted_source_failure'] for q in allq)+spurious
    return dict(episodes=len(scores),source_event_exact_accuracy=sum(all_events)/len(all_events),source_episode_exact_accuracy=sum(s['all_source_events_exact'] for s in scores)/len(scores),spurious_query_outputs=sum(s['spurious_query_outputs'] for s in scores),spurious_accepted_outputs=spurious,all_certificate_outputs=dict(count=accepted,source_failures=fails,source_failure_rate=fails/accepted if accepted else None),final=queries([s['final'] for s in scores],True),all_queries=queries(allq,False))
def audit_heading_history(task,history,parsed):
    ensure(len(history)==len(parsed),'heading history/source length');prev=None;state={};routes=[]
    for i,(entry,p) in enumerate(zip(history,parsed)):
        body={k:v for k,v in entry.items() if k!='sha256'};ensure(hashlib.sha256(canonical(body).encode()).hexdigest()==entry['sha256'],'heading event hash');ensure(entry['index']==i+1 and entry['previous_sha256']==prev and entry['task']==task,'heading hash-chain/order/task');prev=entry['sha256'];result=entry['result'];created=[];links=[];heading=None;version=None
        if not p.get('valid') or p.get('abstained'):
            ensure(entry['event']=={'kind':'extraction_unresolved','parsed':p},'rejected parser history');status='EXTRACTION_UNRESOLVED'
        else:
            ensure(len(p['events'])==1,'parser cardinality');e=p['events'][0];ensure(entry['event']==e,'heading observed event');kind=e.get('kind');child=e.get('child');parent=e.get('parent');value=e.get('value')
            if kind not in ['root','edge','revision','query']:status='UNSUPPORTED'
            elif not isinstance(child,str) or not child.startswith('x') or not child[1:].isdigit():status='MISSING_ENTITY'
            elif kind=='query':
                status='QUERY_RESOLVED' if child in state else 'MISSING_ENTITY'
                if child in state:heading=hid(task,child);version=state[child]['version']
            elif (kind=='edge' and parent is None) or (parent is not None and (not isinstance(parent,str) or not parent.startswith('x') or not parent[1:].isdigit())):status='MISSING_ENTITY'
            elif not finite(value) or parent==child:status='UNSUPPORTED'
            else:
                if kind=='root':parent=None
                is_new=child not in state
                for name in [child]+([parent] if parent is not None else []):
                    if name not in state:state[name]={'version':0,'facets':{}};created.append(hid(task,name))
                heading=hid(task,child);facet='absolute' if parent is None else 'edge';assertion={'parent':parent,'value':float(value)};node=state[child]
                if node['facets'].get(facet)==assertion:status='UNCHANGED'
                else:
                    node['facets'][facet]=assertion;node['version']+=1;status='CREATED' if is_new else 'UPDATED'
                    if parent is not None:links=[dict(type='DEPENDS_ON_AFFINE',source=heading,target=hid(task,parent),source_version=node['version'],offset=float(value),task=task)]
                version=node['version']
        expected=dict(status=status,heading_id=heading,version=version,created=created,links=links);ensure(result==expected,'heading semantic/version/typed-link replay');routes.append(heading or 'main/unresolved_source')
    return routes,state
def audit_db(path,row,case,labels,environment,collect_curve=False):
    result=row['result'];task=case['id'];ensure(sha(path)==row['archive_sha256'],'whole DB SHA256');ensure(not Path(str(path)+'-journal').exists(),'live archive journal')
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    try:
        records_by_title=collections.defaultdict(list);observation_refs=collections.defaultdict(list);trace_groups=collections.defaultdict(list);trace_count=0;registry=None;provenance=[];raw_count=0;first=[]
        for rid,record_task,title,version,raw,digest in db.execute('SELECT id,task,title,version,raw,sha FROM records ORDER BY id'):
            raw_count+=1;ensure(rid==raw_count,'raw ID gap');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'raw payload SHA256');record=json.loads(raw);ensure(record_task==task and record['task']==task and record['title']==title and record['version']==version,'raw metadata match')
            prior_count=len(records_by_title[title]);ensure(version==prior_count+1,'raw title version gap/overwrite');records_by_title[title].append(record);e=record['event'];kind=e['kind']
            if kind=='model_trace':
                trace_count+=1;ensure(title=='main/trace','trace owner');key=(e['source_index'],e['phase'],e['query_index']);trace_groups[key].append(e)
                ensure(e['heading_slot_mapping']=={hid(task,name):slot for name,slot in e['slot_mapping'].items()},'model slot -> heading routing');ensure(set(e['nodes'])==set(e['slot_mapping']),'trace node/slot names')
                ensure(len(e['title_latents'])==10 and len(e['versions'])==10,'trace latent/version slots');ensure(all(finite(v) for v in e['nodes'].values()),'trace finite nodes')
            elif kind=='heading_registry':ensure(registry is None and title=='main/heading_registry','one heading registry');registry=e['payload']
            elif kind=='model_provenance':provenance.append(e['payload']);ensure(title=='main/model_provenance','provenance owner')
            else:
                ensure('source_observation_id' in e,'observed record provenance reference');observation_refs[e['source_observation_id']].append((title,e))
        ensure(raw_count==result['raw_records'],'raw record count');ensure(trace_count==result['total_refinements_executed'],'actual archived trace count != executed refinements');ensure(registry is not None and registry['schema']=='finite-heading-registry-v1','heading registry missing')
        routes,heading_state=audit_heading_history(task,registry['history'],result['parsed']);ensure(len(heading_state)==result['heading_count'],'heading count');ensure(len(provenance)==1,'one model provenance');prov=provenance[0]
        for tag in ['model','parser']:ensure(environment['checkpoint_sha256'].get(prov[tag+'_path'])==prov[tag+'_sha256'],'checkpoint provenance')
        ensure(prov['protocol_sha256']==PIN,'archive protocol provenance')
        observed_count=0;valid_count=0
        for oid,otask,otitle,raw,digest in db.execute('SELECT id,task,title,raw,sha FROM observations ORDER BY id'):
            observed_count+=1;ensure(oid==observed_count,'observation ID gap');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'observation SHA256');obs=json.loads(raw);idx=oid-1;ensure(otask==task and obs['task']==task and otitle==routes[idx] and obs['title']==otitle,'source observation heading routing');ensure(obs['parsed']==result['parsed'][idx],'source parsed preservation')
            p=obs['parsed'];textsha=p.get('provenance',{}).get('text_sha256')
            if textsha is not None:ensure(hashlib.sha256(obs['text'].encode()).hexdigest()==textsha,'original text provenance hash')
            refs=observation_refs.pop(oid,[])
            if p.get('valid'):
                valid_count+=1;ensure(len(refs)==1,'accepted source missing/duplicated raw observation');rt,e=refs[0];ensure(rt==otitle and {k:v for k,v in e.items() if k!='source_observation_id'}==p['events'][0],'accepted event reconstruction')
            else:ensure(not refs,'rejected observation turned into raw event')
        ensure(not observation_refs,'dangling source observation references');ensure(observed_count==len(case['records'])==result['source_observations'],'source observation cardinality')
        ensure(result['compactions']==(valid_count-1)//4 if valid_count else result['compactions']==0,'capacity-only compaction count');ensure(result['premature_compactions']==0 and result['max_active_records']<=4,'active capacity/premature compaction')
        compact_count=0;compact_versions={}
        for cid,ctask,title,version,raw,digest in db.execute('SELECT id,task,title,version,raw,sha FROM compacts ORDER BY id'):
            compact_count+=1;ensure(cid==compact_count,'compact ID gap');ensure(ctask==task,'compact task');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'compact payload SHA256');snapshot=json.loads(raw);ensure(snapshot['version']==version and version>=compact_versions.get(title,0),'compact version regressed');compact_versions[title]=version
            history=[r for r in records_by_title[title] if r['version']<=version];ensure(len(history)==version,'compact references missing raw history');ensure(hashlib.sha256(canonical(history).encode()).hexdigest()==snapshot['sha'],'compact raw-prefix reconstruction SHA');ensure(snapshot['conditions']==assertion_map([r['event'] for r in history]),'compact loses/changes latest condition')
        output_bypos={o['source_index']:o for o in result['outputs']};count_bypos=collections.Counter();current_bypos={};curve=[]
        for (position,phase,qindex),traces in trace_groups.items():
            ensure(phase in ['current_query','replayed_prior_query'],'unknown trace phase');ensure([x['k'] for x in traces]==list(range(1,len(traces)+1)),'actual k trace sequence');ensure(len({x['t'] for x in traces})==1,'refinement consumes token');count_bypos[position]+=len(traces)
            ensure(position in output_bypos,'trace without output')
            if phase=='current_query':ensure(position not in current_bypos,'multiple current query trace groups');current_bypos[position]=traces
        for position,output in output_bypos.items():
            ensure(count_bypos[position]==output.get('refinements_executed',0),'per-query actual trace/refinement mismatch');traces=current_bypos.get(position,[]);ensure(len(traces)==output.get('steps',0),'current query trace steps mismatch')
            if traces:
                last=traces[-1];ensure(abs(last['nodes'][output['query']]-output['prediction'])<1e-7,'reported prediction differs from archived actual node');ensure(last['t']==output['consumed_tokens'],'reported consumed tokens differs from actual trace')
                prior=output['prior_query_rounds'];replayed=[trace_groups[(position,'replayed_prior_query',j)] for j in range(len(prior))];ensure([len(t) for t in replayed]==prior,'prior query replay rounds mismatch')
        ensure(sum(o.get('consumed_tokens',0) for o in result['outputs'])==result['total_consumed_tokens_including_replay'],'aggregate consumed-token count')
        if collect_curve:
            final=labels[-1];position=final['source_index'];output=output_bypos.get(position,{});traces=current_bypos.get(position,[])
            if traces:
                ensure(len(traces)==8,'fixed replay final query lacks8 refinements');ensure(output['prior_query_rounds']==[8,8],'conditional curve prior query budgets not8,8')
            for k in range(1,9):
                trace=next((t for t in traces if t['k']==k),None);query=output.get('query');prediction=trace['nodes'].get(query) if trace else None;emitted=finite(prediction);error=prediction-final['answer'] if emitted else None
                curve.append(dict(k=k,id=case['id'],emitted=emitted,query_matches=query==final['query'],source_correct=bool(emitted and query==final['query'] and abs(error)<=.1),squared_error=error*error if emitted else None,prior_query_rounds=output.get('prior_query_rounds'),status='DIAGNOSTIC_ARCHIVED_CANDIDATE_NOT_CERTIFIED_AT_K'))
        return dict(raw_records=raw_count,observations=observed_count,compacts=compact_count,traces=trace_count,archive_bytes=path.stat().st_size,headings=len(heading_state),conditional_curve=curve)
    finally:db.close()
def validate_complete_manifest():
    # Gate before opening any held-out case/prediction/DB file.
    ensure(sha(OUT/'protocol.json')==PIN,'unexpected protocol pin');ensure((OUT/'protocol.sha256').read_text().strip()==PIN,'protocol sidecar mismatch')
    ensure((RESULT/'metrics.json').exists(),'INCOMPLETE: final metrics.json missing; do not audit running results')
    p=jread(OUT/'protocol.json');m=jread(RESULT/'metrics.json');expected={(s,split,c['name']) for s in p['seeds'] for split in p['datasets'] for c in p['conditions']};actual={(r['seed'],r['split'],r['condition']['name']) for r in m['rows']}
    ensure(len(m['rows'])==24 and actual==expected,'INCOMPLETE: expected24 unique groups');ensure(m['condition_seed_episode_runs']==6144 and m['independent_source_cases']==512,'manifest execution/task count');ensure(m['protocol_sha256']==PIN,'final protocol binding')
    for r in m['rows']:
        group=RESULT/f"{r['condition']['name']}-{r['split']}-s{r['seed']}";ensure((group/'metrics.json').exists() and (group/'predictions.jsonl').exists(),'INCOMPLETE group files');ensure(r['episodes']==256,'INCOMPLETE group episode count')
    return p,m
def main():
    started=time.perf_counter();protocol,completed=validate_complete_manifest();environment=jread(RESULT/'environment.json');ensure(environment['protocol_sha256']==PIN,'environment protocol');ensure(environment['threads']==1,'CPU thread count')
    for mapping in [protocol['source_sha256'],protocol['upstream_protocol_sha256'],environment['checkpoint_sha256']]:
        for path,digest in mapping.items():ensure(sha(path)==digest,'frozen source/checkpoint drift: '+path)
    executions=[json.loads(x) for x in (RESULT/'execution.jsonl').read_text().splitlines()];ensure(any(e['event']=='complete' for e in executions),'INCOMPLETE: completion receipt absent')
    datasets={};max_label_discrepancy=0.
    for split in protocol['datasets']:
        path=RESULT/f'data-{split}.json';receipts=[e for e in executions if e['event']=='fresh_test_first_open' and e['split']==split];ensure(len(receipts)==1 and receipts[0]['sha256']==sha(path),'source dataset hash');ensure(receipts[0]['utc']>protocol['frozen_utc'],'source opened before protocol freeze');cases=jread(path);ensure(len(cases)==256 and len({c['id'] for c in cases})==256,'source dataset cardinality/ID')
        for case in cases:
            labels=independent_labels(case['records']);ensure(len(labels)==3,'source expected3queries');ensure(close(labels,case['queries'],1e-8),'independent graph source labels differ from evaluator linalg');max_label_discrepancy=max(max_label_discrepancy,max(abs(x['answer']-y['answer']) for x,y in zip(labels,case['queries'])));case['_independent_labels']=labels
        datasets[split]=cases
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S');audit_path=BASE/'audit'/f'connected-independent-{stamp}.json';detail_path=BASE/'audit'/f'connected-independent-{stamp}-episodes.jsonl';curve_path=BASE/'audit'/f'connected-independent-{stamp}-depth-curves.json';groups=[];curves=[];totals=collections.Counter();all_ids=set()
    with detail_path.open('x',encoding='utf8') as detail:
        for recorded in completed['rows']:
            seed=recorded['seed'];split=recorded['split'];condition=recorded['condition']['name'];group=RESULT/f'{condition}-{split}-s{seed}';ensure(jread(group/'metrics.json')==recorded,'group/final metrics disagree');pred_path=group/'predictions.jsonl';ensure(sha(pred_path)==recorded['predictions_sha256'],'prediction file SHA256');scores=[];costs=[];allcurve=[];count=0
            with pred_path.open(encoding='utf8') as predictions:
                for index,line in enumerate(predictions):
                    ensure(index<256,'extra episode');row=json.loads(line);case=datasets[split][index];ensure(row['seed']==seed and row['split']==split and row['condition']==recorded['condition'],'prediction group pairing');ensure(row['score']['id']==case['id'],'source example pairing');all_ids.add(case['id']);result=row['result'];labels=case['_independent_labels'];score=score_independent(case,result,labels);ensure(close(score,row['score'],1e-8),'source-relative score mismatch');scores.append(score)
                    dbpath=group/f'episode-{index:04d}.sqlite';ensure(Path(result['archive_path']).resolve()==dbpath.resolve(),'archive path binding');db=audit_db(dbpath,row,case,labels,environment,condition=='fixed-replay-seen');allcurve.extend(db.pop('conditional_curve'))
                    for key in ['raw_records','observations','compacts','traces','archive_bytes']:totals[key]+=db[key]
                    cost={k:result[k] for k in ['total_refinements_executed','total_consumed_tokens_including_replay','seconds','raw_records','compactions','premature_compactions','reopens','max_active_records','heading_count']};cost['archive_bytes']=db['archive_bytes'];costs.append(cost);count+=1
                    detail.write(json.dumps(dict(seed=seed,split=split,condition=condition,id=case['id'],db= db,final=score['final'],all_source_events_exact=score['all_source_events_exact'],spurious_accepted=score['spurious_accepted_outputs']))+'\n')
            ensure(count==256,'INCOMPLETE prediction lines');ensure(len(list(group.glob('episode-*.sqlite')))==256,'INCOMPLETE/extra archives');computed=summarize(scores);ensure(close(computed,recorded,1e-8),'group metric recomputation mismatch')
            for name,mask in [('affected',lambda s:s['affected']),('unaffected',lambda s:not s['affected']),('inverse',lambda s:s['inverse']),('forward',lambda s:not s['inverse'])]:ensure(close(summarize([s for s in scores if mask(s)]),recorded[name],1e-8),'subgroup metric mismatch:'+name)
            ensure(close({k:mean([c[k] for c in costs]) for k in costs[0]},recorded['costs_mean'],1e-8),'cost mean mismatch');ensure(close({k:max(c[k] for c in costs) for k in costs[0]},recorded['costs_max'],1e-8),'cost max mismatch')
            final=[s['final'] for s in scores];certified_yield=sum(q['accepted'] and q['source_correct'] for q in final)/256
            groups.append(dict(seed=seed,split=split,condition=condition,**computed,certified_source_correct_yield=certified_yield,diagnostic_candidate_accuracy=computed['final']['accuracy_all_intended'],status='PASS',predictions_sha256=sha(pred_path)))
            if allcurve:
                for k in range(1,9):
                    subset=[r for r in allcurve if r['k']==k];errors=[r['squared_error'] for r in subset if r['emitted']];curves.append(dict(seed=seed,split=split,k=k,episodes=256,emitted=len(errors),missing=256-len(errors),diagnostic_accuracy_all_intended=sum(r['source_correct'] for r in subset)/256,diagnostic_mse_emitted=mean(errors),prior_query_rounds=[8,8],scope='conditional final-query depth curve from fixed replay actual traces; not whole-system k sweep; not certified answers'))
            print(json.dumps(dict(event='group_audited',seed=seed,split=split,condition=condition,episodes=count,seconds=time.perf_counter()-started)),flush=True)
    ensure(len(groups)==24 and len(all_ids)==512,'completed independent group/task cardinality')
    summary=dict(status='PASS_WITH_SCOPE_LIMITS',protocol_sha256=PIN,executions=6144,shared_source_tasks=512,groups=groups,archive_totals=dict(totals),max_independent_source_label_discrepancy=max_label_discrepancy,conditional_depth_curves=curves,wall_seconds=time.perf_counter()-started,auditor_source_sha256=sha(Path(__file__)),detail_path=str(detail_path),limitations=['Read-only final-state integrity checks support append-only history but cannot prove no privileged actor rewrote complete DBs and all hash receipts.','Three joint parser/model seeds;6144 executions share512 tasks and are not6144 independent examples.','A diagnostic numeric candidate can be correct without being certified; source-certified yield requires both.','All certificate risk includes spurious query positions as source alignment failures.','k1..8 curve varies final-query refinement only with earlier query refinements fixed8,8.','No training or model execution performed by this audit.'])
    with audit_path.open('x',encoding='utf8') as f:json.dump(summary,f,indent=2)
    with curve_path.open('x',encoding='utf8') as f:json.dump(curves,f,indent=2)
    report=BASE/'reports'/'connected-independent-audit.md';lines=['# Independent connected experiment audit','',f'Status: **PASS_WITH_SCOPE_LIMITS**. Audited24 complete groups,6144 episode executions,512 shared source tasks and three paired parser/model seeds. Protocol `{PIN}`.','',f'All source/checkpoint pins, prediction-file hashes and archive-file hashes match. Every raw, source-observation and compact payload SHA was verified. Per-title contiguous versions, preserved source-observation references, heading semantic/history routing and compact raw-prefix hashes were checked. Archive totals: {dict(totals)}.','',f'Original-source labels were independently derived with signed graph propagation, never evaluator least squares. Maximum difference: {max_label_discrepancy:.3g}. All intended queries, missing denominators, source extraction exactness, source-relative candidate errors and certificate failure counts including spurious queries independently match recorded metrics.','', 'Every actual archived model-trace count equals reported executed refinement totals. Replay/current phases, contiguous k, invariant t during refinement, prior replay budgets and reported final candidates agree with archived node values.','', '| Seed | Split | Condition | Diagnostic candidate accuracy | Certified correct yield | Certificate coverage | Accepted source failures | Missing final |','|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['seed']} | {g['split']} | {g['condition']} | {g['diagnostic_candidate_accuracy']:.4f} | {g['certified_source_correct_yield']:.4f} | {g['final']['coverage']:.4f} | {g['final']['accepted_failures']} | {g['final']['missing_outputs']} |")
    lines+=['','## Conditional depth curves','', 'Curves use actual fixed-replay-seen final-query traces for k1..8; earlier queries remain8,8. They describe diagnostic candidate quality, not certification or a whole-system computation-budget sweep. Missing outputs remain in accuracy denominators and are excluded from emitted-only MSE with counts reported.','', '| Seed | Split | k | Candidate accuracy | Emitted MSE | Missing |','|---|---|---:|---:|---:|---:|']
    for c in curves:lines.append(f"| {c['seed']} | {c['split']} | {c['k']} | {c['diagnostic_accuracy_all_intended']:.4f} | {c['diagnostic_mse_emitted']} | {c['missing']} |")
    lines+=['','## Limits','']+[f'- {w}' for w in summary['limitations']]+['',f'Independent JSON: `{audit_path}`. Streaming per-episode checks: `{detail_path}`. Curves: `{curve_path}`.']
    with report.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],report=str(report),audit=str(audit_path),seconds=summary['wall_seconds'])),flush=True)
def self_test():
    records=[dict(kind='edge',child='x1',parent='x0',value=2),dict(kind='root',child='x1',parent=None,value=5),dict(kind='query',child='x0',parent=None,value=None),dict(kind='revision',child='x1',parent=None,value=8),dict(kind='query',child='x0',parent=None,value=None)]
    labels=independent_labels(records);ensure([x['answer'] for x in labels]==[3,6],'inverse propagation/revision self-test')
    parsed=[dict(valid=True,events=[e]) for e in records];case=dict(id='fixture',records=records,queries=labels,affected=True,inverse=True,depth=1)
    result=dict(parsed=parsed,outputs=[dict(source_index=2,query='WRONG',prediction=3.,status='VERIFIED_PARSED_CONSTRAINTS',steps=1),dict(source_index=0,query='x0',prediction=3.,status='VERIFIED_PARSED_CONSTRAINTS',steps=1)])
    scored=score_independent(case,result,labels);summary=summarize([scored]);ensure(summary['final']['missing_outputs']==1 and summary['final']['accuracy_all_intended']==0,'missing denominator self-test');ensure(summary['all_certificate_outputs']['source_failures']==2 and summary['spurious_accepted_outputs']==1,'wrong query/spurious certificate self-test')
    ensure(not scored['queries'][0]['source_correct'] and scored['queries'][0]['accepted_source_failure'],'alias certificate self-test');ensure(event_exact(records[0],dict(records[0])),'tuple self-test');ensure(not event_exact(records[0],dict(records[0],value=4)),'wrong tuple self-test');print(json.dumps(dict(self_test='PASS',fixtures=['inverse graph revision','missing denominator','wrong entity certificate','spurious certificate','tuple exactness'],heldout_files_read=0)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    try:self_test() if args.self_test else main()
    except Exception as error:print(json.dumps(dict(status='AUDIT_REJECTED',error_type=type(error).__name__,reason=str(error))),flush=True);raise
