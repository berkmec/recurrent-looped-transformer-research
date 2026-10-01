"""Independent read-only generated-state window audit, gated on completion."""
import collections,datetime,hashlib,json,sqlite3,time
from pathlib import Path
from verify_connected import canonical,sha,jread,ensure,close,hid,finite,independent_labels,score_independent,summarize,audit_heading_history,mean
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'integration/reasoning-companion';R=OUT/'results';PIN='54e37e67b107af932ceb2a22381547885944cc1139e83210a2bd6d30107c9de9'

def archive(path,row,case,protocol):
    result=row['result'];task=case['id'];enabled=result['compact_reasoning'];ensure(sha(path)==row['archive_sha256'],'DB hash');ensure(not Path(str(path)+'-journal').exists(),'live DB');db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    try:
        histories=collections.defaultdict(list);traces={};groups=collections.defaultdict(list);refs=collections.defaultdict(list);registry=None;provenance=[];raw_count=0;trace_digest=hashlib.sha256();active=[];max_active=0;max_bytes=0;blocks=[]
        for rid,rt,title,version,raw,digest in db.execute('SELECT id,task,title,version,raw,sha FROM records ORDER BY id'):
            raw_count+=1;ensure(rid==raw_count,'raw ID gap');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'raw payload hash');record=json.loads(raw);ensure(rt==task and record['task']==task and record['title']==title and record['version']==version,'raw metadata');ensure(version==len(histories[title])+1,'raw version gap/overwrite');histories[title].append(record);e=record['event']
            if e['kind']=='model_trace':
                ensure(title=='main/trace','trace owner');ensure(e['heading_slot_mapping']=={hid(task,n):i for n,i in e['slot_mapping'].items()},'trace heading mapping');ensure(hid(task,e['query_entity']) in e['heading_slot_mapping'],'query heading absent');ensure(all(finite(x) for x in e['nodes'].values()),'trace nonfinite');traces[rid]=e;groups[e['source_index'],e['phase'],e['query_index']].append(e);trace_digest.update(canonical(e).encode())
                if e['k']==1:active=[]
                # Producer ReasoningWindow receives this row before trace sink
                # wraps source_index/heading_slot_mapping into the archive row.
                generated={k:v for k,v in e.items() if k not in ['source_index','heading_slot_mapping']};active.append((rid,generated));max_active=max(max_active,len(active));payload=sum(len(json.dumps(x,separators=(',',':')).encode()) for _,x in active);max_bytes=max(max_bytes,payload)
                if enabled and len(active)==4:blocks.append([i for i,_ in active]);active=[]
                if enabled:ensure(len(active)<4,'active generated trace overflow')
            elif e['kind']=='heading_registry':ensure(registry is None,'duplicate heading registry');registry=e['payload']
            elif e['kind']=='model_provenance':provenance.append(e['payload'])
            else:refs[e['source_observation_id']].append((title,e))
        ensure(raw_count==result['raw_records'],'raw count');ensure(len(traces)==result['reasoning_generated_traces']==result['total_refinements_executed'],'actual generated/refinement count');ensure(max_active==result['reasoning_max_resident_traces'],'actual active trace max');ensure(max_bytes==result['reasoning_peak_active_serialized_bytes'],'actual serialized active payload peak');ensure(not enabled or max_active<=4,'enabled active window bound')
        ensure(len(provenance)==1 and provenance[0]['protocol_sha256']==PIN,'model protocol provenance')
        for tag in ['model','parser']:ensure(provenance[0][tag+'_sha256']==protocol['checkpoint_sha256'][provenance[0][tag+'_path']],'model checkpoint provenance')
        routes,state=audit_heading_history(task,registry['history'],result['parsed']);ensure(len(state)==result['heading_count'],'heading count');observations=0
        for oid,rt,title,raw,digest in db.execute('SELECT id,task,title,raw,sha FROM observations ORDER BY id'):
            observations+=1;ensure(oid==observations and rt==task,'observation IDs/task');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'observation hash');obs=json.loads(raw);ensure(obs['parsed']==result['parsed'][oid-1] and title==routes[oid-1] and obs['title']==title,'source/heading routing');p=obs['parsed'];textsha=p.get('provenance',{}).get('text_sha256')
            if textsha is not None:ensure(hashlib.sha256(obs['text'].encode()).hexdigest()==textsha,'source text hash')
            linked=refs.pop(oid,[])
            if p.get('valid'):
                ensure(len(linked)==1,'accepted source raw preservation');owner,e=linked[0];ensure(owner==title and {k:v for k,v in e.items() if k!='source_observation_id'}==p['events'][0],'accepted source tuple')
            else:ensure(not linked,'rejected source raw record')
        ensure(not refs and observations==len(case['records']),'source cardinality');compact_count=0;reasoning=[];constraint_count=0
        for cid,rt,title,version,raw,digest in db.execute('SELECT id,task,title,version,raw,sha FROM compacts ORDER BY id'):
            compact_count+=1;ensure(cid==compact_count and rt==task,'compact IDs/task');ensure(hashlib.sha256(raw.encode()).hexdigest()==digest,'compact SHA');s=json.loads(raw);ensure(s['version']==version,'compact version metadata')
            if s.get('kind')=='latent_title_boundary':
                reasoning.append(s);ensure(s['version']==len(reasoning),'reasoning boundary serial version');ids=s['raw_trace_ids'];ensure(ids==blocks[len(reasoning)-1] and len(ids)==4,'full-capacity actual trace boundary');block=[traces[i] for i in ids];last=block[-1];heading=hid(task,last['query_entity']);ensure(title==heading+'/generated_reasoning' and s['heading_id']==heading,'reasoning heading routing');ensure(s['t']==last['t'] and s['k']==last['k'] and s['k'] in [4,8],'reasoning boundary t/k');ensure(s['shape']==[1,10,32] and s['dtype']=='torch.float32','reasoning snapshot shape/dtype');ensure(s['latest_title_state']==[last['title_latents']],'snapshot exact actual latest latent state');ensure(len(s['obligation_sha256'])==64 and all(c in '0123456789abcdef' for c in s['obligation_sha256']),'obligation fingerprint format');ensure(len({x['query_entity'] for x in block})==1,'block mixes heading identities')
            else:
                constraint_count+=1;history=[h for h in histories[title] if h['version']<=version];ensure(len(history)==version and hashlib.sha256(canonical(history).encode()).hexdigest()==s['sha'],'observation compact raw prefix');facets={}
                for h in history:
                    e=h['event']
                    if e['kind']=='query':continue
                    parent=None if e['kind']=='root' else e.get('parent');facets[('absolute:' if parent is None else 'edge:')+e['child']]=[e['child'],parent,float(e['value'])]
                ensure(facets==s['conditions'],'observation compact constraints')
        expected=len(blocks);ensure(len(reasoning)==expected==result['reasoning_compactions']==result['reasoning_rereads'],'actual reasoning compaction/reread count');ensure(result['reasoning_reload_bitwise_equal'] and result['reasoning_maximum_reload_difference']==0.,'recorded reload fidelity');ensure(result['same_heading_reasoning_continuations']==len(reasoning)-len({s['heading_id'] for s in reasoning}),'stable-heading continuations')
        outputs={x['source_index']:x for x in result['outputs']};counts=collections.Counter();current={}
        for (position,phase,qidx),trace in groups.items():
            ensure([x['k'] for x in trace]==list(range(1,len(trace)+1)),'actual refinement k sequence');ensure(len({x['t'] for x in trace})==1,'refinement changed token clock');counts[position]+=len(trace)
            if phase=='current_query':current[position]=trace;ensure(len(trace) in [4,8],'sufficiency outside4/8 boundary')
        for pos,o in outputs.items():
            ensure(counts[pos]==o.get('refinements_executed',0) and len(current.get(pos,[]))==o.get('steps',0),'trace and execution budgets')
            if pos in current:
                last=current[pos][-1];ensure(last['nodes'][o['query']]==o['prediction'],'candidate != actual trace readout');ensure(last['t']==o['consumed_tokens'],'actual token count');ensure([len(groups[pos,'replayed_prior_query',j]) for j in range(len(o['prior_query_rounds']))]==o['prior_query_rounds'],'actual replay budgets')
        ensure(sum(o.get('consumed_tokens',0) for o in outputs.values())==result['total_consumed_tokens_including_replay'],'aggregate tokens')
        return dict(raw_records=raw_count,observations=observations,trace_count=len(traces),trace_sha256=trace_digest.hexdigest(),reasoning_snapshots=len(reasoning),constraint_snapshots=constraint_count,max_generated_active=max_active,peak_generated_active_serialized_bytes=max_bytes,same_heading_continuations=result['same_heading_reasoning_continuations'],archive_bytes=path.stat().st_size)
    finally:db.close()

def main():
    start=time.perf_counter();ensure(sha(OUT/'protocol.json')==PIN and (OUT/'protocol.sha256').read_text().strip()==PIN,'protocol pin');ensure((R/'metrics.json').exists(),'INCOMPLETE: no final companion metrics; do not open heldout files');p=jread(OUT/'protocol.json');m=jread(R/'metrics.json');ensure(m['status']=='COMPLETE' and len(m['rows'])==12 and len(m['comparisons'])==6 and m['episode_executions']==1536,'INCOMPLETE companion manifest');ensure(m['protocol_sha256']==PIN,'manifest pin')
    for mapping in [p['source_sha256'],p['checkpoint_sha256']]:
        for path,digest in mapping.items():ensure(sha(path)==digest,'source/checkpoint drift')
    log=[json.loads(x) for x in (R/'execution.jsonl').read_text().splitlines()];ensure(any(e['event']=='complete' for e in log),'missing completion receipt');datasets={};labelmax=0.
    for split in p['data']:
        path=R/f'data-{split}.json';receipt=next(e for e in log if e['event']=='fresh_test_opened' and e['split']==split);ensure(receipt['sha256']==sha(path) and receipt['utc']>p['frozen_utc'],'data hash/chronology');cases=jread(path);ensure(len(cases)==128,'source task count')
        for c in cases:
            labels=independent_labels(c['records']);ensure(close(labels,c['queries'],1e-8),'independent source label discrepancy');labelmax=max(labelmax,max(abs(a['answer']-b['answer']) for a,b in zip(labels,c['queries'])));c['_labels']=labels
        datasets[split]=cases
    groups=[];pairs={};totals=collections.Counter();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S');detail=BASE/'audit'/f'reasoning-companion-{stamp}-episodes.jsonl'
    with detail.open('x') as stream:
        for recorded in m['rows']:
            seed=recorded['seed'];split=recorded['split'];name=recorded['condition'];folder=R/f'{split}-{name}-s{seed}';ensure(jread(folder/'metrics.json')==recorded,'local/final summary');path=folder/'predictions.jsonl';ensure(sha(path)==recorded['prediction_sha256'],'prediction hash');scores=[];costs=[];paired=[];count=0
            with path.open() as f:
                for index,line in enumerate(f):
                    ensure(index<128,'extra case');row=json.loads(line);case=datasets[split][index];ensure(row['score']['id']==case['id'] and row['seed']==seed and row['condition']['name']==name,'paired ID/group');result=row['result'];score=score_independent(case,result,case['_labels']);ensure(close(score,row['score'],1e-8),'independent score');scores.append(score);dbpath=folder/f'episode-{index:04d}.sqlite';ensure(Path(result['archive_path']).resolve()==dbpath.resolve(),'archive binding');stats=archive(dbpath,row,case,p);count+=1
                    for key in ['raw_records','observations','trace_count','reasoning_snapshots','constraint_snapshots','archive_bytes']:totals[key]+=stats[key]
                    keys=['seconds','total_refinements_executed','total_consumed_tokens_including_replay','reasoning_compactions','reasoning_rereads','reasoning_max_resident_traces','reasoning_generated_traces','reasoning_peak_active_serialized_bytes','reasoning_maximum_reload_difference','same_heading_reasoning_continuations'];cost={k:result[k] for k in keys};cost['archive_bytes']=stats['archive_bytes'];costs.append(cost)
                    paired.append(dict(id=case['id'],candidates=[(q.get('prediction'),q['status'],q['steps']) for q in result['outputs']],refinements=result['total_refinements_executed'],tokens=result['total_consumed_tokens_including_replay'],trace_sha256=stats['trace_sha256']));stream.write(json.dumps(dict(seed=seed,split=split,condition=name,id=case['id'],stats=stats))+'\n')
            ensure(count==128 and len(list(folder.glob('episode-*.sqlite')))==128,'INCOMPLETE group');computed=summarize(scores);ensure(close(computed,recorded,1e-8),'group recomputed metrics');ensure(close({k:mean([x[k] for x in costs]) for k in costs[0]},recorded['cost_mean'],1e-8),'cost means');ensure(close({k:max(x[k] for x in costs) for k in costs[0]},recorded['cost_max'],1e-8),'cost maxima');pairs[seed,split,name]=paired
            groups.append(dict(seed=seed,split=split,condition=name,**computed,certified_correct_yield=sum(s['final']['accepted'] and s['final']['source_correct'] for s in scores)/128,cost_mean=recorded['cost_mean'],cost_max=recorded['cost_max']));print(json.dumps(dict(group=f'{split}-{name}-s{seed}',status='PASS',seconds=time.perf_counter()-start)),flush=True)
    comparisons=[]
    for seed in p['seeds']:
        for split in p['data']:
            a=pairs[seed,split,'uncompacted'];b=pairs[seed,split,'window4'];ensure(a==b,'paired candidate/status/actual-trace divergence');comparisons.append(dict(seed=seed,split=split,episodes=128,candidates_status_budgets_tokens_identical=True,all_actual_generated_trace_payloads_identical=True))
    limitations=['Source/payload hashes, exact paired generated states and matching outputs independently verify observed fidelity. The complete runtime obligation fingerprint cannot be reconstructed solely from archived z; full KV/global-encoder/runtime state is not serialized. Guard completeness/order is supported by pinned source and prior mutation tests, not new full-runtime replay here.','Reread counters match full boundary snapshots and source calls restore before candidate/sufficiency; no separate runtime reread event receipt exists.','Active generated-history cap4 excludes model weights,encoder/KV,graph,live title state,source registry and reconstruction working memory. Serialized payload byte peaks are not measured process RAM.','Uncompacted active history resets at everyquery, so comparison is8 versus4 local records, not unlimited cross-query history.','Both arms certify only at4/8 boundaries; do not conflate these results with primary every-k adaptive stopping.','256 independent source tasks are shared across three seeds/two conditions;1536 executions are not1536 independent cases.','No producer code, frozen tests, model parameters or prior results were modified.']
    output=dict(status='PASS_WITH_SCOPE_LIMITS',protocol_sha256=PIN,executions=1536,shared_source_tasks=256,groups=groups,paired_comparisons=comparisons,archive_totals=dict(totals),max_source_label_discrepancy=labelmax,limitations=limitations,seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),detail_path=str(detail));target=BASE/'audit'/f'reasoning-companion-independent-{stamp}.json'
    with target.open('x') as f:json.dump(output,f,indent=2)
    lines=['# Independent reasoning-window companion audit','',f'Status **PASS_WITH_SCOPE_LIMITS**. All12 complete conditions and6 paired comparisons audited:1536DBs,256 shared tasks. Protocol `{PIN}`.','',f'All source/checkpoint/data/prediction/DB hashes and raw/observation/compact payload hashes match. Independently propagated source labels differ by at most{labelmax:.3g}. All recomputed conditional candidate/certificate metrics match. Archive totals: {dict(totals)}.','', 'All paired generated trace payloads are exactly identical, in addition to candidate values, certification statuses, budgets and consumed tokens. Every generated-state snapshot references exactly4 actual consecutive trace records, occurs at k4/8, stores the exact latest title latent state, preserves the queried heading identity, and has a guarded obligation digest. All enabled maxima are<=4; uncompacted maxima and serialized payload peaks independently reconstruct from raw traces. No raw trace was removed from observed append-only sequences.','', '| Seed | Split | Condition | Diagnostic accuracy | Certified correct yield | Max active generated traces | Mean snapshot/reread count | Mean archive bytes |','|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['seed']} | {g['split']} | {g['condition']} | {g['final']['accuracy_all_intended']:.4f} | {g['certified_correct_yield']:.4f} | {g['cost_max']['reasoning_max_resident_traces']} | {g['cost_mean']['reasoning_compactions']:.2f} | {g['cost_mean']['archive_bytes']:.0f} |")
    lines+=['','## Limits','']+[f'- {x}' for x in limitations]+['',f'Independent detailed output: `{target}`. Streaming episode evidence: `{detail}`.']
    report=BASE/'reports/reasoning-window-independent-audit.md'
    with report.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=output['status'],report=str(report),audit=str(target),seconds=output['seconds'])),flush=True)
if __name__=='__main__':main()
