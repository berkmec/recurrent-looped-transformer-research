"""Fresh, source-relative end-to-end evaluation. Labels never enter inference.

Independent linear algebra labels the rounded original constraints. Missing
queries remain in the intended denominator. Certificate scope remains parsed
constraints, while source-relative acceptance failures are measured separately.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time
import platform
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'integration'/'connected'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def check_sources(hashes):
    for path,expected in hashes.items():
        if sha(path)!=expected:raise ValueError('frozen source drift: '+str(path))


def assert_fresh_results(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError('partial or completed evidence exists: '+str(path))


def source_labels(records):
    """Independent full-rank linear system, not inference's graph verifier."""
    current={};labels=[]
    for position,event in enumerate(records):
        kind=event['kind'];child=event['child'];parent=event.get('parent')
        if kind!='query':
            key=('absolute' if parent is None else 'edge',child)
            current[key]=(child,parent,float(event['value']));continue
        names=sorted({x for c,p,v in current.values() for x in (c,p) if x is not None})
        if child not in names:raise ValueError('unobserved source query')
        mapping={name:i for i,name in enumerate(names)};matrix=[];rhs=[]
        for c,p,value in current.values():
            row=np.zeros(len(names));row[mapping[c]]=1.
            if p is not None:row[mapping[p]]-=1.
            matrix.append(row);rhs.append(value)
        a=np.array(matrix);b=np.array(rhs)
        solution,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
        if rank!=len(names) or np.max(np.abs(a@solution-b))>1e-8:
            raise ValueError('unanchored or inconsistent original source')
        labels.append({'source_index':position,'query':child,'answer':float(solution[mapping[child]])})
    return labels


def make_cases(count,seed,split):
    if count<1 or split not in ('iid','depth8'):raise ValueError('invalid case specification')
    rng=np.random.default_rng(seed);cases=[]
    for index in range(count):
        depth=8 if split=='depth8' else (1 if index%2==0 else 3)
        inverse=bool((index//2)%2) if split=='iid' else bool(index%2)
        affected=bool((index//4)%2) if split=='iid' else bool((index//2)%2)
        ids=rng.permutation(10);chain=list(ids[:depth+1]);other=list(ids[depth+1:])
        parents={chain[j]:chain[j-1] for j in range(1,len(chain))}
        parents.update({other[j]:other[j-1] for j in range(1,len(other))})
        def name(node):return f'x{50000+index*10+int(node)}'
        def numeric(x):return round(float(x),8)
        offsets={n:numeric(rng.uniform(-.35,.35)) for n in parents}
        known=chain[-1] if inverse else chain[0];requested=chain[0] if inverse else chain[-1]
        observations={known:numeric(rng.uniform(-.7,.7)),other[0]:numeric(rng.uniform(-.7,.7))}
        facts=[dict(kind='edge',child=name(c),parent=name(p),value=offsets[c]) for c,p in parents.items()]
        facts += [dict(kind='root',child=name(c),parent=None,value=v) for c,v in observations.items()]
        rng.shuffle(facts)
        query=dict(kind='query',child=name(requested),parent=None,value=None)
        changed=known if affected else other[0]
        replacement=numeric(observations[changed]+rng.choice([-1,1])*rng.uniform(.3,.8))
        revision=dict(kind='revision',child=name(changed),parent=None,value=replacement)
        # Text experiment contains observations and queries, with no invented
        # parser class for the training model's numeric no-write continuation.
        records=facts+[dict(query),revision,dict(query),dict(query)]
        cases.append({'id':f'{split}-{seed}-{index}','index':index,'split':split,'depth':depth,
                      'inverse':inverse,'affected':affected,'records':records,'queries':source_labels(records)})
    return cases


def render_case(case,style):
    from memory.api import render_event
    return [render_event(e,style,(case['index']+position)%6) for position,e in enumerate(case['records'])]


def event_exact(predicted,expected):
    if predicted is None:return False
    if any(predicted.get(k)!=expected.get(k) for k in ('kind','child','parent')):return False
    a,b=predicted.get('value'),expected.get('value')
    if a is None or b is None:return a is None and b is None
    return bool(math.isfinite(float(a)) and abs(float(a)-float(b))<=1e-12)


def score_episode(case,result,tolerance=.1):
    by_position={row['source_index']:row for row in result['outputs']}
    query_rows=[]
    for expected in case['queries']:
        actual=by_position.get(expected['source_index'],{})
        prediction=actual.get('prediction');emitted=isinstance(prediction,(float,int)) and math.isfinite(prediction)
        right_query=actual.get('query')==expected['query']
        error=abs(float(prediction)-expected['answer']) if emitted else None
        correct=bool(emitted and right_query and error<=tolerance)
        accepted=actual.get('status')=='VERIFIED_PARSED_CONSTRAINTS'
        query_rows.append({**expected,'prediction':prediction if emitted else None,
                           'status':actual.get('status','MISSING_QUERY_OUTPUT'),'query_matches':right_query,
                           'absolute_error':error,'squared_error':error*error if error is not None else None,
                           'source_correct':correct,'accepted':accepted,'accepted_source_failure':accepted and not correct,
                           'steps':actual.get('steps',0),'emitted':emitted})
    parsed=result.get('parsed',[])
    exact=[event_exact(row['events'][0] if row.get('valid') and len(row.get('events',[]))==1 else None,expected)
           for row,expected in zip(parsed,case['records'])]
    exact += [False]*(len(case['records'])-len(exact))
    intended_positions={q['source_index'] for q in case['queries']}
    spurious=[row for position,row in by_position.items() if position not in intended_positions]
    return {'id':case['id'],'affected':case['affected'],'inverse':case['inverse'],'depth':case['depth'],
            'queries':query_rows,'final':query_rows[-1],'source_event_exact':exact,
            'all_source_events_exact':all(exact),'spurious_query_outputs':len(spurious),
            'spurious_accepted_outputs':sum(row.get('status')=='VERIFIED_PARSED_CONSTRAINTS' for row in spurious)}


def summarize_scores(scores):
    from integration.audit_stats import _wilson
    if not scores:raise ValueError('empty scores')
    def summarize(rows,independent):
        n=len(rows);accepted=sum(r['accepted'] for r in rows);failed=sum(r['accepted_source_failure'] for r in rows)
        errors=[r['squared_error'] for r in rows if r['squared_error'] is not None]
        return {'intended_queries':n,'emitted_count':len(errors),'coverage':accepted/n,'accepted_count':accepted,
                'accepted_failures':failed,'accepted_error_rate':failed/accepted if accepted else None,
                'accepted_error_rate_ci95':_wilson(failed,accepted) if independent else None,
                'accuracy_all_intended':sum(r['source_correct'] for r in rows)/n,
                'mse_emitted':float(np.mean(errors)) if errors else None,
                'missing_outputs':n-len(errors),'mean_refinements_at_query':float(np.mean([r['steps'] for r in rows])),
                'interval_scope':'fixed-model independent final cases' if independent else 'correlated within episode; no binomial interval'}
    events=[e for s in scores for e in s['source_event_exact']]
    spurious_accepted=sum(s['spurious_accepted_outputs'] for s in scores)
    all_accepted=spurious_accepted+sum(q['accepted'] for s in scores for q in s['queries'])
    all_failed=spurious_accepted+sum(q['accepted_source_failure'] for s in scores for q in s['queries'])
    return {'episodes':len(scores),'source_event_exact_accuracy':sum(events)/len(events),
            'source_episode_exact_accuracy':sum(s['all_source_events_exact'] for s in scores)/len(scores),
            'spurious_query_outputs':sum(s['spurious_query_outputs'] for s in scores),
            'spurious_accepted_outputs':spurious_accepted,
            'all_certificate_outputs':{'count':all_accepted,'source_failures':all_failed,
                'source_failure_rate':all_failed/all_accepted if all_accepted else None,
                'scope':'all emitted certificates, including unsolicited/misparsed query positions as source alignment failures; correlated queries, no binomial interval'},
            'final':summarize([s['final'] for s in scores],True),
            'all_queries':summarize([q for s in scores for q in s['queries']],False)}


def freeze():
    assert_fresh_results(OUT);OUT.mkdir(parents=True,exist_ok=True)
    source_paths=['integration/evaluate_system.py','integration/pipeline.py','integration/audit_stats.py',
                  'learning/api.py','learning/model.py','learning/token_core.py','memory/api.py','memory/headings.py','memory/extractor.py']
    protocol={'version':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'goal':'Measure the actual connected learned-text, headings, archive, RLT feedback and halting pipeline on fresh original-source-relative cases.',
              'seeds':[17,43,91],'model_arm':'combined','checkpoint_step':600,
              'checkpoint_selection':'fixed protocol arm and final step; all three seeds; no held-out selection',
              'parser_pairing':'extractor seed equals RLT seed; joint pipeline seeds, not independent parser/model factorial',
              'datasets':{'iid':{'count':256,'seed':104731},'depth8':{'count':256,'seed':104738}},
              'conditions':[{'name':'adaptive-replay-seen','view':'replay','adaptive':True,'style':'seen'},
                            {'name':'fixed-replay-seen','view':'replay','adaptive':False,'style':'seen'},
                            {'name':'adaptive-compact-seen','view':'compact','adaptive':True,'style':'seen'},
                            {'name':'adaptive-replay-heldout','view':'replay','adaptive':True,'style':'heldout'}],
              'max_refinements_per_query':8,'active_record_capacity':4,'tolerance':.1,'numeric_margin':1e-6,
              'compaction_scope':'actual stable-heading-owned observed constraint boundaries; cross-heading raw ID chronology for replay; reset from compact constraints is a distinct memory policy',
              'labels':'independent full-rank linear least squares on original 8-decimal observations, never passed to inference',
              'primary_metrics':['all-intended final accuracy0.1','final emitted MSE with missing count','parsed-certificate coverage and original-source failure risk','actual refinements including replay','wall time and archive bytes'],
              'secondary_metrics':['source tuple/episode exactness','all3 queries with correlated denominator','affected/inverse subgroups','title count','compaction and reopen counts'],
              'pairing':'same512 source tasks and rephrasings shared across3 seeds/conditions; report3 seeds separately, paired example comparisons; never flatten seeds into independent samples',
              'limitations':['finite controlled English,10 entities,affine constraints','heldout templates known evaluation family from prior parser study; fresh numeric/entity examples','no no-write continuation sentences: observation/query-only transfer stream','first-appearance ID mapping instead of original randomized numeric labels','capacity bounds resident record IDs, not total RAM/reconstruction/model working state','SQLite raw title-state trace persists every actual generated refinement; no generic lossless natural-language reasoning proof','no semantic vector retrieval claim from keyed archive replay','thread contention: measured wall time is this local run, not isolated device efficiency'],
              'source_sha256':{str(ROOT/p):sha(ROOT/p) for p in source_paths},
              'upstream_protocol_sha256':{str(ROOT/p):sha(ROOT/p) for p in ['learning/protocol.json','learning/combined-protocol.json','memory/protocol.json']}}
    write_json(OUT/'protocol.json',protocol)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print(json.dumps({'frozen':str(OUT/'protocol.json'),'sha256':sha(OUT/'protocol.json')}),flush=True)


def main():
    import torch
    from learning.api import load_model
    from memory.api import load_parser
    from integration.pipeline import run_text_episode
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text())
    if sha(OUT/'protocol.json')!=(OUT/'protocol.sha256').read_text().strip():raise ValueError('protocol drift')
    check_sources(protocol['source_sha256']);check_sources(protocol['upstream_protocol_sha256'])
    checkpoints={s:(ROOT/'learning'/'checkpoints'/f'combined-{s}-step600.pt',ROOT/'memory'/'checkpoints'/f'extractor-{s}.pt') for s in protocol['seeds']}
    checkpoint_hashes={str(p):sha(p) for pair in checkpoints.values() for p in pair}
    result_dir=OUT/'results';assert_fresh_results(result_dir);result_dir.mkdir(parents=True,exist_ok=True)
    write_json(result_dir/'environment.json',{'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads(),'protocol_sha256':sha(OUT/'protocol.json'),'checkpoint_sha256':checkpoint_hashes})
    execution=(result_dir/'execution.jsonl').open('x',encoding='utf-8');started=time.perf_counter();summaries=[]
    def log(event,**fields):
        row={'event':event,'utc':datetime.now(timezone.utc).isoformat(),**fields}
        execution.write(json.dumps(row,allow_nan=False)+'\n');execution.flush();print(json.dumps(row),flush=True)
    datasets={}
    for split,spec in protocol['datasets'].items():
        cases=make_cases(spec['count'],spec['seed'],split);datasets[split]=cases
        write_json(result_dir/f'data-{split}.json',cases)
        log('fresh_test_first_open',split=split,path=str(result_dir/f'data-{split}.json'),sha256=sha(result_dir/f'data-{split}.json'))
    for seed in protocol['seeds']:
        model_path,parser_path=checkpoints[seed];model=load_model(model_path);parser=load_parser(parser_path)
        provenance={'model_path':str(model_path),'model_sha256':checkpoint_hashes[str(model_path)],'parser_path':str(parser_path),'parser_sha256':checkpoint_hashes[str(parser_path)],'protocol_sha256':sha(OUT/'protocol.json')}
        for split,cases in datasets.items():
            for condition in protocol['conditions']:
                label=f'{condition["name"]}-{split}-s{seed}';group=result_dir/label;group.mkdir()
                scores=[];costs=[];group_start=time.perf_counter()
                with (group/'predictions.jsonl').open('x',encoding='utf-8') as stream:
                    for case in cases:
                        db=group/f'episode-{case["index"]:04d}.sqlite'
                        result=run_text_episode(render_case(case,condition['style']),model,parser,db,task=case['id'],
                            capacity=protocol['active_record_capacity'],max_steps=protocol['max_refinements_per_query'],
                            view=condition['view'],adaptive=condition['adaptive'],tolerance=protocol['tolerance'],model_provenance=provenance)
                        score=score_episode(case,result,protocol['tolerance']);scores.append(score)
                        costs.append({k:result[k] for k in ('total_refinements_executed','total_consumed_tokens_including_replay','seconds','raw_records','compactions','premature_compactions','reopens','max_active_records','heading_count')})
                        costs[-1]['archive_bytes']=db.stat().st_size
                        stream.write(json.dumps({'seed':seed,'split':split,'condition':condition,'score':score,'result':result,'archive_sha256':sha(db)},allow_nan=False)+'\n');stream.flush()
                        if (case['index']+1)%64==0:log('condition_progress',condition=label,complete=case['index']+1,total=len(cases),seconds=time.perf_counter()-group_start)
                summary={'seed':seed,'split':split,'condition':condition,**summarize_scores(scores),
                         'costs_mean':{key:float(np.mean([r[key] for r in costs])) for key in costs[0]},
                         'costs_max':{key:max(r[key] for r in costs) for key in costs[0]},
                         'wall_seconds':time.perf_counter()-group_start,'predictions_sha256':sha(group/'predictions.jsonl')}
                for name,selector in [('affected',lambda s:s['affected']),('unaffected',lambda s:not s['affected']),('inverse',lambda s:s['inverse']),('forward',lambda s:not s['inverse'])]:
                    summary[name]=summarize_scores([s for s in scores if selector(s)])
                write_json(group/'metrics.json',summary);summaries.append(summary)
                log('condition_complete',condition=label,metrics=summary['final'],wall_seconds=summary['wall_seconds'])
    check_sources(protocol['source_sha256']);check_sources(protocol['upstream_protocol_sha256']);check_sources(checkpoint_hashes)
    write_json(result_dir/'metrics.json',{'protocol_sha256':sha(OUT/'protocol.json'),'rows':summaries,'wall_seconds':time.perf_counter()-started,'training_seeds':3,'independent_source_cases':512,'condition_seed_episode_runs':6144})
    log('complete',wall_seconds=time.perf_counter()-started);execution.close()


if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
