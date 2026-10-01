"""Independent read-only results auditor; writes only the review JSON."""
import hashlib
import json
from datetime import datetime
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); OUT=ROOT/'results/R1'; OLD=Path('E:/rlt-research/2026-09-16-e2e')
DEST=ROOT/'.superpowers/sdd/review-diagnostics-plan'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rat(x): return {'numerator':x.numerator,'denominator':x.denominator}
p=read(OUT/'protocol.json'); ph=sha(OUT/'protocol.json'); comp=read(OUT/'completion.json'); summary=read(OUT/'summary.json')
assert ph==(OUT/'protocol.sha256').read_text().strip()==comp['protocol_sha256']
for rel,h in p['source_pins'].items(): assert sha(ROOT/rel)==h,rel
for seed,h in p['checkpoint_pins'].items(): assert sha(OLD/'learning/checkpoints'/p['checkpoint'].format(seed=seed))==h
assert sha(OLD/'audit/final-manifest.json')==p['old_manifest_sha256']
expected={(s,k) for s in p['sources'] for k in p['model_seeds']}
assert len(comp['receipts'])==len(expected)==12
assert {(r['split'],r['seed']) for r in comp['receipts']}==expected
assert comp['independent_source_problems']==2048 and comp['model_problem_executions']==6144
oracle={}; source_info=[]
for split in p['sources']:
    data=np.load(OUT/f'data-{split}.npz'); opened=read(OUT/f'source-open-{split}.json')
    assert opened['source_sha256']==sha(OUT/f'data-{split}.npz')
    assert opened['source_seed']==p['sources'][split] and opened['protocol_sha256']==ph
    assert datetime.fromisoformat(p['frozen_utc'])<datetime.fromisoformat(opened['opened_utc'])
    parsed=[]
    for tape in data['events']:
        ev=[]; active={}
        for op,child,parent,value in tape:
            op=int(op); child=str(int(child)); parent=str(int(parent)); value=float(value)
            if op==6: continue
            kind={0:'root',1:'edge',2:'query',3:'revision',4:'revision'}[op]
            parent=parent if op in (1,4) else None
            ev.append(dict(kind=kind,child=child,parent=parent,value=None if op==2 else value))
            if op!=2: active[('absolute:' if parent is None else 'edge:')+child]=(child,parent,F(value))
        adj={}; anchors={}
        for facet,(v,u,d) in active.items():
            adj.setdefault(v,[])
            if u is None: anchors[v]=(d,facet)
            else:
                adj.setdefault(u,[]); adj[u].append((v,d,facet)); adj[v].append((u,-d,facet))
        truth={}; anchor_for={}; prev={}; components={}
        for anchor,(value,facet) in anchors.items():
            if anchor in truth:
                assert truth[anchor]==value
                continue
            truth[anchor]=value; anchor_for[anchor]=anchor; prev[anchor]=None; todo=[anchor]
            for v in todo:
                for u,d,key in adj[v]:
                    if u in truth: assert truth[u]==truth[v]+d
                    else:
                        truth[u]=truth[v]+d; anchor_for[u]=anchor; prev[u]=(v,key,d); todo.append(u)
            components[anchor]=set(todo)
        assert len(truth)==10
        q=ev[-1]['child']; a=anchor_for[q]; path=[]; v=q
        while v is not None:
            path.append(v); v=prev[v][0] if prev[v] is not None else None
        path=path[::-1]
        parsed.append((ev,active,truth,q,a,path,prev,components[a],anchors))
    oracle[split]=(data,parsed)
    source_info.append(dict(split=split,sha256=sha(OUT/f'data-{split}.npz'),opened_utc=opened['opened_utc']))
rows=[]; paired=[]; lazy=[]; checked=0; cert_count=0; max_bound_diff=0.; chain_exact_equal=0
for receipt in comp['receipts']:
    split,seed=receipt['split'],receipt['seed']; data,parsed=oracle[split]
    assert receipt==read(OUT/f'receipt-{split}-{seed}.json')
    assert receipt['protocol_sha256']==ph and receipt['source_sha256']==sha(OUT/f'data-{split}.npz')
    names={f'raw-{split}-{seed}.npz',f'certificates-{split}-{seed}.json',f'lazy-{split}-{seed}.json'}
    assert set(receipt['artifacts'])==names
    for rel,h in receipt['artifacts'].items(): assert sha(OUT/rel)==h
    raw=np.load(OUT/f'raw-{split}-{seed}.npz'); nodes=raw['nodes']
    assert nodes.shape==(512,8,10) and np.isfinite(nodes).all()
    for field in ['example_id','affected','inverse','depth']: assert np.array_equal(raw[field],data[field])
    correct=np.zeros((512,8),bool); accept=np.zeros((512,8,2,3),bool)
    certs=read(OUT/f'certificates-{split}-{seed}.json')
    cmap={(r['example_id'],r['round']):r['certificate'] for r in certs}
    assert len(cmap)==len(certs)==receipt['audited_certificates']
    expected_certs=set()
    for i,(ev,active,truth,q,a,path,prev,component,anchors) in enumerate(parsed):
        assert int(raw['query_ids'][i])==int(q) and raw['targets'][i]==float(truth[q])
        for k in range(8):
            vals={str(j):float(x) for j,x in enumerate(nodes[i,k])}; vf={v:F(x) for v,x in vals.items()}
            correct[i,k]=abs(vf[q]-truth[q])<=F(1,10)
            old=0.; exact_component=F(0)
            for facet,(v,u,d) in active.items():
                if v not in component: continue
                old+=abs(vals[v]-float(d)) if u is None else abs(vals[v]-vals[u]-float(d))
                exact_component+=abs(vf[v]-d-(vf[u] if u is not None else 0))
            first=abs(vf[a]-anchors[a][0]); total=first
            terms=[dict(facet=anchors[a][1],residual=rat(first))]
            for v in path[1:]:
                u,facet,d=prev[v]; residual=abs(vf[v]-vf[u]-d); total+=residual
                terms.append(dict(facet=facet,residual=rat(residual)))
            if split!='branched':
                assert total==exact_component
                chain_exact_equal+=1
            else: assert total<=exact_component
            assert raw['bounds'][i,k,0]==old
            assert raw['bounds'][i,k,1]==float(total)
            for t,lim in enumerate(p['thresholds']):
                accept[i,k,0,t]=old<=lim; accept[i,k,1,t]=total<=F(str(lim))
            if accept[i,k,1,2]:
                expected_certs.add((i,k+1)); assert correct[i,k]
                expected_cert=dict(version=1,query=q,path=path,terms=terms,bound=rat(total),threshold=rat(F('0.099999')),source_sha256=digest(ev),candidate_sha256=digest(vals))
                assert cmap[(i,k+1)]==expected_cert,(split,seed,i,k)
                cert_count+=1
            checked+=1
    assert set(cmap)==expected_certs
    assert np.array_equal(correct,raw['correct']) and np.array_equal(accept,raw['acceptance'])
    for j,method in enumerate(['component','path']):
        for policy in ['fixed8','adaptive']:
            selected=np.full(512,7,dtype=int)
            if policy=='adaptive':
                for i in range(512):
                    hits=np.flatnonzero(accept[i,:,j,2])
                    if len(hits): selected[i]=hits[0]
            aa=accept[np.arange(512),selected,j,2]; cc=correct[np.arange(512),selected]
            nacc=int(aa.sum()); ncorr=int(cc.sum()); tp=int((aa&cc).sum()); wrong=int((aa&~cc).sum())
            r=dict(split=split,seed=seed,method=method,policy=policy,n=512,accepted=nacc,correct=ncorr,accepted_correct=tp,accepted_wrong=wrong,coverage=nacc/512,accuracy=ncorr/512,recall=tp/ncorr if ncorr else None,selective_risk=wrong/nacc if nacc else None,mean_final_refinements=float((selected+1).mean()))
            saved=next(x for x in summary['rows'] if x['split']==split and x['seed']==seed and x['method']==method and x['policy']==policy and x['threshold']==p['primary_threshold'])
            for key,value in r.items(): assert saved[key]==value,(key,r,saved)
            rows.append(r)
    delta=accept[:,-1,1,2].astype(int)-accept[:,-1,0,2].astype(int)
    rng=np.random.default_rng(p['bootstrap_seed'])
    bs=delta[rng.integers(0,512,(p['bootstrap_repetitions'],512))].mean(axis=1)
    ci=np.quantile(bs,[.025,.975]).tolist()
    pr=dict(split=split,seed=seed,gained=int((delta>0).sum()),lost=int((delta<0).sum()),coverage_delta=float(delta.mean()),ci95=ci)
    sr=next(x for x in summary['paired_primary'] if x['split']==split and x['seed']==seed)
    assert sr['gained']==pr['gained'] and sr['lost']==pr['lost']
    assert sr['path_minus_component_coverage']==pr['coverage_delta'] and sr['source_paired_bootstrap95']==ci
    paired.append(pr)
    lazyrows=read(OUT/f'lazy-{split}-{seed}.json'); assert len(lazyrows)==32
    assert {(x['example_id'],x['method']) for x in lazyrows}=={(i,m) for i in range(16) for m in ['component','path']}
    for r in lazyrows:
        i=r['example_id']; j=['component','path'].index(r['method'])
        hits=np.flatnonzero(accept[i,:,j,2]); chosen=int(hits[0]) if len(hits) else 7
        assert r['actual_refinements']==chosen+1 and r['prior_refinements']==12
        assert r['accepted']==bool(accept[i,chosen,j,2])
        diff=abs(r['bound']-raw['bounds'][i,chosen,j]); max_bound_diff=max(max_bound_diff,diff)
    lm=dict(split=split,seed=seed,rows=32,halt_matches=32)
    for method in ['component','path']:
        lm[method+'_mean']=float(np.mean([r['actual_refinements'] for r in lazyrows if r['method']==method]))
    lazy.append(lm)
    print(json.dumps({'audited':f'{split}-{seed}','certificates':len(certs),'gained':pr['gained'],'lost':pr['lost']}),flush=True)
result=dict(status='PASS',protocol_sha256=ph,source_pins=p['source_pins'],sources=source_info,conditions=12,independent_source_problems=2048,model_problem_executions=6144,checked_candidate_rounds=checked,checked_acceptance_booleans=checked*6,independently_recomputed_certificates=cert_count,chain_exact_sum_equalities=chain_exact_equal,primary_rows=rows,paired=paired,lazy_receipts=lazy,lazy_receipt_trace_halt_matches=384,max_lazy_vs_batched_bound_abs_difference=max_bound_diff,limitations=['No seed or round pooling; aggregate counts are audit workload only.','This phase validates saved inference outputs and lazy receipts; live replay is separate.'])
(DEST/'r1-main-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print('AUDIT_TOTAL',json.dumps({k:result[k] for k in ['status','checked_candidate_rounds','checked_acceptance_booleans','independently_recomputed_certificates','chain_exact_sum_equalities','max_lazy_vs_batched_bound_abs_difference']}))
