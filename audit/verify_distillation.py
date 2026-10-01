"""Read-only independent distillation audit plus separately frozen initial reference."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,datetime,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
BASE=Path(__file__).resolve().parents[1];D=BASE/'learning/distillation';L=BASE/'learning';R=D/'results'
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jr(p):return json.loads(Path(p).read_text())
def ar(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def measure(pred,target,affected):
    p=pred.astype(float);t=target.astype(float);e=(p-t)**2
    return dict(mse=float(e.mean()),final_mse=float(e[:,-1].mean()),affected_final_mse=float(e[affected,-1].mean()),unaffected_final_mse=float(e[~affected,-1].mean()),delta_mse=float(((p[:,1]-p[:,0])-(t[:,1]-t[:,0])).__pow__(2).mean()),accuracy_0_1=float((abs(p[:,-1]-t[:,-1])<=.1).mean()))
def uncertainty(diff,reps=2000,rngseed=170043):
    diff=np.asarray(diff,float);per=diff.mean(1);mu=float(per.mean());se=float(per.std(ddof=1)/math.sqrt(3));rng=np.random.default_rng(rngseed);boot=[]
    for _ in range(reps):
        rows=rng.integers(3,size=3);cols=rng.integers(diff.shape[1],size=diff.shape[1]);boot.append(float(diff[np.ix_(rows,cols)].mean()))
    return dict(mean_candidate_minus_reference=mu,per_seed_candidate_minus_reference=per.tolist(),seed_t95=[mu-4.30265273*se,mu+4.30265273*se],crossed_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_reps=reps,bootstrap_seed=rngseed,unit='candidate minus reference squared final error; negative is improvement; common examples shared across resampled seed rows')
def facets(tape):
    current={}
    for op,c,p,v in tape:
        op=int(op);c=int(c);p=int(p)
        if op in [0,3]:current[('absolute',c)]=(c,None,float(v))
        elif op in [1,4]:current[('edge',c)]=(c,p,float(v))
    return current
def component(current,query):
    adj=collections.defaultdict(set)
    for c,p,v in current.values():
        adj[c]
        if p is not None:adj[c].add(p);adj[p].add(c)
    seen={query};queue=[query]
    for n in queue:
        for other in adj[n]:
            if other not in seen:seen.add(other);queue.append(other)
    return seen
def residual(current,candidate,query):
    nodes=component(current,query);anchors=[c for c,p,v in current.values() if p is None and c in nodes]
    if not anchors:return math.inf
    return sum(abs(float(candidate[c])-v) if p is None else abs(float(candidate[c])-float(candidate[p])-v) for c,p,v in current.values() if c in nodes)
def main():
    start=time.perf_counter();checks=[];fails=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        if not condition:fails.append(name)
    p=jr(D/'protocol.json');m=jr(R/'metrics.json');closure=jr(R/'closure.json');events=[json.loads(x) for x in (D/'execution.jsonl').read_text().splitlines()]
    check('protocol_sha',sha(D/'protocol.json')==(D/'protocol.sha256').read_text().strip()==m['protocol_sha256']);check('six_students_twelve_predictions',len(m['runs'])==6 and len(m['rows'])==12)
    for filename,digest in p['sources'].items():check('source:'+filename,sha(D/filename)==digest)
    for filename,digest in p['external_sources'].items():check('external:'+filename,sha(BASE/filename)==digest)
    for filename,digest in p['checkpoints'].items():check('checkpoint:'+filename,sha(L/'checkpoints'/filename)==digest)
    dep=jr(D/'dependency-receipt.json');check('dependency_sidecar_sha',sha(D/'dependency-receipt.json')==(D/'dependency-receipt.sha256').read_text().strip());check('dependency_protocol_pins',dep['distillation_protocol_sha256']==sha(D/'protocol.json') and dep['primary_protocol_sha256']==sha(L/'protocol.json'))
    for filename,digest in dep['files'].items():check('transitive:'+filename,sha(BASE/filename)==digest)
    train=ar(L/'results/data-train.npz');check('source_train_sha',sha(L/'results/data-train.npz')==p['train_source_sha256']);source=train['events'][:2048];archive_rows=[];initial_hashes={};sample_replay=[]
    from learning.api import load_model
    for seed in p['seeds']:
        receipt=jr(D/'archives'/f'receipt-{seed}.json');raw=ar(D/'archives'/f'teacher-{seed}.npz');selection=ar(D/'archives'/f'selection-{seed}.npz');decisions=[json.loads(x) for x in (D/'archives'/f'decisions-{seed}.jsonl').read_text().splitlines()]
        for file,key in [(f'teacher-{seed}.npz','archive_sha256'),(f'selection-{seed}.npz','selection_sha256'),(f'decisions-{seed}.jsonl','decisions_sha256')]:check('archivehash:'+file,sha(D/'archives'/file)==receipt[key])
        check('TRAIN_only_observations_'+str(seed),np.array_equal(raw['events'],source) and np.array_equal(raw['example_id'],np.arange(2048)) and 'targets' not in raw and 'node_targets' not in raw)
        accepted=np.zeros((2048,3),bool);full=np.zeros_like(accepted);masks=np.zeros((2048,3,10),np.float32);max_bound_difference=0.;decision_count=0
        for i,tape in enumerate(raw['events']):
            qs=np.where(tape[:,0]==2)[0]
            for q,pos in enumerate(qs):
                current=facets(tape[:pos+1]);query=int(tape[pos,1]);nodes=raw['nodes'][i,q,-1];reach=component(current,query);mask=np.zeros(10,np.float32);mask[list(reach)]=1.;masks[i,q]=mask;bound=residual(current,nodes,query);other=next(n for n in range(10) if n not in reach);otherbound=residual(current,nodes,other);accepted[i,q]=bound<=.05;full[i,q]=accepted[i,q] and otherbound<=.05;d=decisions[decision_count];decision_count+=1
                max_bound_difference=max(max_bound_difference,abs(bound-d['query_check']['bound']),abs(otherbound-d['other_component_check']['bound']))
                if d['query_check']['accepted']!=bool(accepted[i,q]) or d['other_component_check']['accepted']!=bool(otherbound<=.05):fails.append('decision:'+str(seed)+':'+str(i)+':'+str(q))
                if raw['final'][i,q]!=nodes[query]:fails.append('teacher_readout:'+str(seed))
        check('all_observation_only_certificates_'+str(seed),np.array_equal(accepted,raw['accepted']) and np.array_equal(full,raw['full_verified']) and max_bound_difference<1e-12,dict(decisions=decision_count,max_bound_difference=max_bound_difference));check('topology_only_masks_'+str(seed),np.array_equal(masks,raw['component_mask']))
        fully=full.all(1);strata=np.arange(2048)%8;counts=[int((fully&(strata==j)).sum()) for j in range(8)];use_full=int(fully.sum())>=128 and min(counts)>=16;qm=np.repeat(fully[:,None],3,axis=1) if use_full else accepted;nm=np.repeat(qm[:,:,None],10,axis=2).astype(np.float32) if use_full else masks*qm[:,:,None];eligible=np.where(qm.any(1))[0]
        check('selection_derivation_'+str(seed),np.array_equal(qm,selection['query_mask']) and np.array_equal(nm,selection['node_mask']) and np.array_equal(eligible,selection['eligible']) and len(eligible)==receipt['eligible_episodes'] and len(eligible)>=64)
        batch=ar(R/f'batches-{seed}.npz')['indices'];expected=np.random.default_rng(seed+27000).choice(eligible,(300,16),replace=True);check('exact300x16_paired_batches_'+str(seed),np.array_equal(batch,expected) and np.isin(batch,eligible).all())
        initial=torch.load(L/'checkpoints'/f'feedback-{seed}-step600.pt',map_location='cpu',weights_only=False)['state_dict'];initial_hashes[seed]=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        # Bounded independent generation check on16 fixed TRAIN-only tapes per
        # teacher. Existing producer receipt separately reports full6144 replay.
        teacher=load_model(L/'checkpoints'/f'combined-{seed}-step600.pt');max_replay=0.
        with torch.no_grad():
            runtime=teacher.new_runtime(torch.from_numpy(raw['events'][:16]));q=0
            while runtime.consume_to_query():
                for k in range(6):
                    candidate=runtime.refine_one();max_replay=max(max_replay,float(abs(runtime.z.numpy()-raw['states'][:16,q,k]).max()),float(abs(candidate['nodes'].numpy()-raw['nodes'][:16,q,k]).max()))
                max_replay=max(max_replay,float(abs(candidate['prediction'].numpy()-raw['final'][:16,q]).max()));runtime.commit_feedback();q+=1
        check('independent16_episode_actual_teacher_replay_'+str(seed),max_replay==0.,dict(episodes=16,max_difference=max_replay));sample_replay.append(dict(seed=seed,episodes=16,max_difference=max_replay));archive_rows.append(dict(seed=seed,eligible_episodes=len(eligible),query_coverage=float(qm.mean()),node_coverage=float(nm.mean()),full_episodes=int(fully.sum()),selection='full' if use_full else 'queried_component_fallback'))
        del raw,teacher,initial
    for run in m['runs']:
        arm=run['arm'];seed=run['seed'];loss=ar(R/f'loss-{arm}-{seed}.npz')['losses'];ckpath=D/'checkpoints'/f'{arm}-{seed}-step300.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);finalhash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in ck['state_dict'].values())).hexdigest();progress=[e['step'] for e in events if e['event']=='student_training_progress' and e['seed']==seed and e['arm']==arm]
        check('300_updates_evidence:'+arm+str(seed),loss.shape==(300,2) and np.isfinite(loss).all() and ck['step']==300 and run['examples_seen']==4800 and progress==[100,200,300] and finalhash!=initial_hashes[seed]);check('paired_initial_state:'+arm+str(seed),run['initial_tensor_sha256']==initial_hashes[seed]);check('student_checkpoint_sha:'+arm+str(seed),sha(ckpath)==run['checkpoint_sha256'])
    freeze=next(e['utc'] for e in events if e['event']=='protocol_frozen');first_collection=min(e['utc'] for e in events if e['event']=='teacher_archive_closed');first_training=min(e['utc'] for e in events if e['event']=='student_training_start');last_training=max(e['utc'] for e in events if e['event']=='student_training_complete');firsttest=min(e['utc'] for e in events if e['event']=='heldout_first_open');selection_receipt=next(e for e in events if e['event']=='all_student_selection_fixed_before_test')
    check('chronology',freeze<first_collection<dep['utc']<first_training<=last_training<selection_receipt['utc']<firsttest);check('training_receipt_sha',selection_receipt['training_sha256']==sha(R/'training.json'))
    data={split:ar(R/f'data-{split}.npz') for split in ['iid','depth8']};pred={};metric_rows=[];maxdiff=0.
    for row in m['rows']:
        a=ar(row['raw_path']);split=row['split'];key=(row['arm'],row['seed'],split);pred[key]=a['predictions'];check('prediction_sha:'+Path(row['raw_path']).name,sha(row['raw_path'])==row['raw_sha256']);check('pairing:'+str(key),all(np.array_equal(a[k],data[split][k]) for k in ['targets','affected','inverse','example_id']))
        calculated=measure(a['predictions'],a['targets'],a['affected'])
        for name,v in row['metrics'].items():
            delta=abs(calculated['accuracy_0_1' if name=='accuracy_0.1' else name]-v);maxdiff=max(maxdiff,delta)
            if delta>2e-7:fails.append('metric:'+str(key)+name)
        metric_rows.append(dict(arm=key[0],seed=key[1],split=split,metrics=calculated))
    check('12_metric_recomputations',not any(x.startswith('metric:') for x in fails),dict(max_float64_float32_difference=maxdiff))
    paired=[]
    for split in data:
        target=data[split]['targets'][:,-1].astype(float);diff=np.stack([(pred['trajectory',s,split][:,-1].astype(float)-target)**2-(pred['answer_only',s,split][:,-1].astype(float)-target)**2 for s in [17,43,91]]);stat=uncertainty(diff);paired.append(dict(split=split,**stat));recorded=next(x for x in m['comparisons'] if x['split']==split)
        check('reported_seed_effects:'+split,np.max(abs(np.array(stat['per_seed_candidate_minus_reference'])-recorded['per_seed_differences']))<2e-7)
    export=D/'inference-only-export';off=jr(R/'archive-off-receipt.json');export_pred=ar(export/'predictions.npz');check('archive_off_prediction_hash',sha(export/'predictions.npz')==off['predictions_sha256']);check('archive_off_inputs_observations_only',np.array_equal(np.load(export/'events.npy'),data['iid']['events']) and not any(x.name in ['targets.npy','data.py','trace.py'] for x in export.rglob('*')))
    for name in off['models']:
        arm,seed,_=name.removesuffix('.pt').split('-');check('archive_off_actual_student:'+name,sha(export/name)==sha(D/'checkpoints'/name) and np.array_equal(export_pred[name],pred[arm,int(seed),'iid']) and export_pred[name].shape==(512,3))
    check('archive_off_isolation_receipt',off['isolated_import_path'] and not off['oracle_importable'] and not off['trace_package_importable'] and len(off['models'])==6 and off['examples_per_model']==512)
    for name in ['model.py','api.py','token_core.py','__init__.py']:check('export_source:'+name,sha(export/'learning'/name)==sha(L/name))
    # Separately declared post-hoc descriptive reference, not a third original arm.
    refdir=BASE/'audit/distillation-initial-reference';refdir.mkdir(exist_ok=False)
    reference_protocol=dict(scope='post-hoc descriptive initial checkpoint reference authorized after primary distillation results; not preregistered third arm',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=[17,43,91],splits=['iid','depth8'],selection='all six cells; no training, tuning or checkpoint selection',initial='frozen feedback-step600',rounds=6,datasets={s:sha(R/f'data-{s}.npz') for s in data},checkpoints={str(L/'checkpoints'/f'feedback-{s}-step600.pt'):sha(L/'checkpoints'/f'feedback-{s}-step600.pt') for s in [17,43,91]},model_api_sha256=sha(L/'api.py'),audit_source_sha256=sha(Path(__file__)))
    (refdir/'protocol.json').write_text(json.dumps(reference_protocol,indent=2));(refdir/'protocol.sha256').write_text(sha(refdir/'protocol.json'));initial_predictions={};reference_rows=[]
    for seed in [17,43,91]:
        model=load_model(L/'checkpoints'/f'feedback-{seed}-step600.pt')
        for split,d in data.items():
            chunks=[]
            with torch.no_grad():
                for index in range(0,512,64):chunks.append(model(torch.from_numpy(d['events'][index:index+64]))['predictions'].numpy())
            values=np.concatenate(chunks);initial_predictions[seed,split]=values;path=refdir/f'pred-initial-{seed}-{split}.npz';np.savez_compressed(path,predictions=values,targets=d['targets'],example_id=d['example_id'],affected=d['affected'],inverse=d['inverse']);reference_rows.append(dict(seed=seed,split=split,metrics=measure(values,d['targets'],d['affected']),prediction_sha256=sha(path)))
    ref_effects=[]
    for split,d in data.items():
        target=d['targets'][:,-1].astype(float)
        for arm in ['answer_only','trajectory']:
            diff=np.stack([(pred[arm,s,split][:,-1].astype(float)-target)**2-(initial_predictions[s,split][:,-1].astype(float)-target)**2 for s in [17,43,91]]);ref_effects.append(dict(split=split,arm=arm,reference='initial_feedback600',**uncertainty(diff)))
    (refdir/'metrics.json').write_text(json.dumps(dict(protocol_sha256=sha(refdir/'protocol.json'),rows=reference_rows,paired_effects=ref_effects),indent=2))
    limitations=['The original two-arm comparison establishes trajectory versus answer-only adaptation, not improvement over pre-adaptation weights. The separately frozen initial-reference analysis here is explicitly post hoc.','300 updates per student are supported by frozen loop,300 finite loss rows,100/200/300 progress receipts,4800 paired batch entries and changed final weights; optimizer step counters are not persisted, so no independent training replay is claimed.','Final queried-component certificate uses observed constraints only. Intermediate states are genuine generated trajectories but not all certified correct. Teacher weights previously received oracle intermediate supervision.','All18432 query certificate decisions and masks independently recomputed. Actual teacher state reproduction independently sampled16 fixed TRAIN episodes per seed; producer full6144 replay receipt is a separate result.','Three seed t95 intervals are the primary original uncertainty summary. Crossed percentile bootstrap resamples three seeds and shared512 example columns and is descriptive; it can exclude0 while df2 t95 includes0.','Archive-off verifies six actual student models and512 observation-only inputs each against exported prediction artifacts; no OS filesystem sandbox or rerun by this audit.','Shared row IDs require split namespace; generated archives retained rejected examples; selection is strongly shallow-component biased.']
    output=dict(status='PASS_WITH_SCOPE_LIMITS' if not fails else 'FAIL',checks=checks,failures=fails,archive_selection=archive_rows,independent_teacher_replay=sample_replay,recomputed_metrics=metric_rows,paired_effects=paired,posthoc_initial_reference=dict(path=str(refdir),rows=reference_rows,paired_effects=ref_effects),limitations=limitations,wall_seconds=time.perf_counter()-start)
    (BASE/'audit/distillation-independent-audit.json').write_text(json.dumps(output,indent=2));lines=['# Independent actual-trajectory distillation audit','',f"Status **{output['status']}**; {sum(x['passed'] for x in checks)}/{len(checks)} checks pass. Failures: {fails}.",'','Frozen protocol/source/checkpoint and transitive dependency pins match. TRAIN-only collection precedes student training; all six300-update student runs and selection receipts precede first fresh heldout opening. Both arms use the exact regenerated300×16 index batches and source-feedback initialization tensor hash for each seed. Twelve raw prediction SHA values, target/mask/ID pairing and metrics match independent recomputation.','', 'All18432 final-query certificate decisions and topology masks were independently recomputed using observed graph residuals; all three seeds use the predeclared queried-component fallback. No full episode meets the all-component certificate. Six archive-off student checkpoints and6×512 prediction vectors match their actual heldout outputs exactly.','', '## Original paired effect','', '| Split | Trajectory minus answer-only final MSE | Seed-t95 | Crossed bootstrap95 |','|---|---:|---|---|']
    for e in paired:lines.append(f"| {e['split']} | {e['mean_candidate_minus_reference']:.6f} | {e['seed_t95']} | {e['crossed_bootstrap95']} |")
    lines+=['','Positive means worse. IID is worse and depth8 better in all three seeds, while both original seed-t95 intervals include0. This supports MIXED, not a general improvement claim.','', '## Separate post-hoc pre-adaptation reference','', 'The original protocol has no heldout initial-model reference. Before additional inference this audit saved a new descriptive protocol, then evaluated all three frozen feedback600 initial checkpoints on the same existing IID/depth8 observations, six cells total. There was no training, selection or tuning. This is not a preregistered third arm.','', '| Split | Adaptation vs initial feedback600 | Candidate-minus-initial MSE | Per-seed differences | Seed-t95 |','|---|---|---:|---|---|']
    for e in ref_effects:lines.append(f"| {e['split']} | {e['arm']} | {e['mean_candidate_minus_reference']:.6f} | {e['per_seed_candidate_minus_reference']} | {e['seed_t95']} |")
    lines+=['','## Limits','']+[f'- {x}' for x in limitations]+['',f'Post-hoc raw predictions/protocol: `{refdir}`. Full audit: `audit/distillation-independent-audit.json`.']
    (BASE/'reports/distillation-independent-audit.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status=output['status'],checks=len(checks),failures=fails,reference_effects=ref_effects,seconds=output['wall_seconds']),indent=2))
if __name__=='__main__':main()
