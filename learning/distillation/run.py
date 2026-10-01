"""Prespecified later training on actual archived successful teacher trajectories."""
from pathlib import Path
import datetime,hashlib,json,time,sys
import numpy as np
import torch
from learning.api import load_model
from learning.safe_run import assert_protocol_integrity
from .trace import load_training_events,write_archive,certify,observed_records,component_mask,trace_forward,losses

HERE=Path(__file__).parent;PARENT=HERE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(event,**kw):
    with (HERE/'execution.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw})+'\n')
def newjson(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
def verify():
    p=json.loads((HERE/'protocol.json').read_text())
    if sha(HERE/'protocol.json')!=(HERE/'protocol.sha256').read_text().strip():raise ValueError('protocol drift')
    for filename,digest in p['sources'].items():
        if sha(HERE/filename)!=digest:raise ValueError('source drift '+filename)
    for filename,digest in p['external_sources'].items():
        if sha(PARENT.parent/filename)!=digest:raise ValueError('external source drift '+filename)
    for filename,digest in p['checkpoints'].items():
        if sha(PARENT/'checkpoints'/filename)!=digest:raise ValueError('checkpoint drift '+filename)
    return p


def freeze():
    primary=assert_protocol_integrity(PARENT)
    p={'version':1,'primary_protocol_sha256':primary,'seeds':[17,43,91],
       'teacher':'fixed combined-seed-step600; actual model-generated z and numeric candidate states; no label repair',
       'student_initial':'existing feedback-seed-step600, identical for each paired student arm',
       'train_source':'../results/data-train.npz','train_source_sha256':sha(PARENT/'results'/'data-train.npz'),
       'collection_count':2048,'collection_selection':'first2048 TRAIN tapes, all raw successes/failures retained','rounds':6,
       'tolerance':.05,'certifier':'memory.api.verify_candidate component residual sum on observed conditions only',
       'full_episode_rule':'all3 query states certify both observed connected components; require>=128 episodes and>=16 per input-index modulo8 stratum',
       'fallback':'certified queried-component states only; query mask and topology-only node mask. Require>=64 episodes with>=1 certified query; otherwise report unsupported and do not train that seed.',
       'arms':['answer_only','trajectory'],'steps':300,'batch_size':16,'optimizer':{'name':'AdamW','lr':.0005,'weight_decay':.0001,'clip':1.},
       'final_loss':'MSE to SAME actual certified teacher final query output in BOTH arms',
       'trajectory_weight':.1,'trajectory_loss':'mean hidden z MSE + mean numeric all-round candidate MSE, masked to finally certified nodes/queries; no oracle auxiliary target',
       'latent_scope':'architecture-specific teacher state/trajectory distillation; not universal thinking-style learning',
       'evaluation':{'iid':{'count':512,'seed':124331},'depth8':{'count':512,'seed':124338}},
       'selection':'fixed300 updates; no heldout-based selection; no tolerance widening; training-only certificate filtering',
       'uncertainty':'paired per-seed final-MSE differences, df2 t95 and common-example/seed hierarchical bootstrap1000',
       'archive_off':'new subprocess isolated inference package, no trace or generator module importable',
       'device':'cpu','threads':1,'sources':{f.name:sha(f) for f in HERE.glob('*.py')},
       'external_sources':{name:sha(PARENT.parent/name) for name in ['learning/model.py','learning/token_core.py','learning/api.py','memory/api.py','learning/data.py','learning/structural_extra.py']},
       'checkpoints':{f'{arm}-{s}-step600.pt':sha(PARENT/'checkpoints'/f'{arm}-{s}-step600.pt') for s in (17,43,91) for arm in ('combined','feedback')}}
    newjson(HERE/'protocol.json',p);(HERE/'protocol.sha256').write_text(sha(HERE/'protocol.json')+'\n');emit('protocol_frozen',sha256=sha(HERE/'protocol.json'),protocol=p);print(sha(HERE/'protocol.json'))


def collect():
    torch.set_num_threads(1);p=verify()
    (HERE/'archives').mkdir(exist_ok=True);(HERE/'results').mkdir(exist_ok=True);(HERE/'checkpoints').mkdir(exist_ok=True)
    events=load_training_events(PARENT/'results'/'data-train.npz',p['train_source_sha256'],'train',p['collection_count'])
    for seed in p['seeds']:
        target=HERE/'archives'/f'teacher-{seed}.npz'
        if target.exists():raise FileExistsError('teacher archive exists; no overwrite')
        model=load_model(PARENT/'checkpoints'/f'combined-{seed}-step600.pt');t0=time.perf_counter();captured={k:[] for k in ('states','nodes','final')}
        with torch.no_grad():
            for start in range(0,len(events),64):
                result=trace_forward(model,torch.from_numpy(events[start:start+64]),p['rounds'])
                for key in captured:captured[key].append(result[key].numpy())
        arrays={key:np.concatenate(parts) for key,parts in captured.items()};arrays['events']=events
        accepted=np.zeros((len(events),3),bool);full=np.zeros_like(accepted);masks=np.zeros((len(events),3,10),np.float32);decisions=[]
        for index,tape in enumerate(events):
            positions=np.where(tape[:,0]==2)[0]
            for q,t in enumerate(positions):
                records=observed_records(tape[:t+1]);query=str(int(tape[t,1]));candidate={str(n):float(value) for n,value in enumerate(arrays['nodes'][index,q,-1])}
                check=certify(candidate,records,query,p['tolerance']);mask=component_mask(records,query);masks[index,q]=mask;accepted[index,q]=check['accepted']
                other=np.where(mask==0)[0];second=certify(candidate,records,str(int(other[0])),p['tolerance']) if len(other) else check
                full[index,q]=check['accepted'] and second['accepted']
                decisions.append({'example_id':index,'query':q,'token_t':int(t+1),'refinement_k':6,'query_node':query,'query_check':check,'other_component_check':second,'component_mask':mask.tolist()})
        arrays.update(accepted=accepted,full_verified=full,component_mask=masks,example_id=np.arange(len(events)),stratum=np.arange(len(events))%8)
        write_archive(target,arrays)
        with (HERE/'archives'/f'decisions-{seed}.jsonl').open('x',encoding='utf8') as stream:
            for decision in decisions:stream.write(json.dumps(decision)+'\n')
        fully=full.all(1);strata=np.arange(len(events))%8;full_counts=[int((fully&(strata==j)).sum()) for j in range(8)]
        use_full=int(fully.sum())>=128 and min(full_counts)>=16
        query_mask=np.repeat(fully[:,None],3,axis=1) if use_full else accepted
        node_mask=np.repeat(query_mask[:,:,None],10,axis=2).astype(np.float32) if use_full else masks*query_mask[:,:,None]
        eligible=np.where(query_mask.any(1))[0];eligible_counts=[int((query_mask.any(1)&(strata==j)).sum()) for j in range(8)]
        write_archive(HERE/'archives'/f'selection-{seed}.npz',{'eligible':eligible,'query_mask':query_mask.astype(np.float32),'node_mask':node_mask})
        receipt={'seed':seed,'mode':'full_episode' if use_full else 'queried_component_fallback','support':len(eligible)>=64,
                 'full_episodes':int(fully.sum()),'full_strata':full_counts,'eligible_episodes':len(eligible),'eligible_strata':eligible_counts,
                 'certified_query_states':int(query_mask.sum()),'query_coverage':float(query_mask.mean()),'certified_node_coverage':float(node_mask.mean()),
                 'archive_sha256':sha(target),'decisions_sha256':sha(HERE/'archives'/f'decisions-{seed}.jsonl'),'selection_sha256':sha(HERE/'archives'/f'selection-{seed}.npz'),
                 'teacher_checkpoint_sha256':p['checkpoints'][f'combined-{seed}-step600.pt'],'wall_seconds':time.perf_counter()-t0}
        newjson(HERE/'archives'/f'receipt-{seed}.json',receipt);emit('teacher_archive_closed',**receipt);print(json.dumps(receipt),flush=True)


def train():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);p=verify()
    if list((HERE/'checkpoints').glob('*.pt')) or (HERE/'results'/'metrics.json').exists():raise FileExistsError('student output exists')
    t_all=time.perf_counter();runs=[]
    for seed in p['seeds']:
        receipt=json.loads((HERE/'archives'/f'receipt-{seed}.json').read_text())
        if not receipt['support']:emit('student_skipped_insufficient_certificates',seed=seed);continue
        archive=HERE/'archives'/f'teacher-{seed}.npz';selection=HERE/'archives'/f'selection-{seed}.npz'
        if sha(archive)!=receipt['archive_sha256'] or sha(selection)!=receipt['selection_sha256']:raise ValueError('immutable trace changed')
        raw=np.load(archive);selected=np.load(selection);eligible=selected['eligible']
        x=torch.from_numpy(raw['events']);teacher={key:torch.from_numpy(raw[key]) for key in ('states','nodes','final')}
        qm=torch.from_numpy(selected['query_mask']);nm=torch.from_numpy(selected['node_mask'])
        batches=np.random.default_rng(seed+27000).choice(eligible,(p['steps'],p['batch_size']),replace=True)
        write_archive(HERE/'results'/f'batches-{seed}.npz',{'indices':batches})
        for arm in p['arms']:
            model=load_model(PARENT/'checkpoints'/f'feedback-{seed}-step600.pt');model.train()
            init=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);t0=time.perf_counter();curve=[]
            emit('student_training_start',arm=arm,seed=seed,initial_tensor_sha256=init,teacher_archive_sha256=receipt['archive_sha256'])
            for step,index in enumerate(batches,1):
                optimizer.zero_grad();out=trace_forward(model,x[index],6);target={k:v[index] for k,v in teacher.items()}
                final,trajectory=losses(out,target,qm[index],nm[index]);loss=final+(.1*trajectory if arm=='trajectory' else 0)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite student loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                curve.append([float(final.detach()),float(trajectory.detach())])
                if step%100==0:emit('student_training_progress',arm=arm,seed=seed,step=step,final_loss=curve[-1][0],trajectory_loss=curve[-1][1]);print(json.dumps({'arm':arm,'seed':seed,'step':step}),flush=True)
            checkpoint=HERE/'checkpoints'/f'{arm}-{seed}-step300.pt';torch.save({'state_dict':model.state_dict(),'feedback':True,'seed':seed,'arm':arm,'step':300},checkpoint)
            write_archive(HERE/'results'/f'loss-{arm}-{seed}.npz',{'losses':np.array(curve)})
            row={'arm':arm,'seed':seed,'wall_seconds':time.perf_counter()-t0,'initial_tensor_sha256':init,'checkpoint_sha256':sha(checkpoint),'certified_episodes':len(eligible),'examples_seen':4800};runs.append(row);emit('student_training_complete',**row)
    newjson(HERE/'results'/'training.json',runs);emit('all_student_selection_fixed_before_test',training_sha256=sha(HERE/'results'/'training.json'))
    # Only here do test labels/generators become available. No use in trace or train code.
    from learning.data import make_dataset
    from learning.structural_extra import make_extra
    from learning.run import evaluate
    from learning.metrics import summarize
    from learning.analyze import uncertainty
    rows=[];comparisons=[]
    for split,info in p['evaluation'].items():
        data=make_dataset(info['count'],info['seed'],'iid') if split=='iid' else make_extra(info['count'],info['seed'])
        write_archive(HERE/'results'/f'data-{split}.npz',data);emit('heldout_first_open',split=split,**info)
        outputs={}
        for run in runs:
            arm,seed=run['arm'],run['seed'];model=load_model(HERE/'checkpoints'/f'{arm}-{seed}-step300.pt');t0=time.perf_counter();pred,nodes=evaluate(model,data)
            path=HERE/'results'/f'pred-{arm}-{seed}-{split}.npz';write_archive(path,dict(predictions=pred,node_predictions=nodes,targets=data['targets'],affected=data['affected'],inverse=data['inverse'],example_id=data['example_id']))
            row={'arm':arm,'seed':seed,'split':split,'metrics':summarize(pred,data['targets'],data['affected']),'wall_seconds':time.perf_counter()-t0,'raw_path':str(path),'raw_sha256':sha(path)}
            rows.append(row);outputs[arm,seed]=pred;emit('student_evaluation',**row)
        available=[s for s in p['seeds'] if ('trajectory',s) in outputs]
        if len(available)==3:
            diffs=[(outputs['trajectory',s][:,-1]-data['targets'][:,-1])**2-(outputs['answer_only',s][:,-1]-data['targets'][:,-1])**2 for s in available]
            comparisons.append({'split':split,**uncertainty(diffs)})
    newjson(HERE/'results'/'metrics.json',{'protocol_sha256':sha(HERE/'protocol.json'),'runs':runs,'rows':rows,'comparisons':comparisons,'wall_seconds':time.perf_counter()-t_all,'oracle_trace_repair':False})
    emit('distillation_complete',wall_seconds=time.perf_counter()-t_all);print(json.dumps(comparisons),flush=True)

if __name__=='__main__':
    if '--freeze' in sys.argv:freeze()
    elif '--collect' in sys.argv:collect()
    elif '--train' in sys.argv:train()
    else:raise SystemExit('choose --freeze, --collect, or --train')
