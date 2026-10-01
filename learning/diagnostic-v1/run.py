"""Frozen paired learned experiments. CPU one thread, no GPU."""
from pathlib import Path
import datetime,hashlib,json,platform,time,sys
import numpy as np
import torch
from .model import EventModel
from .data import make_dataset
from .metrics import summarize,revision_gate

HERE=Path(__file__).parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2),encoding='utf8')
def log(event,**kw):
    with (HERE/'execution.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw})+'\n')
def saveck(model,path,seed,arm,step):
    torch.save({'state_dict':model.state_dict(),'feedback':model.feedback,'seed':seed,'arm':arm,'step':step},path)
    return sha(path)


def evaluate(model,data,**kwargs):
    predictions=[];nodes=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(data['events']),64):
            out=model(torch.from_numpy(data['events'][start:start+64]),**kwargs)
            predictions.append(out['predictions'].numpy());nodes.append(out['nodes'].numpy())
    return np.concatenate(predictions),np.concatenate(nodes)


def freeze():
    protocol={'version':1,'source_pin':'cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39',
      'scope':'fresh controlled affine forests; independent learned event model, not official RLT implementation',
      'seeds':[17,43,91],'arms':['forward','bidir','wrong','flat','feedback','method'],
      'steps':600,'batch_size':16,'width':32,'heads':2,'ff_width':64,'rounds':6,'swa_window':4,
      'optimizer':{'name':'AdamW','lr':.002,'weight_decay':.0001,'gradient_clip':1.0},
      'method_auxiliary_weight':.3,'method_targets':'TRAIN ONLY exact all-node values at three query states from tree traversal oracle; independently checked linear solve',
      'train':{'count':8192,'seed':81231},'validation':{'count':512,'seed':81232},
      'iid':{'count':512,'seed':91331},'shift':{'count':512,'seed':91332},
      'train_depths':[1,3],'shift_depth':5,'balanced_factors':['inverse_evidence','affected','train_depth'],
      'revision_candidates':[1,2,6],'revision_selection':'minimum affected final validation MSE among candidates with unaffected final deterioration <=0.01; if none feasible minimum total MSE, explicit failed gate',
      'revision_final_tolerance':.01,'revision_rounds':'first query6; post-update6; continuation query6 for every policy',
      'revision_policies':'same learned bidir checkpoint retain,all,validation-selected undirected local radius',
      'feedback_interventions':['none','zero','permute'],'feedback_intervention_timing':'after current readout before next recurrent token state',
      'efficiency_snapshots':[150,300,600],'paired_batches':True,'paired_initialization':True,
      'evaluation':'final-answer MSE and abs-error<=0.1; query delta; affected/unaffected FINAL MSE; forward/inverse and depths separately',
      'uncertainty':'paired seed differences, seed-level t interval df2; hierarchical seed and common example bootstrap1000',
      'no_test_tuning':True,'archive_off_inference':True,'device':'cpu','threads':1,
      'deviations':['No global encoder cross-attention; GRU recurrent token state plus two-head SWA and ff64.',
                    'Title graph inputs use explicit canonical IDs; text extraction evaluated separately.',
                    'Direct facts refresh title anchor; retain means retaining other title states.',
                    'All parameters trainable including title propagation; no symbolic solver in inference.']}
    path=HERE/'protocol.json'
    if path.exists():raise RuntimeError('protocol exists; do not overwrite frozen experiment')
    write(path,protocol)
    sources={p.name:sha(p) for p in HERE.glob('*.py')}
    write(HERE/'source-hashes.json',sources)
    (HERE/'protocol.sha256').write_text(sha(path)+'\n')
    log('protocol_frozen',sha256=sha(path),sources=sources,python=sys.version,torch=torch.__version__,platform=platform.platform(),calibration='10steps seed17 on synthetic16 train-only samples; observed decreasing loss; fixed600 budget from preread requirement; not checkpoint used in main runs')
    return protocol


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=freeze();started=time.perf_counter()
    datasets={}
    for split in ('train','validation'):
        info=protocol[split];datasets[split]=make_dataset(info['count'],info['seed'],split)
        path=HERE/'results'/f'data-{split}.npz';np.savez_compressed(path,**datasets[split]);log('dataset_created',split=split,sha256=sha(path),**info)
    # Verified public method traces are train labels saved separately, unavailable to API/model.
    np.savez_compressed(HERE/'results'/'verified-method-traces-TRAIN-ONLY.npz',events=datasets['train']['events'],intermediates=datasets['train']['node_targets'])
    write(HERE/'results'/'trace-provenance.json',{'oracle':'learning.data.solve_tree','independent_check':'tests/test_learning.py linear least squares32 validation examples','split':'train','count':8192,'sha256':sha(HERE/'results'/'verified-method-traces-TRAIN-ONLY.npz')})
    train=datasets['train'];X=torch.from_numpy(train['events']);Y=torch.from_numpy(train['targets']);Z=torch.from_numpy(train['node_targets'])
    runs=[];selections={}
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed+1000);batches=rng.integers(0,len(X),(protocol['steps'],protocol['batch_size']))
        np.save(HERE/'results'/f'batches-{seed}.npy',batches)
        for arm in protocol['arms']:
            torch.manual_seed(seed);model=EventModel(feedback=arm=='feedback');direction=arm if arm in ('forward','wrong','flat') else 'bidir'
            initial=HERE/'checkpoints'/f'{arm}-{seed}-initial.pt';initial_hash=saveck(model,initial,seed,arm,0)
            # Compare tensors, since serialization metadata can alter byte hashes.
            tensor_hash=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            t0=time.perf_counter();curve=[];train_losses=[];aux_losses=[]
            log('training_start',seed=seed,arm=arm,initial_sha256=initial_hash,initial_tensor_sha256=tensor_hash,parameters=sum(p.numel() for p in model.parameters()))
            for step,index in enumerate(batches,1):
                model.train();optimizer.zero_grad();out=model(X[index],direction=direction)
                final_loss=((out['predictions']-Y[index])**2).mean();aux=((out['nodes']-Z[index])**2).mean()
                loss=final_loss+(.3*aux if arm=='method' else 0)
                if not torch.isfinite(loss):raise FloatingPointError(f'{arm}-{seed}-{step} nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                train_losses.append(float(final_loss.detach()));aux_losses.append(float(aux.detach()))
                if step in (150,300,600):
                    checkpoint=HERE/'checkpoints'/f'{arm}-{seed}-step{step}.pt';digest=saveck(model,checkpoint,seed,arm,step)
                    pred,_=evaluate(model,datasets['validation'],direction=direction)
                    entry={'step':step,'examples_seen':step*16,'wall_seconds':time.perf_counter()-t0,'validation':summarize(pred,datasets['validation']['targets'],datasets['validation']['affected']),'checkpoint_sha256':digest}
                    curve.append(entry);log('training_checkpoint',seed=seed,arm=arm,**entry)
                    print(json.dumps({'seed':seed,'arm':arm,**entry}),flush=True)
            np.savez_compressed(HERE/'results'/f'loss-{arm}-{seed}.npz',final=np.array(train_losses),aux=np.array(aux_losses))
            runs.append({'arm':arm,'seed':seed,'wall_seconds':time.perf_counter()-t0,'initial_tensor_sha256':tensor_hash,'curve':curve})
            if arm=='bidir':
                val=datasets['validation'];base,_=evaluate(model,val)
                candidates=[]
                for radius in protocol['revision_candidates']:
                    p,_=evaluate(model,val,policy='local',radius=radius)
                    gate=revision_gate(base,p,val['targets'],val['affected'])
                    candidates.append({'radius':radius,'metrics':summarize(p,val['targets'],val['affected']),'gate':gate})
                feasible=[row for row in candidates if row['gate']['unaffected_final_change']<=.01]
                chosen=min(feasible,key=lambda row:row['metrics']['affected_final_mse']) if feasible else min(candidates,key=lambda row:row['metrics']['mse'])
                selections[str(seed)]={'radius':chosen['radius'],'candidates':candidates};log('revision_selection',seed=seed,**selections[str(seed)])
    write(HERE/'training.json',runs);write(HERE/'revision-selection.json',selections)
    log('all_selection_frozen_before_test',selection_sha256=sha(HERE/'revision-selection.json'),training_sha256=sha(HERE/'training.json'))
    # Held-out generation and opening occur only after all selections are frozen.
    for split in ('iid','shift'):
        info=protocol[split];datasets[split]=make_dataset(info['count'],info['seed'],split)
        path=HERE/'results'/f'data-{split}.npz';np.savez_compressed(path,**datasets[split]);log('test_first_open',split=split,sha256=sha(path),**info)
    rows=[]
    for run in runs:
        seed,arm=run['seed'],run['arm'];direction=arm if arm in ('forward','wrong','flat') else 'bidir'
        for step in (150,300,600):
            if step!=600 and arm not in ('bidir','method'):continue
            payload=torch.load(HERE/'checkpoints'/f'{arm}-{seed}-step{step}.pt',weights_only=True)
            model=EventModel(feedback=arm=='feedback');model.load_state_dict(payload['state_dict']);model.eval()
            conditions=[('retain',{})]
            if arm=='bidir' and step==600:conditions.extend([('all',{'policy':'all'}),('local',{'policy':'local','radius':selections[str(seed)]['radius']})])
            if arm=='feedback':conditions.extend([('zero',{'intervention':'zero'}),('permute',{'intervention':'permute'})])
            for split in ('iid','shift'):
                data=datasets[split]
                for condition,kwargs in conditions:
                    t0=time.perf_counter();pred,nodes=evaluate(model,data,direction=direction,**kwargs);elapsed=time.perf_counter()-t0
                    path=HERE/'results'/f'pred-{arm}-{seed}-{step}-{split}-{condition}.npz'
                    np.savez_compressed(path,predictions=pred,node_predictions=nodes,targets=data['targets'],affected=data['affected'],inverse=data['inverse'],depth=data['depth'],example_id=data['example_id'])
                    row={'arm':arm,'seed':seed,'step':step,'split':split,'condition':condition,'wall_seconds':elapsed,'raw_path':str(path),'raw_sha256':sha(path),'metrics':summarize(pred,data['targets'],data['affected'])}
                    for name,mask in [('forward_task',~data['inverse']),('inverse_task',data['inverse']),('shallow',data['depth']==1),('deep',data['depth']>=3)]:
                        if mask.any():row[name]=summarize(pred[mask],data['targets'][mask],data['affected'][mask])
                    rows.append(row);log('evaluation',**row)
    write(HERE/'results'/'metrics.json',{'protocol_sha256':sha(HERE/'protocol.json'),'total_wall_seconds':time.perf_counter()-started,'runs':runs,'rows':rows,'revision_selection':selections,'energy':'UNMEASURED','scope':'controlled affine forests only; no archive at inference; no large-scale or unrestricted language claim'})
    log('complete',wall_seconds=time.perf_counter()-started,rows=len(rows))


if __name__=='__main__':main()
