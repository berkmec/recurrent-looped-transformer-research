"""Prespecified feedback x method-supervision interaction follow-up."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
import torch
from .model import EventModel
from .run import evaluate,saveck,log
from .metrics import summarize
from .analyze import uncertainty

HERE=Path(__file__).parent


def assert_fresh_combined(root):
    root=Path(root)
    paths=list((root/'checkpoints').glob('combined-*'))+list((root/'results').glob('*combined*'))
    if paths:raise FileExistsError('Combined evidence exists: '+str(paths[0]))


def freeze():
    protocol={'version':1,'requested_by':'root integration audit before primary held-out opening',
              'primary_protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
              'seeds':[17,43,91],'steps':600,'batch_size':16,'feedback':True,'auxiliary_weight':.3,'direction':'bidir',
              'factorial':'bidir(no feedback,no methods), feedback(yes,no), method(no,yes), combined(yes,yes)',
              'pairing':'same seed initialization, saved train arrays and saved primary batch indices, AdamW lr0.002 wd0.0001 clip1',
              'selection':'fixed final600 checkpoint; all seeds reported; combined seed17 predeclared full-system integration checkpoint',
              'evaluation':['iid','shift','depth8'],'count_per_split':512,'rounds_per_query':6,
              'scope':'controlled affine tasks; title-feedback and exact intermediate supervision interaction; no oracle/archive during inference',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=HERE/'combined-protocol.json'
    if path.exists():raise FileExistsError('combined protocol already frozen')
    path.write_text(json.dumps(protocol,indent=2));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'combined-protocol.sha256').write_text(digest+'\n');log('combined_protocol_frozen',sha256=digest,protocol=protocol);print(digest)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);assert_fresh_combined(HERE)
    protocol=json.loads((HERE/'combined-protocol.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=protocol['source_sha256']:raise AssertionError('combined source changed')
    if not (HERE/'results'/'metrics.json').exists():raise RuntimeError('wait for primary training completion')
    train=np.load(HERE/'results'/'data-train.npz');val=np.load(HERE/'results'/'data-validation.npz')
    X=torch.from_numpy(train['events']);Y=torch.from_numpy(train['targets']);Z=torch.from_numpy(train['node_targets'])
    runs=[];tall=time.perf_counter()
    for seed in protocol['seeds']:
        torch.manual_seed(seed);model=EventModel(feedback=True)
        initial=saveck(model,HERE/'checkpoints'/f'combined-{seed}-initial.pt',seed,'combined',0)
        tensor_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        batches=np.load(HERE/'results'/f'batches-{seed}.npy');optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        t0=time.perf_counter();curve=[];losses=[];auxiliary=[];log('combined_training_start',seed=seed,initial_sha256=initial,initial_tensor_sha256=tensor_hash)
        for step,index in enumerate(batches,1):
            model.train();optimizer.zero_grad();out=model(X[index]);final=((out['predictions']-Y[index])**2).mean();aux=((out['nodes']-Z[index])**2).mean();loss=final+.3*aux
            if not torch.isfinite(loss):raise FloatingPointError('combined nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(final.detach()));auxiliary.append(float(aux.detach()))
            if step in (150,300,600):
                digest=saveck(model,HERE/'checkpoints'/f'combined-{seed}-step{step}.pt',seed,'combined',step);pred,_=evaluate(model,val)
                row={'step':step,'validation':summarize(pred,val['targets'],val['affected']),'wall_seconds':time.perf_counter()-t0,'checkpoint_sha256':digest};curve.append(row);log('combined_training_checkpoint',seed=seed,**row);print(json.dumps({'seed':seed,**row}),flush=True)
        np.savez_compressed(HERE/'results'/f'loss-combined-{seed}.npz',final=losses,aux=auxiliary)
        runs.append({'seed':seed,'initial_tensor_sha256':tensor_hash,'wall_seconds':time.perf_counter()-t0,'curve':curve})
    # All three final checkpoints are fixed before this follow-up opens any test array.
    log('combined_all_training_complete_before_evaluation',seeds=protocol['seeds'])
    rows=[];comparisons=[]
    for split in protocol['evaluation']:
        data=np.load(HERE/'results'/f'data-{split}.npz');preds={}
        for seed in protocol['seeds']:
            model=EventModel(feedback=True);payload=torch.load(HERE/'checkpoints'/f'combined-{seed}-step600.pt',weights_only=True);model.load_state_dict(payload['state_dict'])
            t0=time.perf_counter();pred,nodes=evaluate(model,data);preds[seed]=pred
            path=HERE/'results'/f'pred-combined-{seed}-600-{split}-retain.npz';np.savez_compressed(path,predictions=pred,node_predictions=nodes,targets=data['targets'],affected=data['affected'],inverse=data['inverse'],depth=data['depth'],example_id=data['example_id'])
            row={'arm':'combined','seed':seed,'split':split,'raw_path':str(path),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':summarize(pred,data['targets'],data['affected']),'wall_seconds':time.perf_counter()-t0}
            for name,mask in [('inverse',data['inverse']),('forward',~data['inverse'])]:row[name]=summarize(pred[mask],data['targets'][mask],data['affected'][mask])
            rows.append(row);log('combined_evaluation',**row)
        for baseline in ('bidir','feedback','method'):
            diffs=[]
            for seed in protocol['seeds']:
                base=np.load(HERE/'results'/f'pred-{baseline}-{seed}-600-{split}-retain.npz')['predictions']
                diffs.append((preds[seed][:,-1]-data['targets'][:,-1])**2-(base[:,-1]-data['targets'][:,-1])**2)
            comparisons.append({'candidate':'combined','baseline':baseline,'split':split,**uncertainty(diffs)})
        interaction=[]
        for seed in protocol['seeds']:
            errors={arm:(np.load(HERE/'results'/f'pred-{arm}-{seed}-600-{split}-retain.npz')['predictions'][:,-1]-data['targets'][:,-1])**2 for arm in ('bidir','feedback','method')}
            ce=(preds[seed][:,-1]-data['targets'][:,-1])**2
            interaction.append(ce-errors['method']-errors['feedback']+errors['bidir'])
        comparisons.append({'contrast':'combined-method-feedback+bidir','split':split,**uncertainty(interaction)})
    result={'protocol':protocol,'runs':runs,'rows':rows,'comparisons':comparisons,'total_wall_seconds':time.perf_counter()-tall}
    (HERE/'results'/'combined-metrics.json').write_text(json.dumps(result,indent=2));log('combined_complete',wall_seconds=result['total_wall_seconds'])

if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
