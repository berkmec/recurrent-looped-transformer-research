"""Prespecified genuinely unseen dependency-depth evaluation; oracle labels only."""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
import torch
from .data import solve_tree
from .api import load_model
from .run import evaluate,log
from .metrics import summarize
from .analyze import uncertainty

HERE=Path(__file__).parent


def make_extra(count,seed):
    rng=np.random.default_rng(seed);rows=[];targets=[];nodes=[];affected=[];inverse=[]
    for index in range(count):
        ids=rng.permutation(10);chain=list(ids[:9]);other=int(ids[9]);inv=bool(index%2);effect=bool((index//2)%2)
        parents={chain[j]:chain[j-1] for j in range(1,9)}
        offsets={node:float(rng.uniform(-.35,.35)) for node in parents}
        known=chain[-1] if inv else chain[0];requested=chain[0] if inv else chain[-1]
        observations={known:float(rng.uniform(-.7,.7)),other:float(rng.uniform(-.7,.7))}
        facts=[[1,c,p,offsets[c]] for c,p in parents.items()]+[[0,n,n,v] for n,v in observations.items()];rng.shuffle(facts)
        query=[2,requested,requested,0];before=solve_tree(parents,offsets,observations)
        changed=known if effect else other;replacement=observations[changed]+float(rng.choice([-1,1])*rng.uniform(.3,.8));observations[changed]=replacement
        after=solve_tree(parents,offsets,observations)
        rows.append(facts+[query,[6,0,0,0],[3,changed,changed,replacement],query,[6,0,0,0],query])
        targets.append([before[requested],after[requested],after[requested]]);nodes.append([before,after,after]);affected.append(effect);inverse.append(inv)
    return {'events':np.array(rows,np.float32),'targets':np.array(targets,np.float32),'node_targets':np.array(nodes,np.float32),
            'affected':np.array(affected,bool),'inverse':np.array(inverse,bool),'depth':np.full(count,8,np.int64),'example_id':np.arange(count,dtype=np.int64)}


def freeze():
    protocol={'version':1,'reason':'Primary depth5 is unseen query depth only: all-node train targets include other-component depths up to7. Depth8 is outside all training dependency depths.',
              'train_unchanged':True,'primary_protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
              'count':512,'seed':98338,'seeds':[17,43,91],'arms':['forward','bidir','wrong','flat','feedback','method'],
              'checkpoint_step':600,'rounds_per_query':6,'depth':8,'components':'nine-node chain and isolated observed node',
              'selection':'none; all arms fixed; no tuning from primary or supplemental held-out',
              'metrics':'final MSE, accuracy0.1, inverse/forward subgroups, method-minus-final-only paired seed uncertainty',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=HERE/'structural-extra-protocol.json'
    if path.exists():raise RuntimeError('supplemental protocol already frozen')
    path.write_text(json.dumps(protocol,indent=2));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'structural-extra-protocol.sha256').write_text(digest+'\n');log('structural_extra_protocol_frozen',sha256=digest,reason=protocol['reason'])
    print(digest)


def main():
    torch.set_num_threads(1)
    p=json.loads((HERE/'structural-extra-protocol.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=p['source_sha256']:raise AssertionError('supplement source changed')
    if not (HERE/'results'/'metrics.json').exists():raise RuntimeError('wait for frozen primary run completion')
    data=make_extra(p['count'],p['seed']);path=HERE/'results'/'data-depth8.npz';np.savez_compressed(path,**data)
    log('structural_extra_test_first_open',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=p['seed'])
    rows=[];raws={}
    for seed in p['seeds']:
        for arm in p['arms']:
            model=load_model(HERE/'checkpoints'/f'{arm}-{seed}-step600.pt');direction=arm if arm in ('forward','wrong','flat') else 'bidir'
            t0=time.perf_counter();pred,nodes=evaluate(model,data,direction=direction)
            target=HERE/'results'/f'pred-{arm}-{seed}-600-depth8-retain.npz'
            np.savez_compressed(target,predictions=pred,node_predictions=nodes,targets=data['targets'],affected=data['affected'],inverse=data['inverse'],depth=data['depth'],example_id=data['example_id'])
            row={'arm':arm,'seed':seed,'wall_seconds':time.perf_counter()-t0,'raw_path':str(target),'raw_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'metrics':summarize(pred,data['targets'],data['affected'])}
            for name,mask in [('inverse',data['inverse']),('forward',~data['inverse'])]:row[name]=summarize(pred[mask],data['targets'][mask],data['affected'][mask])
            rows.append(row);raws[arm,seed]=pred;log('structural_extra_evaluation',**row)
    comparisons=[]
    for arm,baseline in [('method','bidir'),('bidir','forward')]:
        diffs=[(raws[arm,s][:,-1]-data['targets'][:,-1])**2-(raws[baseline,s][:,-1]-data['targets'][:,-1])**2 for s in p['seeds']]
        comparisons.append({'candidate':arm,'baseline':baseline,**uncertainty(diffs)})
    output={'protocol':p,'rows':rows,'comparisons':comparisons}
    (HERE/'results'/'structural-extra-metrics.json').write_text(json.dumps(output,indent=2));print(json.dumps(comparisons))

if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
