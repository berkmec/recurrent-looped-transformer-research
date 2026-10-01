"""TRAINING/EVALUATION ORACLE. Never imported by model.py or api.py."""
import numpy as np


def solve_tree(parents,offsets,observations):
    # Independent tree traversal oracle; tests check against linear algebra solve.
    values=np.full(10,np.nan)
    for node,val in observations.items():values[node]=val
    for _ in range(10):
        for child,parent in parents.items():
            if np.isfinite(values[parent]):values[child]=values[parent]+offsets[child]
            elif np.isfinite(values[child]):values[parent]=values[child]-offsets[child]
    return np.nan_to_num(values).astype(np.float32)


def make_dataset(count,seed,split):
    if split not in ('train','validation','iid','shift'):raise ValueError('unknown split')
    rng=np.random.default_rng(seed);records=[];answers=[];node_answers=[];flags=[];depths=[];inverse=[]
    for index in range(count):
        depth=5 if split=='shift' else (1 if index%2==0 else 3)
        is_inverse=bool((index//2)%2);affected=bool((index//4)%2)
        ids=rng.permutation(10);chain=list(ids[:depth+1]);other=list(ids[depth+1:])
        parents={chain[j]:chain[j-1] for j in range(1,len(chain))}
        parents.update({other[j]:other[j-1] for j in range(1,len(other))})
        offsets={node:float(rng.uniform(-.35,.35)) for node in parents}
        known=chain[-1] if is_inverse else chain[0]
        observations={known:float(rng.uniform(-.7,.7)),other[0]:float(rng.uniform(-.7,.7))}
        facts=[[1,c,p,offsets[c]] for c,p in parents.items()]+[[0,n,n,v] for n,v in observations.items()]
        rng.shuffle(facts)
        requested=chain[0] if is_inverse else chain[-1]
        query=[2,requested,requested,0]
        labels=solve_tree(parents,offsets,observations);before=labels[requested]
        # Delayed revision: irrelevant no-write continuation separates the initial query and update.
        changed=known if affected else other[0]
        replacement=observations[changed]+float(rng.choice([-1,1])*rng.uniform(.3,.8))
        observations[changed]=replacement;after_nodes=solve_tree(parents,offsets,observations)
        tape=facts+[query,[6,0,0,0],[3,changed,changed,replacement],query,[6,0,0,0],query]
        records.append(tape);answers.append([before,after_nodes[requested],after_nodes[requested]])
        node_answers.append([labels,after_nodes,after_nodes]);flags.append(affected);depths.append(depth);inverse.append(is_inverse)
    return {'events':np.array(records,np.float32),'targets':np.array(answers,np.float32),
            'node_targets':np.array(node_answers,np.float32),'affected':np.array(flags,bool),
            'depth':np.array(depths,np.int64),'inverse':np.array(inverse,bool),'example_id':np.arange(count,dtype=np.int64)}
