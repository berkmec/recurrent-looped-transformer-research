"""Actual model trace capture and observation-only acceptance. No label oracle."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from memory.api import verify_candidate


def load_training_events(path,expected_sha256,split,count):
    if split!='train':raise ValueError('teacher collection accepts TRAIN only')
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('training receipt mismatch')
    # Deliberately never access targets or node_targets in this source archive.
    with np.load(path) as data:return data['events'][:count].copy()


def write_archive(path,arrays):
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)


def certify(candidate,events,query,tolerance):
    decision=verify_candidate(candidate,events,query,tolerance)
    return {**decision,'accepted':decision['status']=='VERIFIED'}


def observed_records(tape):
    result=[]
    for op,child,parent,value in tape:
        op,child,parent=int(op),str(int(child)),str(int(parent))
        if op in (0,3):result.append(dict(kind='root' if op==0 else 'revision',child=child,parent=None,value=float(value)))
        elif op in (1,4):result.append(dict(kind='edge' if op==1 else 'revision',child=child,parent=parent,value=float(value)))
    return result


def component_mask(records,query,nodes=10):
    adjacency={}
    for record in records:
        child=record['child'];parent=record['parent'];adjacency.setdefault(child,set())
        if parent is not None:adjacency.setdefault(parent,set()).add(child);adjacency[child].add(parent)
    reached={str(query)};queue=[str(query)]
    for item in queue:
        for other in adjacency.get(item,()):
            if other not in reached:reached.add(other);queue.append(other)
    mask=np.zeros(nodes,np.float32)
    for name in reached:mask[int(name)]=1.
    return mask


def trace_forward(model,events,rounds=6):
    runtime=model.new_runtime(events);states=[];numeric=[];final=[]
    while runtime.consume_to_query():
        zs=[];ys=[]
        for _ in range(rounds):
            candidate=runtime.refine_one();zs.append(runtime.z);ys.append(candidate['nodes'])
        states.append(torch.stack(zs,1));numeric.append(torch.stack(ys,1));final.append(candidate['prediction']);runtime.commit_feedback()
    return {'states':torch.stack(states,1),'nodes':torch.stack(numeric,1),'final':torch.stack(final,1)}


def losses(current,teacher,query_mask,node_mask):
    final=(((current['final']-teacher['final'])**2)*query_mask).sum()/query_mask.sum().clamp_min(1)
    k=current['states'].shape[2];d=current['states'].shape[-1]
    state=(((current['states']-teacher['states'])**2)*node_mask[:,:,None,:,None]).sum()/(node_mask.sum()*k*d).clamp_min(1)
    numeric=(((current['nodes']-teacher['nodes'])**2)*node_mask[:,:,None,:]).sum()/(node_mask.sum()*k).clamp_min(1)
    return final,state+numeric
