"""Paired, larger-scale parity training with frozen held-out evaluation.

This is a 1.5M-parameter controlled-task scale check, not large-language-model
validation. Exact core streams are shared with smaller configurations per seed.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.model import Core,generate
OUT=ROOT/'integration/scale'


def exclusive_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)


def weight_hash(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        t=tensor.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(seed,n,length,offset):return generate('parity',n,length,offset+seed+length*101)


def select_checkpoint(history):
    if not history:raise ValueError('no validation evidence')
    return min(history,key=lambda row:(row['validation_loss'],row['step']))['step']


def event(kind,**payload):
    row={'event':kind,'utc':datetime.now(timezone.utc).isoformat(),'monotonic':time.perf_counter(),**payload}
    with (OUT/'execution.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def protocol():
    return {'version':1,'package':14,'scope':'larger local parity model, not unrestricted language or author-scale reproduction',
            'source_pin':'cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39',
            'seeds':[17,43,91],'modes':['rlt','nofeedback'],'width':256,'heads':4,'window':4,'classes':2,
            'encoder_layers':1,'decoder_layers':1,'train_length':32,'batch':32,'steps':2000,
            'lr':.003,'weight_decay':.01,'clip_norm':1.,'loss':'prefix cross entropy',
            'validation_n':256,'validation_every':100,
            'selection':'minimum validation prefix cross entropy; earliest step on tie',
            'test_n':512,'test_lengths':[32,64,128,256,512,1024],
            'training_offset':'100000+step*1009','validation_offset':200000,'test_offset':900000,
            'generator_seed':'offset+model_seed+length*101; matched with core module per model seed',
            'neutral_slots':[0,8,32,128],'neutral_offset':1900000,'k':0,
            'device':'cuda','precision':'float32','cpu_threads':1,'energy':'UNMEASURED',
            'comparisons':'paired same-width shared initialization/batches; active parameter and executed-compute differences reported',
            'scale_boundary':'width and head configuration grow together vs smaller core runs; no fitted scaling law',
            'classification':'report final accuracy per length/seed; short-fit threshold95%, effective horizon requires all preceding tested lengths meet95%',
            'code_hashes':{'model':sha(ROOT/'core/model.py'),'runner':sha(__file__)}}


@torch.no_grad()
def evaluate_arrays(model,x,y,batch):
    model.eval();pred=[];loss=0.
    torch.cuda.synchronize();start=time.perf_counter()
    for start_row in range(0,len(x),batch):
        xx=x[start_row:start_row+batch].cuda();yy=y[start_row:start_row+batch].cuda()
        logits=model(xx);pred.append(logits.argmax(-1).cpu().numpy())
        loss+=float(F.cross_entropy(logits.flatten(0,1),yy.flatten(),reduction='sum').cpu())
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    predictions=np.concatenate(pred)
    return {'final_accuracy':float((predictions[:,-1]==y[:,-1].numpy()).mean()),
            'prefix_accuracy':float((predictions==y.numpy()).mean()),'prefix_loss':loss/y.numel(),
            'seconds':seconds},predictions


def make(seed,mode,p):
    torch.manual_seed(seed)
    return Core(width=p['width'],heads=p['heads'],window=p['window'],classes=2,feedback=mode=='rlt').cuda()


def train(seed,mode,p,ph):
    name=f'parity-w256-{mode}-s{seed}';folder=OUT/name;folder.mkdir(exist_ok=False)
    model=make(seed,mode,p);initial=weight_hash(model.state_dict())
    torch.save(model.state_dict(),folder/'initial.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    vx,vy=data(seed,p['validation_n'],32,p['validation_offset'])
    history=[];best=float('inf');selected=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    event('train_start',condition=name,initial_weight_hash=initial,parameters=model.counts(),protocol_sha256=ph)
    for step in range(1,p['steps']+1):
        model.train();x,y=data(seed,p['batch'],32,100000+step*1009);x,y=x.cuda(),y.cuda()
        opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),p['clip_norm']);opt.step()
        if step%p['validation_every']==0:
            val,_=evaluate_arrays(model,vx,vy,32)
            row={'step':step,'validation_loss':val['prefix_loss'],'validation_accuracy':val['final_accuracy'],
                 'train_loss':float(loss.detach().cpu()),'gradient_norm':float(grad.detach().cpu()),
                 'elapsed':time.perf_counter()-start}
            history.append(row)
            if row['validation_loss']<best:
                best=row['validation_loss'];selected=step
                torch.save(model.state_dict(),folder/f'validation-best-step{step:04d}.pt')
            event('validation',condition=name,**row)
    if selected!=select_checkpoint(history):raise RuntimeError('checkpoint selection contract mismatch')
    torch.save(model.state_dict(),folder/'last.pt')
    chosen=folder/f'validation-best-step{selected:04d}.pt'
    record={'seed':seed,'mode':mode,'name':name,'steps':p['steps'],'parameters':model.counts(),
            'initial_weight_hash':initial,'initial_file_sha256':sha(folder/'initial.pt'),
            'selected_step':selected,'selected_checkpoint':str(chosen),'selected_checkpoint_sha256':sha(chosen),
            'history':history,'train_seconds':time.perf_counter()-start,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'protocol_sha256':ph}
    exclusive_json(folder/'training.json',record);event('train_complete',condition=name,seconds=record['train_seconds'],selected_step=selected)
    del model,opt,x,y,logits,loss,grad;torch.cuda.empty_cache()
    return record


def test(record,p,ph):
    seed,mode=record['seed'],record['mode'];folder=OUT/record['name']
    model=make(seed,mode,p);model.load_state_dict(torch.load(record['selected_checkpoint'],map_location='cuda',weights_only=True));rows=[]
    for length in p['test_lengths']:
        x,y=data(seed,p['test_n'],length,p['test_offset'])
        stats,pred=evaluate_arrays(model,x,y,8 if length>=512 else 32)
        file=folder/f'heldout-L{length}.npz'
        with file.open('xb') as f:np.savez_compressed(f,inputs=x.numpy(),targets=y.numpy(),predictions=pred,example_id=np.arange(len(x)))
        row={'seed':seed,'mode':mode,'length':length,'t':length,'k':0,'examples':len(x),**stats,'file':str(file),'sha256':sha(file)}
        rows.append(row);event('heldout',**row)
    neutral=[];x,y=data(seed,p['test_n'],32,p['neutral_offset'])
    for slots in p['neutral_slots']:
        xx=torch.cat((x,torch.full((len(x),slots),2,dtype=torch.long)),1)
        yy=torch.cat((y,y[:,-1:].expand(-1,slots)),1)
        stats,pred=evaluate_arrays(model,xx,yy,32);file=folder/f'neutral-{slots}.npz'
        with file.open('xb') as f:np.savez_compressed(f,inputs=xx.numpy(),targets=yy.numpy(),predictions=pred)
        neutral.append({'slots':slots,'t':32+slots,'k':0,'useful_input_length':32,**stats,'file':str(file),'sha256':sha(file)})
    result={'seed':seed,'mode':mode,'protocol_sha256':ph,'long_horizon':rows,'neutral_slots':neutral}
    exclusive_json(folder/'evaluation.json',result);del model;torch.cuda.empty_cache();return result


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required by frozen protocol')
    OUT.mkdir(exist_ok=False);p=protocol();exclusive_json(OUT/'protocol.json',p);ph=sha(OUT/'protocol.json')
    exclusive_json(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                   'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                   'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                   'note':'seeded but no bitwise CUDA determinism claim; record numeric prediction tolerances on replay'})
    event('protocol_frozen',sha256=ph)
    training=[train(seed,mode,p,ph) for seed in p['seeds'] for mode in p['modes']]
    for seed in p['seeds']:
        pair=[r['initial_weight_hash'] for r in training if r['seed']==seed]
        if len(set(pair))!=1:raise RuntimeError('paired initial weights differ')
    event('all_training_complete',runs=len(training));event('test_opened',protocol_sha256=ph)
    evaluation=[test(row,p,ph) for row in training]
    if sha(ROOT/'core/model.py')!=p['code_hashes']['model'] or sha(__file__)!=p['code_hashes']['runner']:
        raise RuntimeError('source changed during frozen run')
    exclusive_json(OUT/'metrics.json',{'status':'COMPLETE','protocol_sha256':ph,'training':training,'evaluation':evaluation})
    event('complete',runs=len(training))


if __name__=='__main__':main()
