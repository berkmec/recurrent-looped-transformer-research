"""Frozen-protocol real training/evaluation; write only this independent module."""
import argparse,copy,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core.model import Core,generate
ROOT=Path(__file__).parent;RES=ROOT/'results';CK=ROOT/'checkpoints'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
def event(kind,**kw):
 with (ROOT/'execution.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(event=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),monotonic=time.perf_counter(),**kw))+'\n')
 print(kind,kw,flush=True)
def protocol():
 return dict(version=1,source_pin='cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39',source_url='https://raw.githubusercontent.com/yifanzhang-pro/recurrent-looped-tranformer/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md',seeds=[17,43,91],tasks=['parity','five'],width=32,scale_width=48,heads=2,window=4,encoder_layers=1,decoder_layers=1,ff_multiplier=2,alpha=.1,train_tokens=32,train_operation_includes_identity=True,token_probabilities=[.4375,.4375,.125],batch=32,max_steps=2000,lr=.003,weight_decay=.01,clip=1.,loss='mean prefix cross entropy',validation_n=256,validation_every=100,selection='minimum validation prefix cross entropy over steps100..2000; no early stopping; final-state accuracy is primary reporting',test_n=512,test_lengths=[32,64,128,256,512,1024],tbptt=8,tbptt_route='detach s and decoder SWA every8; encoder and projected global memory retained',conditions='width32: both tasks full RLT and matched nofeedback; parity also TBPTT8. width48: parity full RLT and nofeedback. all3seeds. 21runs',split_seeds=dict(train_offset=100000,validation_offset=200000,test_offset=900000),neutral_slots=[0,8,32,128],neutral_task_tokens=32,k=0,state_noise_std=[0,.001,.01,.1],diagnostic_lags=[1,4,8,16,31],energy='UNMEASURED',device='cpu',threads=1)
def conditions(p):
 for task in p['tasks']:
  for seed in p['seeds']:
   for mode in (['rlt','nofeedback','tbptt'] if task=='parity' else ['rlt','nofeedback']):yield dict(task=task,seed=seed,width=32,mode=mode)
 for seed in p['seeds']:
  for mode in ['rlt','nofeedback']:yield dict(task='parity',seed=seed,width=48,mode=mode)
def ident(c):return f"{c['task']}-w{c['width']}-{c['mode']}-s{c['seed']}"
def make(c):
 torch.manual_seed(c['seed']);return Core(width=c['width'],classes=2 if c['task']=='parity' else 5,feedback=c['mode']!='nofeedback')
def dataset(c,n,l,offset):return generate(c['task'],n,l,offset+c['seed']+l*101+(0 if c['task']=='parity' else 10000))
@torch.no_grad()
def eval_data(m,x,y,batch=32):
 pred=[];loss=0.;start=time.perf_counter()
 for i in range(0,len(x),batch):
  z=m(x[i:i+batch]);pred.append(z.argmax(-1).numpy());loss+=float(F.cross_entropy(z.flatten(0,1),y[i:i+batch].flatten(),reduction='sum'))
 pred=np.concatenate(pred);return dict(final_accuracy=float((pred[:,-1]==y.numpy()[:,-1]).mean()),prefix_accuracy=float((pred==y.numpy()).mean()),loss=loss/y.numel(),seconds=time.perf_counter()-start),pred
def train(c,p):
 name=ident(c);record=RES/(name+'-train.json')
 if record.exists():return json.loads(record.read_text())
 m=make(c);torch.save(m.state_dict(),CK/(name+'-initial.pt'));opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);vx,vy=dataset(c,p['validation_n'],32,200000)
 best=float('inf');history=[];start=time.perf_counter();event('train_start',condition=c,counts=m.counts())
 for step in range(1,p['max_steps']+1):
  x,y=dataset(c,p['batch'],32,100000+step*1009);opt.zero_grad(set_to_none=True);z=m(x,tbptt=8 if c['mode']=='tbptt' else 0);loss=F.cross_entropy(z.flatten(0,1),y.flatten());loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
  if step%100==0:
   val,_=eval_data(m,vx,vy);history.append(dict(step=step,train_loss=float(loss.detach()),grad_norm=grad,validation=val,elapsed=time.perf_counter()-start))
   if val['loss']<best:best=val['loss'];selected=step;torch.save(m.state_dict(),CK/(name+'-selected.pt'))
   event('validation',condition=name,**history[-1])
 torch.save(m.state_dict(),CK/(name+'-last.pt'));out=dict(condition=c,steps=p['max_steps'],selected_step=selected,train_seconds=time.perf_counter()-start,parameters=m.counts(),history=history,checkpoint_hashes={suffix:sha(CK/(name+'-'+suffix+'.pt')) for suffix in ['initial','selected','last']});savejson(record,out);return out
def evaluate(c,p):
 name=ident(c);m=make(c);m.load_state_dict(torch.load(CK/(name+'-selected.pt'),weights_only=True));rows=[]
 for length in p['test_lengths']:
  x,y=dataset(c,512,length,900000);stats,pred=eval_data(m,x,y,batch=8 if length>=512 else 32);path=RES/(name+f'-L{length}.npz');np.savez_compressed(path,inputs=x.numpy(),predictions=pred,targets=y.numpy(),example_id=np.arange(512),split_seed=900000+c['seed']+length*101+(0 if c['task']=='parity' else 10000));row=dict(**c,length=length,consumed_tokens=length,k=0,operation_count=length,mean_nonidentity_count=float((x!=2).sum(1).double().mean()),**stats,predictions_file=str(path),sha256=sha(path));rows.append(row);event('heldout',**row)
 # Same held-out base programs, append information-free identity slots.
 x,y=dataset(c,512,32,1900000);neutral=[]
 for slots in p['neutral_slots']:
  xx=torch.cat([x,torch.full((512,slots),2,dtype=torch.long)],1);yy=torch.cat([y,y[:,-1:].expand(-1,slots)],1);stats,pred=eval_data(m,xx,yy);path=RES/(name+f'-neutral{slots}.npz');np.savez_compressed(path,inputs=xx.numpy(),predictions=pred,targets=yy.numpy());neutral.append(dict(slots=slots,consumed_tokens=32+slots,base_operations=32,mean_nonidentity_count=float((x!=2).sum(1).double().mean()),output_class_count=1,k=0,**stats,file=str(path)))
 out=dict(condition=c,long_horizon=rows,neutral_slots=neutral);savejson(RES/(name+'-eval.json'),out);return out
def main():
 RES.mkdir(exist_ok=True);CK.mkdir(exist_ok=True);pp=ROOT/'protocol.json'
 if not pp.exists():savejson(pp,protocol());(ROOT/'protocol.sha256').write_text(sha(pp));event('protocol_frozen',sha256=sha(pp))
 p=json.loads(pp.read_text());assert sha(pp)==(ROOT/'protocol.sha256').read_text();savejson(ROOT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),cuda_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,device='cpu',threads=torch.get_num_threads(),source_hashes={f.name:sha(f) for f in ROOT.glob('*.py')}))
 cs=list(conditions(p));training=[train(c,p) for c in cs];event('all_training_complete',runs=len(training));evaluation=[]
 for c in cs:
  ep=RES/(ident(c)+'-eval.json');evaluation.append(json.loads(ep.read_text()) if ep.exists() else evaluate(c,p));savejson(RES/'metrics.json',dict(protocol_hash=sha(pp),training=training,evaluation=evaluation,status='RUNNING' if len(evaluation)<len(cs) else 'COMPLETE'))
 event('primary_complete',runs=len(cs))
if __name__=='__main__':main()
