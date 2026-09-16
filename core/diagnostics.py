"""Oracle-loss influence diagnostics, explicitly separate from learned accuracy."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from core.model import Core,generate
def influence(m,x,y,lags,epsilon=.001):
 m.zero_grad(set_to_none=True);states=[];emb=[]
 def capture(module,args,out):out.retain_grad();emb.append(out)
 hook=m.embedding.register_forward_hook(capture);s=None
 for t in range(x.shape[1]):
  z,s=m.step(x[:,t],s);s['s'].retain_grad();states.append(s['s'])
 loss=F.cross_entropy(z,y[:,-1]);loss.backward();hook.remove();direction=torch.ones_like(states[0]);direction=direction/direction.norm();g=states[0].grad;analytical=0. if g is None else float((g*direction).sum());state_norms=[];input_norms=[]
 for lag in lags:
  i=x.shape[1]-1-lag
  if i>=0:
   state_norms.append(dict(lag=lag,norm=0. if states[i].grad is None else float(states[i].grad.norm())));input_norms.append(dict(lag=lag,norm=float(emb[i].grad.norm())))
 def perturbed(amount):
  with torch.no_grad():
   _,s=m.step(x[:,0]);s['s']=s['s']+amount*direction
   for t in range(1,x.shape[1]):z,s=m.step(x[:,t],s)
   return float(F.cross_entropy(z,y[:,-1]))
 fd=(perturbed(epsilon)-perturbed(-epsilon))/(2*epsilon)
 return dict(loss=float(loss.detach()),evaluation_dtype=str(next(m.parameters()).dtype),state_norms=state_norms,embedding_norms=input_norms,directional_autograd=analytical,directional_finite_difference=fd,epsilon=epsilon,oracle='final independently generated state target; no inference oracle')
def checkpoint_check(m,x,y):
 records=[];gradients=[];updates=[];original=copy.deepcopy(m.state_dict())
 for enabled in [False,True]:
  m.load_state_dict(original);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);m.zero_grad(set_to_none=True);start=time.perf_counter();z=checkpoint(m,x,use_reentrant=False) if enabled else m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten());loss.backward();gradients.append({n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None});torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();records.append(dict(checkpoint=enabled,seconds=time.perf_counter()-start,loss=float(loss.detach())));updates.append(copy.deepcopy(m.state_dict()))
 m.load_state_dict(original)
 return dict(steps=records,loss_difference=abs(records[0]['loss']-records[1]['loss']),max_gradient_difference=max(float((gradients[0][n]-gradients[1][n]).abs().max()) for n in gradients[0]),max_parameter_update_difference=max(float((updates[0][n]-updates[1][n]).abs().max()) for n in updates[0]),memory='UNMEASURED: CPU native peak allocation not instrumented; no memory saving claim',scope='whole-forward activation recomputation; real backward and one AdamW update from identical model/optimizer states, restored afterward')
@torch.no_grad()
def noise_predictions(m,x,std,seed):
 gen=torch.Generator().manual_seed(seed);s=None;start=time.perf_counter()
 for t in range(x.shape[1]):
  z,s=m.step(x[:,t],s)
  if std and t<x.shape[1]-1:s['s']=s['s']+std*torch.randn(s['s'].shape,generator=gen,dtype=s['s'].dtype)
 return z.argmax(-1).numpy(),time.perf_counter()-start
def main():
 torch.set_num_threads(1);root=Path(__file__).parent;pp=root/'diagnostics-protocol.json'
 if not pp.exists():
  pp.write_text(json.dumps(dict(frozen_before_diagnostic_evaluation=True,task='parity',width=32,seeds=[17,43,91],modes=['rlt','nofeedback','tbptt'],lags=[1,4,8,16,31],gradient_examples=8,gradient_length=32,gradient_split=2900000,influence_dtype='float64 cast of frozen FP32 checkpoint',finite_difference_epsilon=.001,noise_std=[0,.001,.01,.1],noise_examples=512,noise_length=32,noise_split=3900000,checkpoint='seed17 RLT length32 batch32 3 repeats after warmup; real AdamW update',memory='UNMEASURED',k=0),indent=2));(root/'diagnostics-protocol.sha256').write_text(hashlib.sha256(pp.read_bytes()).hexdigest())
 p=json.loads(pp.read_text());assert hashlib.sha256(pp.read_bytes()).hexdigest()==(root/'diagnostics-protocol.sha256').read_text();rows=[]
 for seed in p['seeds']:
  for mode in p['modes']:
   name=f'parity-w32-{mode}-s{seed}';m=Core(feedback=mode!='nofeedback');m.load_state_dict(torch.load(root/'checkpoints'/(name+'-selected.pt'),weights_only=True));x,y=generate('parity',8,32,p['gradient_split']+seed);r=influence(copy.deepcopy(m).double(),x,y,p['lags']);nx,ny=generate('parity',512,32,p['noise_split']+seed);noise=[]
   for std in p['noise_std']:
    pred,seconds=noise_predictions(m,nx,std,4900000+seed);path=root/'results'/(name+f'-noise{std}.npz');np.savez_compressed(path,inputs=nx.numpy(),predictions=pred,targets=ny[:,-1].numpy());noise.append(dict(std=std,accuracy=float((pred==ny[:,-1].numpy()).mean()),seconds=seconds,file=str(path)))
   row=dict(seed=seed,mode=mode,influence=r,noise=noise)
   if seed==17 and mode=='rlt':
    xx,yy=generate('parity',32,32,5900017);checkpoint_check(m,xx,yy);row['checkpoint_repeats']=[checkpoint_check(m,xx,yy) for _ in range(3)]
   rows.append(row);(root/'results'/'diagnostics.json').write_text(json.dumps(dict(protocol_hash=hashlib.sha256(pp.read_bytes()).hexdigest(),rows=rows),indent=2));print(name,r,flush=True)
if __name__=='__main__':main()
