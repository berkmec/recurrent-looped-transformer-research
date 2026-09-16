"""Separate frozen delayed-error companion; never changes primary protocol."""
import json,time
import torch
from torch.nn import functional as F
from core.run import ROOT,RES,CK,savejson,sha,event,make,dataset,eval_data
def freeze():
 pp=ROOT/'final-only-protocol.json'
 if not pp.exists():
  savejson(pp,dict(version=1,purpose='package4 delayed-final-error learning; separate from primary prefix supervision',seeds=[17,43,91],modes=['rlt','tbptt'],width=32,task='parity',steps=2000,batch=32,length=32,lr=.003,weight_decay=.01,clip=1.,tbptt=8,gradient_routes='TBPTT detaches s and decoder KV only; encoder/global memory stays differentiable',loss='final target only cross entropy',selection='minimum validation final CE at every100 steps',validation_n=256,validation_offset=6200000,train_offset=6100000,test_offset=6900000,test_lengths=[32,128],test_n=512,k=0));(ROOT/'final-only-protocol.sha256').write_text(sha(pp));event('final_only_protocol_frozen',sha256=sha(pp))
 p=json.loads(pp.read_text());assert sha(pp)==(ROOT/'final-only-protocol.sha256').read_text();return p
def run():
 p=freeze();pp=ROOT/'final-only-protocol.json';records=[]
 for seed in p['seeds']:
  for mode in p['modes']:
   c=dict(task='parity',seed=seed,width=32,mode=mode);name=f'finalonly-parity-w32-{mode}-s{seed}';rp=RES/(name+'.json')
   if rp.exists():records.append(json.loads(rp.read_text()));continue
   m=make(c);torch.save(m.state_dict(),CK/(name+'-initial.pt'));opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);vx,vy=dataset(c,256,32,p['validation_offset']);best=float('inf');history=[];start=time.perf_counter();event('final_only_train_start',condition=name)
   for step in range(1,2001):
    x,y=dataset(c,32,32,p['train_offset']+step*1009);opt.zero_grad(set_to_none=True);z=m(x,tbptt=8 if mode=='tbptt' else 0);loss=F.cross_entropy(z[:,-1],y[:,-1]);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
    if step%100==0:
     with torch.no_grad():
      allz=torch.cat([m(vx[i:i+32])[:,-1] for i in range(0,256,32)]);vce=float(F.cross_entropy(allz,vy[:,-1]));vacc=float((allz.argmax(-1)==vy[:,-1]).float().mean())
     row=dict(step=step,train_loss=float(loss.detach()),grad_norm=grad,validation_final_ce=vce,validation_final_accuracy=vacc,elapsed=time.perf_counter()-start);history.append(row)
     if vce<best:best=vce;selected=step;torch.save(m.state_dict(),CK/(name+'-selected.pt'))
     event('final_only_validation',condition=name,**row)
   training_seconds=time.perf_counter()-start;torch.save(m.state_dict(),CK/(name+'-last.pt'));m.load_state_dict(torch.load(CK/(name+'-selected.pt'),weights_only=True));evaluations=[]
   for length in p['test_lengths']:
    x,y=dataset(c,512,length,p['test_offset']);stats,pred=eval_data(m,x,y);import numpy as np
    path=RES/(name+f'-L{length}.npz');np.savez_compressed(path,inputs=x.numpy(),predictions=pred,targets=y.numpy(),example_id=np.arange(512));evaluations.append(dict(length=length,**stats,file=str(path),sha256=sha(path)))
   record=dict(condition=c,supervision='final_only',steps=2000,selected_step=selected,train_seconds=training_seconds,parameters=m.counts(),history=history,evaluation=evaluations,hashes={s:sha(CK/(name+'-'+s+'.pt')) for s in ['initial','selected','last']});savejson(rp,record);records.append(record);savejson(RES/'final-only-metrics.json',dict(protocol_hash=sha(pp),records=records,status='COMPLETE' if len(records)==6 else 'RUNNING'))
 event('final_only_complete',runs=len(records))
if __name__=='__main__':run()
