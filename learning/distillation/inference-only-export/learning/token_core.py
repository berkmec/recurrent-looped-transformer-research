"""Source-style core copied from sibling core/model.py; no task/oracle imports. Event features replace token IDs."""
import torch
from torch import nn
from torch.nn import functional as F
class RMS(nn.Module):
 def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
 def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight
class TokenCore(nn.Module):
 def __init__(self,width=32,heads=2,window=4,classes=2,feedback=True,alpha=.1):
  super().__init__()
  if window<1 or width%heads or (width//heads)%2:raise ValueError('invalid dimensions/window')
  self.width,self.heads,self.window,self.feedback,self.alpha=width,heads,window,feedback,alpha
  self.embedding=nn.Linear(28,width);self.initial=nn.Parameter(torch.zeros(width))
  for name in ['en','efn','pn','dn','cn','dfn','mn','outn']:setattr(self,name,RMS(width))
  for name in ['eq','ek','ev','eo','merge','dq','dk','dv','do','cq','co','mk','mv']:setattr(self,name,nn.Linear(width,width,bias=False))
  self.gate=nn.Linear(width*2,width);self.eff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width));self.dff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width));self.read=nn.Linear(width,classes)
 def split(self,x):return x.reshape(x.shape[0],-1,self.heads,self.width//self.heads).transpose(1,2)
 def rope(self,x,start=0):
  angle=torch.arange(start,start+x.shape[2],device=x.device,dtype=x.dtype)[:,None]*10000**(-torch.arange(0,x.shape[-1],2,device=x.device,dtype=x.dtype)/x.shape[-1]);c=angle.cos()[None,None];s=angle.sin()[None,None]
  return torch.stack((x[...,::2]*c-x[...,1::2]*s,x[...,::2]*s+x[...,1::2]*c),-1).flatten(-2)
 def attend(self,q,k,v,causal=False):
  return F.scaled_dot_product_attention(q,k,v,is_causal=causal).transpose(1,2).reshape(q.shape[0],q.shape[2],self.width)
 def encode(self,x):
  e=self.embedding(x);n=self.en(e);q=self.rope(self.split(self.eq(n)));k=self.rope(self.split(self.ek(n)));v=self.split(self.ev(n));e=e+self.eo(self.attend(q,k,v,True));e=e+self.eff(self.efn(e));return e,k,v
 def decode(self,e,s,mk,mv,t):
  prev=self.initial.expand(e.shape[0],-1) if s is None else s['s']
  z=e
  if self.feedback:
   r=self.pn(prev);z=z+self.alpha*torch.sigmoid(self.gate(torch.cat([e,r],-1)))*self.merge(r)
  n=self.dn(z);q=self.rope(self.split(self.dq(n)),t);k=self.rope(self.split(self.dk(n)),t);v=self.split(self.dv(n))
  if s is not None:k=torch.cat([s['dk'],k],2);v=torch.cat([s['dv'],v],2)
  z=z+self.do(self.attend(q,k,v)[:,0]);q=self.rope(self.split(self.cq(self.cn(z))),t);z=z+self.co(self.attend(q,mk,mv)[:,0]);z=z+self.dff(self.dfn(z))
  keep=self.window-1
  return self.read(self.outn(z)),dict(s=z,dk=k[:,:,-keep:] if keep else k[:,:,:0],dv=v[:,:,-keep:] if keep else v[:,:,:0],mk=mk,mv=mv,t=t+1)
 def step(self,x,state=None):
  t=0 if state is None else state['t'];e=self.embedding(x);n=self.en(e);q=self.rope(self.split(self.eq(n)),t);k=self.rope(self.split(self.ek(n)),t);v=self.split(self.ev(n))
  if state is not None:k=torch.cat([state['ek'],k],2);v=torch.cat([state['ev'],v],2)
  e=e+self.eo(self.attend(q,k,v)[:,0]);e=e+self.eff(self.efn(e));n=self.mn(e);mk=self.rope(self.split(self.mk(n)),t);mv=self.split(self.mv(n))
  if state is not None:mk=torch.cat([state['mk'],mk],2);mv=torch.cat([state['mv'],mv],2)
  y,s=self.decode(e,state,mk,mv,t);s.update(ek=k,ev=v);return y,s
 def forward(self,x,tbptt=0,stream_encoder=False,parallel_control=True):
  if x.shape[1]<1 or tbptt<0:raise ValueError('empty sequence/negative truncation')
  s=None;out=[]
  if not stream_encoder:
   e,_,_=self.encode(x);n=self.mn(e);mk=self.rope(self.split(self.mk(n)));mv=self.split(self.mv(n))
   if not self.feedback and parallel_control and tbptt==0:
    n=self.dn(e);q=self.rope(self.split(self.dq(n)));k=self.rope(self.split(self.dk(n)));v=self.split(self.dv(n));idx=torch.arange(x.shape[1],device=x.device);delta=idx[:,None]-idx[None,:];mask=(delta>=0)&(delta<self.window)
    a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape_as(e);z=e+self.do(a);q=self.rope(self.split(self.cq(self.cn(z))));z=z+self.co(self.attend(q,mk,mv,True));z=z+self.dff(self.dfn(z));return self.read(self.outn(z))
  for t in range(x.shape[1]):
   if tbptt and t and t%tbptt==0:
    s=dict(s);s.update({k:s[k].detach() for k in ['s','dk','dv']})
   if stream_encoder:y,s=self.step(x[:,t],s)
   else:y,s=self.decode(e[:,t],s,mk[:,:,:t+1],mv[:,:,:t+1],t)
   out.append(y)
  return torch.stack(out,1)
 def counts(self):
  allocated=sum(p.numel() for p in self.parameters());inactive=0
  if not self.feedback:inactive=self.initial.numel()+sum(p.numel() for m in [self.pn,self.gate,self.merge] for p in m.parameters())
  return dict(allocated=allocated,active=allocated-inactive)

