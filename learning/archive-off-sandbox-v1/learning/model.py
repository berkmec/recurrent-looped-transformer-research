"""Learned event/title research model; no dataset, solver, target or archive imports.

This is a controlled-domain architecture, not the pinned official RLT decoder.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .token_core import TokenCore


def revision_mask(links,changed,radius,active):
    mask=F.one_hot(changed,links.shape[-1]).to(links.dtype)
    adjacency=((links+links.transpose(1,2))>0).to(links.dtype)
    for _ in range(radius):mask=torch.maximum(mask,(adjacency@mask.unsqueeze(-1)).squeeze(-1).clamp(0,1))
    return mask*active


class EventModel(nn.Module):
    def __init__(self,width=32,heads=2,ff_width=64,max_ids=10,feedback=False,swa_window=4):
        super().__init__();self.width=width;self.max_ids=max_ids;self.feedback=feedback;self.swa_window=swa_window
        if max_ids!=10 or ff_width!=2*width:raise ValueError('current source-style core requires10 IDs and ff_width=2*width')
        self.token_core=TokenCore(width,heads,swa_window,classes=1,feedback=True,alpha=.1)
        self.anchor=nn.Linear(4,width)
        self.self_map=nn.Linear(width,width,bias=False)
        self.parent_map=nn.Linear(width,width,bias=False)
        self.child_map=nn.Linear(width,width,bias=False)
        self.child_anchor=nn.Linear(4,width,bias=False)
        self.title_head=nn.Linear(width,1)
        self.token_head=nn.Linear(width,1)
        self.feedback_map=nn.Linear(width,width,bias=False)

    def refine(self,z,anchors,links,active,direction):
        if direction not in ('forward','bidir','wrong','flat'):raise ValueError('invalid graph direction')
        p=links
        if direction=='wrong':p=torch.roll(p,(1,1),(1,2))
        elif direction=='flat':p=active[:,:,None]*active[:,None,:]
        p=p*active[:,:,None]*active[:,None,:]
        c=p.transpose(1,2)
        p=p/p.sum(-1,keepdim=True).clamp_min(1)
        c=c/c.sum(-1,keepdim=True).clamp_min(1)
        if direction=='forward':c=torch.zeros_like(c)
        proposal=self.self_map(z)+self.parent_map(p@z)+self.child_map(c@z)+self.anchor(anchors)+self.child_anchor(c@anchors)
        return (.25*z+.75*torch.tanh(proposal))*active[:,:,None]

    def _validate(self,events):
        if events.ndim!=3 or events.shape[-1]!=4 or events.shape[1]==0:raise ValueError('events must be nonempty [B,T,4]')
        if not torch.isfinite(events).all():raise ValueError('nonfinite event')
        if not torch.equal(events[:,:,:3],events[:,:,:3].round()):raise ValueError('op and IDs must be integers')
        if ((events[:,:,0]<0)|(events[:,:,0]>6)|(events[:,:,0]==5)).any():raise ValueError('invalid operation')
        if ((events[:,:,1:3]<0)|(events[:,:,1:3]>=self.max_ids)).any():raise ValueError('ID outside capacity')
        q=events[:,:,0]==2
        if ((q.any(0))&(~q.all(0))).any():raise ValueError('query positions must align within batch')

    def new_runtime(self,events,direction='bidir',policy='retain',radius=1,intervention=None,trace=False):
        self._validate(events)
        return Runtime(self,events,direction,policy,radius,intervention,trace)

    def forward(self,events,direction='bidir',policy='retain',rounds=6,radius=1,intervention=None,return_trace=False,post_rounds=None):
        r=self.new_runtime(events,direction,policy,radius,intervention,return_trace)
        predictions=[];nodes=[]
        while r.consume_to_query():
            count=rounds if len(predictions)==0 or post_rounds is None else post_rounds
            for _ in range(count):r.refine_one()
            result=r.candidate();predictions.append(result['prediction']);nodes.append(result['nodes']);r.commit_feedback()
        if not predictions:raise ValueError('at least one query required')
        return {'predictions':torch.stack(predictions,1),'nodes':torch.stack(nodes,1),'trace':r.trace if return_trace else None}


class Runtime:
    """Per-request state. Refinement never consumes a token or writes token cache."""
    def __init__(self,model,events,direction,policy,radius,intervention,trace):
        if policy not in ('retain','all','local'):raise ValueError('invalid policy')
        self.model=model;self.events=events;self.direction=direction;self.policy=policy;self.radius=radius;self.intervention=intervention
        B=events.shape[0];N=model.max_ids;D=model.width
        self.s=events.new_zeros(B,D);self.z=events.new_zeros(B,N,D);self.a=events.new_zeros(B,N,4)
        self.active=events.new_zeros(B,N);self.links=events.new_zeros(B,N,N)
        self.versions=torch.zeros(B,N,dtype=torch.long,device=events.device);self.cache_versions=self.versions.clone()
        self.cache=[];self.pending=None;self.t=0;self.k=0;self.qids=None;self.query_open=False
        op=events[:,:,0].long();ids=events[:,:,1].long();parents=events[:,:,2].long()
        features=torch.cat((F.one_hot(op,7),F.one_hot(ids,N),F.one_hot(parents,N),events[:,:,3:4]),-1).to(events.dtype)
        self.encoder,_,_=model.token_core.encode(features)
        normalized=model.token_core.mn(self.encoder)
        self.global_k=model.token_core.rope(model.token_core.split(model.token_core.mk(normalized)))
        self.global_v=model.token_core.split(model.token_core.mv(normalized))
        self.decoder_state=None
        self.record=trace;self.trace={'token_clocks':[],'cache_appends':0,'refinements':[],'versions':[]}

    def consume_to_query(self):
        if self.query_open:raise RuntimeError('commit previous query before consuming token')
        m=self.model;B=self.events.shape[0]
        while self.t<self.events.shape[1]:
            event=self.events[:,self.t];op=event[:,0].long();ids=event[:,1].long();parent=event[:,2].long()
            previous=self.decoder_state
            if self.pending is not None:
                previous=dict(previous);previous['s']=previous['s']+self.pending;self.pending=None
            _,self.decoder_state=m.token_core.decode(self.encoder[:,self.t],previous,self.global_k[:,:,:self.t+1],self.global_v[:,:,:self.t+1],self.t)
            self.s=self.decoder_state['s']
            self.cache=list(range(self.decoder_state['dk'].shape[2]))
            self.t+=1
            if self.record:self.trace['token_clocks'].append(self.t);self.trace['cache_appends']+=1
            if bool((op==2).all()):
                self.qids=ids;self.k=0;self.query_open=True;return True
            write=(op!=6).to(event.dtype)
            select=F.one_hot(ids,m.max_ids).to(event.dtype)*write[:,None]
            absolute=((op==0)|(op==3)).to(event.dtype)
            edge=((op==1)|(op==4)).to(event.dtype)
            # Absolute evidence on an existing child leaves its offset constraint intact.
            newa=self.a.clone()
            newa[:,:,0]=self.a[:,:,0]*(1-select*edge[:,None])+select*edge[:,None]*event[:,3,None]
            newa[:,:,1]=self.a[:,:,1]*(1-select*absolute[:,None])+select*absolute[:,None]*event[:,3,None]
            newa[:,:,2]=torch.maximum(self.a[:,:,2],select*absolute[:,None])
            newa[:,:,3]=torch.maximum(self.a[:,:,3],select)
            parent_select=F.one_hot(parent,m.max_ids).to(event.dtype)*edge[:,None]
            newa[:,:,3]=torch.maximum(newa[:,:,3],parent_select)
            self.a=newa;self.active=torch.maximum(self.active,torch.maximum(select,parent_select))
            edgesel=select*edge[:,None]
            self.links=self.links*(1-edgesel[:,:,None])+edgesel[:,:,None]*F.one_hot(parent,m.max_ids)[:,None,:]
            self.versions=self.versions+select.long()
            # Directly written title is always refreshed; other titles use policy mask.
            mask=select
            revised=((op==3)|(op==4)).to(event.dtype)
            if self.policy=='all':mask=torch.maximum(mask,self.active*revised[:,None])
            elif self.policy=='local':mask=torch.maximum(mask,revision_mask(self.links,ids,self.radius,self.active)*revised[:,None])
            self.z=self.z*(1-mask[:,:,None])+torch.tanh(m.anchor(self.a))*mask[:,:,None]
            self.cache_versions=torch.where(mask.bool(),self.versions,self.cache_versions)
        return False

    def refine_one(self):
        if not self.query_open:raise RuntimeError('refinement requires pending query')
        t=self.t;cache=len(self.cache)
        self.z=self.model.refine(self.z,self.a,self.links,self.active,self.direction);self.k+=1
        self.cache_versions=self.versions.clone()
        if self.record:self.trace['refinements'].append({'token_before':t,'token_after':self.t,'cache_before':cache,'cache_after':len(self.cache),'k':self.k})
        return self.candidate()

    def candidate(self):
        if not self.query_open:raise RuntimeError('candidate requires pending query')
        values=self.model.title_head(self.z).squeeze(-1)+.1*self.model.token_head(self.s)
        answer=values.gather(1,self.qids[:,None]).squeeze(-1)
        return {'prediction':answer,'nodes':values,'t':self.t,'k':self.k,'versions':self.versions.clone()}

    def commit_feedback(self):
        if not self.query_open:raise RuntimeError('query already committed')
        if self.model.feedback:
            z=self.z
            if self.intervention=='zero':z=torch.zeros_like(z)
            elif self.intervention=='permute':z=torch.roll(z,1,1)
            requested=z[torch.arange(len(z),device=z.device),self.qids]
            self.pending=.2*torch.tanh(self.model.feedback_map(requested))
        if self.record:self.trace['versions'].append(self.versions.detach().cpu().tolist())
        self.query_open=False
