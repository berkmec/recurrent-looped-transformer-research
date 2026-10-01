"""Small learned controlled-English event extractor. No semantic regex parser.

Regex only tokenizes text, classifies copy-eligible entity/number token types,
and validates well-formed output. BiGRU learns event class and C/P/V token roles.
"""
import json, random, re, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
ROOT=Path(__file__).parent
def tokenize(text): return re.findall(r'x\d+|-?\d+(?:\.\d+)?|[A-Za-z_]+|[^\s]',text.lower())
def normalize_token(token):
    if re.fullmatch(r'x\d+',token): return '<entity>'
    if re.fullmatch(r'-?\d+(?:\.\d+)?',token): return '<number>'
    return token

TRAIN_TEMPLATES={
 0:['set {c} to {v} .','{c} starts at {v} .','the value of {c} is {v} .'],
 1:['{c} equals {p} plus {v} .','from {p} add {v} to obtain {c} .','define {c} as {p} shifted by {v} .'],
 2:['revise {c} to {v} .','now set {c} to {v} .','replace the value of {c} with {v} .'],
 3:['revise {c} to {p} plus {v} .','now define {c} as {p} shifted by {v} .','replace the offset from {p} to {c} with {v} .'],
 4:['query {c} .','what is {c} ?','report the value of {c} .']}
HELDOUT_TEMPLATES={
 0:['initially {c} has value {v} .','{v} is the starting value of {c} .'],
 1:['add {v} to {p} giving {c} .','the value of {c} is {v} plus {p} .'],
 2:['the value of {c} is now {v} .','{v} replaces the value of {c} .'],
 3:['the offset from {p} to {c} is now {v} .','revise the offset to {v} from {p} to {c} .'],
 4:['give {c} .','the requested value is {c} .']}

def reconstruct_event(tokens,roles,cls):
    if cls==5: raise ValueError('REJECT')
    selected={role:[t for t,r in zip(tokens,roles) if r==role] for role in (1,2,3)}
    edge=cls in (1,3); query=cls==4
    if [len(selected[i]) for i in (1,2,3)] != [1,int(edge),int(not query)]: raise ValueError('ROLE_CARDINALITY')
    child=selected[1][0]; parent=selected[2][0] if edge else None
    if not re.fullmatch(r'x\d+',child) or (parent is not None and not re.fullmatch(r'x\d+',parent)): raise ValueError('ENTITY_TYPE')
    value=None if query else float(selected[3][0])
    if value is not None and not np.isfinite(value): raise ValueError('NUMBER_TYPE')
    return dict(kind=['root','edge','revision','revision','query'][cls],child=child,parent=parent,value=value)

def make_dataset(split,n,seed):
    rng=random.Random(seed+{'train':0,'validation':100000,'test':200000}[split]); rows=[]
    base={'train':0,'validation':1000,'test':2000}[split]
    for i in range(n):
        negative=i%4==3; cls=rng.randrange(5); c='x'+str(base+rng.randrange(100)); p='x'+str(base+100+rng.randrange(100)); v=f'{rng.uniform(-1,1):.3f}'
        family='heldout' if split=='test' and i%2==0 else 'seen'
        template=rng.choice((HELDOUT_TEMPLATES if family=='heldout' else TRAIN_TEMPLATES)[cls]); text=template.format(c=c,p=p,v=v)
        if negative:
            # Training includes rejection semantics; heldout conflict/negation
            # challenge retains entities and numbers to resist copy-only success.
            noise=rng.choice(['do not '+text,'false : '+text,text+' but this is not a fact .',text+' and '+c+' equals '+str(float(v)+2)+' .'])
            text=noise; out_class=5; event=None
        else: out_class=cls; event=dict(kind=['root','edge','revision','revision','query'][cls],child=c,parent=p if cls in (1,3) else None,value=None if cls==4 else float(v))
        tokens=tokenize(text); roles=[0 if negative else (1 if t==c else 2 if t==p and cls in (1,3) else 3 if t==v and cls!=4 else 0) for t in tokens]
        rows.append(dict(id=f'{split}-{seed}-{i}',graph_id=f'{split}-graph-{seed}-{i}',family=family,text=text,tokens=tokens,roles=roles,**{'class':out_class},event=event))
    return rows

class Tagger(nn.Module):
    def __init__(self,vocab_size,hidden=48):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,32,padding_idx=0); self.gru=nn.GRU(32,hidden,batch_first=True,bidirectional=True); self.tags=nn.Linear(hidden*2,4); self.cls=nn.Linear(hidden*2,6)
    def forward(self,x,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(x),lengths.cpu(),batch_first=True,enforce_sorted=False); out,h=self.gru(packed); out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=x.shape[1]); return self.tags(out),self.cls(torch.cat([h[-2],h[-1]],dim=1))

def encode(rows,vocab):
    lengths=torch.tensor([len(r['tokens']) for r in rows]); x=torch.zeros(len(rows),int(lengths.max()),dtype=torch.long); y=torch.full_like(x,-100)
    for i,r in enumerate(rows):
        x[i,:lengths[i]]=torch.tensor([vocab.get(normalize_token(t),1) for t in r['tokens']]); y[i,:lengths[i]]=torch.tensor(r['roles'])
    return x,lengths,y,torch.tensor([r['class'] for r in rows])

class LearnedParser:
    def __init__(self,checkpoint):
        self.path=Path(checkpoint); ck=torch.load(self.path,map_location='cpu',weights_only=False); self.vocab=ck['vocab']; self.model=Tagger(len(self.vocab)); self.model.load_state_dict(ck['state']); self.model.eval(); self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest(); self.threshold=.65
    def parse_batch(self,texts):
        rows=[dict(tokens=tokenize(t),roles=[0]*len(tokenize(t)),**{'class':0}) for t in texts]
        # Empty text has no learned event and must not crash the recurrent model.
        nonempty=[i for i,r in enumerate(rows) if r['tokens']]; result=[dict(events=[],confidence=0.,valid=False,abstained=True,provenance={'checkpoint_sha256':self.sha,'reason':'empty_text'}) for _ in rows]
        if not nonempty: return result
        x,l,_,_=encode([rows[i] for i in nonempty],self.vocab)
        with torch.no_grad(): tags,classes=self.model(x,l); tp=tags.softmax(-1); cp=classes.softmax(-1)
        for j,i in enumerate(nonempty):
            cls=int(cp[j].argmax()); roles=tp[j,:l[j]].argmax(-1).tolist(); conf=float(cp[j].max()); selected=[float(tp[j,k,r]) for k,r in enumerate(roles) if r>0]
            if selected: conf=min(conf,min(selected))
            try:
                if conf<self.threshold: raise ValueError('LOW_CONFIDENCE')
                event=reconstruct_event(rows[i]['tokens'],roles,cls); events=[event]; valid=True; reason='learned_roles_copied_tokens'
            except (ValueError,IndexError): events=[]; valid=False; reason='learned_rejection_or_invalid_roles'
            result[i]=dict(events=events,confidence=conf,valid=valid,abstained=not valid,provenance=dict(checkpoint_sha256=self.sha,text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),reason=reason,model_class=cls))
        return result
    def parse(self,text): return self.parse_batch([text])[0]

def train_all():
    protocol=json.loads((ROOT/'protocol.json').read_text()); psha=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(); out=ROOT/'results'; out.mkdir(exist_ok=True); (ROOT/'checkpoints').mkdir(exist_ok=True); metrics=[]
    with (ROOT/'execution.jsonl').open('a') as log:
        log.write(json.dumps(dict(event='extractor_train_start',time=time.time(),protocol_sha256=psha))+'\n')
    for seed in protocol['seeds']:
        started=time.perf_counter(); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        train=make_dataset('train',3072,seed); val=make_dataset('validation',384,seed)
        vocab={'<pad>':0,'<unk>':1}
        for row in train:
            for t in row['tokens']:
                if normalize_token(t) not in vocab: vocab[normalize_token(t)]=len(vocab)
        model=Tagger(len(vocab)); initial=ROOT/'checkpoints'/f'extractor-{seed}-initial.pt'; torch.save(dict(state=model.state_dict(),vocab=vocab),initial); opt=torch.optim.Adam(model.parameters(),lr=.003)
        x,l,y,c=encode(train,vocab); losses=[]
        for epoch in range(18):
            model.train(); order=torch.randperm(len(train)); total=0.
            for ids in order.split(64):
                opt.zero_grad(); tags,classes=model(x[ids],l[ids]); loss=nn.functional.cross_entropy(tags.reshape(-1,4),y[ids].reshape(-1))+nn.functional.cross_entropy(classes,c[ids]); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); total+=float(loss)
            losses.append(total/(len(train)/64))
        ck=ROOT/'checkpoints'/f'extractor-{seed}.pt'; torch.save(dict(state=model.state_dict(),vocab=vocab,seed=seed,protocol_sha256=psha),ck); parser=LearnedParser(ck)
        validation=parser.parse_batch([r['text'] for r in val]); vacc=sum((p['events'][0] if p['events'] else None)==r['event'] for p,r in zip(validation,val))/len(val)
        # Final evaluation identities created only after fixed final checkpoint.
        test=make_dataset('test',768,seed); predictions=parser.parse_batch([r['text'] for r in test]); records=[]
        for row,p in zip(test,predictions):
            pred=p['events'][0] if p['events'] else None; records.append({**row,'prediction':p,'exact':pred==row['event'],'oracle_event':row['event']})
        (out/f'extractor-{seed}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        positive=[r for r in records if r['event'] is not None]; negative=[r for r in records if r['event'] is None]; edges=[r for r in positive if r['event']['parent'] is not None]
        m=dict(seed=seed,n=len(records),positive_n=len(positive),negative_n=len(negative),exact_tuple_accuracy=sum(r['exact'] for r in positive)/len(positive),edge_accuracy=sum(r['exact'] for r in edges)/len(edges),all_accuracy=sum(r['exact'] for r in records)/len(records),abstain_rate=sum(r['prediction']['abstained'] for r in records)/len(records),negative_false_accept_rate=sum(r['prediction']['valid'] for r in negative)/len(negative),validation_accuracy=vacc,oracle_exact_tuple_accuracy=1.,wall_seconds=time.perf_counter()-started,initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),checkpoint_sha256=parser.sha,losses=losses)
        for family in ['seen','heldout']:
            group=[r for r in positive if r['family']==family]; m[family+'_exact_accuracy']=sum(r['exact'] for r in group)/len(group)
        metrics.append(m); print(json.dumps(m),flush=True)
        np.savez_compressed(out/f'extractor-{seed}.npz',exact=np.array([r['exact'] for r in records]),accepted=np.array([r['prediction']['valid'] for r in records]),negative=np.array([r['event'] is None for r in records]),ids=np.array([r['id'] for r in records]))
        with (ROOT/'execution.jsonl').open('a') as log: log.write(json.dumps(dict(event='extractor_seed_complete',time=time.time(),protocol_sha256=psha,**m))+'\n')
    (out/'extractor-metrics.json').write_text(json.dumps(metrics,indent=2)); return metrics

if __name__=='__main__': train_all()
