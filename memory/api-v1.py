"""Public engineering contracts. Symbolic checks never replace learned answers.

Events: kind root/edge/revision/query, child entity, parent entity or None,
value numeric or None. Revision with parent=None revises an absolute root;
revision with parent revises an edge offset. Multiple events are chronological.
"""
import hashlib, json, math, sqlite3, re
from pathlib import Path
import numpy as np

def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()
def embed(text,dim=128):
    """Deterministic signed hashed bag-of-words; no semantic embedding claim."""
    v=np.zeros(dim,dtype=np.float32)
    for word in re.findall(r'[a-z0-9_]+',text.lower()):
        h=hashlib.sha256(word.encode()).digest(); v[int.from_bytes(h[:4],'little')%dim]+=1 if h[4]%2 else -1
    return v/max(float(np.linalg.norm(v)),1e-9)

class Archive:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        self.db.execute('CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,task TEXT,title TEXT,version INTEGER,raw TEXT,sha TEXT,vector BLOB)')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS versions ON records(task,title,version)'); self.db.commit()
    def append(self,task,title,event):
        version=self.db.execute('SELECT COALESCE(MAX(version),0)+1 FROM records WHERE task=? AND title=?',(task,title)).fetchone()[0]
        record=dict(task=task,title=title,version=version,event=event); raw=canonical(record); sha=hashlib.sha256(raw.encode()).hexdigest(); vector=embed(raw).tobytes()
        cur=self.db.execute('INSERT INTO records(task,title,version,raw,sha,vector) VALUES(?,?,?,?,?,?)',(task,title,version,raw,sha,vector)); self.db.commit(); return cur.lastrowid
    def read(self,key):
        row=self.db.execute('SELECT raw,sha FROM records WHERE id=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError('RAW_HASH_MISMATCH')
        return json.loads(row[0])
    def history(self,task,title): return [self.read(r[0]) for r in self.db.execute('SELECT id FROM records WHERE task=? AND title=? ORDER BY version',(task,title)).fetchall()]
    def keys(self,task,title): return [r[0] for r in self.db.execute('SELECT id FROM records WHERE task=? AND title=? ORDER BY version',(task,title)).fetchall()]
    def applicable(self,key,task,title,child=None):
        r=self.read(key)
        if r['task']!=task or r['title']!=title: return False
        if child is not None and r['event'].get('child')!=child: return False
        # Semantic version is per child assertion, not global title version:
        # adding an unrelated constraint does not invalidate an older valid edge.
        if r['event'].get('kind')=='query': return False
        later=[x for x in self.history(task,title) if x['version']>r['version'] and x['event'].get('kind')!='query' and x['event'].get('child')==r['event'].get('child')]
        return not later
    def search(self,text,k=5):
        q=embed(text); rows=self.db.execute('SELECT id,vector FROM records').fetchall()
        scores=[(key,float(np.dot(q,np.frombuffer(v,dtype=np.float32)))) for key,v in rows]
        return [dict(id=key,score=score) for key,score in sorted(scores,key=lambda x:(-x[1],x[0]))[:k]]
    def count(self): return self.db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    def close(self): self.db.close()

def constraints(events):
    result={}
    for e in events:
        kind=e.get('kind')
        if kind=='query': continue
        if kind not in ('root','edge','revision'): raise ValueError('UNSUPPORTED')
        child=e.get('child'); parent=e.get('parent'); value=e.get('value')
        if not isinstance(child,str) or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError('INVALID')
        if kind=='root': parent=None
        if kind=='edge' and parent is None: raise ValueError('INVALID_EDGE')
        result[child]=(parent,float(value))
    return result

def symbolic_values(events):
    """Explicit symbolic oracle, only for engineering metrics and labels."""
    c=constraints(events); values={}
    def visit(key,seen):
        if key in values: return values[key]
        if key not in c or key in seen: raise ValueError('MISSING_OR_CYCLIC')
        parent,value=c[key]; values[key]=value if parent is None else visit(parent,seen|{key})+value; return values[key]
    for key in c: visit(key,set())
    return values

class TitleController:
    """Active budget counts resident raw records. Compact scopes live in archive.

    Capacity excludes storage and compact metadata; reconstruction is an explicit
    external read buffer, not resident title memory. Compaction preserves every
    currently applicable affine constraint, not only a computed answer. Stable
    title identity comes from caller-defined distinct subproblem keys.
    """
    def __init__(self,capacity,archive):
        if capacity<2: raise ValueError('capacity must be >=2')
        self.capacity=capacity; self.archive=archive; self.active=[]; self.summaries={}; self.titles=set(); self.compactions=0; self.premature_compactions=0; self.reopens=0; self.read_records=0; self.max_active=0
    @property
    def active_count(self): return len(self.active)
    def _compact(self):
        if len(self.active)<self.capacity: self.premature_compactions+=1; raise ValueError('PREMATURE_COMPACTION')
        for key in {(self.archive.read(i)['task'],self.archive.read(i)['title']) for i in self.active}:
            history=self.archive.history(*key); self.read_records+=len(history)
            self.summaries[key]=dict(version=history[-1]['version'],conditions=constraints([x['event'] for x in history]),sha=digest(history),valid=True)
        self.active.clear(); self.compactions+=1
    def append(self,task,title,event):
        if self.active_count==self.capacity: self._compact()
        key=(task,title); self.titles.add(key)
        if key in self.summaries and event.get('kind')!='query':
            # All late constraints require reconstruction; revisions explicitly
            # invalidate frozen scope obligations even if values happen to match.
            self.summaries[key]['valid']=False
        raw_id=self.archive.append(task,title,event); self.active.append(raw_id); self.max_active=max(self.max_active,self.active_count); return raw_id
    def summary_status(self,task,title):
        summary=self.summaries.get((task,title)); return 'COMPACTION_REJECTED' if summary and not summary['valid'] else 'CONTINUE'
    def reconstruct(self,task,title):
        key=(task,title); history=self.archive.history(task,title); self.read_records+=len(history)
        if key in self.summaries and not self.summaries[key]['valid']:
            self.reopens+=1; self.summaries[key]=dict(version=history[-1]['version'],conditions=constraints([x['event'] for x in history]),sha=digest(history),valid=True)
        return [x['event'] for x in history]
    def compact_events(self,task,title):
        # Reopen first when obligations expired; serialize constraint boundary.
        events=self.reconstruct(task,title); c=constraints(events)
        return [dict(kind='root' if p is None else 'edge',child=child,parent=p,value=v) for child,(p,v) in c.items()]

def verify_candidate(candidate,events,query,tolerance=0.1):
    """Path residual sum bounds absolute query error without computing an answer.

    Needs candidate values at every node of the queried path. Missing/unsupported
    evidence or a cycle cannot certify. Tolerance applies in raw numeric units.
    """
    if tolerance<0: raise ValueError('negative tolerance')
    try:
        c=constraints(events); key=query; seen=set(); bound=0.; steps=0
        while True:
            if key in seen or key not in c or key not in candidate: raise ValueError('MISSING_OR_CYCLIC')
            seen.add(key); parent,value=c[key]; pred=float(candidate[key])
            if not math.isfinite(pred): raise ValueError('NONFINITE')
            if parent is None: residual=abs(pred-value)
            else:
                if parent not in candidate or not math.isfinite(float(candidate[parent])): raise ValueError('MISSING_PARENT')
                residual=abs(pred-float(candidate[parent])-value)
            bound+=residual; steps+=1
            if parent is None: break
            key=parent
        return dict(status='VERIFIED' if bound<=tolerance else 'CONTINUE',bound=bound,reason='path_residual_bound',checked_edges=steps)
    except (ValueError,TypeError,KeyError): return dict(status='UNRESOLVED_BUDGET',bound=None,reason='unsupported_or_incomplete_evidence',checked_edges=0)

def halt_candidates(iterator,events,query,tolerance=0.1,max_steps=8):
    result=dict(status='UNRESOLVED_BUDGET',bound=None,steps=0,candidate=None)
    iterator=iter(iterator)
    for step in range(1,max_steps+1):
        try: candidate=next(iterator)
        except StopIteration: break
        check=verify_candidate(candidate,events,query,tolerance)
        result={**check,'steps':step,'candidate':candidate}
        if check['status']=='VERIFIED': return result
    result['status']='UNRESOLVED_BUDGET'; return result

_PARSERS={}
def parse_text(text,checkpoint=None):
    """Learned finite-language extraction; returns observations, never answers."""
    from .extractor import LearnedParser
    checkpoint=str(checkpoint or Path(__file__).parent/'checkpoints'/'extractor-17.pt')
    if checkpoint not in _PARSERS: _PARSERS[checkpoint]=LearnedParser(checkpoint)
    return _PARSERS[checkpoint].parse(text)
