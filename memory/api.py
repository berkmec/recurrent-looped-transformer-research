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
def facet(event): return ('absolute' if event.get('parent') is None or event.get('kind')=='root' else 'edge')+':'+str(event.get('child'))
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
        self.db.execute('CREATE TABLE IF NOT EXISTS compacts(id INTEGER PRIMARY KEY,task TEXT,title TEXT,version INTEGER,raw TEXT,sha TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY,task TEXT,title TEXT,raw TEXT,sha TEXT)')
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
        later=[x for x in self.history(task,title) if x['version']>r['version'] and x['event'].get('kind')!='query' and facet(x['event'])==facet(r['event'])]
        return not later
    def search(self,text,k=5):
        q=embed(text); rows=self.db.execute('SELECT id,vector FROM records').fetchall()
        scores=[(key,float(np.dot(q,np.frombuffer(v,dtype=np.float32)))) for key,v in rows]
        return [dict(id=key,score=score) for key,score in sorted(scores,key=lambda x:(-x[1],x[0]))[:k]]
    def count(self): return self.db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    def save_compact(self,task,title,snapshot):
        raw=canonical(snapshot); sha=hashlib.sha256(raw.encode()).hexdigest()
        cur=self.db.execute('INSERT INTO compacts(task,title,version,raw,sha) VALUES(?,?,?,?,?)',(task,title,snapshot['version'],raw,sha)); self.db.commit(); return cur.lastrowid
    def read_compact(self,key):
        row=self.db.execute('SELECT raw,sha FROM compacts WHERE id=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError('COMPACT_HASH_MISMATCH')
        return json.loads(row[0])
    def append_observation(self,task,title,text,parsed):
        raw=canonical(dict(task=task,title=title,text=text,parsed=parsed)); sha=hashlib.sha256(raw.encode()).hexdigest()
        cur=self.db.execute('INSERT INTO observations(task,title,raw,sha) VALUES(?,?,?,?)',(task,title,raw,sha)); self.db.commit(); return cur.lastrowid
    def read_observation(self,key):
        row=self.db.execute('SELECT raw,sha FROM observations WHERE id=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError('OBSERVATION_HASH_MISMATCH')
        return json.loads(row[0])
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
        result[facet(e)]=(child,parent,float(value))
    return result

def symbolic_values(events):
    """Explicit symbolic oracle, only for engineering metrics and labels."""
    c=constraints(events); values={}; graph={}; queue=[]
    for child,parent,value in c.values():
        graph.setdefault(child,[])
        if parent is None:
            if child in values and abs(values[child]-value)>1e-9: raise ValueError('INCONSISTENT_ANCHOR')
            values[child]=value; queue.append(child)
        else:
            graph.setdefault(parent,[]); graph[parent].append((child,value)); graph[child].append((parent,-value))
    for key in queue:
        for other,offset in graph[key]:
            expected=values[key]+offset
            if other in values:
                if abs(values[other]-expected)>1e-8: raise ValueError('INCONSISTENT_CONSTRAINT')
            else: values[other]=expected; queue.append(other)
    if set(graph)-set(values): raise ValueError('UNANCHORED_COMPONENT')
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
            self._persist_summary(key,history)
        self.active.clear(); self.compactions+=1
    def _persist_summary(self,key,history):
        snapshot=dict(version=history[-1]['version'],conditions=constraints([x['event'] for x in history]),sha=digest(history))
        compact_id=self.archive.save_compact(*key,snapshot)
        self.summaries[key]=dict(version=snapshot['version'],sha=snapshot['sha'],compact_id=compact_id,valid=True)
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
    def observe_text(self,task,title,text,parsed=None):
        if parsed is None: parsed=parse_text(text)
        observation_id=self.archive.append_observation(task,title,text,parsed)
        ids=[self.append(task,title,{**e,'source_observation_id':observation_id}) for e in parsed.get('events',[]) if parsed.get('valid',False)]
        return dict(observation_id=observation_id,record_ids=ids,parsed=parsed)
    def reconstruct(self,task,title):
        key=(task,title); history=self.archive.history(task,title); self.read_records+=len(history)
        if key in self.summaries: self.archive.read_compact(self.summaries[key]['compact_id'])
        if key in self.summaries and not self.summaries[key]['valid']:
            self.reopens+=1; self._persist_summary(key,history)
        return [x['event'] for x in history]
    def compact_events(self,task,title):
        # Reopen first when obligations expired; serialize constraint boundary.
        events=self.reconstruct(task,title); c=constraints(events)
        return [dict(kind='root' if p is None else 'edge',child=child,parent=p,value=v) for child,p,v in c.values()]

def verify_candidate(candidate,events,query,tolerance=0.1):
    """Connected-component residual sum bounds query error on anchored affine data.

    Needs candidate values at every node of the queried path. Missing/unsupported
    evidence or a cycle cannot certify. Tolerance applies in raw numeric units.
    """
    if tolerance<0: raise ValueError('negative tolerance')
    try:
        c=constraints(events); graph={}; anchors={}
        for child,parent,value in c.values():
            graph.setdefault(child,[])
            if parent is None: anchors[child]=value
            else: graph.setdefault(parent,[]); graph[child].append((parent,-value)); graph[parent].append((child,value))
        if query not in graph: raise ValueError('MISSING_QUERY')
        # Relative potentials are an evidence consistency check. They are not
        # substituted into candidate predictions or returned as model answers.
        potentials={query:0.}; queue=[query]
        for key in queue:
            for other,offset in graph[key]:
                expected=potentials[key]+offset
                if other in potentials:
                    if abs(potentials[other]-expected)>1e-8: raise ValueError('INCONSISTENT_CYCLE')
                else: potentials[other]=expected; queue.append(other)
        implied_origins=[value-potentials[key] for key,value in anchors.items() if key in potentials]
        if not implied_origins or max(implied_origins)-min(implied_origins)>1e-8: raise ValueError('UNANCHORED_OR_INCONSISTENT')
        bound=0.; steps=0
        for child,parent,value in c.values():
            if child not in potentials: continue
            if child not in candidate or not math.isfinite(float(candidate[child])): raise ValueError('MISSING_CANDIDATE')
            if parent is None: residual=abs(float(candidate[child])-value)
            else:
                if parent not in candidate or not math.isfinite(float(candidate[parent])): raise ValueError('MISSING_PARENT')
                residual=abs(float(candidate[child])-float(candidate[parent])-value)
            bound+=residual; steps+=1
        return dict(status='VERIFIED' if bound<=tolerance else 'CONTINUE',bound=bound,reason='component_residual_bound',checked_edges=steps)
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
def load_parser(checkpoint=None):
    from .extractor import LearnedParser
    checkpoint=str(checkpoint or Path(__file__).parent/'checkpoints'/'extractor-17.pt')
    if checkpoint not in _PARSERS: _PARSERS[checkpoint]=LearnedParser(checkpoint)
    return _PARSERS[checkpoint]

def render_event(event,style='seen',variant=0):
    """Finite experimental renderer; no parser tuning from rendered results."""
    from .extractor import TRAIN_TEMPLATES,HELDOUT_TEMPLATES
    kind=event['kind']; parent=event.get('parent')
    cls=4 if kind=='query' else (2 if parent is None else 3) if kind=='revision' else 0 if kind=='root' else 1
    templates=(TRAIN_TEMPLATES if style=='seen' else HELDOUT_TEMPLATES)[cls]
    value=event.get('value'); textvalue='0' if value is None else format(float(value),'.8f').rstrip('0').rstrip('.')
    if textvalue in ('','-0'): textvalue='0'
    return templates[variant%len(templates)].format(c=event['child'],p=parent,v=textvalue)

def parse_text(text,checkpoint=None):
    """Learned finite-language extraction; returns observations, never answers."""
    return load_parser(checkpoint).parse(text)
