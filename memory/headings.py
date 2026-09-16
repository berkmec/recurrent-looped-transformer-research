"""Deterministic finite-affine heading organization of learned parsed events.

Contract: H(task,entity)=SHA256(canonical task/entity pair). Wording is not an
identity input. A semantic state S is a partial map {absolute,edge}->assertion.
v increases iff a new assertion changes S. Old (v,S_v) snapshots are immutable.
An edge child=parent+offset appends a typed child-heading->parent-heading link.
Only the latest edge facet determines the active link; link history persists.
Queries resolve existing identities without creating them. No numeric solving,
oracle labels, inferred answers, model training, or unrestricted topic discovery.
"""
import copy,hashlib,json,math,re

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def identity(task,entity): return 'heading:'+hashlib.sha256(canonical([task,entity]).encode()).hexdigest()

class HeadingRegistry:
    def __init__(self): self._nodes={}; self._history=[]; self._links=[]
    def _ensure(self,task,entity):
        key=(task,entity)
        if key in self._nodes: return self._nodes[key],False
        node=dict(id=identity(task,entity),task=task,entity=entity,objective=f'Determine affine value of {entity}',version=0,versions=[],facets={})
        self._nodes[key]=node; return node,True
    def _record(self,task,event,result):
        # Deep copying prevents a caller mutating old input/output evidence.
        payload=dict(index=len(self._history)+1,task=task,event=copy.deepcopy(event),result=copy.deepcopy(result),previous_sha256=self._history[-1]['sha256'] if self._history else None)
        payload['sha256']=hashlib.sha256(canonical(payload).encode()).hexdigest(); self._history.append(payload); return copy.deepcopy(result)
    def consume(self,task,event):
        if not isinstance(task,str) or not task: raise ValueError('task must be a nonempty string')
        kind=event.get('kind'); child=event.get('child'); parent=event.get('parent'); value=event.get('value')
        empty=dict(heading_id=None,version=None,created=[],links=[])
        if kind not in ('root','edge','revision','query'): return self._record(task,event,dict(status='UNSUPPORTED',**empty))
        if not isinstance(child,str) or not re.fullmatch(r'x\d+',child): return self._record(task,event,dict(status='MISSING_ENTITY',**empty))
        if kind=='query':
            node=self._nodes.get((task,child))
            return self._record(task,event,dict(status='QUERY_RESOLVED' if node else 'MISSING_ENTITY',heading_id=node['id'] if node else None,version=node['version'] if node else None,created=[],links=[]))
        if kind=='root': parent=None
        if (kind=='edge' and parent is None) or (parent is not None and (not isinstance(parent,str) or not re.fullmatch(r'x\d+',parent))): return self._record(task,event,dict(status='MISSING_ENTITY',**empty))
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value): return self._record(task,event,dict(status='UNSUPPORTED',**empty))
        if parent==child: return self._record(task,event,dict(status='UNSUPPORTED',**empty))
        node,new=self._ensure(task,child); created=[node['id']] if new else []; links=[]
        if parent is not None:
            parent_node,parent_new=self._ensure(task,parent)
            if parent_new: created.append(parent_node['id'])
        facet='absolute' if parent is None else 'edge'; assertion=dict(parent=parent,value=float(value))
        if node['facets'].get(facet)==assertion: return self._record(task,event,dict(status='UNCHANGED',heading_id=node['id'],version=node['version'],created=created,links=[]))
        node['facets'][facet]=assertion; node['version']+=1
        node['versions'].append(copy.deepcopy(dict(version=node['version'],facets=node['facets'],changed_facet=facet)))
        if parent is not None:
            link=dict(type='DEPENDS_ON_AFFINE',source=node['id'],target=parent_node['id'],source_version=node['version'],offset=float(value),task=task)
            self._links.append(copy.deepcopy(link)); links.append(link)
        return self._record(task,event,dict(status='CREATED' if new else 'UPDATED',heading_id=node['id'],version=node['version'],created=created,links=links))
    def consume_parsed(self,task,parsed):
        if not parsed.get('valid',False) or parsed.get('abstained',False):
            result=self._record(task,{'kind':'extraction_unresolved','parsed':copy.deepcopy(parsed)},dict(status='EXTRACTION_UNRESOLVED',heading_id=None,version=None,created=[],links=[]))
            return dict(status='EXTRACTION_UNRESOLVED',results=[result])
        results=[self.consume(task,e) for e in parsed.get('events',[])]
        return dict(status='PROCESSED' if results else 'EXTRACTION_UNRESOLVED',results=results)
    def get(self,task,entity): return copy.deepcopy(self._nodes.get((task,entity)))
    def headings(self,task): return [copy.deepcopy(v) for (t,_),v in self._nodes.items() if t==task]
    def active_links(self,task):
        # Absolute updates leave the latest edge facet and its source version
        # intact. Historical edge replacement never deletes previous links.
        latest={}
        for link in self._links:
            if link['task']==task: latest[link['source']]=link
        return copy.deepcopy(list(latest.values()))
    def serialize(self): return dict(schema='finite-heading-registry-v1',history=copy.deepcopy(self._history))
    @classmethod
    def from_serialized(cls,payload):
        if payload.get('schema')!='finite-heading-registry-v1': raise ValueError('UNSUPPORTED_SCHEMA')
        registry=cls()
        for row in payload['history']:
            body={k:v for k,v in row.items() if k!='sha256'}
            if hashlib.sha256(canonical(body).encode()).hexdigest()!=row['sha256']: raise ValueError('HISTORY_HASH_MISMATCH')
            if row['event'].get('kind')=='extraction_unresolved': registry.consume_parsed(row['task'],row['event']['parsed'])
            else: registry.consume(row['task'],row['event'])
            if registry._history[-1]!=row: raise ValueError('HISTORY_REPLAY_MISMATCH')
        return registry
