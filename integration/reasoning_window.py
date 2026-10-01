"""Generated-state window with a guarded exact local Markov boundary.

This preserves every generated trace in the archive. At capacity, only the
latest latent title state remains necessary for the next refinement under the
same weights, evidence, token state and graph. Reload is guarded against any
of those conditions changing. It is not free-form semantic summarization, a
bound on total RAM, or a way to resume arbitrary changed-model histories.
"""
import copy,hashlib,json
import torch
from memory.headings import identity


def fingerprint(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode())
            digest.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):digest.update(str(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        else:digest.update(json.dumps(x,sort_keys=True).encode())
    visit(value);return digest.hexdigest()


def obligation(runtime):
    return fingerprint({'weights':runtime.model.state_dict(),'model_training':runtime.model.training,
                        'runtime_config':{name:getattr(runtime,name) for name in ('direction','policy','radius','intervention')},
                        'model_config':{name:getattr(runtime.model,name) for name in ('width','max_ids','feedback','swa_window')},
                        'core_config':{name:getattr(runtime.model.token_core,name) for name in ('width','heads','window','feedback','alpha')},
                        'evidence':runtime.a,'links':runtime.links,'active':runtime.active,
                        'versions':runtime.versions,'cache_versions':runtime.cache_versions,
                        'token_state':runtime.s,'decoder':runtime.decoder_state,'pending':runtime.pending,
                        'events':runtime.events,'encoder':runtime.encoder,
                        'global_k':runtime.global_k,'global_v':runtime.global_v,
                        't':runtime.t,'k':runtime.k,'query':runtime.qids,'query_open':runtime.query_open})


class ReasoningWindow:
    def __init__(self,archive,task,capacity=4,enabled=True):
        if not isinstance(capacity,int) or capacity<2:raise ValueError('reasoning capacity must be >=2')
        self.archive=archive;self.task=task;self.capacity=capacity;self.enabled=enabled
        self.active=[];self.max_active=0;self.compactions=0;self.rereads=0;self.generated=0
        self.last_compact_id=None;self.same_heading_continuations=0;self._seen_headings=set()
        self.reload_equal=True;self.maximum_reload_difference=0.;self.last_heading=None
        self.max_active_serialized_payload_bytes=0

    @torch.no_grad()
    def observe(self,runtime,row,raw_id=None):
        if runtime.k==1:self.active=[]
        heading=identity(self.task,row['query_entity']);self.last_heading=heading
        if raw_id is None:
            raw_id=self.archive.append(self.task,heading+'/generated_reasoning',{'kind':'model_trace',**row})
        self.active.append({'raw_id':raw_id,'generated_state':copy.deepcopy(row)})
        self.generated+=1;self.max_active=max(self.max_active,len(self.active))
        payload_bytes=sum(len(json.dumps(item['generated_state'],separators=(',',':')).encode()) for item in self.active)
        self.max_active_serialized_payload_bytes=max(self.max_active_serialized_payload_bytes,payload_bytes)
        if not self.enabled or len(self.active)<self.capacity:return False
        if len(self.active)!=self.capacity:raise ValueError('reasoning capacity exceeded')
        snapshot={'version':self.compactions+1,'kind':'latent_title_boundary','heading_id':heading,
                  'raw_trace_ids':[item['raw_id'] for item in self.active],'t':runtime.t,'k':runtime.k,
                  'obligation_sha256':obligation(runtime),'shape':list(runtime.z.shape),
                  'dtype':str(runtime.z.dtype),'latest_title_state':runtime.z.detach().cpu().tolist(),
                  'scope':'exact next local refinement under unchanged evidence, graph, token state and weights'}
        key=self.archive.save_compact(self.task,heading+'/generated_reasoning',snapshot)
        self.last_compact_id=key;self.compactions+=1
        before=runtime.z.clone();self.restore(runtime,key)
        self.reload_equal=self.reload_equal and torch.equal(before,runtime.z)
        self.maximum_reload_difference=max(self.maximum_reload_difference,float((before-runtime.z).abs().max()))
        if heading in self._seen_headings:self.same_heading_continuations+=1
        self._seen_headings.add(heading);self.active=[]
        return True

    @torch.no_grad()
    def restore(self,runtime,key):
        snapshot=self.archive.read_compact(key)
        if snapshot['obligation_sha256']!=obligation(runtime):raise ValueError('stale reasoning boundary obligations')
        if list(runtime.z.shape)!=snapshot['shape'] or str(runtime.z.dtype)!=snapshot['dtype']:
            raise ValueError('stale reasoning boundary shape/dtype')
        runtime.z=torch.tensor(snapshot['latest_title_state'],device=runtime.z.device,dtype=runtime.z.dtype)
        self.rereads+=1
