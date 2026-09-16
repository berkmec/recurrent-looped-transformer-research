"""Connected learned extraction, immutable evidence, RLT titles and verification.

The verifier certifies parsed affine constraints only. It neither validates the
original language interpretation nor replaces the learned numeric answer.
Replay and current-constraint compaction are separate explicit memory policies.
Every actual title refinement can be archived, including replayed refinements.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import torch
from learning.api import events_from_records
from memory.api import Archive,TitleController,verify_candidate
from memory.headings import HeadingRegistry


def _validated(records):
    if not records:raise ValueError('empty observed records')
    result=[]
    for source in records:
        e=dict(source);kind=e.get('kind');child=e.get('child');parent=e.get('parent');value=e.get('value')
        if kind not in ('root','edge','revision','query','continuation') or not isinstance(child,str) or not child:
            raise ValueError('invalid event identity or kind')
        if parent is not None and (not isinstance(parent,str) or not parent):raise ValueError('invalid parent')
        if kind=='root' and parent is not None:raise ValueError('absolute observation must have no parent')
        if kind=='edge' and (parent is None or parent==child):raise ValueError('invalid self/missing edge')
        if kind in ('root','edge','revision') and (not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError('finite observed value required')
        result.append(e)
    if result[-1]['kind']!='query':raise ValueError('request must end at a query')
    return result


@torch.no_grad()
def run_from_records(model,records,max_steps=8,prior_rounds=6,direction='bidir',policy='retain',radius=1,
                     tolerance=.1,numeric_margin=1e-6,trace_sink=None,adaptive=True):
    records=_validated(records)
    if max_steps<1 or not isinstance(max_steps,int):raise ValueError('positive refinement budget required')
    if not math.isfinite(tolerance) or tolerance<0 or not math.isfinite(numeric_margin) or numeric_margin<0:
        raise ValueError('finite nonnegative verification tolerance/margin required')
    tensor,mapping=events_from_records(records)
    device=next(model.parameters()).device;tensor=tensor.to(device)
    query_positions=[i for i,r in enumerate(records) if r['kind']=='query']
    if isinstance(prior_rounds,int):past=[prior_rounds]*(len(query_positions)-1)
    else:past=list(prior_rounds)
    if len(past)!=len(query_positions)-1 or any(not isinstance(x,int) or x<1 for x in past):
        raise ValueError('one positive observed budget per earlier query required')
    runtime=model.new_runtime(tensor,direction=direction,policy=policy,radius=radius)
    total=0;query_index=0;last=None;start=time.perf_counter()

    def refine(phase):
        nonlocal total
        candidate=runtime.refine_one();total+=1
        node_values={name:float(candidate['nodes'][0,index].cpu()) for name,index in mapping.items()}
        if trace_sink is not None:
            trace_sink({'kind':'model_trace','phase':phase,'query_index':query_index,
                        't':int(candidate['t']),'k':int(candidate['k']),
                        'nodes':node_values,'slot_mapping':dict(mapping),
                        'title_latents':runtime.z[0].detach().cpu().tolist(),
                        'versions':candidate['versions'][0].detach().cpu().tolist()})
        return candidate,node_values

    while runtime.consume_to_query():
        if query_index<len(past):
            for _ in range(past[query_index]):refine('replayed_prior_query')
            runtime.commit_feedback();query_index+=1;continue
        query=records[query_positions[-1]]['child']
        for step in range(1,max_steps+1):
            candidate,nodes=refine('current_query')
            check=verify_candidate(nodes,[r for r in records if r['kind']!='continuation'],query,tolerance)
            bound=check.get('bound')
            accepted=(check['status']=='VERIFIED' and bound is not None and math.isfinite(bound)
                      and bound+numeric_margin<=tolerance)
            prediction=float(candidate['prediction'][0].cpu())
            last={'status':'VERIFIED_PARSED_CONSTRAINTS' if accepted else 'UNRESOLVED_BUDGET',
                  'prediction':prediction,'nodes':nodes,'bound':bound,'steps':step,
                  'query':query,'verification_reason':check['reason'],'tolerance':tolerance,
                  'numeric_margin':numeric_margin,'source_faithfulness':'UNVERIFIED',
                  'prior_query_rounds':past,'mapping':mapping}
            if accepted and adaptive:break
        runtime.commit_feedback();break
    if last is None:raise RuntimeError('no final query candidate generated')
    last.update(refinements_executed=total,consumed_tokens=runtime.t,seconds=time.perf_counter()-start)
    return last


def run_text_episode(texts,model,parser,db_path,task='episode',title='main',capacity=4,max_steps=8,
                     prior_rounds=6,view='replay',tolerance=.1,adaptive=True,model_provenance=None):
    if view not in ('replay','compact'):raise ValueError('unknown memory view')
    path=Path(db_path)
    if path.exists():raise FileExistsError('episode archive already exists')
    start=time.perf_counter();archive=Archive(path);controller=TitleController(capacity,archive)
    parsed=parser.parse_batch(list(texts));outputs=[];past_budgets=[];rejected=0;total_rounds=0;total_tokens=0
    quarantine=False;headings=HeadingRegistry()
    archive.append(task,title+'/model_provenance',{'kind':'model_provenance',
                    'payload':model_provenance or {'scope':'unbound fixture; not a scientific checkpoint claim'}})
    try:
        for index,(text,interpretation) in enumerate(zip(texts,parsed)):
            controller.observe_text(task,title,text,interpretation)
            headings.consume_parsed(task,interpretation)
            if not interpretation.get('valid',False):
                rejected+=1;quarantine=True;continue
            events=interpretation.get('events',[])
            if len(events)!=1:quarantine=True;continue
            event=events[0]
            if event['kind']!='query':continue
            if quarantine:
                outputs.append({'source_index':index,'status':'UNRESOLVED_SOURCE','prediction':None,
                                'steps':0,'query':event['child'],'source_faithfulness':'UNVERIFIED'})
                continue
            if view=='replay':
                observed=controller.reconstruct(task,title)
                previous=past_budgets
            else:
                observed=controller.compact_events(task,title)+[event]
                previous=[]
            def archive_trace(row):
                archive.append(task,title+'/trace',{'source_index':index,**row})
            try:
                out=run_from_records(model,observed,max_steps=max_steps,prior_rounds=previous,
                                     tolerance=tolerance,trace_sink=archive_trace,adaptive=adaptive)
            except ValueError as error:
                quarantine=True;out={'status':'UNRESOLVED_INPUT','prediction':None,'steps':0,
                                     'query':event['child'],'reason':str(error),'source_faithfulness':'UNVERIFIED',
                                     'refinements_executed':0,'consumed_tokens':0}
            out['source_index']=index;outputs.append(out)
            if out['steps']:past_budgets.append(out['steps'])
            total_rounds+=out['refinements_executed'];total_tokens+=out['consumed_tokens']
        archive.append(task,title+'/heading_registry',{'kind':'heading_registry','payload':headings.serialize()})
        return {'outputs':outputs,'parsed':parsed,'rejected_observations':rejected,
                'source_observations':len(texts),'compactions':controller.compactions,
                'premature_compactions':controller.premature_compactions,'reopens':controller.reopens,
                'max_active_records':controller.max_active,'capacity':capacity,
                'total_refinements_executed':total_rounds,'total_consumed_tokens_including_replay':total_tokens,
                'raw_records':archive.count(),'seconds':time.perf_counter()-start,'memory_view':view,
                'heading_count':len(headings.headings(task)),
                'archive_path':str(path),'scope':'learned finite-language observations; verifier conditional on parsed constraints'}
    finally:archive.close()
