"""Inference-only API. No datasets, traces, labels, archive or solver required."""
from pathlib import Path
import torch
from .model import EventModel


def load_model(checkpoint):
    payload=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
    model=EventModel(feedback=payload['feedback']);model.load_state_dict(payload['state_dict']);model.eval()
    return model


def events_from_records(records):
    names=[]
    for record in records:
        for key in ('child','parent'):
            value=record.get(key)
            if value is not None and value not in names:names.append(value)
    if len(names)>10:raise ValueError('model capacity is ten distinct IDs')
    mapping={name:i for i,name in enumerate(names)};events=[]
    for record in records:
        kind=record['kind'];child=mapping[record['child']];parent=mapping.get(record.get('parent'),child)
        if kind=='root':op=0
        elif kind=='edge':op=1
        elif kind=='query':op=2
        elif kind=='revision':op=3 if record.get('parent') is None or parent==child else 4
        elif kind=='continuation':op=6
        else:raise ValueError('unsupported event kind')
        events.append([op,child,parent,0. if record.get('value') is None else float(record['value'])])
    return torch.tensor([events],dtype=torch.float32),mapping


@torch.no_grad()
def predict_records(records,checkpoint,rounds=6):
    model=load_model(checkpoint);events,mapping=events_from_records(records)
    out=model(events,rounds=rounds)
    return {'predictions':out['predictions'][0].tolist(),
            'nodes':[{name:float(row[index]) for name,index in mapping.items()} for row in out['nodes'][0]],'mapping':mapping}


@torch.no_grad()
def final_query_candidates(model,events,max_rounds=6,prior_rounds=6,direction='bidir',policy='retain',radius=1):
    """Yield last query candidates lazily; all earlier queries receive fixed budget.

    Iterator closure ends the request; full continuation users should use Runtime
    and call commit_feedback after selecting a candidate.
    """
    runtime=model.new_runtime(events,direction,policy,radius)
    last=int(torch.where(events[0,:,0]==2)[0][-1])+1
    while runtime.consume_to_query():
        if runtime.t==last:
            for _ in range(max_rounds):yield runtime.refine_one()
            return
        for _ in range(prior_rounds):runtime.refine_one()
        runtime.commit_feedback()
