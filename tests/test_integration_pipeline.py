import importlib
from pathlib import Path
import pytest
import torch


def api():
    try:return importlib.import_module('integration.pipeline')
    except ModuleNotFoundError:pytest.fail('connected learned memory pipeline not implemented')


def zero_model():
    from learning.model import EventModel
    m=EventModel(feedback=True)
    with torch.no_grad():
        for p in m.parameters():p.zero_()
    m.eval();return m


def records(value):
    return [{'kind':'root','child':'x0','parent':None,'value':value},
            {'kind':'query','child':'x0','parent':None,'value':None}]


def test_pipeline_never_substitutes_symbolic_answer_for_wrong_candidate():
    out=api().run_from_records(zero_model(),records(2.),max_steps=3)
    assert out['status']=='UNRESOLVED_BUDGET'
    assert out['prediction']==0.
    assert out['bound']==2.
    assert out['steps']==3


def test_real_early_acceptance_executes_one_refinement_only():
    trace=[]
    out=api().run_from_records(zero_model(),records(0.),max_steps=8,trace_sink=trace.append)
    assert out['status']=='VERIFIED_PARSED_CONSTRAINTS'
    assert out['steps']==1
    assert out['refinements_executed']==1
    assert len(trace)==1
    assert trace[0]['t']==2


def test_replay_uses_actual_recorded_prior_query_budget():
    tape=records(0.)+[{'kind':'revision','child':'x0','parent':None,'value':2.},records(0.)[-1]]
    out=api().run_from_records(zero_model(),tape,max_steps=3,prior_rounds=[1])
    assert out['prior_query_rounds']==[1]
    assert out['refinements_executed']==4
    assert out['consumed_tokens']==4


def test_ill_formed_or_oversized_model_inputs_fail_explicitly():
    m=zero_model()
    with pytest.raises(ValueError):api().run_from_records(m,records(float('nan')))
    many=[{'kind':'root','child':f'x{i}','parent':None,'value':0.} for i in range(11)]+records(0.)[-1:]
    with pytest.raises(ValueError):api().run_from_records(m,many)


def test_connected_text_archive_revision_preserves_raw_source_and_wrong_answer(tmp_path):
    from memory.api import load_parser,Archive
    p=api();db=tmp_path/'episode.sqlite'
    texts=['set x0 to 0 .','query x0 .','revise x0 to 2 .','query x0 .']
    out=p.run_text_episode(texts,zero_model(),load_parser(),db,max_steps=3,capacity=2)
    assert out['outputs'][0]['status']=='VERIFIED_PARSED_CONSTRAINTS'
    assert out['outputs'][-1]['status']=='UNRESOLVED_BUDGET'
    assert out['outputs'][-1]['prediction']==0.
    assert out['compactions']>=1
    assert out['max_active_records']<=2
    assert out['heading_count']==1
    a=Archive(db)
    assert [a.read_observation(i+1)['text'] for i in range(4)]==texts
    assert len(a.history('episode','main/trace'))==out['total_refinements_executed']
    a.close()


def test_rejected_text_remains_archived_and_quarantines_future_answers(tmp_path):
    from memory.api import load_parser,Archive
    db=tmp_path/'rejected.sqlite'
    out=api().run_text_episode(['set x0 to 0 .','','query x0 .'],zero_model(),load_parser(),db,max_steps=3)
    assert out['rejected_observations']==1
    assert out['outputs'][-1]['status']=='UNRESOLVED_SOURCE'
    assert out['outputs'][-1]['prediction'] is None
    assert out['total_refinements_executed']==0
    a=Archive(db);assert a.read_observation(2)['text']=='';a.close()


def test_compact_view_uses_latest_evidence_and_counts_actual_consumption(tmp_path):
    from memory.api import load_parser
    out=api().run_text_episode(['set x0 to 0 .','query x0 .','revise x0 to 2 .','query x0 .'],
        zero_model(),load_parser(),tmp_path/'compact.sqlite',max_steps=2,capacity=2,view='compact')
    assert out['outputs'][-1]['bound']==2.
    assert out['outputs'][-1]['consumed_tokens']==2
    assert out['outputs'][-1]['status']=='UNRESOLVED_BUDGET'
