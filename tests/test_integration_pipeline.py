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


def test_contradictory_parser_abstention_is_quarantined_and_raw_output_preserved(tmp_path):
    from memory.api import Archive
    class Contradictory:
        def parse_batch(self,texts):
            return [dict(valid=True,abstained=True,events=[records(0.)[0]]),
                    dict(valid=True,abstained=False,events=[records(0.)[-1]])]
    db=tmp_path/'contradictory.sqlite'
    out=api().run_text_episode(['first','second'],zero_model(),Contradictory(),db)
    assert out['outputs'][-1]['status']=='UNRESOLVED_SOURCE'
    assert out['rejected_observations']==1 and out['heading_count']==0
    a=Archive(db);stored=a.read_observation(1)['parsed']
    assert stored['raw_adapter_output']['valid'] is True
    assert stored['raw_adapter_output']['abstained'] is True
    assert stored['events']==[] and stored['valid'] is False
    a.close()


def test_parser_cannot_silently_drop_original_source_observations(tmp_path):
    class Truncated:
        def parse_batch(self,texts):return []
    db=tmp_path/'truncated.sqlite'
    with pytest.raises(ValueError,match='cardinality'):
        api().run_text_episode(['set x0 to 0 .','query x0 .'],zero_model(),Truncated(),db)
    assert not db.exists()


def test_automatic_heading_ids_drive_storage_and_neural_slot_provenance(tmp_path):
    from memory.api import Archive,load_parser
    from memory.headings import identity
    db=tmp_path/'headings.sqlite'
    out=api().run_text_episode(['set x0 to 0 .','x1 equals x0 plus 1 .','query x1 .',
                               'revise x0 to 2 .','query x1 .'],zero_model(),load_parser(),db,
                              capacity=2,max_steps=2)
    a=Archive(db);h0=identity('episode','x0');h1=identity('episode','x1')
    assert [r['event']['kind'] for r in a.history('episode',h0)]==['root','revision']
    assert [r['event']['kind'] for r in a.history('episode',h1)]==['edge','query','query']
    traces=a.history('episode','main/trace')
    assert all(set(r['event']['heading_slot_mapping'])=={h0,h1} for r in traces)
    assert out['outputs'][-1]['consumed_tokens']==5
    assert out['outputs'][-1]['bound']==3.
    assert out['max_active_records']<=2 and out['heading_count']==2
    a.close()
