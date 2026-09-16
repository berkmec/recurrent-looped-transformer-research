import importlib
import pytest
import torch
from learning.model import EventModel
from learning.api import events_from_records
from memory.api import Archive,load_parser


def module(name):
    try:return importlib.import_module(name)
    except ModuleNotFoundError:pytest.fail('generated reasoning-window continuation is not implemented')


def setup_runtime():
    torch.manual_seed(331);model=EventModel(feedback=True);model.eval()
    events,_=events_from_records([dict(kind='root',child='x0',parent=None,value=.7),
                                  dict(kind='edge',child='x1',parent='x0',value=.2),
                                  dict(kind='query',child='x1',parent=None,value=None)])
    runtime=model.new_runtime(events);assert runtime.consume_to_query()
    return runtime


def test_only_full_generated_window_compacts_and_raw_states_remain(tmp_path):
    cls=module('integration.reasoning_window').ReasoningWindow
    a=Archive(tmp_path/'w.sqlite');w=cls(a,'task',capacity=4);r=setup_runtime()
    for k in range(1,9):
        with torch.no_grad():r.refine_one()
        w.observe(r,{'query_entity':'x1','title_latents':r.z[0].tolist()})
        assert w.compactions==k//4
    assert w.rereads==2 and w.max_active==4 and w.generated==8
    assert a.count()==8
    assert a.db.execute('select count(*) from compacts').fetchone()[0]==2
    assert w.same_heading_continuations==1
    a.close()


def test_boundary_reload_is_bitwise_equal_to_uncompacted_real_refinement(tmp_path):
    cls=module('integration.reasoning_window').ReasoningWindow
    a=Archive(tmp_path/'equiv.sqlite');w=cls(a,'task',capacity=4)
    x,y=setup_runtime(),setup_runtime()
    with torch.no_grad():
        for _ in range(16):
            px=x.refine_one();py=y.refine_one()
            w.observe(y,{'query_entity':'x1','title_latents':y.z[0].tolist()})
            assert torch.equal(x.z,y.z)
            assert torch.equal(px['prediction'],py['prediction'])
    assert w.compactions==w.rereads==4
    a.close()


def test_reopening_boundary_under_changed_evidence_or_weights_is_refused(tmp_path):
    cls=module('integration.reasoning_window').ReasoningWindow
    a=Archive(tmp_path/'stale.sqlite');w=cls(a,'task',capacity=2);r=setup_runtime()
    with torch.no_grad():
        for _ in range(2):r.refine_one();w.observe(r,{'query_entity':'x1','title_latents':r.z[0].tolist()})
        compact_id=w.last_compact_id;r.a[0,0,0]+=1.
        with pytest.raises(ValueError,match='stale'):w.restore(r,compact_id)
    a.close()


def test_compaction_boundary_delays_halting_until_reread_and_keeps_same_heading(tmp_path):
    p=module('integration.window_pipeline');model=EventModel(feedback=True)
    with torch.no_grad():
        for weight in model.parameters():weight.zero_()
    out=p.run_text_episode(['set x0 to 0 .','query x0 .'],model,load_parser(),tmp_path/'end.sqlite',
                           max_steps=8,reasoning_capacity=4)
    assert out['outputs'][0]['steps']==4
    assert out['outputs'][0]['status']=='VERIFIED_PARSED_CONSTRAINTS'
    assert out['reasoning_compactions']==out['reasoning_rereads']==1
    assert out['reasoning_max_resident_traces']==4
    assert out['total_refinements_executed']==4


def test_sufficiency_is_only_evaluated_after_boundary_reread(tmp_path,monkeypatch):
    p=module('integration.window_pipeline');model=EventModel(feedback=True)
    with torch.no_grad():
        for weight in model.parameters():weight.zero_()
    original=p.verify_candidate;calls=[]
    def verify(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(p,'verify_candidate',verify)
    out=p.run_text_episode(['set x0 to 0 .','query x0 .'],model,load_parser(),tmp_path/'checks.sqlite',max_steps=8)
    assert out['reasoning_rereads']==1 and len(calls)==1


def test_changed_transition_direction_invalidates_saved_boundary(tmp_path):
    cls=module('integration.reasoning_window').ReasoningWindow
    a=Archive(tmp_path/'direction.sqlite');w=cls(a,'task',capacity=2);r=setup_runtime()
    with torch.no_grad():
        for _ in range(2):r.refine_one();w.observe(r,{'query_entity':'x1','title_latents':r.z[0].tolist()})
    r.direction='forward'
    with pytest.raises(ValueError,match='stale'):w.restore(r,w.last_compact_id)
    a.close()


def test_verifier_and_answer_read_actual_post_reload_candidate(tmp_path,monkeypatch):
    p=module('integration.window_pipeline');model=EventModel(feedback=True)
    with torch.no_grad():
        for weight in model.parameters():weight.zero_()
        model.title_head.weight.fill_(1.)
    original=p.ReasoningWindow.observe
    def changed_reload(self,runtime,*args,**kwargs):
        boundary=original(self,runtime,*args,**kwargs)
        if boundary:runtime.z.fill_(1.)
        return boundary
    monkeypatch.setattr(p.ReasoningWindow,'observe',changed_reload)
    out=p.run_text_episode(['set x0 to 0 .','query x0 .'],model,load_parser(),tmp_path/'reload-spy.sqlite',max_steps=4)
    assert out['outputs'][0]['prediction']==32.
    assert out['outputs'][0]['status']=='UNRESOLVED_BUDGET'
