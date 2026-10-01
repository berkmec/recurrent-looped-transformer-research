import unittest
import importlib.util
import inspect
import numpy as np
import torch


class LearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def modules(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.model'), 'learned event model not implemented')
        from learning.model import EventModel, revision_mask
        return EventModel, revision_mask

    def test_forward_one_step_cannot_read_child(self):
        Model, _ = self.modules()
        torch.manual_seed(17); m=Model()
        z=torch.zeros(1,10,32); other=z.clone(); other[:,1]=1
        a=torch.zeros(1,10,4); p=torch.zeros(1,10,10); p[:,1,0]=1
        active=torch.ones(1,10)
        one=m.refine(z,a,p,active,'forward'); two=m.refine(other,a,p,active,'forward')
        torch.testing.assert_close(one[:,0],two[:,0],rtol=0,atol=0)
        self.assertGreater(float((m.refine(z,a,p,active,'bidir')[:,0]-m.refine(other,a,p,active,'bidir')[:,0]).abs().max().detach()),0)

    def test_feedback_changes_future_only_and_preserves_clocks(self):
        Model,_=self.modules(); torch.manual_seed(17); m=Model(feedback=True)
        e=torch.tensor([[[0,0,0,.3],[1,1,0,.2],[2,1,1,0],[3,0,0,.9],[2,1,1,0],[6,0,0,0],[2,1,1,0]]])
        x=m(e,return_trace=True); y=m(e,intervention='zero',return_trace=True)
        torch.testing.assert_close(x['predictions'][:,0],y['predictions'][:,0],rtol=0,atol=0)
        self.assertGreater(float((x['predictions'][:,1:]-y['predictions'][:,1:]).abs().max().detach()),0)
        self.assertEqual(x['trace']['token_clocks'],list(range(1,8)))
        self.assertEqual(x['trace']['cache_appends'],7)
        self.assertTrue(all(v['token_before']==v['token_after'] and v['cache_before']==v['cache_after'] for v in x['trace']['refinements']))
        e2=e.clone();e2[:,3,3]=-1
        torch.testing.assert_close(m(e2)['predictions'][:,0],x['predictions'][:,0],rtol=0,atol=0)

    def test_revision_policy_has_no_target_or_effect_input(self):
        _,mask=self.modules()
        self.assertEqual(set(inspect.signature(mask).parameters),{'links','changed','radius','active'})
        p=torch.zeros(1,10,10);p[:,1,0]=1;p[:,2,1]=1;p[:,4,3]=1
        active=torch.zeros(1,10);active[:,:5]=1
        v=mask(p,torch.tensor([0]),1,active)
        torch.testing.assert_close(v[0,:5],torch.tensor([1.,1.,0.,0.,0.]))

    def test_final_error_gate_rejects_no_drift_but_wrong_answer(self):
        self.modules()
        from learning.metrics import revision_gate
        baseline=np.array([[0.,0.],[0.,0.]])
        candidate=np.array([[0.,1.],[1.,1.]])
        truth=np.array([[0.,1.],[0.,0.]])
        self.assertFalse(revision_gate(baseline,candidate,truth,np.array([True,False]),.01)['pass'])

    def test_labels_match_independent_linear_solve(self):
        self.modules()
        from learning.data import make_dataset
        d=make_dataset(32,310,'validation')
        for tape,labels,nodes in zip(d['events'],d['targets'],d['node_targets']):
            edges={};observed={};q=0
            for op,i,p,v in tape:
                i,p=int(i),int(p)
                if op in (0,3): observed[i]=float(v)
                elif op in (1,4):edges[i]=(p,float(v))
                elif op==2:
                    rows=[];rhs=[]
                    for child,(parent,offset) in edges.items():
                        row=np.zeros(10);row[child]=1;row[parent]=-1;rows.append(row);rhs.append(offset)
                    for node,val in observed.items():
                        row=np.zeros(10);row[node]=1;rows.append(row);rhs.append(val)
                    answer=np.linalg.lstsq(rows,rhs,rcond=None)[0]
                    self.assertAlmostEqual(answer[i],float(labels[q]),places=5)
                    np.testing.assert_allclose(answer,nodes[q],atol=1e-6);q+=1

    def test_oracle_unavailable_at_inference(self):
        Model,_=self.modules()
        import builtins
        original=builtins.__import__
        def restricted(name,*args,**kwargs):
            if name.startswith('learning.data') or 'oracle' in name:raise AssertionError('oracle import during inference')
            return original(name,*args,**kwargs)
        builtins.__import__=restricted
        try:
            result=Model()(torch.tensor([[[0,0,0,.3],[2,0,0,0]]]))
            self.assertEqual(tuple(result['predictions'].shape),(1,1))
        finally:builtins.__import__=original

    def test_rejects_mixed_query_alignment_and_fractional_ids(self):
        Model,_=self.modules();m=Model()
        with self.assertRaises(ValueError):m(torch.tensor([[[2,0,0,0]],[[0,0,0,0]]],dtype=torch.float32))
        with self.assertRaises(ValueError):m(torch.tensor([[[2,.5,0,0]]]))

    def test_causal_global_encoder_matches_streamed_core(self):
        Model,_=self.modules();m=Model()
        self.assertTrue(hasattr(m,'token_core'),'causal global encoder and source-style recurrent decoder are absent')
        torch.manual_seed(9);x=torch.randn(2,7,28)
        a=m.token_core(x);b=m.token_core(x,stream_encoder=True)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_unobserved_parent_is_active_for_inverse_evidence(self):
        Model,_=self.modules();m=Model()
        e=torch.tensor([[[1,1,0,.2],[0,1,1,.7],[2,0,0,0]]])
        runtime=m.new_runtime(e);self.assertTrue(runtime.consume_to_query())
        self.assertEqual(float(runtime.active[0,0]),1.,'edge-referenced unknown parent must have a title')
        self.assertEqual(float(runtime.links[0,1,0]),1.)
        self.assertEqual(int(runtime.versions[0,0]),0,'unobserved parent activation must not invent an observation version')

    def test_incremental_refinement_preserves_cache_values_and_single_commit(self):
        Model,_=self.modules();m=Model(feedback=True)
        e=torch.tensor([[[0,0,0,.3],[1,1,0,.2],[2,1,1,0],[3,0,0,.4],[2,1,1,0]]])
        runtime=m.new_runtime(e);runtime.consume_to_query()
        state={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in runtime.decoder_state.items()}
        t=runtime.t
        for _ in range(4):runtime.refine_one()
        self.assertEqual(runtime.t,t);self.assertEqual(runtime.k,4)
        for key,before in state.items():
            if isinstance(before,torch.Tensor):torch.testing.assert_close(before,runtime.decoder_state[key],rtol=0,atol=0)
        runtime.commit_feedback()
        with self.assertRaises(RuntimeError):runtime.commit_feedback()
        runtime.consume_to_query();self.assertEqual(int(runtime.versions[0,0]),2)

    def test_lazy_candidate_iterator_does_not_execute_unrequested_rounds(self):
        Model,_=self.modules()
        from learning.api import final_query_candidates
        m=Model();e=torch.tensor([[[0,0,0,.3],[2,0,0,0]]])
        calls=[]
        handle=m.parent_map.register_forward_hook(lambda *args:calls.append(1))
        iterator=final_query_candidates(m,e,max_rounds=6)
        first=next(iterator);self.assertEqual(first['k'],1);self.assertEqual(len(calls),1)
        iterator.close();self.assertEqual(len(calls),1);handle.remove()

if __name__=='__main__':unittest.main()
