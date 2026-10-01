import unittest,importlib.util,tempfile,hashlib,copy
from pathlib import Path
import torch,numpy as np


class ActualTraceContracts(unittest.TestCase):
    def mod(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.distillation.trace'),'actual-teacher-trace implementation absent')
        from learning.distillation import trace
        return trace

    def test_false_teacher_candidate_is_rejected_without_repair(self):
        module=self.mod();events=[dict(kind='root',child='0',parent=None,value=.2),dict(kind='edge',child='1',parent='0',value=.3)]
        candidate={'0':.2,'1':.8};before=copy.deepcopy(candidate)
        import memory.api
        original=memory.api.symbolic_values
        memory.api.symbolic_values=lambda *_: (_ for _ in ()).throw(AssertionError('oracle forbidden'))
        try:result=module.certify(candidate,events,'1',.05)
        finally:memory.api.symbolic_values=original
        self.assertFalse(result['accepted']);self.assertEqual(candidate,before)
        good={'0':.21,'1':.51};self.assertTrue(module.certify(good,events,'1',.05)['accepted'])

    def test_heldout_or_wrong_hash_cannot_be_loaded_for_trace_training(self):
        module=self.mod();owner=Path(__file__).resolve().parents[1]/'learning'/'distillation'
        with tempfile.TemporaryDirectory(dir=owner) as temp:
            path=Path(temp)/'data.npz';np.savez(path,events=np.zeros((1,1,4),np.float32))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):module.load_training_events(path,digest,'iid',1)
            with self.assertRaises(ValueError):module.load_training_events(path,'wrong','train',1)
            self.assertEqual(module.load_training_events(path,digest,'train',1).shape,(1,1,4))

    def test_archive_does_not_overwrite_generated_trace(self):
        module=self.mod();owner=Path(__file__).resolve().parents[1]/'learning'/'distillation'
        with tempfile.TemporaryDirectory(dir=owner) as temp:
            path=Path(temp)/'trace.npz';module.write_archive(path,{'predictions':np.array([.3])});before=path.read_bytes()
            with self.assertRaises(FileExistsError):module.write_archive(path,{'predictions':np.array([8.])})
            self.assertEqual(path.read_bytes(),before)

    def test_trace_capture_is_same_actual_model_computation(self):
        module=self.mod();from learning.model import EventModel
        torch.set_num_threads(1);torch.manual_seed(17);model=EventModel(feedback=True)
        events=torch.tensor([[[0,0,0,.2],[1,1,0,.3],[2,1,1,0],[3,0,0,.4],[2,1,1,0]]])
        captured=module.trace_forward(model,events,6);ordinary=model(events)
        torch.testing.assert_close(captured['final'],ordinary['predictions'],rtol=0,atol=0)
        self.assertEqual(tuple(captured['states'].shape),(1,2,6,10,32))

    def test_uncertified_node_values_never_affect_auxiliary_loss(self):
        module=self.mod()
        states=torch.zeros(1,1,2,3,4,requires_grad=True);nodes=torch.zeros(1,1,2,3,requires_grad=True)
        current={'final':torch.zeros(1,1,requires_grad=True),'states':states,'nodes':nodes}
        target={k:v.detach().clone() for k,v in current.items()};mask=torch.tensor([[[1.,0.,0.]]]);queries=torch.ones(1,1)
        first=module.losses(current,target,queries,mask)[1]
        target['states'][:,:,:,1:]=100;target['nodes'][:,:,:,1:]=100
        second=module.losses(current,target,queries,mask)[1]
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_empty_evaluation_cannot_be_classified_positive(self):
        from learning.distillation import postprocess
        self.assertTrue(hasattr(postprocess,'validate_expected_results'),'complete-result guard absent')
        with self.assertRaises(ValueError):postprocess.validate_expected_results({'runs':[],'rows':[],'comparisons':[]})

if __name__=='__main__':unittest.main()
