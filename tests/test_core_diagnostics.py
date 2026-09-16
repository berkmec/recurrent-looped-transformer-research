import unittest, torch
from core.model import Core,generate
from core.diagnostics import influence, checkpoint_check
torch.set_num_threads(1)
class Diagnostics(unittest.TestCase):
 def test_state_influence_oracle_and_finite_difference(self):
  torch.manual_seed(11);m=Core(width=16).double();x,y=generate('parity',3,8,19);r=influence(m,x,y,[1,4,7],epsilon=1e-4)
  self.assertGreater(r['state_norms'][0]['norm'],0);self.assertLess(abs(r['directional_autograd']-r['directional_finite_difference']),1e-6)
  m.feedback=False;r=influence(m,x,y,[1,4,7],epsilon=1e-4);self.assertEqual(r['directional_autograd'],0);self.assertAlmostEqual(r['directional_finite_difference'],0,places=10);self.assertGreater(r['embedding_norms'][-1]['norm'],0)
 def test_checkpoint_real_grad_equivalence(self):
  torch.manual_seed(11);m=Core(width=16).double();x,y=generate('parity',3,5,19);r=checkpoint_check(m,x,y);self.assertLess(r['max_gradient_difference'],1e-9);self.assertLess(r['loss_difference'],1e-9);self.assertLess(r['max_parameter_update_difference'],1e-9)
if __name__=='__main__':unittest.main()
