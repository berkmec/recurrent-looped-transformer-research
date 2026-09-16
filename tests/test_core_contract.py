import unittest, torch
from core.model import Core, oracle, generate
torch.set_num_threads(1)

class Contract(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(8); self.m=Core(width=16,window=4).double(); self.x=torch.tensor([[0,1,0,1,1,0]])
 def test_causal_future(self):
  y=self.m(self.x); x=self.x.clone();x[:,4:]=2
  torch.testing.assert_close(y[:,:4],self.m(x)[:,:4])
 def test_stream_output_gradient(self):
  y=self.m(self.x); y.square().sum().backward(); g={n:p.grad.clone() for n,p in self.m.named_parameters() if p.grad is not None};self.m.zero_grad()
  z=self.m(self.x,stream_encoder=True);z.square().sum().backward();torch.testing.assert_close(y,z,atol=1e-9,rtol=1e-8)
  for n,p in self.m.named_parameters():
   if n in g:torch.testing.assert_close(g[n],p.grad,atol=1e-8,rtol=1e-7)
 def test_swa_boundary(self):
  for w in [1,2,4]:
   m=Core(width=16,window=w).double();s=None
   for i in range(6):
    _,s=m.step(self.x[:,i],s);self.assertEqual(s['dk'].shape[2],min(i+1,w-1))
 def test_no_feedback_hidden_independent_cache_present(self):
  self.m.feedback=False;_,s=self.m.step(self.x[:,0]);a,sa=self.m.step(self.x[:,1],s);s2=dict(s);s2['s']=s['s']+100;b,sb=self.m.step(self.x[:,1],s2)
  torch.testing.assert_close(a,b);self.assertEqual(sb['mk'].shape[2],2);self.assertEqual(sb['dk'].shape[2],2)
 def test_oracles(self):
  self.assertEqual(oracle([1,1,0,2],'parity'),[1,0,0,0]);self.assertEqual(oracle([0,1],'five'),[1,0]);self.assertEqual(oracle([1,0],'five'),[1,2])
 def test_tbptt_changes_recurrent_grad_not_forward(self):
  a=self.m(self.x);a[:,-1].square().sum().backward();g=self.m.initial.grad.clone();self.m.zero_grad();b=self.m(self.x,tbptt=2);b[:,-1].square().sum().backward();torch.testing.assert_close(a,b);self.assertFalse(torch.allclose(g,self.m.initial.grad))
  self.assertGreater(float(self.m.embedding.weight.grad.abs().sum()),0)
 def test_invalid_budgets(self):
  with self.assertRaises(ValueError):Core(window=0)
  with self.assertRaises(ValueError):self.m(self.x[:,:0])
  with self.assertRaises(ValueError):self.m(self.x,tbptt=-1)
  with self.assertRaises(ValueError):generate('parity',2,0,1)
 def test_parallel_no_feedback_matches_stream(self):
  self.m.feedback=False;a=self.m(self.x,parallel_control=True);a.square().sum().backward();g={n:p.grad.clone() for n,p in self.m.named_parameters() if p.grad is not None};self.m.zero_grad();b=self.m(self.x,stream_encoder=True);b.square().sum().backward();torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
  for n,p in self.m.named_parameters():
   if n in g:torch.testing.assert_close(g[n],p.grad,atol=1e-8,rtol=1e-7)

if __name__=='__main__':unittest.main()
