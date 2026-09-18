from __future__ import annotations
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
import torch
from rlt_lab.core import protected_update, recurrent_scan, affine_scan
from rlt_lab.memory import Archive, StaleRevision, HeadingPage, affected_descendants, reusable
from rlt_lab.protocol import (atomic_json,make_receipt,validate_receipt,zero_error_upper_bound,
                              holm,summarize_records,manifest_jobs)


torch.set_num_threads(1)

class CoreTests(unittest.TestCase):
    def test_identity_bitwise_and_no_alias(self):
        z=torch.randn(3,7,dtype=torch.float64)
        out=protected_update(z,torch.randn_like(z),torch.zeros(3,dtype=torch.bool))
        self.assertTrue(torch.equal(z,out))
        self.assertNotEqual(z.data_ptr(),out.data_ptr())

    def test_selective_write(self):
        z=torch.zeros(2,3); p=torch.ones_like(z)
        out=protected_update(z,p,torch.tensor([False,True]))
        self.assertTrue(torch.equal(out[0],z[0]))
        self.assertTrue(torch.equal(out[1],p[1]))

    def test_wrong_hold_decision_remains_wrong(self):
        z=torch.zeros(1); required=torch.ones(1)
        out=protected_update(z,required,torch.tensor(False))
        self.assertFalse(torch.equal(out,required))  # mechanism does not discover semantics

    def test_preserving_z_does_not_preserve_cache_readout(self):
        z=torch.ones(1); cache=torch.zeros(1)
        before=z+cache
        for _ in range(16): cache=0.9*cache+0.1
        self.assertTrue(torch.equal(z,torch.ones(1)))
        self.assertFalse(torch.equal(before,z+cache))

    def test_nonfinite_refused(self):
        with self.assertRaises(ValueError):
            protected_update(torch.zeros(1),torch.tensor([float('nan')]),torch.tensor(False))

    def test_wrong_mask_refused(self):
        with self.assertRaises(TypeError):
            protected_update(torch.zeros(1),torch.ones(1),torch.tensor(0.0))

    def test_soft_gate_is_not_exact(self):
        z=torch.zeros(1,dtype=torch.float64)
        p=torch.ones_like(z)
        for _ in range(1000):
            z=(1-1e-5)*z+1e-5*p
        self.assertGreater(float(z),0.009)

    def _grad_trial(self,segment,dropout=False):
        torch.manual_seed(101)
        w=torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*0.07)
        z=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
        c=torch.zeros(2,4,dtype=torch.float64,requires_grad=True)
        x=torch.randn(37,2,4,dtype=torch.float64,requires_grad=True)
        def step(state,inp):
            h,cache=state
            v=torch.tanh(inp@w+0.2*cache)
            if dropout:
                v=torch.nn.functional.dropout(v,p=0.2,training=True)
            return h+0.05*v, 0.9*cache+0.1*h
        out=recurrent_scan(step,(z,c),x,segment)
        loss=out[0].square().mean()+0.1*out[1].square().mean()
        loss.backward()
        return loss.detach(),[q.grad for q in (w,z,c,x)]

    def test_remat_loss_and_all_state_gradients(self):
        l,g=self._grad_trial(None)
        l2,g2=self._grad_trial(8)
        torch.testing.assert_close(l,l2,rtol=0,atol=0)
        for a,b in zip(g,g2):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_remat_dropout_rng(self):
        l,g=self._grad_trial(None,True)
        l2,g2=self._grad_trial(7,True)
        torch.testing.assert_close(l,l2,rtol=0,atol=0)
        for a,b in zip(g,g2):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_final_only_tbptt_disconnects_initial_state(self):
        z=torch.ones(1,requires_grad=True)
        w=torch.tensor(0.98,requires_grad=True)
        h=z
        for t in range(32):
            if t and t%8==0:
                h=h.detach()
            h=h*w+0.01
        h.square().sum().backward()
        self.assertIsNone(z.grad)
        self.assertIsNotNone(w.grad)

    def test_remat_restores_initial_gradient(self):
        z=torch.ones(1,requires_grad=True)
        w=torch.tensor(0.98,requires_grad=True)
        out=recurrent_scan(lambda st,x:(st[0]*w+x,), (z,), torch.ones(32,1)*0.01,8)
        out[0].square().sum().backward()
        self.assertGreater(float(z.grad.abs().sum()),0)

    def test_integer_position_state_survives_remat(self):
        z=torch.ones(1,requires_grad=True)
        pos=torch.tensor(0,dtype=torch.long)
        def step(st,x):
            h,t=st
            return h+x*(t+1),t+1
        out=recurrent_scan(step,(z,pos),torch.ones(17,1),4)
        self.assertEqual(out[1].item(),17)
        self.assertEqual(out[0].item(),154)
        out[0].sum().backward()
        self.assertEqual(z.grad.item(),1)

    def test_scan_empty(self):
        z=torch.randn(2,4)
        self.assertTrue(torch.equal(recurrent_scan(lambda s,x:s,(z,),torch.zeros(0,2,4),8)[0],z))

    def test_affine_scan_value_and_gradient(self):
        torch.manual_seed(17)
        a=(torch.rand(65,2,4,dtype=torch.float64)*0.05+0.95).requires_grad_()
        b=(torch.randn_like(a)*0.05).requires_grad_()
        z=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
        h=z; values=[]
        for at,bt in zip(a,b):
            h=at*h+bt; values.append(h)
        serial=torch.stack(values)
        parallel=affine_scan(a,b,z)
        torch.testing.assert_close(serial,parallel,rtol=1e-12,atol=1e-12)
        g1=torch.autograd.grad(serial.square().sum(),(a,b,z),retain_graph=True)
        g2=torch.autograd.grad(parallel.square().sum(),(a,b,z))
        for x,y in zip(g1,g2):
            torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)

    def test_affine_invalid_shape(self):
        with self.assertRaises(ValueError):
            affine_scan(torch.zeros(2,3),torch.zeros(2,3),torch.zeros(4))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.a=Archive(self.root/'archive.sqlite')

    def tearDown(self):
        self.a.close(); self.temp.cleanup()

    def _append(self,heading='A',version=0,deps=None):
        return self.a.append(heading,'ham düşünce\r\nİı'.encode(), 'özet',model='model:abc',
                             expected_version=version,dependencies=deps)

    def test_raw_is_lossless(self):
        row=self._append()
        self.assertEqual(self.a.raw(row['digest']),'ham düşünce\r\nİı'.encode())

    def test_summary_never_overwrites_raw(self):
        one=self._append()
        two=self.a.append('A',b'new full thought','tiny',model='model:abc',expected_version=1)
        self.assertNotEqual(one['digest'],two['digest'])
        self.assertEqual(self.a.raw(one['digest']),'ham düşünce\r\nİı'.encode())
        self.assertEqual(self.a.verify(),2)

    def test_stale_writer_rejected(self):
        self._append()
        with self.assertRaises(StaleRevision): self._append()
        self.assertEqual(self.a.verify(),1)

    def test_unknown_dependency_rejected(self):
        with self.assertRaises(ValueError): self._append(deps={'B':9})
        self.assertEqual(self.a.verify(),0)

    def test_dependency_is_versioned(self):
        self._append(); self._append('B',deps={'A':1}); self._append(version=1)
        self.assertEqual(self.a.verify(),3)
        self.assertFalse(reusable({'A':1},{'A':2},saved_model='m',current_model='m'))

    def test_model_change_prevents_latent_reuse(self):
        self.assertFalse(reusable({'A':1},{'A':1},saved_model='old',current_model='new'))

    def test_update_forbidden(self):
        self._append()
        with self.assertRaises(sqlite3.IntegrityError):
            self.a.db.execute("UPDATE events SET model='other'")

    def test_delete_forbidden(self):
        self._append()
        with self.assertRaises(sqlite3.IntegrityError): self.a.db.execute('DELETE FROM events')
        with self.assertRaises(sqlite3.IntegrityError): self.a.db.execute('DELETE FROM blobs')

    def test_backup_copies_wal_safely(self):
        self._append(); self._append(version=1)
        backup=self.root/'copy.sqlite'; self.a.backup(backup)
        b=Archive(backup)
        try: self.assertEqual(b.verify(),2)
        finally: b.close()

    def test_backup_same_path_rejected(self):
        with self.assertRaises(ValueError): self.a.backup(self.a.path)

    def test_no_early_compaction(self):
        p=HeadingPage('H',0,16).consume(15)
        self.assertFalse(p.should_compact)
        with self.assertRaises(ValueError): p.next_page(raw_persisted=True,compacted=True)

    def test_same_heading_after_compaction(self):
        p=HeadingPage('H',0,16).consume(16)
        q=p.next_page(raw_persisted=True,compacted=True)
        self.assertEqual(q.heading,'H'); self.assertEqual(q.page,1)

    def test_no_compaction_without_archive(self):
        p=HeadingPage('H',0,16).consume(16)
        with self.assertRaises(ValueError): p.next_page(raw_persisted=False,compacted=True)

    def test_cycle_revision_closure(self):
        deps={'A':{'C'},'B':{'A'},'C':{'B'},'D':set()}
        self.assertEqual(affected_descendants({'A'},deps),{'A','B','C'})

    def test_unaffected_branch_is_preserved(self):
        deps={'A':set(),'B':{'A'},'C':set(),'D':{'C'}}
        self.assertEqual(affected_descendants({'A'},deps),{'A','B'})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.f=self.root/'result.json'; atomic_json(self.f,{'result':1})

    def tearDown(self): self.tmp.cleanup()

    def receipt(self,**kw):
        opts=dict(returncode=0,expected_items=3,completed_items=3,tests_passed=True);opts.update(kw)
        return make_receipt('run1',{'result':self.f},**opts)

    def test_receipt_success(self):
        self.assertTrue(validate_receipt(self.receipt(),{'result':self.f},expected_run_id='run1',expected_items=3))

    def test_receipt_string_boolean_rejected(self):
        with self.assertRaises(ValueError): self.receipt(tests_passed='false')

    def test_empty_manifest_rejected(self):
        with self.assertRaises(ValueError): manifest_jobs([],['a'],[1])

    def test_bad_exit_not_complete(self):
        self.assertEqual(self.receipt(returncode=1)['status'],'INCOMPLETE')

    def test_missing_items_not_complete(self):
        self.assertEqual(self.receipt(completed_items=2)['status'],'INCOMPLETE')

    def test_failed_tests_not_complete(self):
        self.assertEqual(self.receipt(tests_passed=False)['status'],'INCOMPLETE')

    def test_tampered_artifact_fails(self):
        r=self.receipt(); self.f.write_bytes(b'tampered')
        self.assertFalse(validate_receipt(r,{'result':self.f},expected_run_id='run1',expected_items=3))

    def test_wrong_run_fails(self):
        self.assertFalse(validate_receipt(self.receipt(),{'result':self.f},expected_run_id='run2',expected_items=3))

    def test_utf8_and_newline(self):
        atomic_json(self.f,{'t':'Türkçe ıİğ'})
        self.assertEqual(json.loads(self.f.read_bytes())['t'],'Türkçe ıİğ')
        self.assertNotIn(b'\r\n',self.f.read_bytes())

    def test_full_log_read_not_first_rows(self):
        rows=[dict(id=str(i),prediction=0 if i<16 else 1,accepted=True,correct=True) for i in range(100)]
        s=summarize_records(rows,{str(i) for i in range(100)})
        self.assertEqual(s['prediction_counts'],{'0':16,'1':84})

    def test_duplicate_records_rejected(self):
        r=dict(id='a',prediction=0)
        with self.assertRaises(ValueError): summarize_records([r,r],{'a'})

    def test_abstention_not_success(self):
        s=summarize_records([dict(id='a',prediction=0,accepted=False)],{'a','b'})
        self.assertEqual(s['status'],'INCOMPLETE')
        self.assertIsNone(s['accepted_risk']);self.assertEqual(s['coverage'],0)

    def test_risk_bound(self):
        self.assertLess(zero_error_upper_bound(299),0.01)
        self.assertGreater(zero_error_upper_bound(298),0.01)

    def test_holm(self):
        self.assertEqual(holm({'a':0.001,'b':0.02,'c':0.2}),{'a':True,'b':True,'c':False})

    def test_full_3000_problem_manifest(self):
        jobs=manifest_jobs([str(i) for i in range(3000)],['base','a','b','c'],[1])
        self.assertEqual(len(jobs),12000)
        self.assertEqual(len({j['problem'] for j in jobs}),3000)

    def test_duplicate_manifest_rejected(self):
        with self.assertRaises(ValueError): manifest_jobs(['1','1'],['a'],[1])

if __name__ == '__main__': unittest.main()
