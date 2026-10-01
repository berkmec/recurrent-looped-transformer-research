import unittest, tempfile, sqlite3
from pathlib import Path
from memory.api import Archive, TitleController, verify_candidate, halt_candidates, symbolic_values

def root(value=1): return dict(kind='root',child='x',parent=None,value=value)
def edge(child='y',parent='x',value=2): return dict(kind='edge',child=child,parent=parent,value=value)

class MemoryContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)/'a.db'; self.a=Archive(self.path)
    def tearDown(self): self.a.close(); self.temp.cleanup()
    def test_raw_hash_tamper_detected(self):
        key=self.a.append('task','title',root()); self.a.db.execute('UPDATE records SET raw=? WHERE id=?',('{"tampered":true}',key)); self.a.db.commit()
        with self.assertRaises(ValueError): self.a.read(key)
    def test_stale_and_cross_task_never_applicable(self):
        old=self.a.append('A','same',root(1)); new=self.a.append('A','same',root(3)); cross=self.a.append('B','same',root(3))
        self.assertFalse(self.a.applicable(old,'A','same')); self.assertTrue(self.a.applicable(new,'A','same')); self.assertFalse(self.a.applicable(cross,'A','same'))
    def test_capacity_cycles_and_restoration(self):
        c=TitleController(4,self.a)
        events=[root()]+[edge('n'+str(i),'x',i) for i in range(20)]
        for event in events:
            c.append('T','stable',event); self.assertLessEqual(c.active_count,4)
        self.assertEqual(c.reconstruct('T','stable'),events); self.assertGreaterEqual(c.compactions,4); self.assertEqual(c.premature_compactions,0)
        self.assertEqual(self.a.count(),len(events)); self.assertEqual(c.titles,{('T','stable')})
    def test_late_internal_revision_invalidates_and_repairs(self):
        c=TitleController(3,self.a)
        for event in [root(),edge(),edge('z','y',3),edge('q','z',4)]: c.append('T','A',event)
        c.append('T','A',dict(kind='revision',child='y',parent='x',value=9))
        self.assertEqual(c.summary_status('T','A'),'COMPACTION_REJECTED'); before=c.reopens
        values=symbolic_values(c.reconstruct('T','A')); self.assertEqual(values['q'],17); self.assertGreater(c.reopens,before)
    def test_naive_answer_summary_loses_revision_condition(self):
        events=[root(),edge(),edge('z','y',3)]; old=symbolic_values(events)['z']; revised=symbolic_values(events+[dict(kind='revision',child='y',parent='x',value=9)])['z']
        self.assertEqual(old,6); self.assertEqual(revised,13); self.assertNotEqual(old,revised)
    def test_false_stable_rejected_and_actual_early_halt(self):
        events=[root(),edge()]; self.assertNotEqual(verify_candidate({'x':9,'y':9},events,'y')['status'],'VERIFIED')
        count=[]
        def candidates():
            for candidate in [{'x':9,'y':9},{'x':1,'y':3},{'x':0,'y':0}]: count.append(1); yield candidate
        result=halt_candidates(candidates(),events,'y',max_steps=8)
        self.assertEqual(result['status'],'VERIFIED'); self.assertEqual(result['steps'],2); self.assertEqual(len(count),2)
    def test_unsupported_missing_and_cycle_unresolved(self):
        for events in [[edge()],[edge('x','y'),edge('y','x')],[{'kind':'unsupported'}]]:
            self.assertNotEqual(verify_candidate({'x':1,'y':3},events,'y')['status'],'VERIFIED')
    def test_persistence_reopen(self):
        key=self.a.append('T','A',root()); self.a.close(); self.a=Archive(self.path); self.assertEqual(self.a.read(key)['event'],root())
    def test_inverse_observation_preserves_independent_edge_facet(self):
        c=TitleController(3,self.a)
        events=[edge('y','x',2),dict(kind='root',child='y',parent=None,value=5)]
        ids=[c.append('T','inverse',e) for e in events]
        self.assertEqual(symbolic_values(events)['x'],3)
        self.assertTrue(self.a.applicable(ids[0],'T','inverse','y'))
        compact=c.compact_events('T','inverse'); self.assertEqual(len(compact),2)
        self.assertEqual(verify_candidate({'x':3,'y':5},compact,'x')['status'],'VERIFIED')
        self.assertNotEqual(verify_candidate({'x':5,'y':5},compact,'x')['status'],'VERIFIED')
    def test_inconsistent_anchors_do_not_certify(self):
        events=[root(),edge(),dict(kind='root',child='y',parent=None,value=99)]
        self.assertNotEqual(verify_candidate({'x':1,'y':3},events,'y')['status'],'VERIFIED')
    def test_compact_payload_disk_backed_and_hash_verified(self):
        c=TitleController(3,self.a)
        for event in [root(),edge(),edge('z','y',3),edge('q','z',4)]: c.append('T','A',event)
        self.assertNotIn('conditions',c.summaries[('T','A')])
        compact_id=c.summaries[('T','A')]['compact_id']; snapshot=self.a.read_compact(compact_id)
        self.assertIn('conditions',snapshot)
        self.a.db.execute('UPDATE compacts SET raw=? WHERE id=?',('{}',compact_id)); self.a.db.commit()
        with self.assertRaises(ValueError): self.a.read_compact(compact_id)
    def test_original_text_and_abstention_are_preserved(self):
        c=TitleController(3,self.a)
        parsed=dict(events=[root()],valid=True,confidence=.9,abstained=False,provenance={'model':'test'})
        result=c.observe_text('T','A','x is initially one',parsed)
        observed=self.a.read_observation(result['observation_id'])
        self.assertEqual(observed['text'],'x is initially one'); self.assertEqual(observed['parsed'],parsed)
        rejected=c.observe_text('T','A','do not use this',dict(events=[],valid=False,abstained=True))
        self.assertEqual(self.a.read_observation(rejected['observation_id'])['text'],'do not use this')
        self.assertEqual(rejected['record_ids'],[]); self.assertEqual(self.a.count(),1)

if __name__=='__main__': unittest.main()
