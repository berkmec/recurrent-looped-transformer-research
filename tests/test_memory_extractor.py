import unittest
from memory.extractor import make_dataset, tokenize, normalize_token, reconstruct_event

class ExtractorTests(unittest.TestCase):
    def test_split_entities_and_heldout_layouts(self):
        train=make_dataset('train',64,17); test=make_dataset('test',64,17)
        self.assertTrue(set(x['id'] for x in train).isdisjoint(x['id'] for x in test))
        self.assertTrue(any(x['family']=='heldout' for x in test)); self.assertFalse(any(x['family']=='heldout' for x in train))
    def test_copy_keeps_unseen_numbers_and_entities(self):
        event=reconstruct_event(['x999','is','-0.375'],[1,0,3],0)
        self.assertEqual(event,dict(kind='root',child='x999',parent=None,value=-.375))
        self.assertEqual(normalize_token('x999'),'<entity>'); self.assertEqual(normalize_token('-0.375'),'<number>')
    def test_ambiguous_roles_fail_closed(self):
        with self.assertRaises(ValueError): reconstruct_event(['x1','x2','2'],[1,1,3],0)
        with self.assertRaises(ValueError): reconstruct_event(['hello','2'],[1,3],0)
    def test_generation_ground_truth_and_no_answer(self):
        for row in make_dataset('train',64,17):
            self.assertNotIn('answer',row)
            if row['event'] is not None: self.assertEqual(reconstruct_event(row['tokens'],row['roles'],row['class']),row['event'])

if __name__=='__main__': unittest.main()
