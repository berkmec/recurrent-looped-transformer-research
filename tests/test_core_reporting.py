import unittest,numpy as np
from core.summarize import actual_state_changes,read
class Reporting(unittest.TestCase):
 def test_read_utf8_bom_receipt(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'receipt.json';p.write_text('{"status":"COMPLETE"}',encoding='utf-8-sig')
   self.assertEqual(read(p),{'status':'COMPLETE'})
 def test_swap_can_fix_the_current_five_state(self):
  # cycle0->1, swap1->0, swap0->1, identity, cycle1->2, swap fixes2.
  targets=np.array([[1,0,1,1,2,2]])
  np.testing.assert_array_equal(actual_state_changes(targets),[[True,True,True,False,True,False]])
if __name__=='__main__':unittest.main()
import tempfile
from pathlib import Path
