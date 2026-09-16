import unittest,importlib.util
import numpy as np


class TraceVerification(unittest.TestCase):
    def test_corrupted_intermediate_target_is_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.verify_traces'),'independent trace archive verifier absent')
        from learning.verify_traces import verify_arrays
        events=np.array([[[0,0,0,.2],[1,1,0,.3],[2,1,1,0]]],np.float32)
        nodes=np.array([[[.2,.5]]],np.float32)
        self.assertLess(verify_arrays(events,nodes,2)['max_absolute_error'],1e-6)
        nodes[0,0,0]=.7
        with self.assertRaises(AssertionError):verify_arrays(events,nodes,2)

if __name__=='__main__':unittest.main()
