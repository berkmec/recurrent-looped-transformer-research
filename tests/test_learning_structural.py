import unittest,importlib.util
import numpy as np


class StructuralTransfer(unittest.TestCase):
    def test_supplement_has_genuinely_unseen_depth_and_consistent_labels(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.structural_extra'),'structurally unseen-depth evaluation absent')
        from learning.structural_extra import make_extra
        data=make_extra(16,88112)
        self.assertTrue(np.all(data['depth']==8))
        for tape,target in zip(data['events'],data['targets']):
            edges={};observations={};q=0
            for op,i,p,v in tape:
                i,p=int(i),int(p)
                if op in (0,3):observations[i]=float(v)
                elif op in (1,4):edges[i]=(p,float(v))
                elif op==2:
                    rows=[];rhs=[]
                    for c,(parent,offset) in edges.items():
                        row=np.zeros(10);row[c]=1;row[parent]=-1;rows.append(row);rhs.append(offset)
                    for n,value in observations.items():
                        row=np.zeros(10);row[n]=1;rows.append(row);rhs.append(value)
                    result=np.linalg.solve(np.asarray(rows),np.asarray(rhs))
                    self.assertAlmostEqual(result[i],float(target[q]),places=5);q+=1

if __name__=='__main__':unittest.main()
