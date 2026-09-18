from pathlib import Path
import json
import platform
import sys
import time
import unittest
import torch
from rlt_lab.protocol import atomic_json

class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.rows=[]
    def addSuccess(self,test):
        super().addSuccess(test); self.rows.append({'test':test.id(),'status':'PASS'})
    def addFailure(self,test,err):
        super().addFailure(test,err);self.rows.append({'test':test.id(),'status':'FAIL','error':str(err[1])})
    def addError(self,test,err):
        super().addError(test,err);self.rows.append({'test':test.id(),'status':'ERROR','error':str(err[1])})
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.rows.append({'test':test.id(),'status':'SKIP','reason':reason})

if __name__=='__main__':
    root=Path(__file__).parent
    suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
    report=dict(scope='Independent mechanism/contract tests; NOT RLT training or benchmark',
        python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.cuda.is_available(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        seconds=time.perf_counter()-start,success=result.wasSuccessful(),tests=result.rows)
    atomic_json(root/'results'/'tests.json',report)
    sys.exit(0 if result.wasSuccessful() else 1)
