"""Training-only GPU timing to choose a feasible larger-model experiment.

Not an accuracy experiment. No final test data are generated or evaluated.
"""
import hashlib
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.model import Core,generate


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA calibration requires actual GPU')
    rows=[]
    for width in (128,256):
        for feedback in (True,False):
            torch.manual_seed(17)
            model=Core(width=width,heads=4,window=4,classes=2,feedback=feedback).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            x,y=generate('parity',32,32,8817)
            x,y=x.cuda(),y.cuda();times=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(9):
                torch.cuda.synchronize();start=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                pred=model(x);loss=torch.nn.functional.cross_entropy(pred.flatten(0,1),y.flatten())
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if step>=3:times.append(elapsed)
            row={'width':width,'feedback':feedback,**model.counts(),
                 'step_seconds':times,'median_step_seconds':float(np.median(times)),
                 'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'last_training_loss':float(loss.detach().cpu())}
            rows.append(row);print(json.dumps(row),flush=True)
            del optimizer,model,x,y,pred,loss
            torch.cuda.empty_cache()
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(),'cpu_threads':1,'precision':'float32',
            'batch':32,'tokens':32,'warmup_steps':3,'measured_steps':6,
            'source_sha256':hashlib.sha256((ROOT/'core/model.py').read_bytes()).hexdigest(),
            'scope':'training-only calibration; no final-test accuracy or speedup conclusion'}
    with (ROOT/'integration/scale_calibration.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)


if __name__=='__main__':main()
