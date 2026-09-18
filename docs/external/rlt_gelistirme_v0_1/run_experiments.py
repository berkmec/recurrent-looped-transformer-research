"""Falsifiable mechanism probes. None measures trained-RLT intelligence/speed."""
from pathlib import Path
import platform
import sys
import tempfile
import time
import torch
from rlt_lab.core import protected_update, recurrent_scan, affine_scan
from rlt_lab.memory import Archive
from rlt_lab.protocol import atomic_json

def invariance_probe():
    torch.manual_seed(13)
    z=torch.randn(8,8,dtype=torch.float64)
    original=z.clone(); work=torch.zeros_like(z)
    gate=torch.zeros(8,dtype=torch.bool)
    checkpoints=[]; start=time.perf_counter()
    with torch.no_grad():
        for t in range(1,65537):
            work=torch.tanh(0.7*work+0.2*torch.sin(original+t*0.001))
            proposal=z+0.01*work
            z=protected_update(z,proposal,gate)
            if t in {128,512,4096,16384,65536}:
                checkpoints.append(dict(steps=t,bitwise_equal=torch.equal(z,original),
                    max_change=float((z-original).abs().max()),work_norm=float(work.norm())))
    return dict(name='explicit_hold_contract',scope='Externally supplied all-false write gate; NOT learned neutral detection',
                checkpoints=checkpoints,seconds=time.perf_counter()-start)

def credit_trial(mode):
    torch.manual_seed(123)
    w=torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*0.05)
    z=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    cache=torch.zeros_like(z,requires_grad=True)
    xs=torch.randn(128,2,4,dtype=torch.float64,requires_grad=True)
    def step(st,x):
        h,c=st
        return h+0.01*torch.tanh(x@w+0.1*c),0.9*c+0.1*h
    if mode=='tbptt8':
        out=(z,cache)
        for t,x in enumerate(xs):
            if t and t%8==0:
                out=tuple(v.detach() for v in out)
            out=step(out,x)
    else:
        out=recurrent_scan(step,(z,cache),xs,None if mode=='full' else 8)
    loss=out[0].square().mean()+0.03*out[1].square().mean()
    loss.backward()
    tensors=[w,z,cache,xs]
    grads=[torch.zeros_like(t) if t.grad is None else t.grad.detach().clone() for t in tensors]
    return dict(mode=mode,loss=float(loss.detach()),initial_gradient_present=z.grad is not None,
                initial_gradient_norm=float(grads[1].norm()),first_input_gradient_norm=float(grads[3][0].norm())),grads

def credit_probe():
    full,g=credit_trial('full'); cp,g2=credit_trial('remat8'); truncated,g3=credit_trial('tbptt8')
    return dict(name='terminal_loss_credit_path',scope='Untrained 2-state nonlinear recurrence, CPU float64',
                modes=[full,cp,truncated],max_gradient_error_full_vs_remat=max(float((a-b).abs().max()) for a,b in zip(g,g2)),
                warning='Nonzero gradient/equivalence is not successful delayed-reward learning.')

def scan_probe():
    torch.manual_seed(55)
    a=torch.rand(4096,2,4,dtype=torch.float64)*0.05+0.95
    b=torch.randn_like(a)*0.01; z=torch.randn(2,4,dtype=torch.float64)
    out=affine_scan(a,b,z); h=z; serial=[]
    for at,bt in zip(a,b):
        h=at*h+bt;serial.append(h)
    reference=torch.stack(serial)
    return dict(name='associative_transport_path',scope='Diagonal affine path ONLY; not nonlinear RLT',length=4096,
                max_abs_error=float((out-reference).abs().max()),cuda_speed='UNMEASURED')

def archive_probe():
    with tempfile.TemporaryDirectory() as tmp:
        a=Archive(Path(tmp)/'a.sqlite')
        raw=[]
        for i in range(64):
            text=(f'Başlık A, sürüm {i+1}: ham kayıt, yanlışlar da korunuyor.\r\n').encode()
            row=a.append('A',text,'compact',model='toy-model-v1',expected_version=i)
            raw.append((row['digest'],text))
        a.backup(Path(tmp)/'backup.sqlite')
        result=dict(name='immutable_raw_and_versions',rows_verified=a.verify(),
                    raw_exact=all(a.raw(d)==v for d,v in raw),backup_verified=True,
                    scope='Local SQLite application-level contract; not disk-failure-proof hardware')
        a.close();return result

if __name__=='__main__':
    torch.set_num_threads(1)
    report=dict(scope='MECHANISM_PROBES_ONLY',python=sys.version,torch=torch.__version__,
                cuda=torch.cuda.is_available(),platform=platform.platform(),
                experiments=[invariance_probe(),credit_probe(),scan_probe(),archive_probe()])
    root=Path(__file__).parent
    atomic_json(root/'results'/'mechanisms.json',report)
    for item in report['experiments']:print(item)
