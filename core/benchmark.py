"""Fresh-process real training-step timing and native Windows peak working set."""
import argparse,ctypes,hashlib,json,subprocess,sys,time
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).parent
def rss():
 class Counters(ctypes.Structure):
  _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
 counter=Counters();counter.cb=ctypes.sizeof(counter);kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE;api=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];api.restype=wintypes.BOOL
 if not api(kernel.GetCurrentProcess(),ctypes.byref(counter),counter.cb):raise ctypes.WinError(ctypes.get_last_error())
 return dict(working_set_bytes=counter.WorkingSetSize,peak_working_set_bytes=counter.PeakWorkingSetSize,private_bytes=counter.PrivateUsage)
def freeze():
 p=ROOT/'benchmark-protocol.json'
 if not p.exists():
  p.write_text(json.dumps(dict(repeats=3,width=32,seed=17,checkpoint='parity-w32-rlt-s17-selected.pt',length=32,batch=32,modes=['full','tbptt8','checkpoint'],losses=['prefix','final_only'],data_seed=7900017,optimizer='AdamW lr.003 weight_decay.01 clip1',threads=1,warmup='one no-grad forward',memory='Windows native whole-process PeakWorkingSetSize; not activation-only allocation',purpose='real one-step gradient/update and process-memory comparison from paired weights/data in fresh processes'),indent=2));(ROOT/'benchmark-protocol.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==(ROOT/'benchmark-protocol.sha256').read_text();return json.loads(p.read_text())
def child(mode,loss):
 import torch
 from torch.nn import functional as F
 from torch.utils.checkpoint import checkpoint
 from core.model import Core,generate
 torch.set_num_threads(1);torch.set_num_interop_threads(1);m=Core();m.load_state_dict(torch.load(ROOT/'checkpoints'/'parity-w32-rlt-s17-selected.pt',weights_only=True));x,y=generate('parity',32,32,7900017);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01)
 with torch.no_grad():m(x)
 before=rss();start=time.perf_counter();z=checkpoint(m,x,use_reentrant=False) if mode=='checkpoint' else m(x,tbptt=8 if mode=='tbptt8' else 0);value=F.cross_entropy(z.flatten(0,1),y.flatten()) if loss=='prefix' else F.cross_entropy(z[:,-1],y[:,-1]);value.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();seconds=time.perf_counter()-start;after=rss();print(json.dumps(dict(mode=mode,loss=loss,seconds=seconds,before=before,after=after,loss_value=float(value.detach()),gradient_norm=grad,torch=torch.__version__,threads=torch.get_num_threads())))
def main():
 p=freeze();rows=[];target=ROOT/'results'/'benchmark.json'
 if target.exists():raise RuntimeError('Benchmark evidence exists; refuse overwrite')
 for loss in p['losses']:
  for mode in p['modes']:
   for repeat in range(p['repeats']):
    process=subprocess.run([sys.executable,'-m','core.benchmark','--child',mode,'--loss',loss],cwd=ROOT.parent,capture_output=True,text=True,check=True);r=json.loads(process.stdout);r['repeat']=repeat;rows.append(r)
 with target.open('x',encoding='utf8') as f:json.dump(dict(protocol_hash=hashlib.sha256((ROOT/'benchmark-protocol.json').read_bytes()).hexdigest(),rows=rows,scope='fresh child processes run sequentially; external system load not controlled'),f,indent=2)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--child');a.add_argument('--loss');a.add_argument('--freeze-only',action='store_true');args=a.parse_args()
 if args.freeze_only:freeze()
 elif args.child:child(args.child,args.loss)
 else:main()
