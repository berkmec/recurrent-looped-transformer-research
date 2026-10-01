"""Correct independent replay's numerical batch mismatch; preserve first audit."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from learning.api import load_model
rows=[]
for seed in [17,43,91]:
    model=load_model(BASE/'learning/checkpoints'/f'combined-{seed}-step600.pt')
    with np.load(BASE/'learning/distillation/archives'/f'teacher-{seed}.npz') as data:
        events=data['events'][:64];states=data['states'][:64];nodes=data['nodes'][:64];final=data['final'][:64]
    maximum=0.
    with torch.no_grad():
        runtime=model.new_runtime(torch.from_numpy(events));q=0
        while runtime.consume_to_query():
            for k in range(6):
                candidate=runtime.refine_one();maximum=max(maximum,float(abs(runtime.z.numpy()-states[:,q,k]).max()),float(abs(candidate['nodes'].numpy()-nodes[:,q,k]).max()))
            maximum=max(maximum,float(abs(candidate['prediction'].numpy()-final[:,q]).max()));runtime.commit_feedback();q+=1
    rows.append(dict(seed=seed,episodes=64,batch_size=64,max_difference=maximum))
result=dict(status='PASS' if all(r['max_difference']==0 for r in rows) else 'FAIL',rows=rows,reason='Original independent16 batch changed floating-point execution shape; original collection64 restored for exact comparison. No tolerance widened.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=BASE/'audit/distillation-replay-samebatch.json'
with path.open('x') as f:json.dump(result,f,indent=2)
if result['status']=='PASS':
    main=BASE/'audit/distillation-independent-audit.json';report=BASE/'reports/distillation-independent-audit.md'
    shutil.copy2(main,BASE/'audit/distillation-independent-audit-initial-batch16.json');shutil.copy2(report,BASE/'audit/distillation-independent-audit-initial-batch16.md')
    audit=json.loads(main.read_text());corrected=[]
    for c in audit['checks']:
        if c['name'].startswith('independent16_episode_actual_teacher_replay_'):
            seed=int(c['name'].rsplit('_',1)[1]);r=next(r for r in rows if r['seed']==seed);c.update(name='independent64_episode_actual_teacher_replay_'+str(seed),passed=True,detail=r);corrected.append(seed)
    audit['failures']=[x for x in audit['failures'] if not x.startswith('independent16_episode_actual_teacher_replay_')];audit['status']='PASS_WITH_SCOPE_LIMITS' if not audit['failures'] else 'FAIL';audit['independent_teacher_replay']=rows;audit['replay_numerical_correction']=result
    audit['limitations']=[s.replace('sampled16 fixed TRAIN episodes per seed','sampled64 fixed TRAIN episodes per seed using original batch size64') for s in audit['limitations']];main.write_text(json.dumps(audit,indent=2))
    text=report.read_text().replace('Status **FAIL**; 108/111 checks pass. Failures:', 'Status **PASS_WITH_SCOPE_LIMITS**; 111/111 checks pass. Initial batch16 replay diagnostic failures, subsequently resolved with original batch64:')
    text=text.replace('sampled16 fixed TRAIN episodes per seed','sampled64 fixed TRAIN episodes per seed using original batch size64')
    text+='\n## Replay numerical correction\n\nThe first independent check used batch16 and required bitwise equality against batch64 collection. Its maximum differences were2.38e-7 to2.98e-7. Repeating with the original batch64 yielded exactly0 for all three teachers across192 independently replayed TRAIN episodes. No tolerance was widened. The first audit and detailed correction receipt remain in audit/. This was an audit execution-shape mismatch, not altered teacher evidence.\n'
    report.write_text(text)
print(json.dumps(result,indent=2))
