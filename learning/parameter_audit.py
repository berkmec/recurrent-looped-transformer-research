"""Train-only gradient-bearing parameter audit, no final-test access."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from .model import EventModel


def main():
    torch.set_num_threads(1);root=Path(__file__).parent
    data=np.load(root/'results'/'data-train.npz');index=np.load(root/'results'/'batches-17.npy')[0]
    x=torch.from_numpy(data['events'][index]);y=torch.from_numpy(data['targets'][index]);z=torch.from_numpy(data['node_targets'][index])
    rows=[]
    for arm in ('forward','bidir','wrong','flat','feedback','method'):
        torch.manual_seed(17);model=EventModel(feedback=arm=='feedback');direction=arm if arm in ('forward','wrong','flat') else 'bidir'
        out=model(x,direction=direction);loss=((out['predictions']-y)**2).mean()
        if arm=='method':loss=loss+.3*((out['nodes']-z)**2).mean()
        loss.backward();details=[]
        for name,p in model.named_parameters():details.append({'name':name,'allocated':p.numel(),'gradient_present':p.grad is not None,'nonzero_gradient_entries':0 if p.grad is None else int(torch.count_nonzero(p.grad))})
        rows.append({'arm':arm,'allocated':sum(p.numel() for p in model.parameters()),
                     'gradient_present_parameters':sum(d['allocated'] for d in details if d['gradient_present']),
                     'nonzero_gradient_parameters':sum(d['nonzero_gradient_entries'] for d in details),'details':details})
    result={'scope':'same seed17 initial tensors and first TRAIN batch; observed gradient support is a probe, not all possible active computation',
            'matching_limit':'Dense title shapes, allocated parameters, samples and optimizer steps match. Forward zeros reverse-channel inputs, feedback enables an extra title projection, method adds auxiliary backward work. Effective parameters and compute are NOT exactly matched.',
            'train_data_sha256':hashlib.sha256((root/'results'/'data-train.npz').read_bytes()).hexdigest(),'rows':rows}
    (root/'results'/'parameter-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='details'} for r in rows]))

if __name__=='__main__':main()
