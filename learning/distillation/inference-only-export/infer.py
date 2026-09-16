from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.append('C:\\Users\\pc\\AppData\\Roaming\\Python\\Python311\\site-packages')
import torch,numpy as np
from learning.api import load_model
torch.set_num_threads(1)
assert importlib.util.find_spec('learning.data') is None
assert importlib.util.find_spec('learning.distillation') is None
x=np.load('events.npy');outputs={}
for name in ['answer_only-17-step300.pt', 'answer_only-43-step300.pt', 'answer_only-91-step300.pt', 'trajectory-17-step300.pt', 'trajectory-43-step300.pt', 'trajectory-91-step300.pt']:
    model=load_model(name);rows=[]
    with torch.no_grad():
        for start in range(0,len(x),64):rows.append(model(torch.from_numpy(x[start:start+64]))['predictions'].numpy())
    outputs[name]=np.concatenate(rows)
np.savez_compressed('predictions.npz',**outputs)
Path('receipt.json').write_text(json.dumps({'isolated_import_path':bool(sys.flags.isolated),'oracle_importable':False,'trace_package_importable':False,'models':['answer_only-17-step300.pt', 'answer_only-43-step300.pt', 'answer_only-91-step300.pt', 'trajectory-17-step300.pt', 'trajectory-43-step300.pt', 'trajectory-91-step300.pt'],'examples_per_model':len(x)}))
