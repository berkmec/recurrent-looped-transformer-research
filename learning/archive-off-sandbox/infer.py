from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.append('__USER_HOME__\\AppData\\Roaming\\Python\\Python311\\site-packages')
import torch,numpy as np
from learning.api import load_model
torch.set_num_threads(1)
assert importlib.util.find_spec('learning.data') is None
assert importlib.util.find_spec('learning.run') is None
model=load_model('model.pt');x=np.load('events.npy');out=[]
with torch.no_grad():
    for start in range(0,len(x),64):out.append(model(torch.from_numpy(x[start:start+64]))['predictions'].numpy())
np.save('predictions.npy',np.concatenate(out))
Path('receipt.json').write_text(json.dumps({'oracle_module_available':False,'training_runner_available':False,'examples':len(x),'exported_input_fields':'events only','isolated_python':sys.flags.isolated}))
