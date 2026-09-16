"""Export inference-only package and verify archive/oracle-free import-path run."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np

HERE=Path(__file__).parent


def main():
    root=HERE/'archive-off-sandbox'
    if root.exists():raise FileExistsError('archive-off evidence already exists')
    package=root/'learning';package.mkdir(parents=True)
    for name in ('__init__.py','model.py','token_core.py','api.py'):shutil.copy2(HERE/name,package/name)
    shutil.copy2(HERE/'checkpoints'/'method-17-step600.pt',root/'model.pt')
    # Inputs contain observations only. No targets or training trace are exported.
    source=np.load(HERE/'results'/'data-iid.npz');np.save(root/'events.npy',source['events'])
    script='''from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
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
'''
    dependency_site=str(Path(np.__file__).resolve().parent.parent)
    script=script.replace('import torch,numpy as np',f'sys.path.append({dependency_site!r})\nimport torch,numpy as np')
    (root/'infer.py').write_text(script)
    completed=subprocess.run([sys.executable,'-I',str(root/'infer.py')],cwd=root,capture_output=True,text=True)
    (root/'stdout.txt').write_text(completed.stdout);(root/'stderr.txt').write_text(completed.stderr)
    if completed.returncode:raise RuntimeError(completed.stderr)
    actual=np.load(root/'predictions.npy');expected=np.load(HERE/'results'/'pred-method-17-600-iid-retain.npz')['predictions']
    error=float(abs(actual-expected).max())
    if error>1e-7:raise AssertionError('isolated inference differs')
    receipt=json.loads((root/'receipt.json').read_text());receipt.update({'max_prediction_difference':error,'checkpoint_sha256':hashlib.sha256((root/'model.pt').read_bytes()).hexdigest(),
              'events_sha256':hashlib.sha256((root/'events.npy').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((root/'predictions.npy').read_bytes()).hexdigest(),
              'boundary':'self-contained export and isolated import path, not an operating-system filesystem sandbox; original research directories are not supplied to inference'})
    (HERE/'results'/'archive-off-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
