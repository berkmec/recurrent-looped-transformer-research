"""Independent receipt/oracle/metric audit of the completed width256 experiment.

Does not import the experiment generator, evaluator or model. Prefix cross
entropy is not independently reconstructed because raw logits were not saved.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCALE=ROOT/'integration'/'scale'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads((SCALE/'protocol.json').read_text());m=json.loads((SCALE/'metrics.json').read_text())
    assert m['status']=='COMPLETE' and m['protocol_sha256']==sha(SCALE/'protocol.json')
    assert sha(ROOT/'core/model.py')==p['code_hashes']['model']
    assert sha(ROOT/'integration/run_scale.py')==p['code_hashes']['runner']
    assert len(m['training'])==len(m['evaluation'])==6
    logs=[json.loads(s) for s in (SCALE/'execution.jsonl').read_text().splitlines()]
    first_test=next(i for i,r in enumerate(logs) if r['event']=='test_opened')
    assert sum(r['event']=='train_complete' for r in logs[:first_test])==6
    assert not any(r['event']=='heldout' for r in logs[:first_test])
    train={};tables=[];predictions={};checked=[]
    for row in m['training']:
        key=(row['seed'],row['mode']);assert key not in train;train[key]=row
        assert row['steps']==2000 and len(row['history'])==20
        assert [r['step'] for r in row['history']]==list(range(100,2001,100))
        assert all(math.isfinite(r['validation_loss']) for r in row['history'])
        selected=min(row['history'],key=lambda r:(r['validation_loss'],r['step']))['step']
        assert row['selected_step']==selected
        assert sha(row['selected_checkpoint'])==row['selected_checkpoint_sha256']
        assert sha(SCALE/row['name']/'initial.pt')==row['initial_file_sha256']
    import torch
    torch.set_num_threads(1)
    for seed in p['seeds']:
        a,b=[train[seed,mode] for mode in p['modes']]
        assert a['initial_weight_hash']==b['initial_weight_hash']
        weights=[torch.load(SCALE/r['name']/'initial.pt',map_location='cpu',weights_only=True) for r in (a,b)]
        assert weights[0].keys()==weights[1].keys()
        assert all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0])
    evaluation_keys=set()
    for evaluation in m['evaluation']:
        seed,mode=evaluation['seed'],evaluation['mode'];key=(seed,mode)
        assert key in train and key not in evaluation_keys;evaluation_keys.add(key)
        assert evaluation['protocol_sha256']==m['protocol_sha256']
        assert [r['length'] for r in evaluation['long_horizon']]==p['test_lengths']
        assert [r['slots'] for r in evaluation['neutral_slots']]==p['neutral_slots']
        rows=[]
        for family in ('long_horizon','neutral_slots'):
            for r in evaluation[family]:
                assert sha(r['file'])==r['sha256']
                data=np.load(r['file']);x,y,pred=[data[k] for k in ('inputs','targets','predictions')]
                length=r['length'] if family=='long_horizon' else 32+r['slots']
                assert x.shape==y.shape==pred.shape==(512,length)
                assert np.isin(x,[0,1,2]).all() and np.isin(pred,[0,1]).all()
                oracle=np.cumsum(x==1,axis=1)%2
                assert np.array_equal(y,oracle)
                final=float(np.mean(pred[:,-1]==y[:,-1]));prefix=float(np.mean(pred==y))
                assert abs(final-r['final_accuracy'])<1e-12 and abs(prefix-r['prefix_accuracy'])<1e-12
                index=r['length'] if family=='long_horizon' else r['slots']
                predictions[seed,mode,family,index]=(x,y,pred)
                record={'seed':seed,'mode':mode,'family':family,'index':index,'t':x.shape[1],
                        'final_accuracy':final,'prefix_accuracy':prefix,'seconds':r['seconds'],
                        'per_example_milliseconds':1000*r['seconds']/len(x),'sha256':r['sha256']}
                tables.append(record);checked.append(r['file'])
                if family=='long_horizon':rows.append(record)
        horizon=None
        for row in rows:
            if row['final_accuracy']<.95:break
            horizon=row['t']
        train[key]['audited_effective_tested_horizon95']=horizon
    assert evaluation_keys==set(train)
    for seed in p['seeds']:
        for family,indices in [('long_horizon',p['test_lengths']),('neutral_slots',p['neutral_slots'])]:
            for index in indices:
                a,b=[predictions[seed,mode,family,index] for mode in p['modes']]
                assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
        for mode in p['modes']:
            base=predictions[seed,mode,'neutral_slots',0]
            for slots in p['neutral_slots']:
                x,y,_=predictions[seed,mode,'neutral_slots',slots]
                assert np.array_equal(x[:,:32],base[0]) and (x[:,32:]==2).all()
                assert np.array_equal(y[:,-1],base[1][:,-1])
    result={'status':'PASS','prediction_files_checked':len(checked),'independent_prefix_targets_checked':sum(np.prod(v[0].shape) for v in predictions.values()).item(),
            'same_seed_inputs_targets_match':True,'actual_initial_tensors_match':True,
            'all_training_before_test_open':True,'frozen_sources_unchanged':True,'rows':tables,
            'horizons':[{'seed':r['seed'],'mode':r['mode'],'length':r['audited_effective_tested_horizon95'],'selected_step':r['selected_step']} for r in train.values()],
            'limitations':['prefix CE not independently recomputed: no raw logits','three seed-specific example sets: do not use crossed shared-example bootstrap','width/head count grow together; no scaling law','fixed optimizer across scales; negative fit is protocol-specific','recorded chronology is internal consistency, not external timestamp attestation']}
    path=ROOT/'audit'/'scale-independent-audit.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','limitations')}))


if __name__=='__main__':main()
