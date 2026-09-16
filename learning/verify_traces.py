"""Independent linear-system verification of every public TRAIN trace."""
from pathlib import Path
import hashlib,json,time
import numpy as np


def verify_arrays(events,intermediates,node_count=10):
    maximum=0.;queries=0
    for tape,expected in zip(events,intermediates):
        edges={};observed={};q=0
        for op,i,p,v in tape:
            i,p=int(i),int(p)
            if op in (0,3):observed[i]=float(v)
            elif op in (1,4):edges[i]=(p,float(v))
            elif op==2:
                rows=[];rhs=[]
                for child,(parent,offset) in edges.items():
                    row=np.zeros(node_count);row[child]=1;row[parent]=-1;rows.append(row);rhs.append(offset)
                for node,value in observed.items():
                    row=np.zeros(node_count);row[node]=1;rows.append(row);rhs.append(value)
                solution=np.linalg.solve(np.asarray(rows),np.asarray(rhs))
                error=float(abs(solution-expected[q]).max());maximum=max(maximum,error)
                if error>1e-6:raise AssertionError(f'corrupt trace at query{queries}: {error}')
                q+=1;queries+=1
    return {'examples':len(events),'query_states':queries,'node_targets_checked':queries*node_count,'max_absolute_error':maximum,'tolerance':1e-6}


def main():
    root=Path(__file__).parent;path=root/'results'/'verified-method-traces-TRAIN-ONLY.npz';data=np.load(path);t0=time.perf_counter()
    result=verify_arrays(data['events'],data['intermediates'])
    result.update({'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verifier':'independent10x10 linear solve, no generator oracle import','wall_seconds':time.perf_counter()-t0})
    (root/'results'/'trace-full-independent-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
