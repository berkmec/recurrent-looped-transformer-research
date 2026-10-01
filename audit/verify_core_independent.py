"""Independent, completion-gated core audit; imports no experimental generator/model.

Only --check-ready is appropriate while producers run. Final execution needs the
owner's quiescent evidence-binding and completed report/recomputation artifacts.
No model fitting, test tuning, overwrite, or binding mutation is performed.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, json, math, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'core'
OUT=ROOT/'audit'/'core-independent-audit.json'
REPORT=ROOT/'reports'/'core-independent-audit.md'
PREPARATION=ROOT/'audit'/'core-independent-preparation.json'
SEEDS=(17,43,91)
LENGTHS=(32,64,128,256,512,1024)
SLOTS=(0,8,32,128)
PRIMARY={(task,32,mode,seed) for task in ('parity','five') for seed in SEEDS
         for mode in (('rlt','nofeedback','tbptt') if task=='parity' else ('rlt','nofeedback'))}
PRIMARY|={('parity',48,mode,seed) for seed in SEEDS for mode in ('rlt','nofeedback')}
FINAL={('parity',32,mode,seed) for seed in SEEDS for mode in ('rlt','tbptt')}

def require(test,message):
    if not test:raise AssertionError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def key(condition):return (condition['task'],condition['width'],condition['mode'],condition['seed'])
def name(k):return f'{k[0]}-w{k[1]}-{k[2]}-s{k[3]}'
def equal(a,b,message):require(np.array_equal(a,b),message)

def inside(path):
    p=Path(path)
    if not p.is_absolute():p=CORE/p
    p=p.resolve()
    require(p.is_relative_to(CORE.resolve()),'evidence path escapes core: '+str(path))
    require(p.is_file(),'missing evidence: '+str(p))
    return p

def ready():
    """Read completion metadata only; never access prediction arrays here."""
    needed=['results/metrics.json','results/final-only-metrics.json','results/diagnostics.json',
            'results/benchmark.json','results/recomputed.json','evidence-binding.json',
            'evaluation-seals.jsonl','report.md']
    missing=[rel for rel in needed if not (CORE/rel).is_file()]
    if missing:return False,{'missing':missing}
    try:
        primary=read(CORE/needed[0]);final=read(CORE/needed[1]);diag=read(CORE/needed[2]);bench=read(CORE/needed[3])
        counts={'primary_training':len(primary.get('training',[])),'primary_evaluation':len(primary.get('evaluation',[])),
                'final_only':len(final.get('records',[])),'diagnostics':len(diag.get('rows',[])),
                'benchmark':len(bench.get('rows',[])),'first_observation_seals':len(lines(CORE/'evaluation-seals.jsonl'))}
        complete=(primary.get('status')==final.get('status')=='COMPLETE' and
                  list(counts.values())==[21,21,6,9,18,21])
        return complete,{'counts':counts,'owner_binding_present':True}
    except (ValueError,OSError,KeyError) as error:return False,{'unready_metadata':str(error)}

def independent_targets(inputs,task):
    """XOR parity or arithmetic cycle/transposition; no experimental PERMS import."""
    x=np.asarray(inputs);require(x.ndim==2 and np.isin(x,[0,1,2]).all(),'invalid task tokens')
    require(task in ('parity','five'),'unknown task')
    state=np.zeros(len(x),dtype=np.int64);out=np.empty(x.shape,dtype=np.int64)
    for t in range(x.shape[1]):
        token=x[:,t]
        if task=='parity':state=(state+(token==1))%2
        else:
            cycle=(state+1)%5
            swap=np.where(state==0,1,np.where(state==1,0,state))
            state=np.where(token==0,cycle,np.where(token==1,swap,state))
        out[:,t]=state
    return out

def seeded_inputs(task,seed,length,offset,n=512):
    stream=offset+seed+101*length+(10000 if task=='five' else 0)
    return np.random.default_rng(stream).choice(3,(n,length),p=[.4375,.4375,.125])

def condition_map(rows,expected,field='condition'):
    keys=[key(row[field]) for row in rows]
    require(len(keys)==len(expected) and set(keys)==expected,'missing/duplicate condition cells')
    return dict(zip(keys,rows))

def run_audit():
    ok,state=ready()
    if not ok:print(json.dumps({'status':'NOT_READY',**state}));return 2
    require(not OUT.exists(),'independent audit already exists; no overwrite')
    require(not REPORT.exists(),'independent audit report already exists; no overwrite')
    # Only immutable completed artifacts are hashed; no broad live-directory scan.
    checked={};tables=[];tensor_pairs=[];array_pairs={};diagnostic_checks=[]
    def receipt(path,expected=None):
        p=inside(path);digest=sha(p)
        if expected is not None:require(digest==expected,'receipt drift: '+str(p))
        checked[p.relative_to(CORE).as_posix()]=digest
        return p
    primary=read(receipt('results/metrics.json'));final=read(receipt('results/final-only-metrics.json'))
    diagnostics=read(receipt('results/diagnostics.json'));benchmark=read(receipt('results/benchmark.json'))
    binding=read(receipt('evidence-binding.json'))
    # Existing binding remains untouched. Audit selected finished evidence against it.
    def bound(path):
        p=inside(path);rel=p.relative_to(CORE).as_posix()
        require(rel in binding['files'],'unbound audited artifact: '+rel)
        return receipt(p,binding['files'][rel])
    protocols={}
    for stem,metrics in [('protocol',primary),('final-only-protocol',final),('diagnostics-protocol',diagnostics),('benchmark-protocol',benchmark)]:
        path=bound(stem+'.json');digest=sha(path);protocols[stem]=read(path)
        require((bound(stem+'.sha256')).read_text().strip()==digest,'protocol digest drift')
        require(metrics['protocol_hash']==digest,'result protocol mismatch: '+stem)
    require(protocols['protocol']['source_pin']=='cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39','source pin')
    require(protocols['protocol']['tbptt']==8 and protocols['final-only-protocol']['loss']=='final target only cross entropy','supervision scope mismatch')
    preparation=read(PREPARATION)
    for relative,digest in preparation['frozen_files'].items():require(sha(ROOT/relative)==digest,'pre-audit source drift: '+relative)
    for environment in ['environment.json','final-only-environment.json','diagnostics-environment.json']:
        env=read(bound(environment));require(env['device']=='cpu' and env['threads']==1,'execution environment mismatch')
        for filename,digest in env['source_hashes'].items():
            receipt(filename,digest);receipt(Path('source-snapshots')/digest/filename,digest)
    observer=read(bound('evaluation-observer-protocol.json'))
    require(observer['primary_protocol_sha256']==sha(CORE/'protocol.json'),'observer primary pin')
    receipt('observe_evaluation.py',observer['observer_source_sha256'])
    require(bound('evaluation-observer-protocol.sha256').read_text().strip()==sha(CORE/'evaluation-observer-protocol.json'),'observer protocol digest')
    seals=condition_map(lines(bound('evaluation-seals.jsonl')),PRIMARY)
    for seal in seals.values():
        require(seal['protocol_hash']==sha(CORE/'evaluation-observer-protocol.json'),'seal protocol')
        for relative,digest in seal['files'].items():receipt(relative,digest)
    training=condition_map(primary['training'],PRIMARY);evaluations=condition_map(primary['evaluation'],PRIMARY)
    companions=condition_map(final['records'],FINAL)
    import torch
    torch.set_num_threads(1)
    initial={}
    def training_record(k,row,is_final=False):
        prefix=('finalonly-' if is_final else '')+name(k)
        suffix='.json' if is_final else '-train.json'
        require(read(bound('results/'+prefix+suffix))==row,'training aggregation disagreement')
        history=row['history'];require(row['steps']==2000 and [h['step'] for h in history]==list(range(100,2001,100)),'optimizer/validation budget')
        loss=lambda h:h['validation_final_ce'] if is_final else h['validation']['loss']
        require(all(math.isfinite(loss(h)) for h in history),'nonfinite validation')
        selected=min(history,key=lambda h:(loss(h),h['step']))['step']
        require(row['selected_step']==selected,'checkpoint chosen against declared validation criterion')
        hashes=row['hashes'] if is_final else row['checkpoint_hashes']
        require(set(hashes)=={'initial','selected','last'},'checkpoint receipt set')
        for typ,digest in hashes.items():receipt(f'checkpoints/{prefix}-{typ}.pt',digest)
        state=torch.load(CORE/'checkpoints'/f'{prefix}-initial.pt',map_location='cpu',weights_only=True)
        initial[is_final,k]=state
        allocated=sum(t.numel() for t in state.values())
        inactive=sum(t.numel() for n,t in state.items() if n=='initial' or n.startswith(('pn.','gate.','merge.'))) if k[2]=='nofeedback' else 0
        require(row['parameters']=={'allocated':allocated,'active':allocated-inactive},'active parameter accounting')
        require(row['train_seconds']>0,'invalid recorded duration')
    for k,row in training.items():training_record(k,row)
    for k,row in companions.items():training_record(k,row,True)
    def paired_state(a,b):
        sa,sb=initial[a],initial[b]
        require(sa.keys()==sb.keys() and all(torch.equal(sa[n],sb[n]) for n in sa),'unpaired actual initial tensors')
        tensor_pairs.append([str(a),str(b)])
    for task,width in [('parity',32),('five',32),('parity',48)]:
        for seed in SEEDS:
            paired_state((False,(task,width,'rlt',seed)),(False,(task,width,'nofeedback',seed)))
            if task=='parity' and width==32:paired_state((False,(task,width,'rlt',seed)),(False,(task,width,'tbptt',seed)))
    for seed in SEEDS:paired_state((True,('parity',32,'rlt',seed)),(True,('parity',32,'tbptt',seed)))
    def array_record(k,row,family,length,expected_x,path,expected_hash=None,is_final=False):
        p=bound(path)
        if expected_hash:receipt(p,expected_hash)
        with np.load(p,allow_pickle=False) as archive:
            x,y,pred=(archive[n] for n in ('inputs','targets','predictions'))
            require(x.shape==y.shape==pred.shape==(512,length),'wrong declared token length/array dimensions')
            equal(x,expected_x,'inputs differ from declared independent seed stream')
            oracle=independent_targets(x,k[0]);equal(y,oracle,'independent task oracle mismatch')
            require(np.isin(pred,range(2 if k[0]=='parity' else 5)).all(),'invalid model class')
            if family!='neutral':
                require('example_id' in archive,'missing example identity')
                equal(archive['example_id'],np.arange(512),'example identity')
            if family=='primary':
                require('split_seed' in archive and archive['split_seed'].shape==(),'missing scalar split seed')
                require(int(archive['split_seed'])==900000+k[3]+101*length+(10000 if k[0]=='five' else 0),'split seed receipt')
            acc=float((pred[:,-1]==y[:,-1]).mean());prefix=float((pred==y).mean())
            require(acc==row['final_accuracy'] and prefix==row['prefix_accuracy'],'prediction metric mismatch')
            previous=np.concatenate((np.zeros((512,1),dtype=y.dtype),y[:,:-1]),axis=1)
            changes=float((y!=previous).sum(1).mean())
            input_hash=hashlib.sha256(x.tobytes()).hexdigest();target_hash=hashlib.sha256(y.tobytes()).hexdigest()
            pairing=(is_final,k[0],k[3],family,length)
            if pairing in array_pairs:require(array_pairs[pairing]==(input_hash,target_hash),'unpaired same-seed examples')
            array_pairs[pairing]=(input_hash,target_hash)
            tables.append({'condition':list(k),'family':family,'final_only':is_final,'length':length,
                           'final_accuracy':acc,'prefix_accuracy':prefix,'true_state_changes_mean':changes,
                           'non_neutral_inputs_mean':float((x!=2).sum(1).mean()),'path':p.relative_to(CORE).as_posix()})
    for k,ev in evaluations.items():
        require(read(bound('results/'+name(k)+'-eval.json'))==ev,'evaluation aggregation disagreement')
        require([r['length'] for r in ev['long_horizon']]==list(LENGTHS),'primary length sweep')
        require([r['slots'] for r in ev['neutral_slots']]==list(SLOTS),'neutral sweep')
        for row in ev['long_horizon']:
            length=row['length'];require(row['k']==0 and row['consumed_tokens']==length,'token/refinement clock')
            array_record(k,row,'primary',length,seeded_inputs(k[0],k[3],length,900000),row['predictions_file'],row['sha256'])
        base=seeded_inputs(k[0],k[3],32,1900000)
        for row in ev['neutral_slots']:
            slots=row['slots'];x=np.concatenate((base,np.full((512,slots),2,dtype=base.dtype)),axis=1)
            require(row['k']==0 and row['consumed_tokens']==32+slots,'neutral token clock')
            array_record(k,row,'neutral',32+slots,x,row['file'])
            # Independent oracle establishes token2 preserves both parity and five state.
            equal(independent_targets(x,k[0])[:,-1],independent_targets(base,k[0])[:,-1],'neutral suffix changes target')
    for k,row in companions.items():
        require(row['supervision']=='final_only' and [r['length'] for r in row['evaluation']]==[32,128],'delayed-error companion scope')
        for ev in row['evaluation']:
            length=ev['length'];array_record(k,ev,'final_only',length,seeded_inputs('parity',k[3],length,6900000),ev['file'],ev['sha256'],True)
    # Chronology is a consistency check on saved events, not an external timestamp proof.
    events=lines(bound('execution.jsonl'));counts=Counter(e['event'] for e in events)
    expected_counts={'protocol_frozen':1,'train_start':21,'validation':420,'all_training_complete':1,'heldout':126,'primary_complete':1,
                     'final_only_protocol_frozen':1,'final_only_train_start':6,'final_only_validation':120,'final_only_complete':1}
    require(all(counts[k]==v for k,v in expected_counts.items()),'missing/duplicate execution events')
    completed_at=next(i for i,e in enumerate(events) if e['event']=='all_training_complete')
    require(all(i>completed_at for i,e in enumerate(events) if e['event']=='heldout'),'primary final opened before all training completed')
    for is_final,records in [(False,training),(True,companions)]:
        for k,row in records.items():
            label=('finalonly-' if is_final else '')+name(k);kind='final_only_validation' if is_final else 'validation'
            found=[(i,e) for i,e in enumerate(events) if e['event']==kind and e['condition']==label]
            require([e['step'] for _,e in found]==list(range(100,2001,100)),'per-condition step chronology')
            starts=[i for i,e in enumerate(events) if e['event']==('final_only_train_start' if is_final else 'train_start') and (e['condition']==label if is_final else key(e['condition'])==k)]
            require(len(starts)==1 and starts[0]<found[0][0],'validation precedes training start')
            if not is_final:require(found[-1][0]<completed_at,'training completion precedes final validation')
            for (_,event),history in zip(found,row['history']):
                require(all(event[n]==v for n,v in history.items()),'training history/event disagreement')
    # Diagnostics are saved full-forward derivatives at trained checkpoints, not TBPTT training gradients.
    drows=diagnostics['rows'];require(len(drows)==9 and {(r['seed'],r['mode']) for r in drows}=={(s,m) for s in SEEDS for m in ('rlt','nofeedback','tbptt')},'diagnostic cells')
    require(sum('checkpoint_repeats' in row for row in drows)==1,'missing/duplicate checkpoint equivalence diagnostic')
    for row in drows:
        inf=row['influence'];require(inf['evaluation_dtype']=='torch.float64','diagnostic dtype')
        require([v['lag'] for v in inf['state_norms']]==[1,4,8,16,31],'state influence lags')
        require([v['lag'] for v in inf['embedding_norms']]==[1,4,8,16,31],'embedding influence lags')
        g,fd=inf['directional_autograd'],inf['directional_finite_difference']
        require(all(math.isfinite(v) for v in [g,fd,inf['loss']]),'nonfinite derivative')
        if row['mode']=='nofeedback':require(g==0 and fd==0 and all(v['norm']==0 for v in inf['state_norms']),'no-feedback recurrent sensitivity')
        diagnostic_checks.append({'seed':row['seed'],'mode':row['mode'],'finite_difference_absolute_gap':abs(g-fd),
                                  'reported_gradient':g,'reported_central_difference':fd})
        require([n['std'] for n in row['noise']]==[0,.001,.01,.1],'noise sweep')
        expected_x=np.random.default_rng(3900000+row['seed']).choice(3,(512,32),p=[.4375,.4375,.125])
        for n in row['noise']:
            path=receipt(n['file'],n['sha256']);bound(path)
            with np.load(path,allow_pickle=False) as raw:
                equal(raw['inputs'],expected_x,'noise inputs seed stream')
                equal(raw['targets'],independent_targets(expected_x,'parity')[:,-1],'noise labels')
                require(raw['predictions'].shape==(512,) and np.isin(raw['predictions'],[0,1]).all(),'noise class/shape')
                require(float((raw['predictions']==raw['targets']).mean())==n['accuracy'],'noise accuracy')
        if 'checkpoint_repeats' in row:
            require(row['seed']==17 and row['mode']=='rlt' and len(row['checkpoint_repeats'])==3,'checkpoint diagnostic scope')
            for repeat in row['checkpoint_repeats']:
                require([s['checkpoint'] for s in repeat['steps']]==[False,True],'checkpoint pair')
                for metric in ('loss_difference','max_gradient_difference','max_parameter_update_difference'):
                    require(math.isfinite(repeat[metric]) and 0<=repeat[metric]<=1e-6,'recorded checkpoint equivalence '+metric)
    # Fresh-process whole-RSS benchmark: do not attribute these bytes to activations.
    brows=benchmark['rows'];expected={(loss,mode,rep) for loss in ('prefix','final_only') for mode in ('full','tbptt8','checkpoint') for rep in range(3)}
    require(len(brows)==18 and {(r['loss'],r['mode'],r['repeat']) for r in brows}==expected,'benchmark completeness')
    for row in brows:
        require(row['threads']==1 and math.isfinite(row['seconds']) and row['seconds']>0,'benchmark timing/threads')
        for phase in ('before','after'):
            mem=row[phase];require(all(isinstance(v,int) and v>0 for v in mem.values()),'benchmark process memory')
            require(mem['peak_working_set_bytes']>=mem['working_set_bytes'],'peak working set bounds')
        require(row['after']['peak_working_set_bytes']>=row['before']['peak_working_set_bytes'],'process peak cannot decrease')
        require(math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0,'benchmark gradient')
    for loss in ('prefix','final_only'):
        rr=[r for r in brows if r['loss']==loss]
        require(max(r['loss_value'] for r in rr)-min(r['loss_value'] for r in rr)<=1e-6,'BPTT/checkpoint forward losses differ')
    # Compare independent recomputation to the worker's completed summary.
    summary=read(bound('results/recomputed.json'))
    lookup={tuple(t['condition'])+(t['length'],t['final_only']):t for t in tables if t['family']!='neutral'}
    for field,is_final in [('primary_recomputed',False),('final_only_recomputed',True)]:
        require(len(summary[field])==(126 if not is_final else 12),'summary row count')
        for row in summary[field]:
            ours=lookup[key(row['condition'])+(row['length'],is_final)]
            require(row['accuracy']==ours['final_accuracy'],'summary accuracy')
            if not is_final:require(abs(row['true_state_changing_operation_count']-ours['true_state_changes_mean'])<1e-12,'summary true state changes')
    # The owner intentionally excludes Markdown from its machine-evidence binding.
    # Capture this completed report separately without inventing a binding contract.
    receipt('report.md');bound('results/recomputed.json')
    # Freeze only the audited finished subset; refuse a concurrent byte change.
    for relative,digest in checked.items():
        require(sha(CORE/relative)==digest,'artifact changed during audit: '+relative)
        if relative not in ('report.md','evidence-binding.json'):
            require(binding['files'].get(relative)==digest,'audited receipt missing from final binding: '+relative)
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(__file__),
            'conditions':{'primary':21,'final_only':6,'diagnostics':9,'benchmark':18},
            'prediction_arrays_recomputed':len(tables)+36,'paired_initial_tensor_checks':len(tensor_pairs),
            'source_and_receipt_files_checked':len(checked),'checked_files':checked,'prediction_rows':tables,
            'influence_arithmetic':diagnostic_checks,'paired_initializations':tensor_pairs,
            'limitations':['No model training or tuning performed; recorded optimizer execution is not externally attested.',
              'Saved initial tensors and seed-defined input streams verified; every training batch was not saved.',
              'Cross entropy/validation CE not independently recomputed because raw logits were not saved.',
              'Diagnostic autograd/finite-difference values and checkpoint gradient/update differences checked from receipts; no new gradient run.',
              'Influence uses full untruncated forward maps of TBPTT-trained weights; not delivered training-time truncated gradients.',
              'TBPTT cuts hidden/decoder KV only; encoder/global-memory gradients remain.',
              'Primary prefix and delayed-final supervision use different predefined streams and unequal label budgets.',
              'Benchmark peak working set is whole process including interpreter/model/optimizer/native libraries, not activation-only memory.',
              'CPU timings include external contention and cannot establish isolated throughput or energy.',
              'Hash bindings and chronology are internal evidence consistency, not trusted external signatures.']}
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    report_lines=[
        '# Independent core evidence audit', '', '**Status: PASS within the stated evidence scope.**', '',
        f"Completed {len(training)} primary conditions, {len(companions)} final-only conditions, {len(drows)} diagnostics and {len(brows)} benchmark rows.",
        f"Independently reconstructed {result['prediction_arrays_recomputed']} raw prediction arrays and checked {len(tensor_pairs)} actual initial-tensor pairings.", '',
        'Parity and five-state targets were reconstructed without importing the experimental generator or model. Exact seed-defined inputs, prediction accuracy, true state changes, neutral suffix semantics, condition completeness, validation checkpoint selection, execution chronology, source preparation pins, snapshots and final evidence-binding hashes passed.', '',
        '## Delayed-final supervision: observed accuracy', '',
        '| Seed | Training mode | Length | Final accuracy |', '|---|---|---:|---:|']
    for row in tables:
        if row['final_only']:
            report_lines.append(f"| {row['condition'][3]} | {row['condition'][2]} | {row['length']} | {row['final_accuracy']:.8f} |")
    report_lines+=['', '## Benchmark process memory', '',
        '| Loss | Mode | Mean peak working set after update (bytes) |', '|---|---|---:|']
    for loss in ('prefix','final_only'):
        for mode in ('full','tbptt8','checkpoint'):
            rows=[r for r in brows if r['loss']==loss and r['mode']==mode]
            report_lines.append(f"| {loss} | {mode} | {sum(r['after']['peak_working_set_bytes'] for r in rows)/len(rows):.2f} |")
    report_lines+=['', '## Limits', '']+['- '+s for s in result['limitations']]
    report_lines+=['', f"Audit source SHA256: `{result['audit_source_sha256']}`.",
                   f"Full machine-readable evidence: [core-independent-audit.json]({OUT.as_posix()}).", '']
    with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(report_lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('checked_files','prediction_rows','influence_arithmetic','paired_initializations','limitations')}))
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-ready',action='store_true');args=parser.parse_args()
    if args.check_ready:
        ok,detail=ready();print(json.dumps({'status':'READY' if ok else 'NOT_READY',**detail}));return 0 if ok else 2
    return run_audit()

if __name__=='__main__':raise SystemExit(main())
