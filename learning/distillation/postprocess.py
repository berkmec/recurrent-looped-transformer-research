"""Independent replay, dependency audit, archive-off export and scoped report."""
from pathlib import Path
import hashlib,json,subprocess,sys,shutil,time
import numpy as np
import torch
from learning.api import load_model
from .trace import trace_forward,certify,observed_records,component_mask
from .run import verify,sha,newjson,emit

HERE=Path(__file__).parent;ROOT=HERE.parent.parent


def dependencies():
    path=HERE/'dependency-receipt.json'
    if sha(path)!=(HERE/'dependency-receipt.sha256').read_text().strip():raise ValueError('dependency receipt drift')
    receipt=json.loads(path.read_text())
    if sha(HERE/'protocol.json')!=receipt['distillation_protocol_sha256'] or sha(ROOT/'learning'/'protocol.json')!=receipt['primary_protocol_sha256']:raise ValueError('protocol dependency drift')
    for name,digest in receipt['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('dependency drift: '+name)
    return receipt


def audit_archives():
    torch.set_num_threads(1);p=verify();dependencies();results=[]
    for seed in p['seeds']:
        receipt=json.loads((HERE/'archives'/f'receipt-{seed}.json').read_text());path=HERE/'archives'/f'teacher-{seed}.npz'
        if sha(path)!=receipt['archive_sha256']:raise AssertionError('trace altered')
        if sha(HERE/'archives'/f'selection-{seed}.npz')!=receipt['selection_sha256'] or sha(HERE/'archives'/f'decisions-{seed}.jsonl')!=receipt['decisions_sha256']:raise AssertionError('selection/decision altered')
        with np.load(path) as source:raw={key:source[key] for key in source.files}
        model=load_model(HERE.parent/'checkpoints'/f'combined-{seed}-step600.pt');maximum=0.;decisions=0
        with torch.no_grad():
            for start in range(0,len(raw['events']),64):
                regenerated=trace_forward(model,torch.from_numpy(raw['events'][start:start+64]))
                for key in ('states','nodes','final'):
                    error=float(abs(regenerated[key].numpy()-raw[key][start:start+64]).max());maximum=max(maximum,error)
        if maximum!=0:raise AssertionError('stored trace differs from actual teacher replay')
        groups={}
        for i,tape in enumerate(raw['events']):
            qs=np.where(tape[:,0]==2)[0];first_records=observed_records(tape[:qs[0]+1]);query=str(int(tape[qs[0],1]));mask=component_mask(first_records,query)
            depth=int(mask.sum()-1);inverse=query not in {e['child'] for e in first_records if e['parent'] is not None}
            update=int(tape[np.where(tape[:,0]==3)[0][0],1]);affected=bool(mask[update])
            key=f'depth{depth}-inverse{int(inverse)}-affected{int(affected)}';g=groups.setdefault(key,{'episodes':0,'eligible_episodes':0,'certified_queries':0});g['episodes']+=1;g['eligible_episodes']+=int(raw['accepted'][i].any());g['certified_queries']+=int(raw['accepted'][i].sum())
            for q,t in enumerate(qs):
                records=observed_records(tape[:t+1]);requested=str(int(tape[t,1]));candidate={str(n):float(v) for n,v in enumerate(raw['nodes'][i,q,-1])}
                accepted=certify(candidate,records,requested,p['tolerance'])['accepted']
                if bool(raw['accepted'][i,q])!=accepted:raise AssertionError('certificate differs')
                component=component_mask(records,requested)
                np.testing.assert_array_equal(component,raw['component_mask'][i,q]);decisions+=1
                others=np.where(component==0)[0]
                other_ok=certify(candidate,records,str(int(others[0])),p['tolerance'])['accepted'] if len(others) else accepted
                if bool(raw['full_verified'][i,q])!=(accepted and other_ok):raise AssertionError('whole-episode certificate differs')
                qid=int(tape[t,1])
                if raw['final'][i,q]!=raw['nodes'][i,q,-1,qid]:raise AssertionError('final target is not teacher readout')
        fully=raw['full_verified'].all(1);strata=np.arange(len(fully))%8
        counts=[int((fully&(strata==j)).sum()) for j in range(8)];use_full=int(fully.sum())>=128 and min(counts)>=16
        expected_q=np.repeat(fully[:,None],3,axis=1) if use_full else raw['accepted']
        expected_n=np.repeat(expected_q[:,:,None],10,axis=2).astype(np.float32) if use_full else raw['component_mask']*expected_q[:,:,None]
        selected=np.load(HERE/'archives'/f'selection-{seed}.npz')
        np.testing.assert_array_equal(selected['query_mask'],expected_q.astype(np.float32));np.testing.assert_array_equal(selected['node_mask'],expected_n)
        np.testing.assert_array_equal(selected['eligible'],np.where(expected_q.any(1))[0])
        results.append({'seed':seed,'teacher_trace_max_replay_difference':maximum,'certificate_decisions_rechecked':decisions,'selection_derivation_rechecked':True,'groups':groups,'archive_sha256':sha(path)})
    output={'replay':'all stored actual latent/numeric states regenerated from pinned teachers and observation-only tapes','rows':results}
    newjson(HERE/'results'/'archive-replay-audit-v2.json',output);print(json.dumps({'teacher_replay_max_difference':max(r['teacher_trace_max_replay_difference'] for r in results),'decisions':sum(r['certificate_decisions_rechecked'] for r in results),'selection_masks':'RECOMPUTED_MATCH'}))


def archive_off():
    root=HERE/'inference-only-export'
    if root.exists():raise FileExistsError('archive-off export exists')
    package=root/'learning';package.mkdir(parents=True)
    for name in ('__init__.py','model.py','token_core.py','api.py'):shutil.copy2(HERE.parent/name,package/name)
    models=[]
    for path in sorted((HERE/'checkpoints').glob('*.pt')):shutil.copy2(path,root/path.name);models.append(path.name)
    test=np.load(HERE/'results'/'data-iid.npz');np.save(root/'events.npy',test['events'])
    site=str(Path(np.__file__).resolve().parent.parent)
    script=f'''from pathlib import Path
import sys,importlib.util,json
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.append({site!r})
import torch,numpy as np
from learning.api import load_model
torch.set_num_threads(1)
assert importlib.util.find_spec('learning.data') is None
assert importlib.util.find_spec('learning.distillation') is None
x=np.load('events.npy');outputs={{}}
for name in {models!r}:
    model=load_model(name);rows=[]
    with torch.no_grad():
        for start in range(0,len(x),64):rows.append(model(torch.from_numpy(x[start:start+64]))['predictions'].numpy())
    outputs[name]=np.concatenate(rows)
np.savez_compressed('predictions.npz',**outputs)
Path('receipt.json').write_text(json.dumps({{'isolated_import_path':bool(sys.flags.isolated),'oracle_importable':False,'trace_package_importable':False,'models':{models!r},'examples_per_model':len(x)}}))
'''
    (root/'infer.py').write_text(script);process=subprocess.run([sys.executable,'-I',str(root/'infer.py')],cwd=root,text=True,capture_output=True)
    (root/'stdout.txt').write_text(process.stdout);(root/'stderr.txt').write_text(process.stderr)
    if process.returncode:raise RuntimeError(process.stderr)
    predictions=np.load(root/'predictions.npz');maximum=0.
    for name in models:
        arm,seed=name.removesuffix('-step300.pt').rsplit('-',1)
        saved=np.load(HERE/'results'/f'pred-{arm}-{seed}-iid.npz')['predictions'];error=float(abs(predictions[name]-saved).max());maximum=max(maximum,error)
    if maximum!=0:raise AssertionError('archive-off student predictions differ')
    receipt=json.loads((root/'receipt.json').read_text());receipt.update(max_prediction_difference=maximum,predictions_sha256=sha(root/'predictions.npz'),boundary='isolated inference export/import path, not OS filesystem sandbox; no archive or targets exported')
    newjson(HERE/'results'/'archive-off-receipt.json',receipt)


def validate_expected_results(metrics):
    expected_runs={(arm,seed) for arm in ('answer_only','trajectory') for seed in (17,43,91)}
    expected_rows={(arm,seed,split) for arm,seed in expected_runs for split in ('iid','depth8')}
    if len(metrics['runs'])!=6 or {(r['arm'],r['seed']) for r in metrics['runs']}!=expected_runs:raise ValueError('missing/duplicate student runs')
    if len(metrics['rows'])!=12 or {(r['arm'],r['seed'],r['split']) for r in metrics['rows']}!=expected_rows:raise ValueError('missing/duplicate evaluation rows')
    if len(metrics['comparisons'])!=2 or {r['split'] for r in metrics['comparisons']}!={'iid','depth8'}:raise ValueError('missing/duplicate comparisons')
    if any(len(r['per_seed_differences'])!=3 or not np.isfinite(r['per_seed_differences']).all() for r in metrics['comparisons']):raise ValueError('invalid paired seed differences')


def finish():
    verify();dependencies();metrics=json.loads((HERE/'results'/'metrics.json').read_text());rows=metrics['rows'];from learning.metrics import summarize
    validate_expected_results(metrics)
    for row in rows:
        path=Path(row['raw_path'])
        if sha(path)!=row['raw_sha256']:raise AssertionError('raw prediction drift')
        raw=np.load(path);computed=summarize(raw['predictions'],raw['targets'],raw['affected'])
        for key,value in computed.items():
            if value!=row['metrics'][key]:raise AssertionError('metric mismatch')
    for seed in (17,43,91):
        initial=[r['initial_tensor_sha256'] for r in metrics['runs'] if r['seed']==seed]
        if len(initial)!=2 or len(set(initial))!=1:raise AssertionError('unpaired student starts')
    comparisons=metrics['comparisons'];status='MEASURED_POSITIVE' if all(c['seed_t95_df2'][1]<0 for c in comparisons) else 'MEASURED_NEGATIVE' if all(c['seed_t95_df2'][0]>0 for c in comparisons) else 'MIXED'
    archive=json.loads((HERE/'results'/'archive-replay-audit-v2.json').read_text());off=json.loads((HERE/'results'/'archive-off-receipt.json').read_text())
    expected_exports={f'{arm}-{seed}-step300.pt' for arm in ('answer_only','trajectory') for seed in (17,43,91)}
    if len(off['models'])!=6 or set(off['models'])!=expected_exports or off['max_prediction_difference']!=0:raise ValueError('incomplete archive-off parity')
    lines=['# Actual successful teacher traces to later student weights','',f'Path implementation/evaluation: **IMPLEMENTED_TESTED**. Paired transfer result: **{status}**.',
      'This separate experiment uses actual teacher-generated refinement states/numeric candidates stored before student training. It does not reuse oracle intermediate labels as distillation targets. Teacher combined600 checkpoints themselves originate from the earlier supervised experiment; this is not unsupervised emergence or universal thinking-style learning.',
      '',f'Protocol SHA256 `{sha(HERE/"protocol.json")}`. Three seeds17/43/91, two student arms,300 updates each; both start from the same seed-matched feedback600 weights and receive identical actual teacher final targets, example batches and query masks. The trajectory arm additionally minimizes0.1 times hidden-state MSE plus numeric trajectory MSE on certified components.',
      '', '## Collection, successful-final criterion and selection bias','',
      'Each fixed combined600 teacher generated actual six-round traces at all three query states on2048 TRAIN-only tapes. A0.05 residual-sum tolerance was frozen before collection. No full ten-node, three-query episode met that certificate. The predeclared queried-component fallback was used in every seed. Certification concerns the final candidate only; intermediate latent/numeric steps are preserved actual states and are not mathematical proofs.',
      '', '| Seed | Eligible episodes /2048 | Certified query fraction | Certified node fraction |','|---|---:|---:|---:|']
    for seed in (17,43,91):
        receipt=json.loads((HERE/'archives'/f'receipt-{seed}.json').read_text());lines.append(f'| {seed} | {receipt["eligible_episodes"]} | {receipt["query_coverage"]:.5f} | {receipt["certified_node_coverage"]:.5f} |')
    lines+=['','All rejected traces and decisions remain in immutable-by-writer archives. No target is solved, repaired, substituted or widened after inspection. Filtering is strongly biased toward shallow successful components; some deep/direction/affected cells contain only1–3 eligible episodes. This is a material external-validity limit, not balanced proof of a general method.',
       '', '| Seed | Observed depth | Inverse | Affected | Eligible / total | Certified queries / all queries |','|---|---:|---:|---:|---:|---:|']
    for row in archive['rows']:
        for key,g in sorted(row['groups'].items()):
            depth,inverse,affected=key.split('-');lines.append(f'| {row["seed"]} | {depth[5:]} | {inverse[7:]} | {affected[8:]} | {g["eligible_episodes"]}/{g["episodes"]} | {g["certified_queries"]}/{g["episodes"]*3} |')
    lines+=['','## Fresh held-out outcomes','', '| Split | Final-answer-only MSE | +Actual trajectory MSE | Final-answer-only accuracy0.1 | +Actual trajectory accuracy0.1 |','|---|---:|---:|---:|---:|']
    def avg(arm,split,key):return float(np.mean([r['metrics'][key] for r in rows if r['arm']==arm and r['split']==split]))
    for split in ('iid','depth8'):lines.append(f'| {split} | {avg("answer_only",split,"final_mse"):.6f} | {avg("trajectory",split,"final_mse"):.6f} | {avg("answer_only",split,"accuracy_0.1"):.4f} | {avg("trajectory",split,"accuracy_0.1"):.4f} |')
    lines+=['', '| Seed | Student arm | IID final MSE | Depth8 final MSE |','|---|---|---:|---:|']
    for seed in (17,43,91):
        for arm in ('answer_only','trajectory'):
            values={r['split']:r['metrics']['final_mse'] for r in rows if r['seed']==seed and r['arm']==arm}
            lines.append(f'| {seed} | {arm} | {values["iid"]:.6f} | {values["depth8"]:.6f} |')
    lines+=['', 'The seed-t intervals are the primary uncertainty summary. Hierarchical bootstrap intervals are exploratory with only three independently trained seeds; they do not override the mixed primary conclusion.']
    for comparison in comparisons:lines.append(f'- {comparison["split"]}: trajectory-minus-final-only MSE {comparison["mean_difference"]:.6f}; per-seed {comparison["per_seed_differences"]}; seed-t95 {comparison["seed_t95_df2"]}; hierarchical bootstrap95 {comparison["hierarchical_bootstrap95"]}.')
    lines+=['','New512-example IID and512-example true-depth8 sets were first generated only after all student checkpoints were fixed. Common example/seed clustering is preserved. With three seeds, intervals remain limited. A result can show no benefit or harm while still demonstrating the actual trace-to-later-weights path.',
      '', '## Provenance, archive-off inference and costs','',
      f'- All actual teacher latent/numeric traces replayed exactly: maximum difference0 across all6144 collected episodes. All18432 final-query acceptance decisions rechecked from observations. Candidate readouts match archived teacher final targets exactly.',
      f'- Independently recomputed {len(rows)} raw held-out prediction files and verified hashes. Each paired student starts with identical weights.',
      f'- Isolated inference-only student export reproduced {len(off["models"])*off["examples_per_model"]} example prediction vectors with maximum difference {off["max_prediction_difference"]}; no generator or distillation/trace package was importable and no archive/targets were exported. This is not an OS filesystem sandbox.',
      '- Frozen source, teacher/student checkpoint pins, train receipt, and additional evaluation dependency sidecar checked before training and after evaluation. No frozen primary/combined source was changed.',
      f'- Student training/evaluation wall time: {metrics["wall_seconds"]:.2f}s on CPU one thread. Teacher collection receipts record separate per-seed costs. Energy is UNMEASURED.',
      '- Both arms compute the same forward trajectories. Auxiliary backward supervision adds compute; equal examples/updates do not imply equal effective cost.',
      '- The initially reported interop-import concern was independently disproved by live source and a post-work import reproduction; the reviewer retracted it. No scientific protocol amendment or source mutation was made.',
      '', 'Six contract tests cover false-candidate no-repair, TRAIN/hash-only loading, immutable archive writes, actual Runtime trace parity, exclusion of uncertified nodes from auxiliary loss and rejection of incomplete/empty evaluation before classification. Selection query/node masks and eligible indices were independently recomputed from the frozen full/fallback rule. Independent root review remains separate from worker recomputation.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    newjson(HERE/'results'/'closure.json',{'path_status':'IMPLEMENTED_TESTED','effect_status':status,'raw_files_recomputed':len(rows),'teacher_replay_max_difference':0.,'paired_initialization':True,'archive_off':off,'root_independent_audit':'SEPARATE'})
    newjson(HERE/'postprocessing-source-hashes.json',{p.name:sha(p) for p in HERE.glob('*.py')});print(json.dumps({'status':status,'report':str(HERE/'report.md'),'comparisons':comparisons}))

if __name__=='__main__':
    if '--audit-archives' in sys.argv:audit_archives()
    elif '--archive-off' in sys.argv:archive_off()
    elif '--finish' in sys.argv:finish()
    else:dependencies();print('DEPENDENCY_PIN_PASS')
