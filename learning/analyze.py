"""Recompute metrics, paired uncertainty and scoped report from frozen raw outputs."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from .metrics import summarize,revision_gate

HERE=Path(__file__).parent

def uncertainty(differences,seed=12345):
    # Rows are seeds, columns are the SAME held-out examples: common resampling.
    a=np.asarray(differences,dtype=float);means=a.mean(1);mean=float(means.mean())
    se=float(means.std(ddof=1)/np.sqrt(len(means)));rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(1000):
        s=rng.integers(0,len(a),len(a));j=rng.integers(0,a.shape[1],a.shape[1]);boot.append(float(a[s][:,j].mean()))
    return {'mean_difference':mean,'per_seed_differences':means.tolist(),
            'seed_t95_df2':[mean-4.30265273*se,mean+4.30265273*se],
            'hierarchical_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_replicates':1000,
            'unit':'squared final prediction error candidate minus baseline','warning':'Only3 independent training seeds; percentile interval exploratory.'}


def main():
    source=HERE/'results'/'metrics.json';payload=json.loads(source.read_text());rows=payload['rows'];checked=0;lookup={}
    for row in rows:
        path=Path(row['raw_path']);actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['raw_sha256']:raise AssertionError('raw hash mismatch')
        raw=np.load(path);calc=summarize(raw['predictions'],raw['targets'],raw['affected'])
        for key,value in calc.items():
            if abs(value-row['metrics'][key])>1e-10:raise AssertionError('recomputation mismatch')
        lookup[(row['arm'],row['seed'],row['step'],row['split'],row['condition'])]=raw;checked+=1
    comparisons=[]
    plans=[('bidir','forward','retain','retain'),('bidir','wrong','retain','retain'),('bidir','flat','retain','retain'),
           ('bidir','bidir','local','retain'),('bidir','bidir','all','retain'),
           ('feedback','bidir','retain','retain'),('method','bidir','retain','retain')]
    for split in ('iid','shift'):
        for candidate,baseline,condition,base_condition in plans:
            for subset in ('all','inverse','forward'):
                diffs=[]
                for seed in (17,43,91):
                    a=lookup[(candidate,seed,600,split,condition)];b=lookup[(baseline,seed,600,split,base_condition)]
                    mask=np.ones(len(a['targets']),bool) if subset=='all' else a['inverse'] if subset=='inverse' else ~a['inverse']
                    ae=(a['predictions'][:,-1]-a['targets'][:,-1])**2;be=(b['predictions'][:,-1]-b['targets'][:,-1])**2
                    diffs.append((ae-be)[mask])
                comparisons.append({'candidate':candidate,'baseline':baseline,'condition':condition,'baseline_condition':base_condition,'split':split,'subset':subset,**uncertainty(diffs)})
    gates=[];interventions=[]
    for split in ('iid','shift'):
        for seed in (17,43,91):
            base=lookup[('bidir',seed,600,split,'retain')]
            for condition in ('all','local'):
                candidate=lookup[('bidir',seed,600,split,condition)]
                gates.append({'split':split,'seed':seed,'condition':condition,**revision_gate(base['predictions'],candidate['predictions'],base['targets'],base['affected'])})
            base=lookup[('feedback',seed,600,split,'retain')]
            for condition in ('zero','permute'):
                change=lookup[('feedback',seed,600,split,condition)]['predictions']-base['predictions']
                interventions.append({'split':split,'seed':seed,'condition':condition,'prefix_max_change':float(abs(change[:,0]).max()),
                                      'future_mean_absolute_change':float(abs(change[:,1:]).mean()),'future_max_change':float(abs(change[:,1:]).max()),
                                      'future_changed_examples':int((abs(change[:,1:]).max(1)>0).sum())})
    analysis={'raw_files_recomputed':checked,'paired_comparisons':comparisons,'revision_gates':gates,'feedback_interventions':interventions}
    (HERE/'results'/'paired-analysis.json').write_text(json.dumps(analysis,indent=2))
    def avg(arm,split,condition='retain',metric='final_mse',step=600):
        return float(np.mean([r['metrics'][metric] for r in rows if r['arm']==arm and r['step']==step and r['split']==split and r['condition']==condition]))
    def comp(arm,base,split,subset='all',condition='retain'):
        return next(c for c in comparisons if c['candidate']==arm and c['baseline']==base and c['condition']==condition and c['split']==split and c['subset']==subset)
    local_pass=sum(g['pass'] for g in gates if g['condition']=='local')
    feedback_causal=all(i['prefix_max_change']==0 and i['future_max_change']>0 for i in interventions)
    graph_positive=all(comp('bidir','forward',s,'inverse')['seed_t95_df2'][1]<0 for s in ('iid','shift'))
    method_positive=all(comp('method','bidir',s)['seed_t95_df2'][1]<0 for s in ('iid','shift'))
    status={'6':'MEASURED_POSITIVE' if graph_positive else 'MIXED','7':'MEASURED_POSITIVE' if local_pass==6 else 'MEASURED_NEGATIVE' if local_pass==0 else 'MIXED',
            '12':'MIXED' if feedback_causal else 'INCONCLUSIVE','13':'MEASURED_POSITIVE' if method_positive else 'MIXED'}
    analysis['package_status']=status;(HERE/'results'/'paired-analysis.json').write_text(json.dumps(analysis,indent=2))
    lines=['# Learning experiments6,7,12,13','',f'Protocol SHA256: `{payload["protocol_sha256"]}`.',
           f'Primary runs:18 learned models, seeds17/43/91,600 updates each,9600 examples seen/run, CPU one thread. Training/evaluation wall time {payload["total_wall_seconds"]:.1f}s.',
           'All comparisons are on512 IID and512 structural-shift examples. These share example identities across seeds/arms; uncertainty respects paired seeds and common example clusters.',
           '', '## Main final-answer results','', '| Arm | IID final MSE | Depth5 final MSE | IID error<=0.1 | Depth5 error<=0.1 |','|---|---:|---:|---:|---:|']
    for arm in ('forward','bidir','wrong','flat','feedback','method'):
        lines.append(f'| {arm} | {avg(arm,"iid"):.6f} | {avg(arm,"shift"):.6f} | {avg(arm,"iid",metric="accuracy_0.1"):.4f} | {avg(arm,"shift",metric="accuracy_0.1"):.4f} |')
    lines+=['','##6: Graph direction','',f'Status: **{status["6"]}**. Paired bidirectional-minus-forward inverse-task final-MSE differences:']
    for split in ('iid','shift'):
        c=comp('bidir','forward',split,'inverse');lines.append(f'- {split}: mean {c["mean_difference"]:.6f}; per-seed {c["per_seed_differences"]}; seed-t95 {c["seed_t95_df2"]}.')
    lines+=['','These tasks genuinely observe a downstream absolute value and query an upstream unknown. One-step forward-title propagation cannot read reverse edges. Global causal token attention still sees all observations, so direction superiority is not a proof that the complete forward arm cannot solve inverse tasks. Wrong and flat controls have identical dense title shapes and budgets.','',
            '##7: Revision','',f'Status: **{status["7"]}**. Validation-selected local policy passed affected-improvement plus unaffected-FINAL-error<=0.01 gate in {local_pass}/6 seed/split cells. First and subsequent query budgets remain six in every policy.',
            '', '| Split | Policy | Final MSE | Affected final MSE | Unaffected final MSE | Delta MSE |','|---|---|---:|---:|---:|---:|']
    for split in ('iid','shift'):
        for policy in ('retain','all','local'):
            lines.append(f'| {split} | {policy} | {avg("bidir",split,policy):.6f} | {avg("bidir",split,policy,"affected_final_mse"):.6f} | {avg("bidir",split,policy,"unaffected_final_mse"):.6f} | {avg("bidir",split,policy,"delta_mse"):.6f} |')
    lines+=['','Per-seed policy radii and validation selection are frozen in revision-selection.json; held-out gates are in results/paired-analysis.json. Retain refreshes the directly written title, while retaining other states. Local is a validation-selected graph-radius policy, not an oracle effect mask. No claim extends to edge-offset revisions, which are supported but not in this primary training distribution.','',
            '##12: Title feedback into future recurrent computation','',f'Status: **{status["12"]}**. Future causal intervention contract passed: **{feedback_causal}**. Readout-only and recurrent-feedback models share initialization/batches/budgets. The source-style core uses causal encoder memory, gated recurrent hidden merge, global cross-attention and local KV cache.',
            'Feedback is injected after query readout into the next token hidden merge. Zero and permuted title projections preserve all already-produced first-query predictions exactly. Refinement does not increment token clocks or duplicate KV cache entries. Intervention magnitudes and changed-example counts are reported without equating a tiny change to useful reasoning.']
    for split in ('iid','shift'):
        c=comp('feedback','bidir',split);lines.append(f'- {split} feedback-minus-readout final MSE: {c["mean_difference"]:.6f}; seed-t95 {c["seed_t95_df2"]}.')
    lines+=['','##13: Verified intermediate supervision and archive-off transfer','',f'Status: **{status["13"]}**. Method traces are exact procedural all-node supervision from TRAIN ONLY. Inference imports no dataset/solver/archive. Novel graph IDs, values and depth5 structure are tested. This is not evidence for unrestricted thinking-style learning.',
            '', '| Updates | Baseline IID MSE | Method IID MSE | Baseline depth5 MSE | Method depth5 MSE |','|---:|---:|---:|---:|---:|']
    for step in (150,300,600):lines.append(f'| {step} | {avg("bidir","iid",step=step):.6f} | {avg("method","iid",step=step):.6f} | {avg("bidir","shift",step=step):.6f} | {avg("method","shift",step=step):.6f} |')
    for split in ('iid','shift'):
        c=comp('method','bidir',split);lines.append(f'- {split} method-minus-baseline final MSE {c["mean_difference"]:.6f}; per-seed {c["per_seed_differences"]}; seed-t95 {c["seed_t95_df2"]}.')
    base_time=np.mean([r['wall_seconds'] for r in payload['runs'] if r['arm']=='bidir']);method_time=np.mean([r['wall_seconds'] for r in payload['runs'] if r['arm']=='method'])
    lines += [f'','Observed mean training-plus-validation wall time: baseline {base_time:.2f}s, method {method_time:.2f}s, ratio {method_time/base_time:.3f}. Auxiliary loss weight0.3 adds backward supervision; shared final examples/steps are matched. Timing is CPU wall time, not energy or pure kernel throughput.',
              '', '## Verification, failed attempt and artifacts','',f'- Recomputed {checked} raw prediction files and checked their hashes against metrics.json.',
              '- Eight test contracts passed before primary training: forward direction isolation, future feedback/prefix immutability, token/cache clocks, final-error gate, policy inputs, independent linear-solve labels, inference without oracle imports, validated IDs/query alignment, and causal encoder streaming parity (some contracts share a test).',
              '- RED evidence: red-tests.txt (missing implementation) and red-core-parity.txt (source-core gap). GREEN evidence: green-tests.txt, green-core-parity.txt.',
              '- The first GRU diagnostic was stopped by root audit before its150-step checkpoint. No test set was opened. Protocol/source/initial checkpoint remain in diagnostic-v1; execution.jsonl preserves the attempted run. This failed architecture attempt is not counted among18 corrected primary models.',
              '- Data receipts: results/data-{train,validation,iid,shift}.npz and execution.jsonl. Training traces: verified-method-traces-TRAIN-ONLY.npz plus trace-provenance.json. Checkpoints include initialization and150/300/600 snapshots. source-hashes.json and protocol.sha256 bind the run.',
              '- Public API: learning/api.py and Runtime in model.py. FORMALIZATION.md describes valid event codes, version semantics, incremental refinement/commit contracts and limits.',
              '', '## Remaining limits','',
              'Independent root audit is required before scientific closure. Three-seed uncertainty is limited; hierarchical bootstrap intervals are exploratory. Small synthetic affine forests do not establish large language-model scalability, unrestricted text extraction, arbitrary graph families, measured energy, RL, physical performance, or general thought transfer. Root integration separately measures parser/controller/verifier use. Validation-selected policy usefulness is judged by the reported final-error gate, never by prediction drift alone.']
    (HERE.parent/'reports'/'learning-report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'checked':checked,'status':status,'local_pass':local_pass,'feedback_causal':feedback_causal,'report':str(HERE.parent/'reports'/'learning-report.md')}))

if __name__=='__main__':main()
