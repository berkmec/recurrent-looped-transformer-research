"""Final evidence recomputation and report extension; does not train/select models."""
from pathlib import Path
import json,hashlib
import numpy as np
from .metrics import summarize
from .analyze import main as primary_report
from .safe_run import assert_protocol_integrity

HERE=Path(__file__).parent


def main():
    assert_protocol_integrity(HERE)
    primary_report()
    primary=json.loads((HERE/'results'/'metrics.json').read_text())
    extra=json.loads((HERE/'results'/'structural-extra-metrics.json').read_text())
    combined=json.loads((HERE/'results'/'combined-metrics.json').read_text())
    analysis=json.loads((HERE/'results'/'paired-analysis.json').read_text())
    checked=len(primary['rows'])
    for row in extra['rows']+combined['rows']:
        path=Path(row['raw_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['raw_sha256']:raise AssertionError('raw hash mismatch')
        raw=np.load(path);metrics=summarize(raw['predictions'],raw['targets'],raw['affected'])
        for name,value in metrics.items():
            if abs(value-row['metrics'][name])>1e-10:raise AssertionError('metric mismatch')
        checked+=1
    for seed in (17,43,91):
        hashes=[r['initial_tensor_sha256'] for r in primary['runs'] if r['seed']==seed]+[r['initial_tensor_sha256'] for r in combined['runs'] if r['seed']==seed]
        if len(hashes)!=7 or len(set(hashes))!=1:raise AssertionError('unpaired initial weights')
    sources=json.loads((HERE/'source-hashes.json').read_text())
    for name,digest in sources.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise AssertionError('primary source changed')
    for filename,protocol in [('structural_extra.py',extra['protocol']),('combined_run.py',combined['protocol'])]:
        if hashlib.sha256((HERE/filename).read_bytes()).hexdigest()!=protocol['source_sha256']:raise AssertionError('supplemental source changed')
    events=[json.loads(line) for line in (HERE/'execution.jsonl').read_text().splitlines()]
    first_test=min(e['utc'] for e in events if e['event']=='test_first_open')
    for label in ('structural_extra_protocol_frozen','combined_protocol_frozen'):
        if not any(e['event']==label and e['utc']<first_test for e in events):raise AssertionError('late supplemental freeze')
    def avg(rows,arm,metric='final_mse',split=None):
        return float(np.mean([r['metrics'][metric] for r in rows if r['arm']==arm and (split is None or r['split']==split)]))
    transfer=next(c for c in extra['comparisons'] if c['candidate']=='method')
    graph_extra=next(c for c in extra['comparisons'] if c['candidate']=='bidir')
    status=analysis['package_status'].copy()
    if transfer['seed_t95_df2'][1]>=0:status['13']='MIXED'
    if graph_extra['mean_difference']>=0:status['6']='MIXED'
    report=HERE.parent/'reports'/'learning-report.md';text=report.read_text(encoding='utf8')
    base_time=float(np.mean([r['wall_seconds'] for r in primary['runs'] if r['arm']=='bidir']))
    method_time=float(np.mean([r['wall_seconds'] for r in primary['runs'] if r['arm']=='method']))
    text=text.replace('baseline {base_time:.2f}s, method {method_time:.2f}s, ratio {method_time/base_time:.3f}',f'baseline {base_time:.2f}s, method {method_time:.2f}s, ratio {method_time/base_time:.3f}')
    text=text.replace('Novel graph IDs, values and depth5 structure are tested.','Novel graph IDs and values, plus a held-out query-depth5 shift, are tested; genuine unseen structural depth is evaluated separately below.')
    text=text.replace('All comparisons are on512 IID and512 structural-shift examples.','Primary comparisons are on512 IID and512 query-depth-shift examples.')
    text=text.replace('Primary runs:18 learned models','Full evaluated-scope classification: **packages6/7/12/13 are MIXED**. There are21 corrected trained models including the combined follow-up. Positive primary graph results, positive depth8 method-transfer results, failed generalization and uncertainty are separated below.\n\nPrimary runs:18 learned models')
    text=text.replace('Eight test contracts passed before primary training','Nine test contracts passed before primary training; sixteen test cases passed in the final expanded suite')
    text=text.replace('Independent root audit is required before scientific closure.','Independent root audit is required before scientific closure; this worker recomputation is not substituted for that independent audit.')
    text+='\n## Supplemental scope correction and genuinely unseen depth8\n\n'
    text+='The main depth5 evaluation is held-out query depth only. TRAIN method targets include the other chain up to depth7. A separately frozen protocol, created before any primary test opening, evaluates512 new depth8 examples per arm/seed. Every dependency path in this supplemental query exceeds all supervised train depths. The model, training and six-round-per-query budget are unchanged.\n\n'
    text+='| Arm | Depth8 final MSE | Depth8 error<=0.1 |\n|---|---:|---:|\n'
    for arm in ('forward','bidir','wrong','flat','feedback','method'):
        text+=f'| {arm} | {avg(extra["rows"],arm):.6f} | {avg(extra["rows"],arm,"accuracy_0.1"):.4f} |\n'
    text+=f'\nMethod-minus-final-only depth8 MSE: {transfer["mean_difference"]:.6f}; per seed {transfer["per_seed_differences"]}; seed-t95 {transfer["seed_t95_df2"]}. Final package13 classification including the genuine structural shift: **{status["13"]}**.\n'
    text+=f'\nBidirectional-minus-forward depth8 MSE: {graph_extra["mean_difference"]:.6f}; per seed {graph_extra["per_seed_differences"]}; seed-t95 {graph_extra["seed_t95_df2"]}. The primary inverse-task benefit does not establish uniform deep generalization. Package6 classification over the full evaluated scope: **{status["6"]}**.\n'
    text+='\n## Combined feedback and method supervision: complete2×2\n\n'
    text+='Three additional models were predeclared before test opening: feedback enabled plus0.3 intermediate supervision, same initialization/batches/600updates. This gives21 corrected trained models overall. Combined seed17 is the predeclared root full-system integration artifact; no test-best selection is used. All three combined checkpoints were fixed before this follow-up opened test arrays.\n\n'
    text+='| Split | Combined final MSE | Combined error<=0.1 |\n|---|---:|---:|\n'
    for split in ('iid','shift','depth8'):
        text+=f'| {split} | {avg(combined["rows"],"combined",split=split):.6f} | {avg(combined["rows"],"combined","accuracy_0.1",split):.4f} |\n'
    text+='\nThe interaction contrast on final squared error is combined−method−feedback+bidir. Negative values indicate an additional reduction beyond the sum of isolated error differences; with three seeds intervals remain uncertain.\n'
    for c in combined['comparisons']:
        if 'contrast' in c:text+=f'- {c["split"]}: interaction {c["mean_difference"]:.8f}, per-seed {c["per_seed_differences"]}, seed-t95 {c["seed_t95_df2"]}.\n'
    text+='\n## Effective parameters, full trace verification and evidence preservation\n\n'
    audit=json.loads((HERE/'results'/'parameter-audit.json').read_text())
    text+='All primary arms allocate29475 parameters. They are **not fully matched in active parameters or effective compute**. Observed nonzero-gradient entries on the same first training batch are:\n\n'
    for row in audit['rows']:text+=f'- {row["arm"]}: {row["nonzero_gradient_parameters"]}; gradient-present allocation {row["gradient_present_parameters"]}.\n'
    text+='\nThe forward arm masks reverse inputs; feedback enables an extra1024-parameter projection; auxiliary training adds backward work. These are the mechanisms being isolated, and equal allocation alone does not make their costs equal.\n'
    trace=json.loads((HERE/'results'/'trace-full-independent-verification.json').read_text())
    text+=f'\nEvery TRAIN trace was independently checked: {trace["examples"]} examples, {trace["query_states"]} query states and {trace["node_targets_checked"]} node targets. Maximum linear-solve discrepancy {trace["max_absolute_error"]:.3g}, tolerance1e-6. The verifier imports no generator oracle. A corruption regression verifies rejection.\n'
    archive=json.loads((HERE/'results'/'archive-off-receipt.json').read_text())
    text+=f'\nArchive-off export check: a separate inference-only package, containing model/API/checkpoint and observation inputs only, ran512 IID examples in isolated Python import mode. Neither learning.data nor the training runner was importable. Predictions matched the primary method checkpoint outputs exactly (max difference {archive["max_prediction_difference"]}). This is an isolated import-path/export check, not an OS filesystem sandbox. The first export attempt could not import Torch because it lives in the user dependency site; that failed sandbox is preserved, and the corrected attempt adds only the dependency site, not the research source path.\n'
    text+='\nA second failed attempt, v2, had the corrected recurrent core but omitted activation of unobserved parents mentioned only by an edge. It was stopped before test access. Three completed600-step diagnostic models and one partial arm remain in diagnostic-v2, alongside all sources/data/protocols. v3 fixed this with a regression and fresh test seeds. Both the v1 and v2 failures are retained, not counted as successful primary models.\n'
    text+='\nThe original frozen runner could overwrite evidence on restart. It is preserved as historical execution source; use only the tested successor safe_run.py, which fails closed on nonempty output/checkpoint destinations. The combined runner also rejects existing combined artifacts. Never restart the original runner against this evidence directory.\n'
    text+='\nExternal protocol integrity guards also verify protocol bytes against the SHA256 sidecar, source-hash index against the original frozen execution receipt, actual source hashes, and combined-to-primary protocol pins. Protocol/model-plus-hash-index tampering regressions fail closed. The frozen primary and combined training sources were not edited.\n'
    resources=[json.loads(s) for s in (HERE/'resource-samples.jsonl').read_text().splitlines()]
    peak=max(r['peak_working_set_bytes'] for r in resources)
    text+=f'\nMaximum OS-reported primary-process peak working set observed by5-second sampling: {peak/1048576:.2f}MiB. This is not total machine RAM or energy. Primary training/evaluation wall {primary["total_wall_seconds"]:.1f}s; combined follow-up wall {combined["total_wall_seconds"]:.1f}s. Supplemental evaluation has per-run wall times in its metrics.\n'
    text+=f'\nFinal worker audit recomputed **{checked}** held-out raw prediction files, verified their hashes, confirmed all seven arms share initialization tensors within each seed, checked unchanged frozen source hashes, and verified supplemental protocols predate first test access. Root independent audit and connected extraction/memory measurements remain separate.\n'
    report.write_text(text,encoding='utf8')
    closure={'recomputed_raw_files':checked,'matched_initialization_arms_per_seed':7,'frozen_sources_unchanged':True,'supplement_protocols_before_test':True,'package_status':status,
             'primary_corrected_models':18,'combined_models':3,'failed_attempts_preserved':['diagnostic-v1','diagnostic-v2'],'independent_root_audit':'REQUIRED_SEPARATELY',
             'peak_primary_working_set_bytes_sampled':peak,'energy':'UNMEASURED','report':str(report)}
    (HERE/'results'/'closure.json').write_text(json.dumps(closure,indent=2))
    final_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (HERE/'postprocessing-source-hashes.json').write_text(json.dumps(final_sources,indent=2));print(json.dumps(closure))

if __name__=='__main__':main()
