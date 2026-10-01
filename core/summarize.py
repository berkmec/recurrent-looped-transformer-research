"""Recompute aggregate tables from saved per-example predictions, without tuning."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(p,n=512):
 z=1.959963984540054;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0.,center-half),min(1.,center+half)]
def actual_state_changes(targets):
 previous=np.concatenate([np.zeros((len(targets),1),dtype=targets.dtype),targets[:,:-1]],axis=1);return targets!=previous
def tensor_hash(path):
 h=hashlib.sha256()
 for n,t in sorted(torch.load(path,weights_only=True).items()):h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
def main():
 metrics=read(ROOT/'results'/'metrics.json');final=read(ROOT/'results'/'final-only-metrics.json');diag=read(ROOT/'results'/'diagnostics.json');assert metrics['status']==final['status']=='COMPLETE';recomputed=[];groups={};lines=['# Core research measured report','',f"Primary protocol SHA256: `{metrics['protocol_hash']}`.",'','Scope: packages1–4 and the model-width portion of14. Independent tiny implementation, not an official accuracy reproduction. Pinned source commit cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39: [official README](https://raw.githubusercontent.com/yifanzhang-pro/recurrent-looped-tranformer/cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39/README.md). All experimental models and oracle labels are local synthetic tasks. No infinite-depth or unrestricted-language conclusions.','','## Execution and controls','','All 27 trained conditions completed 2,000 optimizer updates each, batch 32, training length 32;21 prefix-supervised primary conditions and6 separate final-target-only conditions. Seeds 17, 43, 91. Validation-only checkpoint selection, no test-based tuning. Same initialization tensors and input streams are paired within task/width/seed. No-feedback keeps encoder, global cross-attention and SWA; only hidden feedback is disabled. All allocated merge parameters remain present but are inactive in no-feedback. FLOPs differ; no equal-FLOP claim. CPU 1 thread per training process; training seconds include validation/checkpoint overhead and concurrent system activity. Primary and companion later run concurrently on separate CPU threads, so their durations must not be added as elapsed wall time.','','## Primary final-state accuracy','','Each cell lists seed17 /43 /91 as percentages (512 independent examples per seed/condition). Length is actual consumed tokens and generated operations, including identity. Seeds are replication units; do not pool1536 examples as independent model replications.','', '| Task | Width | Mode |32|64|128|256|512|1024|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
 for ev in metrics['evaluation']:
  c=ev['condition'];key=(c['task'],c['width'],c['mode']);groups.setdefault(key,{})[c['seed']]={}
  for r in ev['long_horizon']:
   a=np.load(r['predictions_file']);acc=float((a['predictions'][:,-1]==a['targets'][:,-1]).mean());assert acc==r['final_accuracy'];groups[key][c['seed']][r['length']]=acc
   changed=actual_state_changes(a['targets']);recomputed.append(dict(condition=c,length=r['length'],accuracy=acc,conditional_example_wilson95=wilson(acc),inputs_sha256=hashlib.sha256(a['inputs'].tobytes()).hexdigest(),targets_sha256=hashlib.sha256(a['targets'].tobytes()).hexdigest(),non_neutral_input_count=float((a['inputs']!=2).sum(1).mean()),true_state_changing_operation_count=float(changed.sum(1).mean()),file=r['predictions_file'],sha256=sha(Path(r['predictions_file']))))
 for key,seeds in groups.items():lines.append('|'+f'{key[0]}|{key[1]}|{key[2]}|'+'|'.join(' / '.join(f'{seeds[s][l]*100:.2f}' for s in [17,43,91]) for l in [32,64,128,256,512,1024])+'|')
 lines+=['','Metadata correction: raw `mean_nonidentity_count` was computed as input tokens other than designated neutral symbol2. For parity, bit0 is also mathematically an identity; therefore the correct raw-field interpretation is **non-neutral input count**. Supplemental recomputation separately records true state-changing operations by comparing each oracle target with the previous target, starting at state0. The five-state swap fixes states2,3,4, so token identity alone cannot count actual state changes. Operation counts are workload measures, not proved lower bounds on minimal algorithmic depth. Raw observations are preserved.','','## Checkpoint selection, active parameters and actual training times','', '| Condition | Selected step | Best validation final accuracy | Allocated / active parameters | Training seconds |','|---|---:|---:|---:|---:|']
 for r in metrics['training']:
  c=r['condition'];v=next(h['validation']['final_accuracy'] for h in r['history'] if h['step']==r['selected_step']);lines.append(f"|{c['task']} w{c['width']} {c['mode']} s{c['seed']}|{r['selected_step']}|{v*100:.2f}%|{r['parameters']['allocated']} / {r['parameters']['active']}|{r['train_seconds']:.2f}|")
 lines+=['','## Extra neutral-token computation (k=0)','', 'Same base input and final answer; suffix symbols are identity operations. The useful base input remains unchanged. Output is one state classification; every added slot consumes another token. This measures token-clock recurrence, not a title-refinement loop. Each entry is seed-averaged final accuracy; individual seed rows and actual latency are in per-condition evaluation files.','', '| Task | Width | Mode | slots0 | slots8 | slots32 | slots128 |','|---|---:|---|---:|---:|---:|---:|']
 for key in groups:
  selected=[ev for ev in metrics['evaluation'] if (ev['condition']['task'],ev['condition']['width'],ev['condition']['mode'])==key];lines.append('|'+f'{key[0]}|{key[1]}|{key[2]}|'+'|'.join(f"{np.mean([ev['neutral_slots'][j]['final_accuracy'] for ev in selected])*100:.2f}%" for j in range(4))+'|')
 lines+=['','## Delayed-final-error companion','', 'Separate frozen protocol; final-target-only CE and final-CE validation selection. TBPTT8 cuts only recurrent hidden and decoder SWA gradients; encoder and global memory remain differentiable. Prefix-supervised success does not establish learning from delayed errors. Within each supervision regime full/TBPTT share initialization, examples and2,000 updates. Across regimes there are different predefined example streams; both see64,000 programs and2,048,000 input tokens, but prefix training supplies2,048,000 targets versus64,000 final-only targets. This is not an equal-label-budget comparison.','', '| Mode | Seed | Selected step | Validation final accuracy | Test32 | Test128 | Training seconds |','|---|---:|---:|---:|---:|---:|---:|']
 final_recomputed=[]
 for r in final['records']:
  h=next(h for h in r['history'] if h['step']==r['selected_step']);vals=[]
  for ev in r['evaluation']:
   a=np.load(ev['file']);acc=float((a['predictions'][:,-1]==a['targets'][:,-1]).mean());assert acc==ev['final_accuracy'];vals.append(acc);final_recomputed.append(dict(condition=r['condition'],length=ev['length'],accuracy=acc,conditional_example_wilson95=wilson(acc),inputs_sha256=hashlib.sha256(a['inputs'].tobytes()).hexdigest(),targets_sha256=hashlib.sha256(a['targets'].tobytes()).hexdigest(),file=ev['file']))
  lines.append(f"|{r['condition']['mode']}|{r['condition']['seed']}|{r['selected_step']}|{h['validation_final_accuracy']*100:.2f}%|{vals[0]*100:.2f}%|{vals[1]*100:.2f}%|{r['train_seconds']:.2f}|")
 lines+=['','## Influence and checkpoint diagnostics','', 'Oracle final-state CE provides a meaningful loss. Recurrent-state and embedding lag norms, plus central finite differences of the first hidden state, are saved in diagnostics.json. No-feedback has zero sensitivity to explicitly supplied previous output, while early embeddings can still influence the final loss through the encoder/global-memory path. These influence diagnostics use the prefix-trained selected checkpoints. A nonzero gradient is not evidence of successful learning; magnitudes also depend on terminal loss/confidence and parameterization, so a small gradient at an already-correct checkpoint does not diagnose learning failure. Gaussian state-noise sweeps are diagnostics, not equivalent arithmetic precision experiments. Whole-forward activation checkpointing compares the same real loss and parameter gradients; peak activation memory is UNMEASURED.','']
 for r in diag['rows']:
  if 'checkpoint_repeats' in r:lines.append(f"Checkpoint 3 repeats: maximum gradient difference {max(x['max_gradient_difference'] for x in r['checkpoint_repeats']):.3g}; standard seconds {[round(x['steps'][0]['seconds'],4) for x in r['checkpoint_repeats']]}; recomputation seconds {[round(x['steps'][1]['seconds'],4) for x in r['checkpoint_repeats']]}. Timing is scope-limited.")
 benchmark=ROOT/'results'/'benchmark.json'
 if benchmark.exists():
  rows=read(benchmark)['rows'];lines+=['','## Fresh-process real-step memory/time comparison','','Each repetition starts a fresh Python process from the same frozen checkpoint and data, warms one inference pass, then performs one real AdamW update. Peak working set includes the interpreter/native libraries/model/optimizer; it is not isolated activation memory. External system load is not controlled.','','| Loss | Mode | Mean seconds ± sample SD | Process peak MiB range |','|---|---|---:|---:|']
  for loss in ['prefix','final_only']:
   for mode in ['full','tbptt8','checkpoint']:
    rr=[r for r in rows if r['loss']==loss and r['mode']==mode];seconds=[r['seconds'] for r in rr];peaks=[r['after']['peak_working_set_bytes']/2**20 for r in rr];lines.append(f'|{loss}|{mode}|{np.mean(seconds):.4f} ± {np.std(seconds,ddof=1):.4f}|{min(peaks):.1f}–{max(peaks):.1f}|')
 resources=ROOT/'resource-samples.jsonl'
 if resources.exists():
  samples=[json.loads(l) for l in resources.read_text(encoding='utf-8-sig').splitlines()];lines+=['',f"Primary process observed peak working set: {max(x['peak_working_set_bytes'] for x in samples)/2**20:.1f}MiB. This includes Python, model, native libraries and allocations; it is not isolated activation memory and is not GPU VRAM."]
 paired=[]
 for task,width in [('parity',32),('five',32),('parity',48)]:
  for l in [32,64,128,256,512,1024]:
   d=[groups[(task,width,'rlt')][s][l]-groups[(task,width,'nofeedback')][s][l] for s in [17,43,91]];paired.append(dict(task=task,width=width,length=l,seeds=[17,43,91],paired_accuracy_deltas=d,mean=float(np.mean(d)),sample_sd=float(np.std(d,ddof=1)),replication_units=3))
 hashes={f.name:tensor_hash(f) for f in (ROOT/'checkpoints').glob('*-initial.pt')};matching=[]
 for task,width in [('parity',32),('five',32),('parity',48)]:
  for seed in [17,43,91]:
   a=f'{task}-w{width}-rlt-s{seed}-initial.pt';b=f'{task}-w{width}-nofeedback-s{seed}-initial.pt';assert hashes[a]==hashes[b];matching.append([a,b,hashes[a]])
 for seed in [17,43,91]:
  for prefix in ['parity-w32','finalonly-parity-w32']:
   a=f'{prefix}-rlt-s{seed}-initial.pt';b=f'{prefix}-tbptt-s{seed}-initial.pt';assert hashes[a]==hashes[b];matching.append([a,b,hashes[a]])
 for task in ['parity','five']:
  for seed in [17,43,91]:
   for length in [32,64,128,256,512,1024]:
    input_hashes={r['inputs_sha256'] for r in recomputed if r['condition']['task']==task and r['condition']['seed']==seed and r['length']==length};assert len(input_hashes)==1
 summary=dict(primary_recomputed=recomputed,final_only_recomputed=final_recomputed,paired_seed_deltas=paired,initial_tensor_hashes=hashes,matched_initializations=matching,metadata_correction='raw mean_nonidentity_count means designated non-neutral token count; true state-changing count provided separately',independent_audit='This is worker self-recomputation only; consult the separate root audit record for independent scientific closure')
 (ROOT/'results'/'recomputed.json').write_text(json.dumps(summary,indent=2));lines+=['','## Scope and status boundaries','','Package1 IMPLEMENTED_TESTED: causal state/cache contract, matched information control, serial/full encoder and parallel-control numerical+gradient parity. Package2 measured at finite length up to1024; interpret numerical tables, not an unbounded-reasoning claim. Package3 measured neutral-token computation with no new useful facts, k=0; title refinement remains outside this module. Package4 measured full/TBPTT learning and delayed-error companion, with path-specific gradient diagnostics; no complete removal of all long paths and no memory-saving claim. Package14 measures width32 versus48 on parity; not a model-law, natural-language, energy, RL or large-scale result.','','Eighteen tests passed in the final core/green-final-18.log; earlier RED/GREEN logs are preserved. Independent root audit is recorded separately; scientific closure requires that external audit. Source, environment, checkpoint and prediction hashes are included in core artifacts. The original runner is unsafe for in-place partial restart; this uninterrupted run is attributed to its exact preserved source snapshots. Do not rerun legacy runners in this evidence directory. The separate safe_entrypoint verifies completed evidence, rejects partial or drifted receipts, and creates only a post-run consistency binding, not a signed execution receipt. Checkpoints contain model weights, not resumable optimizer state. CPU timings under shared load cannot be treated as isolated hardware throughput. The held-out examples differ across model/data seeds; input and target hashes are recorded per condition, and paired models within a seed share cases. Three model/data seeds provide limited precision; per-seed outcomes and paired deltas are retained instead of pooling correlated prefix predictions. Supplemental Wilson95 intervals use512 final outcomes conditional on each fixed checkpoint; they are not confidence intervals over the population of model seeds.']
 report='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(report,encoding='utf8');(ROOT.parent/'reports').mkdir(exist_ok=True);(ROOT.parent/'reports'/'core-report.md').write_text(report,encoding='utf8');manifest={str(f.relative_to(ROOT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in ROOT.rglob('*') if f.is_file() and f.name!='manifest.json' and '__pycache__' not in str(f)};(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()














