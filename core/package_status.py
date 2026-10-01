"""Derived, descriptive package outcomes; never rewrite raw scientific aggregates."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).parent
def main():
 m=json.loads((R/'results/metrics.json').read_text());f=json.loads((R/'results/final-only-metrics.json').read_text())
 assert m['status']==f['status']=='COMPLETE'
 def accuracies(task,width,mode,length):
  return [next(x['final_accuracy'] for x in e['long_horizon'] if x['length']==length) for e in m['evaluation'] if e['condition']['task']==task and e['condition']['width']==width and e['condition']['mode']==mode]
 def slots(task,n):
  return [next(x['final_accuracy'] for x in e['neutral_slots'] if x['slots']==n) for e in m['evaluation'] if e['condition']['task']==task and e['condition']['width']==32 and e['condition']['mode']=='rlt']
 evidence={'parity_width32_rlt_L1024':accuracies('parity',32,'rlt',1024),'five_width32_rlt_L1024':accuracies('five',32,'rlt',1024),'parity_width48_rlt_L1024':accuracies('parity',48,'rlt',1024),'five_width32_rlt_neutral_slots':{str(n):slots('five',n) for n in [0,8,32,128]},'final_only_accuracy_range':[min(x['final_accuracy'] for r in f['records'] for x in r['evaluation']),max(x['final_accuracy'] for r in f['records'] for x in r['evaluation'])]}
 rows=[(1,'IMPLEMENTED_TESTED','Causal recurrent/cache contract and matched-information no-feedback control pass numerical and oracle-gradient checks.','Independent tiny implementation; no production state manager or official checkpoint reproduction.'),(2,'MIXED','Finite long-horizon accuracy is strong, but task/seed-dependent errors occur; per-seed 1024-token values below.','Tests stop at1024 consumed tokens; no infinite-depth or arbitrary-task claim.'),(3,'MEASURED_NEGATIVE','Extra neutral tokens do not establish a useful-computation benefit; five-state cases degrade while consuming additional measured time.','Token-clock recurrence with k=0; not the separate title-refinement mechanism.'),(4,'MIXED','Dense-prefix learning succeeds; final-target-only full and TBPTT models remain near chance. Checkpoint gradient/update equivalence holds, without observed whole-process peak-memory saving.','TBPTT cuts recurrent output/decoder KV only; global encoder routes remain. Whole-process memory is not activation-only allocation.'),(14,'INCONCLUSIVE','Widths32 and48 were actually trained/evaluated on parity; these two tiny widths do not establish a scaling law or a width advantage.','Core-width portion only; external module covers other package14 scopes. Energy, RL, large-model and unrestricted-language scaling unmeasured.')]
 out={'derivation':'Descriptive scope judgments after the preregistered measurements; not additional preregistered hypothesis tests. Raw metrics.json preserved unchanged.','packages':[{'package':n,'status':s,'measured_scope':text,'boundary':boundary} for n,s,text,boundary in rows],'evidence':evidence}
 with (R/'results/package-status.json').open('x',encoding='utf8') as h:json.dump(out,h,indent=2)
 lines=['','## Descriptive package outcomes','','These labels describe the measured scope; they are not new significance tests. Raw metric files remain unchanged.','','| Package | Status | Finding | Boundary |','|---|---|---|---|']
 lines += [f'|{n}|{s}|{text}|{boundary}|' for n,s,text,boundary in rows]
 lines += ['','At1024 tokens, RLT seed17/43/91 final accuracies: parity width32 '+str(evidence['parity_width32_rlt_L1024'])+'; five-state width32 '+str(evidence['five_width32_rlt_L1024'])+'; parity width48 '+str(evidence['parity_width48_rlt_L1024'])+'.','', 'Five-state RLT neutral-suffix accuracy by slots, each list seed17/43/91: '+str(evidence['five_width32_rlt_neutral_slots'])+'. All slots preserve the base answer; degradation is observed, not attributed to an unmeasured causal explanation.','', 'Final-only full/TBPTT accuracies across seeds and both lengths range from '+str(evidence['final_only_accuracy_range'])+'. Failure to learn in both arms is not evidence of general equivalence between gradient methods.','']
 for path in [R/'report.md',R.parent/'reports/core-report.md']:
  with path.open('a',encoding='utf8') as h:h.write('\n'.join(lines))
if __name__=='__main__':main()
