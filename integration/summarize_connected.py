"""Descriptive paired summaries for the frozen connected experiment.

Candidate quality is explicitly distinct from a certified answer. Shared
source examples are one sampling axis across all three joint training seeds.
"""
from pathlib import Path
import json
import numpy as np
from integration.audit_stats import paired_summary

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'integration/connected/results'


def main():
    completed=json.loads((RESULT/'metrics.json').read_text())
    protocol=json.loads((RESULT.parent/'protocol.json').read_text())
    assert len(completed['rows'])==24
    raw={};rows=[]
    for row in completed['rows']:
        seed,split,name=row['seed'],row['split'],row['condition']['name']
        path=RESULT/f'{name}-{split}-s{seed}'/'predictions.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(records)==256
        raw[name,split,seed]=records
        final=[r['score']['final'] for r in records]
        certified_correct=sum(r['accepted'] and r['source_correct'] for r in final)
        rows.append({'seed':seed,'split':split,'condition':name,
                     'diagnostic_candidate_accuracy':row['final']['accuracy_all_intended'],
                     'certified_source_correct_yield':certified_correct/len(final),
                     'certified_coverage':row['final']['coverage'],'accepted_source_failures':row['final']['accepted_failures'],
                     'risk_if_accepted':row['final']['accepted_error_rate'],
                     'emitted_mse':row['final']['mse_emitted'],'missing_final_candidates':row['final']['missing_outputs'],
                     'source_episode_exactness':row['source_episode_exact_accuracy'],
                     'source_event_exactness':row['source_event_exact_accuracy'],
                     'all_certificate_outputs':row['all_certificate_outputs'],
                     'costs':row['costs_mean']})
    comparisons=[]
    for split in protocol['datasets']:
        for candidate,reference in [('adaptive-replay-seen','fixed-replay-seen'),
                                    ('adaptive-compact-seen','adaptive-replay-seen'),
                                    ('adaptive-replay-heldout','adaptive-replay-seen')]:
            source=[];target=[];cert_source=[];cert_target=[];sq_source=[];sq_target=[];cost_rows=[]
            mse_complete=True
            for seed in protocol['seeds']:
                base=raw[reference,split,seed];other=raw[candidate,split,seed]
                assert [r['score']['id'] for r in base]==[r['score']['id'] for r in other]
                def final_rows(items):return [r['score']['final'] for r in items]
                a,b=final_rows(base),final_rows(other)
                source.append([float(not r['source_correct']) for r in a]);target.append([float(not r['source_correct']) for r in b])
                cert_source.append([float(not(r['accepted'] and r['source_correct'])) for r in a])
                cert_target.append([float(not(r['accepted'] and r['source_correct'])) for r in b])
                if any(r['squared_error'] is None for r in a+b):mse_complete=False
                sq_source.append([r['squared_error'] for r in a]);sq_target.append([r['squared_error'] for r in b])
                costs={}
                for metric in ('total_refinements_executed','total_consumed_tokens_including_replay','seconds'):
                    av=float(np.mean([r['result'][metric] for r in base]));bv=float(np.mean([r['result'][metric] for r in other]))
                    costs[metric]={'reference':av,'candidate':bv,'relative_saved':1-bv/av if av else None}
                cost_rows.append({'seed':seed,**costs})
            comparisons.append({'split':split,'candidate':candidate,'reference':reference,
                                'candidate_error_reduction':paired_summary(source,target),
                                'certified_yield_improvement':paired_summary(cert_source,cert_target),
                                'mse_improvement':paired_summary(sq_source,sq_target) if mse_complete else None,
                                'mse_pairing_note':'all intended outputs present' if mse_complete else 'not computed: at least one missing candidate; never zero-filled',
                                'per_seed_actual_cost':cost_rows})
    result={'rows':rows,'paired_comparisons':comparisons,
            'definitions':{'candidate_accuracy':'diagnostic numeric candidate vs original intended query, even if status remains unresolved',
                           'certified_source_correct_yield':'correct AND accepted final answers / all intended source episodes',
                           'certified_coverage':'accepted parsed-constraint certificate / all intended source episodes',
                           'risk':'source failures / accepted certificates; null at zero acceptance',
                           'population':'512 source tasks shared across seeds and conditions;6144 episodes are repeated executions, not6144 independent tasks'},
            'limitations':['three joint parser/model seeds','controlled language and affine tasks only','compact resets neural history and may reorder current facts/slots','actual wall time under concurrent local workload; not isolated speedup or energy']}
    path=ROOT/'integration/connected/summary.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(rows),'comparisons':len(comparisons),'summary':str(path)}))


if __name__=='__main__':main()
