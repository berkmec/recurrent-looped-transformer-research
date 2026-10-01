"""Prespecified generated-state compaction companion; original run unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
import numpy as np
from integration.evaluate_system import ROOT,sha,write_json,check_sources,assert_fresh_results,make_cases,render_case,score_episode,summarize_scores

OUT=ROOT/'integration'/'reasoning-companion'


def freeze():
    assert_fresh_results(OUT);OUT.mkdir(parents=True,exist_ok=True)
    sources=['integration/evaluate_window.py','integration/reasoning_window.py','integration/window_pipeline.py',
             'integration/evaluate_system.py','integration/audit_stats.py','learning/api.py','learning/model.py',
             'learning/token_core.py','memory/api.py','memory/headings.py','memory/extractor.py']
    p={'version':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),
       'reason':'Original finite windows bounded observations; this companion binds actual generated title-state history compaction and reread to continuation.',
       'seeds':[17,43,91],'arm':'combined final600','parser':'matched seed final extractor','selection':'none; all3 fixed prior checkpoints',
       'data':{'iid':{'count':128,'seed':134331},'depth8':{'count':128,'seed':134338}},
       'conditions':[{'name':'uncompacted','compact_reasoning':False},{'name':'window4','compact_reasoning':True}],
       'source_style':'seen','source_view':'replay','observation_capacity':4,'reasoning_capacity':4,'max_steps':8,'tolerance':.1,
       'halting':'both arms evaluate sufficiency only after4/8 actual refinements, after reread in window4',
       'compressed_state':'latest latent title state z, guarded by exact fingerprints of unchanged model/evidence/graph/token/cache/global-memory/query obligations',
       'raw_policy':'every actual generated z/numeric candidate, original source, model provenance and heading mapping persist in SQLite; compaction never deletes raw',
       'hypotheses':['all prediction values, statuses and executed budgets identical to paired uncompacted baseline','every full boundary reload bitwise exact','generated active history capacity<=4 with no early compaction','continuation under same stable heading recorded'],
       'metrics':['conditional candidate quality and certified source yield','actual refinements/tokens','archive bytes','compaction/reread counts','max resident generated-history records','serialized active-history payload bytes','wall time'],
       'limits':['local latent Markov sufficiency, not arbitrary semantic compression','shared generated-history window for current queried heading, not simultaneous per-heading quotas','serialized payload bytes are not measured process RAM','model encoder/KV/graph/title working state remains resident outside history window','same model quality expected by exact continuation; this tests engineering fidelity, not improved reasoning skill','scientific counterpart to primary adaptive checks at everyk; do not conflate stopping schedules'],
       'source_sha256':{str(ROOT/file):sha(ROOT/file) for file in sources},
       'checkpoint_sha256':{str(ROOT/folder/file):sha(ROOT/folder/file) for seed in [17,43,91] for folder,file in [('learning/checkpoints',f'combined-{seed}-step600.pt'),('memory/checkpoints',f'extractor-{seed}.pt')]}}
    write_json(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print(json.dumps({'protocol':str(OUT/'protocol.json'),'sha256':sha(OUT/'protocol.json')}))


def main():
    import torch
    from learning.api import load_model
    from memory.api import load_parser
    from integration.window_pipeline import run_text_episode
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    check_sources(p['source_sha256']);check_sources(p['checkpoint_sha256'])
    result=OUT/'results';assert_fresh_results(result);result.mkdir(parents=True,exist_ok=True)
    execution=(result/'execution.jsonl').open('x');start=time.perf_counter();rows=[];comparisons=[]
    def log(event,**fields):
        r={'event':event,'utc':datetime.now(timezone.utc).isoformat(),**fields}
        execution.write(json.dumps(r)+'\n');execution.flush();print(json.dumps(r),flush=True)
    cases={split:make_cases(v['count'],v['seed'],split) for split,v in p['data'].items()}
    for split,data in cases.items():
        write_json(result/f'data-{split}.json',data);log('fresh_test_opened',split=split,sha256=sha(result/f'data-{split}.json'))
    for seed in p['seeds']:
        model_path=ROOT/'learning/checkpoints'/f'combined-{seed}-step600.pt'
        parser_path=ROOT/'memory/checkpoints'/f'extractor-{seed}.pt'
        model=load_model(model_path);parser=load_parser(parser_path)
        provenance={'model_path':str(model_path),'model_sha256':p['checkpoint_sha256'][str(model_path)],
                    'parser_path':str(parser_path),'parser_sha256':p['checkpoint_sha256'][str(parser_path)],
                    'protocol_sha256':sha(OUT/'protocol.json')}
        for split,data in cases.items():
            paired={}
            for condition in p['conditions']:
                name=condition['name'];label=f'{split}-{name}-s{seed}';directory=result/label;directory.mkdir()
                scored=[];costs=[];outputs=[];group_start=time.perf_counter()
                with (directory/'predictions.jsonl').open('x') as stream:
                    for case in data:
                        db=directory/f'episode-{case["index"]:04d}.sqlite'
                        out=run_text_episode(render_case(case,p['source_style']),model,parser,db,task=case['id'],
                            capacity=p['observation_capacity'],reasoning_capacity=p['reasoning_capacity'],max_steps=p['max_steps'],
                            compact_reasoning=condition['compact_reasoning'],tolerance=p['tolerance'],model_provenance=provenance)
                        score=score_episode(case,out,p['tolerance']);scored.append(score)
                        fields=('seconds','total_refinements_executed','total_consumed_tokens_including_replay',
                                'reasoning_compactions','reasoning_rereads','reasoning_max_resident_traces',
                                'reasoning_generated_traces','reasoning_peak_active_serialized_bytes',
                                'reasoning_maximum_reload_difference','same_heading_reasoning_continuations')
                        costs.append({key:out[key] for key in fields});costs[-1]['archive_bytes']=db.stat().st_size
                        outputs.append({'candidates':[(q.get('prediction'),q['status'],q['steps']) for q in out['outputs']],
                                        'refinements':out['total_refinements_executed'],'tokens':out['total_consumed_tokens_including_replay'],
                                        'bitwise_reload':out['reasoning_reload_bitwise_equal']})
                        stream.write(json.dumps({'seed':seed,'split':split,'condition':condition,'score':score,'result':out,'archive_sha256':sha(db)},allow_nan=False)+'\n');stream.flush()
                        if (case['index']+1)%64==0:log('progress',condition=label,completed=case['index']+1,seconds=time.perf_counter()-group_start)
                summary={'seed':seed,'split':split,'condition':name,**summarize_scores(scored),
                         'cost_mean':{k:float(np.mean([c[k] for c in costs])) for k in costs[0]},
                         'cost_max':{k:max(c[k] for c in costs) for k in costs[0]},
                         'all_reloads_bitwise_equal':all(row['bitwise_reload'] for row in outputs),
                         'prediction_sha256':sha(directory/'predictions.jsonl'),'seconds':time.perf_counter()-group_start}
                write_json(directory/'metrics.json',summary);rows.append(summary);paired[name]=outputs
                log('condition_complete',condition=label,metrics=summary['final'])
            a,b=paired['uncompacted'],paired['window4']
            comparison={'seed':seed,'split':split,'episodes':len(data),
                        'candidate_status_budget_identical':all(x['candidates']==y['candidates'] for x,y in zip(a,b)),
                        'actual_refinements_identical':all(x['refinements']==y['refinements'] for x,y in zip(a,b)),
                        'actual_tokens_identical':all(x['tokens']==y['tokens'] for x,y in zip(a,b)),
                        'all_reloads_bitwise_equal':all(x['bitwise_reload'] for x in b)}
            comparisons.append(comparison);log('paired_comparison',**comparison)
    check_sources(p['source_sha256']);check_sources(p['checkpoint_sha256'])
    write_json(result/'metrics.json',{'status':'COMPLETE','rows':rows,'comparisons':comparisons,'seconds':time.perf_counter()-start,
                                     'independent_source_tasks':256,'training_seeds':3,'episode_executions':1536,'protocol_sha256':sha(OUT/'protocol.json')})
    log('complete',seconds=time.perf_counter()-start);execution.close()


if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
