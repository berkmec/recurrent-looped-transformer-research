"""Recompute stored predictions and assemble scoped worker report."""
import hashlib,json,math,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from .api import Archive,TitleController,load_parser,render_event
ROOT=Path(__file__).parent
def readrows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    learned=json.loads((ROOT/'results'/'extractor-metrics.json').read_text()); eng=json.loads((ROOT/'results-v3'/'engineering-metrics.json').read_text()); checks=[]
    for m in learned:
        rows=readrows(ROOT/'results'/f"extractor-{m['seed']}.jsonl"); positives=[r for r in rows if r['event'] is not None]; negatives=[r for r in rows if r['event'] is None]
        recomputed=sum((r['prediction']['events'][0] if r['prediction']['events'] else None)==r['event'] for r in positives)/len(positives)
        assert abs(recomputed-m['exact_tuple_accuracy'])<1e-12
        assert sum(r['prediction']['valid'] for r in negatives)==0
        checks.append(dict(name=f"extractor_seed_{m['seed']}",pass_check=True,examples=len(rows)))
    rows=readrows(ROOT/'results-v3'/'memory-examples.jsonl')
    for condition,m in eng['memory'].items():
        value=sum(abs(r[condition]-r['target']) for r in rows)/len(rows); assert abs(value-m['mean_absolute_error'])<1e-12
    checks.append(dict(name='memory_errors_recomputed',pass_check=True,examples=len(rows)))
    retrieval=readrows(ROOT/'results-v3'/'retrieval-examples.jsonl')
    for k in ['1','3','5']:
        recall=sum(r['correct'] in [x['id'] for x in r['raw'][:int(k)]] for r in retrieval)/len(retrieval); assert abs(recall-eng['retrieval'][k]['recall_at_k'])<1e-12
    checks.append(dict(name='retrieval_recall_recomputed',pass_check=True,examples=len(retrieval)))
    halting=readrows(ROOT/'results-v3'/'halting-examples.jsonl')
    for k in ['1','2','4','8']:
        group=[r for r in halting if r['budget']==int(k)]; saved=1-sum(r['executed_steps'] for r in group)/len(group)/int(k); assert abs(saved-eng['halting'][k]['saved_step_fraction'])<1e-12
    checks.append(dict(name='actual_halting_steps_recomputed',pass_check=True,examples=len(halting)))
    # Integration smoke preserves source text, parse provenance, and abstentions.
    smoke_path=ROOT/'provenance-smoke.sqlite'
    if smoke_path.exists(): raise RuntimeError('Preserve previous smoke evidence')
    a=Archive(smoke_path); c=TitleController(3,a); parser=load_parser(); smoke=[]
    events=[dict(kind='root',child='x0',parent=None,value=.4),dict(kind='edge',child='x1',parent='x0',value=-.2),dict(kind='revision',child='x1',parent='x0',value=.3),dict(kind='query',child='x1',parent=None,value=None)]
    texts=[render_event(e) for e in events]+['do not set x0 to 0.4 .','']
    for text,parsed in zip(texts,parser.parse_batch(texts)): smoke.append(c.observe_text('smoke','title',text,parsed))
    a.close(); code="from memory.api import Archive; import json; a=Archive(r'%s'); ids=[r[0] for r in a.db.execute('SELECT id FROM observations')]; records=[a.read_observation(i) for i in ids]; print(json.dumps({'observations':len(ids),'source_texts':[r['text'] for r in records],'abstentions':sum(r['parsed']['abstained'] for r in records)})); a.close()"%smoke_path
    fresh=subprocess.run([sys.executable,'-c',code],cwd=ROOT.parent,text=True,capture_output=True,check=True); smoke_reload=json.loads(fresh.stdout); assert smoke_reload['observations']==6 and smoke_reload['abstentions']==2
    (ROOT/'provenance-smoke.json').write_text(json.dumps(dict(rows=smoke,fresh_process=smoke_reload),indent=2))
    checks.append(dict(name='source_text_abstention_fresh_process',pass_check=True,examples=6))
    statuses={'5':'MIXED','8':'IMPLEMENTED_TESTED','9':'MEASURED_POSITIVE','10':'MEASURED_NEGATIVE','11':'MEASURED_POSITIVE'}
    combined=dict(package_status=statuses,extractor_protocol_sha256=sha(ROOT/'protocol-v1.json'),engineering_protocol_sha256=sha(ROOT/'protocol.json'),extractor=learned,engineering=eng,recomputation=checks,independent_audit='root worker requested; this recomputation is not an independent audit',generalization_limits=['controlled English only; unseen paraphrase failure','title boundaries supplied by caller, not learned segmentation','resident raw-record window bounded, not total RAM or transient reconstruction buffer','structured affine compaction, not learned general reasoning summary','very poor deterministic vector retrieval recall','halting trajectories synthetic; learned integration separately owned by root'])
    (ROOT/'results'/'metrics.json').write_text(json.dumps(combined,indent=2)); (ROOT/'recomputation.json').write_text(json.dumps(checks,indent=2))
    sources={str(p.relative_to(ROOT.parent)):sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT.parent/'tests').glob('test_memory*.py'))}
    (ROOT/'final-source-manifest.json').write_text(json.dumps(dict(time=time.time(),python=sys.version,platform=platform.platform(),source_sha256=sources,protocol_sha256=sha(ROOT/'protocol.json'),checkpoint_sha256={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt')}),indent=2))
    lines=['# Learned extraction and non-destructive title memory: measured scope','',
      'Packages 5, 8, 9, 10, 11 have runnable implementations and scoped measurements. This worker did not implement RLT decoder internals; the official cc05cfe0b24bc5eb353c6f3c14990ea4e7be0a39 decoder pin applies to the separate core worker. The extractor is a deliberately separate small BiGRU, not a source-faithful RLT language model.', '',
      '## Formal contracts','',
      '- Stable title identity is `(task,title)` supplied by the caller for a genuine subproblem. Versions append monotonically. Learned title segmentation is not claimed.',
      '- Event facets `(absolute,child)` and `(edge,child)` are independently versioned: an absolute observation on a non-root node must preserve its parent relation. This supports forward and inverse affine evidence.',
      '- Raw event records, source-text observations, parser provenance, and compact snapshots are append-only SQLite rows with canonical JSON SHA256. Hashing provides detection, not filesystem access control or protection against a privileged attacker rewriting both payload and hash.',
      '- The active capacity counts resident raw assertion records. Compaction occurs only when the next append finds the window full. Semantic compact payloads are disk-backed; title registry metadata and transient reconstruction buffers are separately unbounded. This is not a total-RAM bounded algorithm.',
      '- Structured compaction retains all latest affine facets and raw-history hash/version obligations. Late evidence invalidates the scope; rereading verifies raw hashes and appends a replacement compact snapshot. No raw row is overwritten by the API.',
      '- Vector retrieval is deterministic 128-dimensional signed hashed bag-of-words cosine. Applicability requires task, title, entity and latest facet version. Exact-key access is a separate oracle upper bound.',
      '- For consistent affine evidence, summing absolute candidate residuals over the anchored connected component upper-bounds query error along any anchor-to-query path. Relative graph potentials check evidence consistency only; they never replace the candidate. Unsupported, inconsistent, unanchored or incomplete candidates cannot certify.',
      '- `CONTINUE`, `VERIFIED`, `UNRESOLVED_BUDGET`, `COMPACTION_REJECTED` have distinct meanings. VERIFIED concerns extracted constraints only; it does not certify faithfulness to original text. Root integration must assess source-relative false acceptances separately.', '',
      '## Protocol and evidence preservation','',
      'Extractor protocol v1 was frozen before training/test evaluation. The last of 18 epochs and confidence threshold 0.65 were predeclared; no test-based selection or subsequent retuning occurred. v1/v2/v3 protocols, API/measurement source snapshots, archives, outputs and execution history remain preserved. v2 fixed independent absolute/edge facets after integration review, using fresh seed20260917 tasks. v3 moved compact semantic payload to disk and made retrieval distractors numerically wrong, using fresh seed20260918 tasks. These are declared engineering corrections, not three confirmations of one frozen experiment.',
      f"Final engineering protocol SHA256: `{sha(ROOT/'protocol.json')}`. Extractor protocol SHA256: `{sha(ROOT/'protocol-v1.json')}`.", '',
      '## Package 5: MIXED, genuine learned extraction','',
      'Supervised BiGRU learns token roles and event classes; exact entity/number copying occurs only after learned selection. Regex is limited to tokenization/type validation. Each seed trains on 3072 sentences, validates on384 and evaluates768 disjoint-identity sentences (576 positive,192 rejection challenges). Entity IDs are disjoint across splits. Lexical/layout challenge templates are withheld. Oracle extracted events score1.0 by definition and are separately labeled.', '',
      '| Seed | Seen-layout exact | Heldout paraphrase exact | All positive exact | Edge exact | Abstain rate | Negative false acceptance |','|---|---:|---:|---:|---:|---:|---:|']
    for m in learned: lines.append(f"| {m['seed']} | {m['seen_exact_accuracy']:.4f} | {m['heldout_exact_accuracy']:.4f} | {m['exact_tuple_accuracy']:.4f} | {m['edge_accuracy']:.4f} | {m['abstain_rate']:.4f} | {m['negative_false_accept_rate']:.4f} |")
    lines+=['','The main negative result is weak transfer to withheld wording/order, despite perfect seen-layout exactness. Rejection challenges share finite negative construction families with training; this is not arbitrary negation comprehension. Do not pool the three seeds as independent model populations. Checkpoints and per-example predictions are retained. Training wall seconds per seed: '+', '.join(f"{m['seed']}={m['wall_seconds']:.2f}" for m in learned)+'.','',
      '## Packages 8 and 9: structured controller engineering','',
      'Fresh192 tasks:96 forward and96 inverse, lengths8/16/32/64, capacities4/8/16, three competing stable titles, and delayed internal revisions. There were1216 actual capacity-triggered compactions, zero premature compactions and zero resident raw-record budget violations. All192 raw histories restored exactly; all192 stale scopes were detected and repaired by reopening. There were10304 raw records and1584 compact snapshots, all hash-verified in a fresh process.','',
      '| Condition | Mean absolute final error |','|---|---:|']
    for name,m in eng['memory'].items(): lines.append(f"| {name} | {m['mean_absolute_error']:.6f} |")
    lines+=['',f"Boundary-preserving compact/raw bytes averaged {eng['controller']['mean_compact_to_raw_bytes']:.4f}; this is only modest compression because most constraints remain necessary. Mean raw records read per example={eng['controller']['mean_read_records']:.1f}. All answer-quality values in this table are explicit symbolic engineering checks, not learned-model predictions. Learned controller integration is a separate root experiment.", '',
      '## Package 10: MEASURED_NEGATIVE retrieval','',
      '| k | Unfiltered recall | Filtered precision | Filtered coverage | Unfiltered top1 numeric damage |','|---|---:|---:|---:|---:|']
    for k,m in eng['retrieval'].items(): lines.append(f"| {k} | {m['recall_at_k']:.4f} | {m['filtered_precision']} | {m['filtered_coverage']:.4f} | {m['unfiltered_numeric_damage']:.4f} |")
    lines+=['','Filtering prevents returned stale/cross-task/entity-invalid records, but very low coverage means this is not successful retrieval. Exact-key oracle coverage is1.0 and is not used to hide vector misses. Missing-reference coverage is0.0. Final decoys use99/-99 while valid values are2; earlier v1/v2 decoys coincidentally shared value2, so their zero numeric damage was not informative and remains preserved.', '',
      '## Package 11: actual sequential halting, synthetic candidates','',
      '| Maximum steps | Accepted coverage | Wrong acceptances | Mean executed steps | Saved steps |','|---|---:|---:|---:|---:|']
    for k,m in eng['halting'].items(): lines.append(f"| {k} | {m['accepted_coverage']:.4f} | {m['wrong_accepted']} | {m['mean_executed_steps']:.4f} | {m['saved_step_fraction']:.4f} |")
    lines+=['','Candidates are explicitly synthetic shared-offset residual trajectories, including25% stable-wrong examples; labels are not learned inference. At budget8, coverage0.75 with zero wrong acceptances and28.125% actual saved calls. Budgets1/2/4 have zero coverage and are unsuccessful certification settings. This does not establish learned-model halting gains; root owns that integration.','',
      '## Public integration API','',
      '```python\nfrom memory.api import Archive, TitleController, load_parser, render_event, verify_candidate, halt_candidates\nparser = load_parser("memory/checkpoints/extractor-17.pt")\nparsed = parser.parse(text)          # or parser.parse_batch(list_of_texts)\narchive = Archive(path)\ncontroller = TitleController(capacity=4, archive=archive)\ncontroller.observe_text(task, title, text, parsed)  # preserves abstentions and source text\nraw_events = controller.reconstruct(task, title)\ncompact_events = controller.compact_events(task, title)\ncheck = verify_candidate(candidate_by_original_entity, compact_events, query, tolerance=0.1)\nresult = halt_candidates(lazy_candidate_iterator, compact_events, query, tolerance=0.1, max_steps=8)\n```','',
      '`render_event(event, style="seen"|"heldout", variant=0)` provides fixed public finite-language templates. Canonicalize entity names to numeric IDs per problem outside the parser; map learned node predictions back to original names for verification. Events contain observations only, never answer targets. A fresh-process smoke test preserves four accepted observations and two abstentions including original text.','',
      '## Verification and limits','',
      '16 unit tests pass, with preserved RED/GREEN evidence. Controls include raw/compact tampering, stale applicability, competing titles, multiple full compactions, late internal revisions, naive-summary counterexample, false stable candidates, inverse facet preservation, inconsistent anchors, source-text/abstention persistence, and no execution after early halt. Recomputed saved per-example extractor/memory/retrieval/halting metrics match. This worker recomputation is not an independent audit; root review remains a distinct closure gate.','',
      'Energy, unrestricted natural-language understanding, learned title segmentation, general reasoning summarization, production vector retrieval, total-RAM bounds and large-model scale remain UNMEASURED. Main experiment artifacts are `memory/results/metrics.json`, `memory/results/extractor-*.jsonl`, `memory/results-v3/*.jsonl`, `.npz`, checkpoints, archives, protocol files, execution.jsonl and source/environment manifests.']
    report=ROOT.parent/'reports'/'memory-report.md'; report.parent.mkdir(exist_ok=True); report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(report),checks=checks,package_status=statuses),indent=2))

if __name__=='__main__': run()
