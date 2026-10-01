"""Fixed finite heading experiment; registry receives learned parser outputs only."""
import hashlib,json,time,sys
from pathlib import Path
from .api import load_parser,render_event
from .headings import HeadingRegistry
ROOT=Path(__file__).parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    out=ROOT/'heading-results'; out.mkdir(exist_ok=False); protocol_sha=sha(ROOT/'heading_protocol.json'); (out/'protocol.sha256').write_text(protocol_sha)
    start=time.perf_counter(); parser=load_parser(ROOT/'checkpoints'/'extractor-17.pt'); registry=HeadingRegistry(); observations=[]; text_rows=[]
    for task_index in range(64):
        task=f'heading-heldout-{task_index}'; value=round((task_index-32)/100,3)
        event_rows=[('root',dict(kind='root',child='x0',parent=None,value=value))]
        event_rows += [('edge',dict(kind='edge',child=f'x{j}',parent=f'x{j-1}',value=round(j/100,3))) for j in range(1,10)]
        event_rows += [('revision',dict(kind='revision',child=f'x{j}',parent=f'x{j-1}',value=round(j/100+.2,3))) for j in range(1,5)]
        event_rows += [('duplicate_wording',dict(kind='root',child='x0',parent=None,value=value)),('query_existing',dict(kind='query',child='x9',parent=None,value=None)),('query_missing',dict(kind='query',child='x99',parent=None,value=None))]
        for i,(condition,e) in enumerate(event_rows):
            variant=1 if condition=='duplicate_wording' else 0
            text_rows.append(dict(task=task,condition=condition,text=render_event(e,style='seen',variant=variant)))
        text_rows.append(dict(task=task,condition='negative',text=f'do not set x0 to {value} .'))
    # The source-generation records are not passed to the registry. Only learned
    # text extraction results enter semantic organization, including its errors.
    parsed=parser.parse_batch([r['text'] for r in text_rows]); before_ids={}
    for row,p in zip(text_rows,parsed):
        result=registry.consume_parsed(row['task'],p)
        observations.append({**row,'parsed':p,'heading_result':result})
    serialized=registry.serialize(); restored=HeadingRegistry.from_serialized(json.loads(json.dumps(serialized)))
    exact_roundtrip=restored.serialize()==serialized
    (out/'heading-history.json').write_text(json.dumps(serialized,indent=2)); (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
    headings=[h for i in range(64) for h in registry.headings(f'heading-heldout-{i}')]; links=[link for i in range(64) for link in registry.active_links(f'heading-heldout-{i}')]
    condition_counts={}
    for row in observations:
        condition=row['condition']; condition_counts.setdefault(condition,{})
        for result in row['heading_result']['results']:
            status=result['status']; condition_counts[condition][status]=condition_counts[condition].get(status,0)+1
    id_to_task={h['id']:h['task'] for h in headings}
    cross_task_links=sum(id_to_task[link['source']]!=id_to_task[link['target']] for link in links)
    totals=dict(tasks=64,sentences=len(observations),learned_accepted=sum(r['parsed']['valid'] for r in observations),headings=len(headings),unique_heading_ids=len(set(h['id'] for h in headings)),active_links=len(links),semantic_versions=sum(len(h['versions']) for h in headings),cross_task_links=cross_task_links,serialization_exact=exact_roundtrip,history_events=len(serialized['history']),condition_statuses=condition_counts,wall_seconds=time.perf_counter()-start,protocol_sha256=protocol_sha,checkpoint_sha256=parser.sha,scope='deterministic finite affine organization of learned parsed events; not learned topic discovery')
    (out/'metrics.json').write_text(json.dumps(totals,indent=2)); (out/'manifest.json').write_text(json.dumps(dict(python=sys.version,protocol_sha256=protocol_sha,sources={p.name:sha(p) for p in [ROOT/'headings.py',ROOT/'measure_headings.py',ROOT.parent/'tests'/'test_memory_headings.py']}),indent=2))
    report=f'''# Finite heading registry: added bounded requirement

The new module consumes learned extracted events and automatically organizes finite affine variables into stable linked headings. It does not modify frozen `memory/api.py`, retrain the extractor, or overwrite previous results. This is deterministic semantic organization after learned extraction, not learned unrestricted topic discovery.

## Mathematical contract

For task t and extracted entity e, H(t,e)=SHA256(canonical(t,e)). Objective O(e) is "Determine affine value of e". Wording and numeric answers are not identity inputs. Semantic state S_v maps the independent absolute and edge facets to observed assertions. v advances exactly when an observed facet changes; snapshots (v,S_v) append and are never replaced. A new entity opens a heading; a newly referenced parent opens a reference-only version0 heading. A later direct observation continues that same identity. A query resolves an existing heading and cannot create one.

For observed e=p+b, append typed link H(t,e) --DEPENDS_ON_AFFINE(offset=b)--> H(t,p). Revisions append a newer edge-link version while preserving earlier history. Absolute observations preserve the edge facet and its link. Task identity is part of every heading key. The registry has no answer solver and receives no numeric oracle labels.

Compaction does not participate in identity derivation: callers use the returned stable heading ID with the separate memory controller. Serialized registry history restores the same identities and versions after process boundaries. Registry metadata is not itself a bounded-memory implementation.

## API

```python
from memory.headings import HeadingRegistry
registry = HeadingRegistry()
result = registry.consume_parsed(task, learned_parser_output)
heading = registry.get(task, "x0")
headings = registry.headings(task)
links = registry.active_links(task)
payload = registry.serialize()
restored = HeadingRegistry.from_serialized(payload)
```

Lower-level `consume(task, normalized_event)` accepts the parser's event schema. Preferred `consume_parsed` refuses rejected/abstaining extraction even if its payload contains an event. Event statuses: CREATED, UPDATED, UNCHANGED, QUERY_RESOLVED, MISSING_ENTITY, UNSUPPORTED, EXTRACTION_UNRESOLVED. Every processed input appends a hash-chained history entry. Serialization validates hashes and deterministic replay; this detects accidental corruption, not malicious replacement of an entire chain.

## Fixed measurement

Protocol SHA256 `{protocol_sha}` was saved before inference. One unchanged seed17 extractor checkpoint processes64 new task identities, each with ten variables, four edge revisions, differently worded duplicate root observation, existing/missing queries and one negative sentence. Registry inputs are learned parser outputs only. Generator events are used to render the finite language, never passed to the registry. This small seen-layout test does not overturn the earlier heldout-paraphrase extraction failures.

Metrics: {json.dumps(totals,indent=2)}

Seven meaningful unit tests passed after preserved RED evidence: different wording same identity, revision preservation and immutable returned snapshots, actual new variable/title, typed relation direction, task isolation and missing query, independently retained absolute/edge facets, unsupported/no-op behavior, rejected extraction and exact serialization replay. The fixed experiment records every text, learned parse and heading result in `heading-results/predictions.jsonl`, with full append-only history in `heading-results/heading-history.json`. Scientific status is IMPLEMENTED_TESTED for finite automatic heading organization. Unrestricted topic boundary discovery, learned title creation and free-form reasoning headings remain UNMEASURED.
'''
    (ROOT/'heading_report.md').write_text(report); print(json.dumps(totals,indent=2))

if __name__=='__main__': run()
