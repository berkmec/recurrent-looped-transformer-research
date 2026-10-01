# Finite heading registry: added bounded requirement

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

Protocol SHA256 `a39b9bd27cea71ee98f73c8c0ed6ba2077e1211a005468de8a294f51bde18260` was saved before inference. One unchanged seed17 extractor checkpoint processes64 new task identities, each with ten variables, four edge revisions, differently worded duplicate root observation, existing/missing queries and one negative sentence. Registry inputs are learned parser outputs only. Generator events are used to render the finite language, never passed to the registry. This small seen-layout test does not overturn the earlier heldout-paraphrase extraction failures.

Metrics: {
  "tasks": 64,
  "sentences": 1152,
  "learned_accepted": 1088,
  "headings": 640,
  "unique_heading_ids": 640,
  "active_links": 576,
  "semantic_versions": 896,
  "cross_task_links": 0,
  "serialization_exact": true,
  "history_events": 1152,
  "condition_statuses": {
    "root": {
      "CREATED": 64
    },
    "edge": {
      "CREATED": 576
    },
    "revision": {
      "UPDATED": 256
    },
    "duplicate_wording": {
      "UNCHANGED": 64
    },
    "query_existing": {
      "QUERY_RESOLVED": 64
    },
    "query_missing": {
      "MISSING_ENTITY": 64
    },
    "negative": {
      "EXTRACTION_UNRESOLVED": 64
    }
  },
  "wall_seconds": 3.694021499948576,
  "protocol_sha256": "a39b9bd27cea71ee98f73c8c0ed6ba2077e1211a005468de8a294f51bde18260",
  "checkpoint_sha256": "97d6dc86f668f4071689c84a461b5c1866a38886f56be9b0d10b1ef1e641121b",
  "scope": "deterministic finite affine organization of learned parsed events; not learned topic discovery"
}

Eight meaningful unit tests passed after preserved RED evidence: different wording same identity, revision preservation and immutable returned snapshots, actual new variable/title, typed relation direction, task isolation and missing query, independently retained absolute/edge facets, unsupported/no-op behavior, rejected extraction, exact serialization replay, and identity retention across four actual controller compactions. The fixed experiment records every text, learned parse and heading result in `heading-results/predictions.jsonl`, with full append-only history in `heading-results/heading-history.json`. Scientific status is IMPLEMENTED_TESTED for finite automatic heading organization. Unrestricted topic boundary discovery, learned title creation and free-form reasoning headings remain UNMEASURED.
