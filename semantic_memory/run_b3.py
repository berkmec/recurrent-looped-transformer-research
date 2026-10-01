"""B3 runner: multi-hop retrieval with an LM-named bridge (B2 follow-up), on the same frozen MuSiQue pins.

What B2 established: the cheap capitalised-term bridge produces generic terms and leaves a candidate union whose
ceiling is 0.488, so reranking cannot reach the targets. B3 therefore reports the CEILING as a first-class gate
(`ceiling_gate`) and measures bridge quality directly (`bridge_quality`: does a bridge occur in a gold passage the
first hop missed?) — both are reported whether or not recall improves, so a negative result still says why.

CLI: --freeze [--protocol-json '{...}'] | --development | --final   (--out results/B3)
"""
import argparse
import copy
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from semantic_memory import lm_bridge as lb
from semantic_memory import multihop as mh
from semantic_memory import retrieval as rt
from semantic_memory import run_b2 as b2

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('semantic_memory/retrieval.py', 'semantic_memory/multihop.py', 'semantic_memory/lm_bridge.py',
           'semantic_memory/embedding.py', 'semantic_memory/rerank.py', 'language/generator.py',
           'semantic_memory/run_b2.py', 'semantic_memory/run_b3.py')


def protocol():
    p = copy.deepcopy(b2.protocol())
    p.update(version=1, purpose='B3 multi-hop retrieval with an LM-named bridge; union ceiling is a first-class gate and bridge quality is measured directly',
             extractor='lm_bridge_v1', generator_model='__RLT_ROOT__/models/Qwen3-4B', quant='nf4', decoding='greedy, thinking disabled',
             gen_batch=8, gen_max_new_tokens=48, development_queries=100, reference='results/B2',
             targets=dict(recall_at_10=.95, coverage_at_10=.90, union_ceiling=.90),
             note='same MuSiQue pins, encoder settings and endpoints as B2; only the bridge extractor changes. One LM call per unique question, batched and cached.')
    return p


def source_hashes():
    return {name: rt.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=rt.digest(p), sources=source_hashes(), environment=dict(python=platform.python_version()), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    rt.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(rt.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if rt.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != rt.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    return manifest


def make_generator(p):
    from language.generator import QwenGenerator
    return QwenGenerator(p['generator_model'])


def run(out, split, encoder=None, reranker=None, generator=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if split not in ('development', 'final'):
        raise ValueError('split must be development or final')
    path = out / f'{split}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    data = rt.load_musique(p[f'{split}_path'], p[f'{split}_queries'])
    queries = data['queries']; texts = [data['corpus'][d] for d in data['order']]
    encoder = encoder or rt.make_encoder(p); reranker = reranker if reranker is not None else rt.make_reranker(p)
    generator = generator or make_generator(p)
    t0 = time.perf_counter(); dense = encoder.encode(texts, batch_size=p['encode_batch']); encode_seconds = time.perf_counter() - t0
    index = rt.Index(data['order'], texts, dense, np.zeros((len(texts), 1)),
                     dict(top_k=p['top_k'], fusion_depth=p['top_k'], rrf_k=p['rrf_k'], rerank_depth=p['rerank_depth'], rerank_batch=p['rerank_batch']))
    cache = {}

    def encode_query(text):
        if text not in cache:
            cache[text] = encoder.encode([text], batch_size=1, is_query=True)[0]
        return cache[text]

    # first hop once per query: feeds both the LM prompt and the missed-gold diagnostic
    first = {q['id']: index.top(index.dense @ np.asarray(encode_query(q['question'])), p['k1']) for q in queries}
    t0 = time.perf_counter()
    table = lb.precompute(queries, lambda q: [index.texts[index.position[d]] for d in first[q['id']][:p['branches']]],
                          generator, batch_size=p['gen_batch'], max_new_tokens=p['gen_max_new_tokens'], n=p['branches'])
    bridge_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    ev = mh.evaluate(index, queries, encode_query, table.as_extractor(), reranker,
                     dict(ks=p['ks'], top_k=p['top_k'], k1=p['k1'], branches=p['branches'], k2=p['k2'], rrf_k=p['rrf_k'],
                          rerank_depth=p['rerank_depth'], rerank_batch=p['rerank_batch']))
    retrieval_seconds = time.perf_counter() - t0

    # bridge quality: measured against the gold passages the FIRST hop missed, independent of any ranking gain
    quality = []
    for q, row in zip(queries, ev['per_query']):
        missed = [index.texts[index.position[d]] for d in q['gold'] if d not in first[q['id']]]
        r = lb.bridge_hits(row['bridges'], missed)
        row['bridge_quality'] = r; row['raw_bridge_output'] = table.raw.get(q['question'], '')
        quality.append(r)
    measured = [r for r in quality if r['status'] == 'MEASURED']
    bq = dict(queries_with_missed_gold=len(measured), any_hit_rate=float(np.mean([r['any_hit'] for r in measured])) if measured else None,
              mean_hits=float(np.mean([r['hits'] for r in measured])) if measured else None,
              no_missed_gold=len(quality) - len(measured))

    groups = [q['group'] for q in queries]; k = p['top_k']; arms = {}
    for arm in p['arms']:
        rows = [rt.rank_metrics(pq['ranked'][arm], pq['gold'], p['ks']) for pq in ev['per_query']]
        metrics = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        for key in (f'recall@{k}', f'coverage@{k}'):
            metrics[key + '_ci95'] = rt.bootstrap_mean([r[key] for r in rows], groups, p['bootstrap'], p['bootstrap_seed'])['ci95']
        entry = dict(n_queries=len(queries), metrics=metrics)
        if split == 'final':
            entry['gates'] = dict(recall=metrics[f'recall@{k}'] >= p['targets']['recall_at_10'], coverage=metrics[f'coverage@{k}'] >= p['targets']['coverage_at_10'])
            entry['status'] = 'MEASURED_POSITIVE' if all(entry['gates'].values()) else 'MEASURED_NEGATIVE'
        arms[arm] = entry
    ceiling = ev['union_ceiling']['all_gold_in_union']
    per_query_path = out / f'{split}-per-query.json'
    rt.write_json(per_query_path, dict(split=split, queries=ev['per_query']))
    result = dict(split=split, status='DEVELOPMENT' if split == 'development' else arms['multihop_rerank']['status'],
                  protocol_hash=manifest['protocol_hash'],
                  data=dict(path=data['path'], sha256=data['sha256'], queries=len(queries), corpus=len(texts), gold_per_query=dict(Counter(len(q['gold']) for q in queries))),
                  arms=arms, union_ceiling=ev['union_ceiling'],
                  ceiling_gate=dict(value=ceiling, target=p['targets']['union_ceiling'], passed=ceiling >= p['targets']['union_ceiling'],
                                    reference_b2=0.487795, note='if the ceiling stays low the bridge is the bottleneck, not the ranking'),
                  bridges=dict(**table.stats(), seconds=bridge_seconds), bridge_quality=bq,
                  timing=dict(encode_seconds=encode_seconds, bridge_seconds=bridge_seconds, retrieval_seconds=retrieval_seconds,
                              per_query_seconds=(bridge_seconds + retrieval_seconds) / max(1, len(queries))),
                  encoder=dict(name=getattr(encoder, 'name', None), encoded=getattr(encoder, 'encoded', None)),
                  reranker=dict(name=getattr(reranker, 'name', None), scored_pairs=getattr(reranker, 'scored', None)),
                  generator=getattr(generator, 'stats', lambda: None)(),
                  per_query_file=per_query_path.name, per_query_sha256=rt.sha(per_query_path), targets=p['targets'], turkish_slice='UNMEASURED')
    rt.write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--development', action='store_true'); group.add_argument('--final', action='store_true')
    parser.add_argument('--out', default='results/B3'); parser.add_argument('--protocol-json')
    args = parser.parse_args()
    overrides = json.loads(args.protocol_json) if args.protocol_json else {}
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out, dict(protocol(), **overrides))['protocol_hash'])
    else:
        r = run(args.out, 'development' if args.development else 'final')
        result = dict(status=r['status'], ceiling=r['ceiling_gate'], bridge_quality=r['bridge_quality'], bridges=r['bridges'],
                      arms={a: {k: round(v, 3) for k, v in e['metrics'].items() if not k.endswith('_ci95')} for a, e in r['arms'].items()})
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
