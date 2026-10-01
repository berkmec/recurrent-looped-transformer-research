"""B2 runner (package P4): multi-hop retrieval on the B1 MuSiQue pins with the B1-selected encoder settings.

Arms: dense (first hop only, top-k), multihop (RRF union of first + bridge-driven second hops, top-k), multihop_rerank
(reranker over the union head, top-k). Metrics on the final list cut at k only; the union ceiling (all gold inside the
union) is reported separately as the bound a perfect reranker could reach. Targets apply to the final split only.
CLI: --freeze [--protocol-json '{...}'] | --development | --final   (--out results/B2)
"""
import argparse
import copy
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from semantic_memory import multihop as mh
from semantic_memory import retrieval as rt

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('semantic_memory/retrieval.py', 'semantic_memory/multihop.py', 'semantic_memory/embedding.py', 'semantic_memory/rerank.py', 'semantic_memory/run_b2.py')
ARMS = ('dense', 'multihop', 'multihop_rerank')
EXTRACTORS = {'capitalised_terms_v1': mh.bridge_terms}


def protocol():
    b = rt.protocol()
    return dict(version=1, purpose='B2 multi-hop retrieval: bridge terms from the first hop drive a second query; final top-10 is exactly ten documents',
                development_path=b['development_path'], development_queries=100, final_path=b['final_path'], final_queries=b['final_queries'],
                embedding_model=b['embedding_model'], reranker_model=b['reranker_model'], dense_max_length=256, dense_instruction=b['dense_instruction'],
                rerank_max_length=b['rerank_max_length'], encode_batch=b['encode_batch'], rerank_batch=b['rerank_batch'],
                arms=list(ARMS), ks=[1, 5, 10], top_k=10, k1=20, branches=3, k2=20, rrf_k=60, rerank_depth=50, extractor='capitalised_terms_v1',
                bootstrap=1000, bootstrap_seed=b['bootstrap_seed'], device='cuda', dtype='float16',
                targets=dict(recall_at_10=.95, coverage_at_10=.90), reference='results/B1',
                note='same MuSiQue pins as B1; corpus = all paragraphs of the loaded rows; bridge extractor is a cheap heuristic (LM extractor = later arm); Turkish slice UNMEASURED')


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


def run(out, split, encoder=None, reranker=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if split not in ('development', 'final'):
        raise ValueError('split must be development or final')
    path = out / f'{split}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    data = rt.load_musique(p[f'{split}_path'], p[f'{split}_queries'])
    queries = data['queries']; texts = [data['corpus'][d] for d in data['order']]
    encoder = encoder or rt.make_encoder(p); reranker = reranker if reranker is not None else rt.make_reranker(p)
    t0 = time.perf_counter(); dense = encoder.encode(texts, batch_size=p['encode_batch']); encode_seconds = time.perf_counter() - t0
    index = rt.Index(data['order'], texts, dense, np.zeros((len(texts), 1)), dict(top_k=p['top_k'], fusion_depth=p['top_k'], rrf_k=p['rrf_k'], rerank_depth=p['rerank_depth'], rerank_batch=p['rerank_batch']))
    cache = {}

    def encode_query(text):
        if text not in cache:
            cache[text] = encoder.encode([text], batch_size=1, is_query=True)[0]
        return cache[text]

    t0 = time.perf_counter()
    ev = mh.evaluate(index, queries, encode_query, EXTRACTORS[p['extractor']], reranker,
                     dict(ks=p['ks'], top_k=p['top_k'], k1=p['k1'], branches=p['branches'], k2=p['k2'], rrf_k=p['rrf_k'], rerank_depth=p['rerank_depth'], rerank_batch=p['rerank_batch']))
    retrieval_seconds = time.perf_counter() - t0
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
    per_query_path = out / f'{split}-per-query.json'
    rt.write_json(per_query_path, dict(split=split, queries=ev['per_query']))
    result = dict(split=split, status='DEVELOPMENT' if split == 'development' else arms['multihop_rerank']['status'], protocol_hash=manifest['protocol_hash'],
                  data=dict(path=data['path'], sha256=data['sha256'], queries=len(queries), corpus=len(texts), gold_per_query=dict(Counter(len(q['gold']) for q in queries))),
                  arms=arms, union_ceiling=ev['union_ceiling'], bridges=dict(mean_per_query=float(np.mean([len(pq['bridges']) for pq in ev['per_query']])),
                  no_bridge_queries=int(sum(1 for pq in ev['per_query'] if not pq['bridges']))), n_queries=len(queries),
                  timing=dict(encode_seconds=encode_seconds, retrieval_seconds=retrieval_seconds, per_query_seconds=retrieval_seconds / max(1, len(queries))),
                  encoder=dict(name=getattr(encoder, 'name', None), encoded=getattr(encoder, 'encoded', None)), reranker=dict(name=getattr(reranker, 'name', None), scored_pairs=getattr(reranker, 'scored', None)),
                  per_query_file=per_query_path.name, per_query_sha256=rt.sha(per_query_path), targets=p['targets'], turkish_slice='UNMEASURED')
    rt.write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--development', action='store_true'); group.add_argument('--final', action='store_true')
    parser.add_argument('--out', default='results/B2'); parser.add_argument('--protocol-json')
    args = parser.parse_args()
    overrides = json.loads(args.protocol_json) if args.protocol_json else {}
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out, dict(protocol(), **overrides))['protocol_hash'])
    else:
        r = run(args.out, 'development' if args.development else 'final')
        result = dict(status=r['status'], arms={a: {k: round(v, 3) for k, v in e['metrics'].items() if not k.endswith('_ci95')} for a, e in r['arms'].items()}, union_ceiling=r['union_ceiling'], bridges=r['bridges'])
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
