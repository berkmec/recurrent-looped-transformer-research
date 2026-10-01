"""B1: lexical versus semantic retrieval on MuSiQue with exact cosine search.

Arms: hash128 (Phase 1 signed hashed bag-of-words, read-only import), bm25 (own implementation),
dense (Qwen3-Embedding-0.6B), hybrid (reciprocal rank fusion of bm25 and dense), hybrid_rerank
(Qwen3-Reranker-0.6B over the hybrid top-`rerank_depth`). Gold supporting paragraphs are never
shown to any arm; retrieval metrics are computed from saved rankings.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from memory.api import embed as phase1_embed

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('semantic_memory/retrieval.py', 'semantic_memory/embedding.py', 'semantic_memory/rerank.py', 'memory/api.py')
ARMS = ('hash128', 'bm25', 'dense', 'hybrid', 'hybrid_rerank')
TOKEN = re.compile(r'\w+', re.UNICODE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def tokenize(text):
    return [t.lower() for t in TOKEN.findall(text)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.
        df = Counter()
        self.postings = {}
        for i, toks in enumerate(self.tokens):
            counts = Counter(toks)
            for term, c in counts.items():
                df[term] += 1
                self.postings.setdefault(term, []).append((i, c))
        n = len(docs)
        self.idf = {term: math.log((n - d + .5) / (d + .5) + 1) for term, d in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tokens), dtype=np.float32)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, c in self.postings[term]:
                denom = c + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += idf * c * (self.k1 + 1) / denom
        return out


class HashArm:
    name = 'hash128'
    dim = 128

    def encode(self, texts, batch_size=None):
        v = np.stack([phase1_embed(t) for t in texts]).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def rrf(rankings, k=60):
    score = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.) + 1. / (k + rank + 1)
    return sorted(score, key=lambda item: (-score[item], str(item)))


def rank_metrics(ranked, gold, ks=(1, 5, 10)):
    gold = set(gold); out = {}
    hits = [1 if r in gold else 0 for r in ranked]
    for k in ks:
        top = hits[:k]
        out[f'recall@{k}'] = sum(top) / len(gold) if gold else 0.
        out[f'coverage@{k}'] = 1. if sum(top) == len(gold) and gold else 0.
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(top))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
        out[f'ndcg@{k}'] = dcg / ideal if ideal else 0.
    out['mrr'] = next((1. / (i + 1) for i, h in enumerate(hits) if h), 0.)
    return out


def batch_vs_single(encoder, texts, batch_size=32):
    batched = encoder.encode(texts, batch_size=batch_size)
    single = np.stack([encoder.encode([t], batch_size=1)[0] for t in texts])
    cos = np.sum(batched * single, axis=1) / (np.linalg.norm(batched, axis=1) * np.linalg.norm(single, axis=1) + 1e-12)
    return dict(n=len(texts), max_abs_cosine_diff=float(np.max(np.abs(1. - cos))), mean_abs_cosine_diff=float(np.mean(np.abs(1. - cos))))


def rerank_order(query, docs, reranker, batch_size=16):
    scores = reranker.score(query, docs, batch_size=batch_size)
    return sorted(range(len(docs)), key=lambda i: (-scores[i], i))


def load_musique(path, limit=None):
    corpus, order, queries = {}, [], []
    with open(path, encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            gold = []
            for para in row['paragraphs']:
                text = f"{para.get('title', '')}: {para['paragraph_text']}".strip()
                doc_id = 'p' + hashlib.sha256(text.encode('utf8')).hexdigest()[:16]
                if doc_id not in corpus:
                    corpus[doc_id] = text; order.append(doc_id)
                if para.get('is_supporting'):
                    gold.append(doc_id)
            queries.append(dict(id=row['id'], group=row['id'], question=row['question'], gold=sorted(set(gold)), answer=row.get('answer')))
            if limit and len(queries) >= limit:
                break
    return dict(queries=queries, corpus=corpus, order=order, path=str(path), sha256=sha(path))


def load_pool(path, exclude=None):
    """Distractor paragraphs from another split (deduplicated, minus any id in `exclude`)."""
    corpus, order = {}, []
    exclude = exclude or set()
    with open(path, encoding='utf8') as f:
        for line in f:
            for para in json.loads(line)['paragraphs']:
                text = f"{para.get('title', '')}: {para['paragraph_text']}".strip()
                doc_id = 'p' + hashlib.sha256(text.encode('utf8')).hexdigest()[:16]
                if doc_id not in corpus and doc_id not in exclude:
                    corpus[doc_id] = text; order.append(doc_id)
    return dict(corpus=corpus, order=order, path=str(path), sha256=sha(path))


def corpus_subset(data, queries, size, seed, extra=None):
    """Gold of the evaluated queries plus seeded distractors from the split pool, then from `extra` if needed."""
    gold = sorted({g for q in queries for g in q['gold']})
    gold_set = set(gold)
    rest = [d for d in data['order'] if d not in gold_set]
    rng = np.random.default_rng(seed)
    take = max(0, min(size - len(gold), len(rest)))
    chosen = [rest[i] for i in sorted(rng.choice(len(rest), size=take, replace=False))] if take else []
    if extra is not None and len(gold) + len(chosen) < size:
        pool = [d for d in extra['order'] if d not in gold_set and d not in data['corpus']]
        more = max(0, min(size - len(gold) - len(chosen), len(pool)))
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), size=more, replace=False))] if more else []
    return sorted(gold + chosen)


def protocol():
    return dict(version=1, purpose='B1 lexical versus semantic retrieval on MuSiQue with exact cosine search',
                development_path='__RLT_ROOT__/datasets/dgslibisey__MuSiQue/musique_ans_v1.0_train.jsonl', development_queries=300,
                final_path='__RLT_ROOT__/datasets/dgslibisey__MuSiQue/musique_ans_v1.0_dev.jsonl', final_queries=None,
                embedding_model='__RLT_ROOT__/models/Qwen3-Embedding-0.6B', reranker_model='__RLT_ROOT__/models/Qwen3-Reranker-0.6B',
                arms=list(ARMS), ks=[1, 5, 10], rrf_k=60, fusion_depth=100, rerank_depth=50, top_k=100,
                dense_max_length=512, dense_instruction='Given a web search query, retrieve relevant passages that answer the query',
                rerank_max_length=1024, encode_batch=32, rerank_batch=16,
                distractor_path='__RLT_ROOT__/datasets/dgslibisey__MuSiQue/musique_ans_v1.0_train.jsonl',
                ladder_sizes=[1000, 10000, 100000], ladder_queries=250, ladder_seed=2026091801, batch_check_docs=64,
                bootstrap=1000, bootstrap_seed=2026091802, device='cuda', dtype='float16',
                targets=dict(recall_at_10=.95, coverage_at_10=.90),
                note='gold supporting paragraphs are labels only; Turkish slice UNMEASURED (MuSiQue is English)')


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    return manifest


class Index:
    """All arms over one corpus subset; dense/hash vectors are sliced from precomputed matrices."""

    def __init__(self, doc_ids, texts, dense_vectors, hash_vectors, p):
        self.doc_ids = list(doc_ids); self.texts = list(texts); self.p = p
        start = time.perf_counter(); self.bm25 = BM25(self.texts); self.bm25_seconds = time.perf_counter() - start
        self.dense = dense_vectors; self.hash = hash_vectors
        self.position = {d: i for i, d in enumerate(self.doc_ids)}

    def top(self, scores, k):
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind='stable')]
        return [self.doc_ids[i] for i in idx]

    def retrieve(self, query, query_dense, query_hash, reranker):
        p = self.p; k = p['top_k']; out = {}; timing = {}
        t = time.perf_counter(); out['hash128'] = self.top(self.hash @ query_hash, k); timing['hash128'] = time.perf_counter() - t
        t = time.perf_counter(); out['bm25'] = self.top(self.bm25.scores(query), k); timing['bm25'] = time.perf_counter() - t
        t = time.perf_counter(); out['dense'] = self.top(self.dense @ query_dense, k); timing['dense'] = time.perf_counter() - t
        t = time.perf_counter(); out['hybrid'] = rrf([out['bm25'][:p['fusion_depth']], out['dense'][:p['fusion_depth']]], p['rrf_k'])[:k]; timing['hybrid'] = timing['bm25'] + timing['dense'] + time.perf_counter() - t
        t = time.perf_counter()
        head = out['hybrid'][:p['rerank_depth']]
        order = rerank_order(query, [self.texts[self.position[d]] for d in head], reranker, p['rerank_batch']) if reranker is not None else list(range(len(head)))
        out['hybrid_rerank'] = [head[i] for i in order] + out['hybrid'][p['rerank_depth']:]
        timing['hybrid_rerank'] = timing['hybrid'] + time.perf_counter() - t
        return out, timing


def bootstrap_mean(values, groups, n, seed):
    values = np.asarray(values, dtype=np.float64); groups = np.asarray(groups)
    unique = np.unique(groups); per_group = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    means = [per_group[rng.integers(0, len(unique), len(unique))].mean() for _ in range(n)]
    return dict(mean=float(values.mean()), groups=int(len(unique)), ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))])


def evaluate_arms(index, queries, query_dense, query_hash, reranker, p, out, tag):
    ks = p['ks']; per_arm = {arm: [] for arm in p['arms']}; latency = {arm: [] for arm in p['arms']}
    rankings = {arm: [] for arm in p['arms']}
    for i, q in enumerate(queries):
        ranked, timing = index.retrieve(q['question'], query_dense[i], query_hash[i], reranker)
        for arm in p['arms']:
            per_arm[arm].append(rank_metrics(ranked[arm], q['gold'], ks)); latency[arm].append(timing[arm]); rankings[arm].append(ranked[arm][:max(ks + [p['rerank_depth']])])
    result = {}
    groups = [q['group'] for q in queries]
    for arm in p['arms']:
        metrics = {}
        for key in per_arm[arm][0]:
            metrics[key] = bootstrap_mean([m[key] for m in per_arm[arm]], groups, p['bootstrap'], p['bootstrap_seed'])['mean']
        for key in (f"recall@{max(ks)}", f"coverage@{max(ks)}"):
            metrics[key + '_ci95'] = bootstrap_mean([m[key] for m in per_arm[arm]], groups, p['bootstrap'], p['bootstrap_seed'])['ci95']
        raw = Path(out) / f'{tag}-{arm}.json'
        write_json(raw, dict(arm=arm, tag=tag, doc_ids=index.doc_ids if len(index.doc_ids) <= 20000 else None, corpus_size=len(index.doc_ids),
                             queries=[dict(id=q['id'], gold=q['gold'], ranked=r) for q, r in zip(queries, rankings[arm])]))
        lat = np.array(latency[arm])
        result[arm] = dict(n_queries=len(queries), corpus_size=len(index.doc_ids), metrics=metrics, raw_file=raw.name, raw_sha256=sha(raw),
                           latency_seconds=dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)), mean=float(lat.mean())))
    return result


def make_encoder(p):
    from semantic_memory.embedding import QwenEncoder
    return QwenEncoder(p['embedding_model'], device=p['device'], max_length=p['dense_max_length'], instruction=p['dense_instruction'])


def make_reranker(p):
    from semantic_memory.rerank import QwenReranker
    return QwenReranker(p['reranker_model'], device=p['device'], max_length=p['rerank_max_length'])


def run_final(out, encoder=None, reranker=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if (out / 'final.json').exists():
        return json.loads((out / 'final.json').read_text(encoding='utf8'))
    data = load_musique(p['final_path'], p['final_queries'])
    queries = data['queries']
    encoder = encoder or make_encoder(p); reranker = reranker if reranker is not None else make_reranker(p)
    texts = [data['corpus'][d] for d in data['order']]
    t0 = time.perf_counter(); dense = encoder.encode(texts, batch_size=p['encode_batch']); dense_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); hashes = HashArm().encode(texts); hash_seconds = time.perf_counter() - t0
    qdense = encoder.encode([q['question'] for q in queries], batch_size=p['encode_batch'], is_query=True)
    qhash = HashArm().encode([q['question'] for q in queries])
    check = batch_vs_single(encoder, texts[:p['batch_check_docs']], batch_size=p['encode_batch'])
    full = Index(data['order'], texts, dense, hashes, p)
    arms = evaluate_arms(full, queries, qdense, qhash, reranker, p, out, 'final')
    ladder = {}
    rng = np.random.default_rng(p['ladder_seed']); pick = sorted(rng.choice(len(queries), size=min(p['ladder_queries'], len(queries)), replace=False))
    lq = [queries[i] for i in pick]
    extra = load_pool(p['distractor_path'], exclude=set(data['corpus'])) if p.get('distractor_path') and max(p['ladder_sizes']) > len(texts) else None
    extra_vectors = {}
    for size in p['ladder_sizes']:
        ids = corpus_subset(data, lq, size, p['ladder_seed'] + size, extra)
        missing = [d for d in ids if d not in full.position and d not in extra_vectors]
        if missing:
            t0 = time.perf_counter(); vec = encoder.encode([extra['corpus'][d] for d in missing], batch_size=p['encode_batch']); hv = HashArm().encode([extra['corpus'][d] for d in missing])
            extra_vectors.update({d: (vec[i], hv[i]) for i, d in enumerate(missing)}); dense_seconds += time.perf_counter() - t0
        sub_texts = [texts[full.position[d]] if d in full.position else extra['corpus'][d] for d in ids]
        sub_dense = np.stack([dense[full.position[d]] if d in full.position else extra_vectors[d][0] for d in ids])
        sub_hash = np.stack([hashes[full.position[d]] if d in full.position else extra_vectors[d][1] for d in ids])
        sub = Index(ids, sub_texts, sub_dense, sub_hash, p)
        ladder[str(size)] = evaluate_arms(sub, lq, qdense[pick], qhash[pick], reranker, p, out, f'ladder{size}')
        ladder[str(size)]['_from_extra_pool'] = sum(1 for d in ids if d not in full.position)
    ladder['full'] = evaluate_arms(full, lq, qdense[pick], qhash[pick], reranker, p, out, 'ladderfull')
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], data=dict(path=data['path'], sha256=data['sha256'], queries=len(queries), corpus=len(texts),
                  gold_per_query=Counter(len(q['gold']) for q in queries)), index=dict(dense_encode_seconds=dense_seconds, hash_seconds=hash_seconds,
                  bm25_seconds=full.bm25_seconds, dense_dim=int(dense.shape[1]), vram=getattr(encoder, 'vram', lambda: None)()),
                  batch_vs_single=check, arms=arms, ladder=ladder, ladder_queries=len(lq), distractor_pool=None if extra is None else dict(path=extra['path'], sha256=extra['sha256'], size=len(extra['order'])),
                  reranker=dict(name=getattr(reranker, 'name', None), scored_pairs=getattr(reranker, 'scored', None), seconds=getattr(reranker, 'score_seconds', None)),
                  encoder=dict(name=getattr(encoder, 'name', None), encoded=getattr(encoder, 'encoded', None), seconds=getattr(encoder, 'encode_seconds', None)),
                  targets=p['targets'], turkish_slice='UNMEASURED')
    for arm, row in arms.items():
        row['target_recall'] = row['metrics']['recall@10'] >= p['targets']['recall_at_10']
        row['target_coverage'] = row['metrics']['coverage@10'] >= p['targets']['coverage_at_10']
    write_json(out / 'final.json', result)
    return result


def run_development(out, p, encoder_factory, reranker):
    """Configuration selection on the development split only; the choice is written before any final freeze."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = load_musique(p['development_path'], p['development_queries']); queries = data['queries']
    texts = [data['corpus'][d] for d in data['order']]; hashes = HashArm().encode(texts); qhash = HashArm().encode([q['question'] for q in queries])
    table = []
    for max_length in p['dev_max_lengths']:
        for instruction in p['dev_instructions']:
            encoder = encoder_factory(max_length, instruction)
            dense = encoder.encode(texts, batch_size=p['encode_batch']); qdense = encoder.encode([q['question'] for q in queries], batch_size=p['encode_batch'], is_query=True)
            for depth in p['dev_rerank_depths']:
                pp = dict(p, rerank_depth=depth)
                index = Index(data['order'], texts, dense, hashes, pp)
                res = evaluate_arms(index, queries, qdense, qhash, reranker, pp, out, f'dev-L{max_length}-I{int(bool(instruction))}-D{depth}')
                table.append(dict(max_length=max_length, instruction=bool(instruction), rerank_depth=depth,
                                  dense_coverage10=res['dense']['metrics']['coverage@10'], hybrid_coverage10=res['hybrid']['metrics']['coverage@10'],
                                  rerank_coverage10=res['hybrid_rerank']['metrics']['coverage@10'], rerank_recall10=res['hybrid_rerank']['metrics']['recall@10'],
                                  rerank_p50=res['hybrid_rerank']['latency_seconds']['p50']))
    best = max(table, key=lambda r: (r['rerank_coverage10'], r['rerank_recall10'], -r['rerank_p50']))
    selection = dict(table=table, selected=best, queries=len(queries), corpus=len(texts), data_sha256=data['sha256'], role='development only; not final evidence')
    write_json(out / 'development.json', selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--final', action='store_true'); group.add_argument('--development', action='store_true')
    parser.add_argument('--out', required=True); parser.add_argument('--protocol-json', help='overrides for freeze/development as JSON')
    args = parser.parse_args()
    overrides = json.loads(args.protocol_json) if args.protocol_json else {}
    if args.freeze:
        result = freeze(args.out, dict(protocol(), **overrides)); result = dict(protocol_hash=result['protocol_hash'])
    elif args.development:
        p = dict(protocol(), dev_max_lengths=[256, 512], dev_instructions=[protocol()['dense_instruction'], None], dev_rerank_depths=[20, 50], **overrides)
        from semantic_memory.embedding import QwenEncoder
        result = run_development(args.out, p, lambda L, I: QwenEncoder(p['embedding_model'], device=p['device'], max_length=L, instruction=I), make_reranker(p))
        result = dict(selected=result['selected'], table=result['table'])
    else:
        result = run_final(args.out); result = dict(status=result['status'], arms={a: r['metrics'] for a, r in result['arms'].items()})
    print(json.dumps(result, indent=1, allow_nan=False, default=str, ensure_ascii=False))


if __name__ == '__main__':
    main()
