"""B2 (package P4): multi-hop retrieval — the first hop's passages supply bridge terms for a second query.

    question --dense k1--> first hop --bridge terms (top `branches` passages)--> "question + bridge" --dense k2--> second hops
    union = RRF(first, second...) with provenance --(optional reranker over the union head)--> final top-k (exactly k)

Seeing more candidates internally never counts as recall: every metric is computed on the final list cut at k.
`bridge_terms` is the cheap extractor (capitalised passage terms absent from the question); an LM extractor can be passed
with the same signature `extractor(question, texts, n) -> [str]`.
"""
import re

import numpy as np

from semantic_memory.retrieval import rank_metrics, rerank_order, rrf

STOP = {'the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'to', 'is', 'was', 'were', 'are', 'by', 'for', 'with', 'at', 'from', 'as', 'that', 'this',
        'it', 'its', 'he', 'she', 'they', 'his', 'her', 'their', 'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'did', 'does', 'do',
        'has', 'have', 'had', 'be', 'been', 'not', 'no', 'but', 'than', 'then', 'there', 'these', 'those', 'into', 'also', 'after', 'before'}


def bridge_terms(question, texts, n=3):
    """Capitalised terms of the passages (in order of appearance) that do not occur in the question; de-duplicated, at most n."""
    qwords = {w.lower() for w in re.findall(r"[A-Za-z][\w'-]*", question)}
    out = []
    for text in texts:
        for tok in re.findall(r"\b[A-Z][\w'-]*", text):
            low = tok.lower()
            if low in qwords or low in STOP or tok in out:
                continue
            out.append(tok)
            if len(out) >= n:
                return out
    return out


def multihop(index, question, encode_query, k1=10, branches=3, k2=10, extractor=bridge_terms, rrf_k=60):
    q0 = np.asarray(encode_query(question))
    first = index.top(index.dense @ q0, k1)
    passages = [index.texts[index.position[d]] for d in first[:branches]]
    bridges = list(extractor(question, passages, n=branches))
    second = []
    for b in bridges:
        qb = np.asarray(encode_query(f'{question} {b}'))
        second.append(index.top(index.dense @ qb, k2))
    union = rrf([first] + second, rrf_k)
    provenance = {d: dict(hop=1, branch=None) for d in first}
    for i, ranked in enumerate(second):
        for d in ranked:
            provenance.setdefault(d, dict(hop=2, branch=bridges[i]))
    return dict(first=first, bridges=bridges, second=second, union=union, provenance=provenance)


def final_ranking(question, union, index, reranker=None, depth=50, top_k=10, batch_size=16):
    head = list(union[:depth]); tail = list(union[depth:])
    if reranker is not None and head:
        order = rerank_order(question, [index.texts[index.position[d]] for d in head], reranker, batch_size)
        head = [head[i] for i in order]
    return (head + tail)[:top_k]


def evaluate(index, queries, encode_query, extractor, reranker, p):
    """Per-query metrics for arms dense / multihop / multihop_rerank on the final top-k, plus the union ceiling."""
    ks = list(p['ks']); top_k = p['top_k']
    arms = {'dense': [], 'multihop': [], 'multihop_rerank': []}; per_query = []; ceiling = []; sizes = []
    for q in queries:
        r = multihop(index, q['question'], encode_query, k1=p['k1'], branches=p['branches'], k2=p['k2'], extractor=extractor, rrf_k=p.get('rrf_k', 60))
        lists = dict(dense=r['first'][:top_k], multihop=r['union'][:top_k],
                     multihop_rerank=final_ranking(q['question'], r['union'], index, reranker, p['rerank_depth'], top_k, p.get('rerank_batch', 16)))
        gold = set(q['gold'])
        for arm, ranked in lists.items():
            arms[arm].append(rank_metrics(ranked, q['gold'], ks))
        in_union = 1. if gold and gold <= set(r['union']) else 0.
        ceiling.append(in_union); sizes.append(len(r['union']))
        per_query.append(dict(id=q['id'], gold=q['gold'], bridges=r['bridges'], union_size=len(r['union']), all_gold_in_union=in_union,
                              ranked={arm: ranked for arm, ranked in lists.items()}, provenance={d: r['provenance'][d] for d in lists['multihop_rerank'] if d in r['provenance']}))
    summary = {arm: {key: float(np.mean([m[key] for m in rows])) for key in rows[0]} for arm, rows in arms.items() if rows}
    return dict(arms=summary, union_ceiling=dict(all_gold_in_union=float(np.mean(ceiling)) if ceiling else None, mean_union_size=float(np.mean(sizes)) if sizes else None),
                per_query=per_query, n_queries=len(queries))
