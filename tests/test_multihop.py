"""B2 (package P4): multi-hop retrieval contract, written before the implementation.
query -> first dense hop -> bridge terms from the top passages -> second queries -> union (RRF) -> optional rerank -> final top-k.
The final top-k is exactly k documents; seeing more candidates internally never counts as recall."""
import unittest

import numpy as np


def encode(text):
    v = np.array([('Alpha' in text), ('Beta' in text), ('Gamma' in text)], dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.)


class StubReranker:
    name = 'stub'

    def score(self, query, docs, batch_size=16):
        return [2. if d.startswith('Beta:') else 1. if d.startswith('Alpha:') else 0. for d in docs]


class MultihopTests(unittest.TestCase):
    def api(self):
        try:
            from semantic_memory import multihop as mh
        except ImportError as e:
            self.fail('semantic_memory.multihop is not implemented: ' + str(e))
        return mh

    def index(self):
        from semantic_memory.retrieval import Index
        ids = ['d1', 'd2', 'd3', 'd4', 'd5']
        texts = ['Alpha: born in Beta city', 'Beta: Beta city is in Gamma country', 'Xray: an unrelated Alpha-ish note', 'Yankee: nothing here', 'Zulu: Gamma mention']
        dense = np.array([[1., .3, 0.], [0., 1., .5], [.2, 0., .1], [.1, .1, 0.], [0., 0., .2]])
        hashes = np.zeros((5, 4))
        p = dict(top_k=3, fusion_depth=5, rrf_k=60, rerank_depth=3, rerank_batch=4)
        return Index(ids, texts, dense, hashes, p)

    def test_bridge_terms_are_capitalised_passage_terms_not_in_the_question(self):
        mh = self.api()
        b = mh.bridge_terms('In which country was Alpha born?', ['Alpha: born in Beta city', 'Beta: Beta city is in Gamma country'], n=3)
        self.assertEqual(b[0], 'Beta'); self.assertIn('Gamma', b); self.assertNotIn('Alpha', b)

    def test_second_hop_recovers_the_bridge_document_and_final_topk_is_exactly_k(self):
        mh = self.api(); idx = self.index()
        q = 'In which country was Alpha born?'; gold = ['d1', 'd2']
        r = mh.multihop(idx, q, encode, k1=3, branches=1, k2=3, extractor=mh.bridge_terms)
        self.assertEqual(r['first'][:1], ['d1'])
        self.assertNotIn('d2', r['first'])            # single dense hop misses the bridge document
        self.assertEqual(r['bridges'], ['Beta'])
        self.assertIn('d2', r['union'])                # the second hop recovers it
        self.assertEqual(r['provenance']['d2']['hop'], 2)
        final = mh.final_ranking(q, r['union'], idx, reranker=None, depth=3, top_k=3)
        self.assertEqual(len(final), 3)
        final_rr = mh.final_ranking(q, r['union'], idx, reranker=StubReranker(), depth=3, top_k=3)
        self.assertEqual(sorted(final_rr[:2]), ['d1', 'd2'])
        from semantic_memory.retrieval import rank_metrics
        self.assertEqual(rank_metrics(r['first'], gold, ks=(3,))['coverage@3'], 0.)
        self.assertEqual(rank_metrics(final_rr, gold, ks=(3,))['coverage@3'], 1.)

    def test_evaluate_reports_dense_multihop_rerank_and_union_ceiling(self):
        mh = self.api(); idx = self.index()
        queries = [dict(id='q1', group='q1', question='In which country was Alpha born?', gold=['d1', 'd2'])]
        p = dict(ks=[3], k1=3, branches=1, k2=3, rerank_depth=3, top_k=3)
        res = mh.evaluate(idx, queries, encode, mh.bridge_terms, StubReranker(), p)
        self.assertEqual(set(res['arms']), {'dense', 'multihop', 'multihop_rerank'})
        self.assertEqual(res['arms']['dense']['coverage@3'], 0.)
        self.assertEqual(res['arms']['multihop_rerank']['coverage@3'], 1.)
        self.assertEqual(res['union_ceiling']['all_gold_in_union'], 1.)
        self.assertEqual(res['per_query'][0]['union_size'], 4)


if __name__ == '__main__':
    unittest.main()
