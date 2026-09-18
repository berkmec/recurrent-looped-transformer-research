"""B1 behavioral tests (written before the implementation). No model download or GPU is required here."""
import json
import math
import tempfile
from pathlib import Path
import unittest

import numpy as np


def fake_musique(path, n_questions=4, paragraphs=6):
    rows = []
    for q in range(n_questions):
        paras = []
        for i in range(paragraphs):
            supporting = i in (0, 1)
            paras.append(dict(idx=i, title=f'Title {q}-{i}', paragraph_text=f'fact {q} part {i} ' + ('alpha beta' if supporting else 'gamma delta ' * (i + 1)), is_supporting=supporting))
        # one shared paragraph across questions to exercise deduplication
        paras.append(dict(idx=paragraphs, title='Shared', paragraph_text='shared paragraph text', is_supporting=False))
        rows.append(dict(id=f'2hop__{q}', question=f'what is fact {q} alpha beta', answer='x', paragraphs=paras))
    with open(path, 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return rows


class StubEncoder:
    name = 'stub'
    dim = 8

    def encode(self, texts, batch_size=4, is_query=False, max_length=None):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in t.lower().split():
                v[hash(w) % self.dim] += 1.
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)


class StubReranker:
    name = 'stub-rerank'

    def score(self, query, docs, batch_size=4):
        return [float(len(set(query.lower().split()) & set(d.lower().split()))) for d in docs]


class RetrievalContractTests(unittest.TestCase):
    def api(self):
        try:
            from semantic_memory import retrieval as r
        except ImportError as e:
            self.fail('B1 retrieval module is not implemented: ' + str(e))
        return r

    def test_bm25_hand_computed(self):
        r = self.api()
        docs = ['the cat sat', 'the dog sat', 'a cat and a cat']
        index = r.BM25(docs)
        scores = index.scores('cat')
        self.assertGreater(scores[2], scores[0]); self.assertGreater(scores[0], scores[1]); self.assertEqual(scores[1], 0.)
        # idf with the standard +1 smoothing: N=3, df(cat)=2 -> ln((3-2+0.5)/(2+0.5)+1)
        self.assertAlmostEqual(index.idf['cat'], math.log((3 - 2 + .5) / (2 + .5) + 1), places=9)
        self.assertEqual(r.tokenize('Hello, World! hello'), ['hello', 'world', 'hello'])

    def test_rrf_fusion_and_metrics(self):
        r = self.api()
        fused = r.rrf([['a', 'b', 'c'], ['b', 'a', 'd']], k=60)
        self.assertEqual(fused[:2], ['a', 'b']) if fused[0] == 'a' else self.assertEqual(fused[:2], ['b', 'a'])
        self.assertEqual(set(fused), {'a', 'b', 'c', 'd'})
        m = r.rank_metrics(['x', 'y', 'z', 'w'], gold={'y', 'w'}, ks=(1, 2, 3))
        self.assertEqual(m['recall@1'], 0.); self.assertEqual(m['recall@2'], .5); self.assertEqual(m['coverage@2'], 0.); self.assertEqual(m['recall@3'], .5)
        self.assertAlmostEqual(m['mrr'], .5)
        ideal = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(m['ndcg@3'], (1 / math.log2(3)) / ideal)
        full = r.rank_metrics(['y', 'w'], gold={'y', 'w'}, ks=(2,))
        self.assertEqual((full['recall@2'], full['coverage@2'], full['ndcg@2']), (1., 1., 1.))

    def test_musique_loading_dedup_and_ladder(self):
        r = self.api()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'dev.jsonl'
            fake_musique(path)
            data = r.load_musique(path)
            self.assertEqual(len(data['queries']), 4)
            self.assertEqual(len(data['corpus']), 4 * 6 + 1)  # shared paragraph deduplicated
            for q in data['queries']:
                self.assertEqual(len(q['gold']), 2)
                self.assertTrue(all(g in data['corpus'] for g in q['gold']))
                self.assertEqual(q['group'], q['id'])
            ladder = r.corpus_subset(data, queries=data['queries'][:2], size=10, seed=3)
            self.assertEqual(len(ladder), 10)
            for q in data['queries'][:2]:
                self.assertTrue(set(q['gold']) <= set(ladder))
            self.assertEqual(ladder, r.corpus_subset(data, queries=data['queries'][:2], size=10, seed=3))
            self.assertNotEqual(ladder, r.corpus_subset(data, queries=data['queries'][:2], size=10, seed=4))
            extra_path = Path(d) / 'train.jsonl'; fake_musique(extra_path, n_questions=3, paragraphs=9)
            extra = r.load_pool(extra_path, exclude=set(data['corpus']))
            self.assertTrue(set(extra['corpus']).isdisjoint(data['corpus']))
            big = r.corpus_subset(data, queries=data['queries'][:2], size=30, seed=3, extra=extra)  # 25 dev + 9 unique extra docs available
            self.assertEqual(len(big), 30)
            self.assertTrue(any(d in extra['corpus'] for d in big))
            for q in data['queries'][:2]:
                self.assertTrue(set(q['gold']) <= set(big))

    def test_hash128_arm_uses_phase1_embedding(self):
        r = self.api()
        arm = r.HashArm()
        v = arm.encode(['alpha beta', 'gamma'])
        self.assertEqual(v.shape, (2, 128))
        self.assertTrue(np.allclose(np.linalg.norm(v, axis=1), 1.))
        from memory.api import embed
        w = embed('alpha beta'); w = w / np.linalg.norm(w)
        self.assertTrue(np.allclose(v[0], w))

    def test_batch_vs_single_and_rerank_contract(self):
        r = self.api()
        enc = StubEncoder()
        check = r.batch_vs_single(enc, ['alpha beta', 'gamma delta', 'alpha gamma'], batch_size=2)
        self.assertLess(check['max_abs_cosine_diff'], 1e-6)  # float32 renormalisation noise; identical inputs
        self.assertEqual(check['n'], 3)
        rr = StubReranker()
        order = r.rerank_order('alpha beta', ['gamma', 'alpha beta', 'alpha'], rr)
        self.assertEqual(order, [1, 2, 0])

    def test_tiny_end_to_end_with_stubs_and_frozen_refusal(self):
        r = self.api()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data' / 'dev.jsonl'; path.parent.mkdir(); fake_musique(path); fake_musique(path.parent / 'train.jsonl', n_questions=5, paragraphs=9)
            out = Path(d) / 'out'
            p = r.protocol(); p.update(final_path=str(path), development_path=str(path), development_queries=2, final_queries=4,
                                       ladder_sizes=[10, 60], ladder_queries=2, rerank_depth=5, ks=[1, 5, 10], distractor_path=str(path.parent / 'train.jsonl'))
            r.freeze(out, p)
            result = r.run_final(out, encoder=StubEncoder(), reranker=StubReranker())
            self.assertEqual(set(result['arms']), {'hash128', 'bm25', 'dense', 'hybrid', 'hybrid_rerank'})
            for arm, row in result['arms'].items():
                self.assertEqual(row['n_queries'], 4)
                self.assertIn('recall@10', row['metrics']); self.assertIn('coverage@10', row['metrics'])
                self.assertTrue((out / row['raw_file']).exists())
            self.assertIn('10', result['ladder']); self.assertIn('60', result['ladder']); self.assertGreater(result['ladder']['60']['_from_extra_pool'], 0)
            self.assertIn('batch_vs_single', result)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                r.freeze(out, dict(p, rerank_depth=6))


if __name__ == '__main__':
    unittest.main()
