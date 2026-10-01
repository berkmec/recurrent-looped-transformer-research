"""B2 runner contract (written before the implementation): freeze/verify, development and final runs on a MuSiQue-format
fixture with stub encoder/reranker, arms dense / multihop / multihop_rerank with group-bootstrap CIs, targets applied on final only."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class StubEncoder:
    name = 'stub-encoder'; encoded = 0; encode_seconds = 0.

    def encode(self, texts, batch_size=32, is_query=False):
        out = []
        for t in texts:
            v = np.array([('Alpha' in t), ('Beta' in t), ('Gamma' in t)], dtype=np.float64)
            out.append(v / (np.linalg.norm(v) or 1.))
        self.encoded += len(texts)
        return np.stack(out)


class StubReranker:
    name = 'stub-reranker'; scored = 0; score_seconds = 0.

    def score(self, query, docs, batch_size=16):
        self.scored += len(docs)
        return [2. if d.startswith('Beta:') else 1. if d.startswith('Alpha:') or d.startswith('Zulu:') else 0. for d in docs]


def fixture(path):
    rows = [dict(id='q1', question='In which country was Alpha born?', answer='Gamma',
                 paragraphs=[dict(title='Alpha', paragraph_text='born in Beta city', is_supporting=True),
                             dict(title='Beta', paragraph_text='Beta city is in Gamma country', is_supporting=True),
                             dict(title='Xray', paragraph_text='an unrelated Alpha-ish note'), dict(title='Yankee', paragraph_text='nothing here')]),
            dict(id='q2', question='Where is Gamma?', answer='here',
                 paragraphs=[dict(title='Zulu', paragraph_text='Gamma mention', is_supporting=True), dict(title='Yankee', paragraph_text='nothing here')])]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf8')


class RunB2Tests(unittest.TestCase):
    def api(self):
        try:
            from semantic_memory import run_b2 as rb
        except ImportError as e:
            self.fail('semantic_memory.run_b2 is not implemented: ' + str(e))
        return rb

    def test_freeze_development_final(self):
        rb = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'dev.jsonl'; fixture(data)
            p = dict(rb.protocol(), development_path=str(data), final_path=str(data), development_queries=2, final_queries=2, ks=[3], top_k=3, k1=2, branches=2, k2=3,
                     rerank_depth=3, bootstrap=50, device='cpu')
            out = root / 'B2'
            m = rb.freeze(out, p)
            self.assertEqual(rb.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            dev = rb.run(out, 'development', encoder=StubEncoder(), reranker=StubReranker())
            self.assertEqual(dev['status'], 'DEVELOPMENT'); self.assertEqual(dev['n_queries'], 2)
            self.assertEqual(set(dev['arms']), {'dense', 'multihop', 'multihop_rerank'})
            self.assertAlmostEqual(dev['arms']['dense']['metrics']['coverage@3'], 0.5)
            self.assertAlmostEqual(dev['arms']['multihop']['metrics']['coverage@3'], 1.0)
            self.assertIn('coverage@3_ci95', dev['arms']['multihop']['metrics'])
            self.assertAlmostEqual(dev['union_ceiling']['all_gold_in_union'], 1.0)
            self.assertTrue((out / 'development.json').exists() and (out / 'development-per-query.json').exists())
            fin = rb.run(out, 'final', encoder=StubEncoder(), reranker=StubReranker())
            self.assertIn(fin['status'], ('MEASURED_POSITIVE', 'MEASURED_NEGATIVE'))
            self.assertIn('gates', fin['arms']['multihop_rerank'])
            self.assertEqual(rb.run(out, 'final')['status'], fin['status'])  # cached, no models needed
            with self.assertRaises(ValueError):
                rb.freeze(out, dict(p, k1=5))


if __name__ == '__main__':
    unittest.main()
