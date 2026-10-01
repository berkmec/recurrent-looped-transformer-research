"""B3 runner contract (written before the implementation): freeze, LM-bridge precompute, the same three arms as B2,
the union ceiling as the primary reported quantity, and bridge quality measured independently of retrieval gains."""
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_run_b2 import StubEncoder, StubReranker, fixture


class StubGen:
    name = 'stub-gen'

    def __init__(self):
        self.calls = []

    def generate(self, prompts, max_new_tokens=48):
        self.calls.append(len(prompts))
        out = []
        for p in prompts:
            out.append('Beta' if 'Alpha' in p else 'Zulu')
        return out

    def stats(self):
        return dict(name=self.name, calls=len(self.calls))


class RunB3Tests(unittest.TestCase):
    def api(self):
        try:
            from semantic_memory import run_b3 as rb
        except ImportError as e:
            self.fail('semantic_memory.run_b3 is not implemented: ' + str(e))
        return rb

    def test_freeze_development_and_bridge_quality(self):
        rb = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'dev.jsonl'; fixture(data)
            p = dict(rb.protocol(), development_path=str(data), final_path=str(data), development_queries=2, final_queries=2,
                     ks=[3], top_k=3, k1=2, branches=2, k2=3, rerank_depth=3, bootstrap=50, gen_batch=2, device='cpu')
            out = root / 'B3'
            m = rb.freeze(out, p)
            self.assertEqual(rb.verify_frozen(out)['protocol_hash'], m['protocol_hash'])
            gen = StubGen()
            r = rb.run(out, 'development', encoder=StubEncoder(), reranker=StubReranker(), generator=gen)
            self.assertEqual(r['status'], 'DEVELOPMENT')
            self.assertEqual(set(r['arms']), {'dense', 'multihop', 'multihop_rerank'})
            self.assertIn('all_gold_in_union', r['union_ceiling'])
            self.assertIn('mean_bridges', r['bridges'])
            self.assertIn('any_hit_rate', r['bridge_quality'])           # measured, not derived from retrieval
            self.assertIn('queries_with_missed_gold', r['bridge_quality'])
            self.assertEqual(gen.calls, [2])                              # one batch of the two unique questions
            self.assertTrue((out / 'development-per-query.json').exists())
            pq = json.loads((out / 'development-per-query.json').read_text(encoding='utf8'))['queries']
            self.assertIn('bridges', pq[0]); self.assertIn('bridge_quality', pq[0]); self.assertIn('raw_bridge_output', pq[0])
            self.assertEqual(rb.run(out, 'development')['status'], 'DEVELOPMENT')   # cached: no models needed
            with self.assertRaises(ValueError):
                rb.freeze(out, dict(p, k1=5))

    def test_final_applies_gates(self):
        rb = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'dev.jsonl'; fixture(data)
            p = dict(rb.protocol(), development_path=str(data), final_path=str(data), development_queries=2, final_queries=2,
                     ks=[3], top_k=3, k1=2, branches=2, k2=3, rerank_depth=3, bootstrap=50, gen_batch=2, device='cpu')
            out = root / 'B3f'
            rb.freeze(out, p)
            r = rb.run(out, 'final', encoder=StubEncoder(), reranker=StubReranker(), generator=StubGen())
            self.assertIn(r['status'], ('MEASURED_POSITIVE', 'MEASURED_NEGATIVE'))
            self.assertIn('gates', r['arms']['multihop_rerank'])
            self.assertIn('ceiling_gate', r)     # the ceiling is a first-class gate in B3, not a footnote


if __name__ == '__main__':
    unittest.main()
