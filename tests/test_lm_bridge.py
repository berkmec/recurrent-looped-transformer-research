"""B3 (LM bridge extractor) contract, written before the implementation.
The LM names the entity that links hop 1 to hop 2; parsing must be strict, extraction batched and cached,
and bridge quality measurable on its own (does the bridge actually occur in a gold passage hop 1 missed)."""
import unittest


class StubGenerator:
    """Echoes a canned answer per question; records batch sizes so batching can be asserted."""
    name = 'stub-gen'

    def __init__(self, answers):
        self.answers = answers; self.calls = []; self.prompts = []

    def generate(self, prompts, max_new_tokens=48):
        self.calls.append(len(prompts)); self.prompts.extend(prompts)
        out = []
        for p in prompts:
            key = next((k for k in self.answers if k in p), None)
            out.append(self.answers[key] if key else 'NONE')
        return out


class ParseTests(unittest.TestCase):
    def api(self):
        try:
            from semantic_memory import lm_bridge as lb
        except ImportError as e:
            self.fail('semantic_memory.lm_bridge is not implemented: ' + str(e))
        return lb

    def test_parse_strips_decoration_dedupes_and_caps(self):
        lb = self.api()
        raw = '1. "Beta City"\n- Beta city\n* Gamma Country\n\nGamma Country\n3) Delta\n'
        self.assertEqual(lb.parse_bridges(raw, 'Where was Alpha born?', n=3), ['Beta City', 'Gamma Country', 'Delta'])

    def test_terms_already_in_the_question_and_NONE_are_dropped(self):
        lb = self.api()
        self.assertEqual(lb.parse_bridges('Alpha\nBeta City', 'Where was Alpha born?', n=3), ['Beta City'])
        self.assertEqual(lb.parse_bridges('NONE', 'Where was Alpha born?', n=3), [])
        self.assertEqual(lb.parse_bridges('  \n\n', 'q', n=3), [])
        self.assertEqual(lb.parse_bridges('A very long sentence that is clearly an explanation rather than an entity name at all', 'q', n=3), [])


class ExtractTests(unittest.TestCase):
    def api(self):
        from semantic_memory import lm_bridge as lb
        return lb

    def test_precompute_batches_caches_and_serves_as_an_extractor(self):
        lb = self.api()
        gen = StubGenerator({'Alpha born': 'Beta City', 'Gamma located': 'Delta Region'})
        queries = [dict(id='q1', question='Where was Alpha born?'), dict(id='q2', question='Where is Gamma located?'),
                   dict(id='q3', question='Where was Alpha born?')]  # duplicate question must not cost a second call
        passages = {'q1': ['Alpha: born in Beta City'], 'q2': ['Gamma: located in Delta Region'], 'q3': ['Alpha: born in Beta City']}
        table = lb.precompute(queries, lambda q: passages[q['id']], gen, batch_size=2, max_new_tokens=48, n=3)
        self.assertEqual(table.table['Where was Alpha born?'], ['Beta City'])
        self.assertEqual(table.table['Where is Gamma located?'], ['Delta Region'])
        self.assertEqual(gen.calls, [2])  # two unique questions, one batch of 2 — the duplicate was cached
        ex = table.as_extractor()
        self.assertEqual(ex('Where was Alpha born?', ['ignored passage'], n=3), ['Beta City'])
        self.assertEqual(ex('Where was Alpha born?', [], n=1), ['Beta City'])
        self.assertEqual(ex('unseen question', [], n=3), [])
        self.assertIn('Alpha: born in Beta City', gen.prompts[0])

    def test_bridge_quality_measures_hits_in_missed_gold_passages(self):
        lb = self.api()
        # gold passages the first hop missed; a bridge is "useful" only if it occurs in one of them
        self.assertTrue(lb.bridge_hits(['Beta City'], ['Beta City is in Gamma country'])['any_hit'])
        self.assertFalse(lb.bridge_hits(['American'], ['Beta City is in Gamma country'])['any_hit'])
        self.assertEqual(lb.bridge_hits([], ['whatever'])['any_hit'], False)
        r = lb.bridge_hits(['Beta City', 'Nowhere'], ['beta city is in Gamma country'])
        self.assertEqual((r['hits'], r['bridges']), (1, 2))  # case-insensitive
        self.assertEqual(lb.bridge_hits(['Beta'], [])['status'], 'NO_MISSED_GOLD')


if __name__ == '__main__':
    unittest.main()
