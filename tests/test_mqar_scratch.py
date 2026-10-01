"""A small Core trained from scratch on multi-query associative recall (MQAR) rows only.

Its purpose is a question the 30M fine-tunes cannot answer: is key-addressed retrieval something this architecture can learn
at all, and does the single-fact needle task it was first taught make that harder? The runner must therefore (1) build the
sealed Core at any width without touching it, (2) draw a batch from (seed, step) alone so a resume is the same run, and
(3) score the key use with the very probe the 30M runs were scored with.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from core.model import Core
from mline import checkpoint as ck
from mline import decoy_train as dt
from mline import mqar_scratch as ms


def tiny(out, steps, **overrides):
    kwargs = dict(out=out, width=16, heads=2, feedback=False, steps=steps, batch=2, lr=1e-3, warmup=2, seed=3, facts=2,
                  queries=2, key_length=(1, 2), alphabet=b'ABCDEFG', filler=(0, 8), pad_to=128, eval_every=2, eval_items=4,
                  eval_configs=[(0, 0)], eval_alphabets=('TRAIN_KEYS',), eval_key_lengths=(1,), verbose=False)
    kwargs.update(overrides)
    return ms.train(**kwargs)


class BuildTests(unittest.TestCase):
    def test_a_small_core_has_the_asked_size_and_a_byte_vocabulary(self):
        model = ms.build_small(width=32, heads=2, feedback=False, seed=1)
        self.assertIsInstance(model, Core)
        self.assertEqual((model.width, model.heads, model.window), (32, 2, 4))
        self.assertEqual(model.embedding.num_embeddings, 256)
        self.assertEqual(model.read.out_features, 256)
        self.assertFalse(model.feedback)
        self.assertTrue(ms.build_small(width=32, heads=2, feedback=True, seed=1).feedback)

    def test_the_seed_fixes_the_weights(self):
        a = ms.build_small(32, 2, False, 1).state_dict()
        b = ms.build_small(32, 2, False, 1).state_dict()
        c = ms.build_small(32, 2, False, 2).state_dict()
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(all(torch.equal(a[k], c[k]) for k in a))

    def test_it_maps_bytes_to_byte_logits(self):
        model = ms.build_small(16, 2, False, 1)
        x = torch.randint(0, 256, (2, 40))
        self.assertEqual(tuple(model(x).shape), (2, 40, 256))


class BatchTests(unittest.TestCase):
    KW = dict(facts=3, queries=3, key_length=(1, 3), alphabet=b'ABCDEFG', filler=(0, 10), pad_to=200)

    def test_a_batch_depends_only_on_seed_and_step(self):
        x1, y1 = ms.draw(5, seed=7, step=3, **self.KW)
        x2, y2 = ms.draw(5, seed=7, step=3, **self.KW)
        x3, _ = ms.draw(5, seed=7, step=4, **self.KW)
        x4, _ = ms.draw(5, seed=8, step=3, **self.KW)
        self.assertTrue(torch.equal(x1, x2) and torch.equal(y1, y2))
        self.assertFalse(torch.equal(x1, x3))
        self.assertFalse(torch.equal(x1, x4))

    def test_rows_are_the_padded_width_next_byte_pairs_with_mqar_content(self):
        x, y = ms.draw(4, seed=1, step=1, **self.KW)
        self.assertEqual(tuple(x.shape), (4, 200))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        for row in x:
            text = bytes(row.tolist()).decode('ascii')
            self.assertEqual(text.count('The code for'), 3 + 3)      # three facts, then three questions

    def test_the_keys_come_only_from_the_given_alphabet(self):
        import re
        x, _ = ms.draw(6, seed=2, step=1, **dict(self.KW, alphabet=b'XYZ'))
        for row in x:
            for key in re.findall(r'The code for ([A-Z]+) is', bytes(row.tolist()).decode('ascii')):
                self.assertTrue(set(key) <= set('XYZ'))


class SummaryTests(unittest.TestCase):
    ROWS = [dict(alphabet=al, key_length=kl, a=0, b=0, ask=ask, gap_to_question=27, decoy_to_question=0, exact_match=.5,
                 usable=8, asked={'first': 1.0, 'decoy': .25}[ask], decoy=.5, neither=.25)
            for al in ('TRAIN_KEYS', 'HELD_KEYS') for kl in (1, 4) for ask in ('first', 'decoy')]

    def test_key_use_is_the_asked_rate_when_the_decoy_is_asked_per_alphabet_and_key_length(self):
        s = ms.summarize(self.ROWS)
        self.assertEqual(set(s), {'TRAIN_KEYS/1', 'TRAIN_KEYS/4', 'HELD_KEYS/1', 'HELD_KEYS/4'})
        self.assertAlmostEqual(s['HELD_KEYS/4']['decoy'], .25)
        self.assertAlmostEqual(s['HELD_KEYS/4']['first'], 1.0)

    def test_it_averages_over_layouts(self):
        rows = [dict(r) for r in self.ROWS if r['alphabet'] == 'TRAIN_KEYS' and r['key_length'] == 1]
        rows += [dict(r, a=10, asked=0.75 if r['ask'] == 'decoy' else 1.0) for r in rows]
        self.assertAlmostEqual(ms.summarize(rows)['TRAIN_KEYS/1']['decoy'], .5)


class TrainTests(unittest.TestCase):
    def test_it_logs_every_step_and_checkpoints_the_last_one(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=4)
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2, 3, 4])
            self.assertTrue(all(r['loss'] is not None and r['loss'] > 0 for r in rows))
            self.assertIn('eval', rows[1])                      # eval_every=2
            self.assertNotIn('eval', rows[0])
            self.assertEqual(ck.load(Path(d, ck.LATEST))['step'], 4)
            header = json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))
            self.assertEqual((header['width'], header['facts'], header['queries']), (16, 2, 2))

    def test_the_eval_rows_carry_the_summary_the_run_is_judged_by(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=2)
            row = json.loads(Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()[1])
            self.assertEqual(set(row['eval']['summary']), {'TRAIN_KEYS/1'})
            self.assertEqual(set(row['eval']['summary']['TRAIN_KEYS/1']), {'first', 'decoy'})
            self.assertGreater(len(row['eval']['rows']), 0)

    def test_a_resumed_run_is_the_same_run(self):
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            tiny(straight, steps=6)
            tiny(split, steps=4)
            tiny(split, steps=6)                               # resumes from the step-4 checkpoint
            a = ck.load(Path(straight, ck.LATEST))['model']
            b = ck.load(Path(split, ck.LATEST))['model']
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            steps = [json.loads(l)['step'] for l in Path(split, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(steps, [1, 2, 3, 4, 5, 6])        # no duplicate rows from the restart

    def test_it_starts_from_given_weights(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as cont:
            tiny(base, steps=2)
            tiny(cont, steps=1, init_from=str(Path(base, ck.LATEST)), warmup=0, lr=0.0)
            a = ck.load(Path(base, ck.LATEST))['model']
            b = ck.load(Path(cont, ck.LATEST))['model']
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))   # lr 0: the weights are exactly the ones it began from
            header = json.loads(Path(cont, 'probe.json').read_text(encoding='utf8'))
            self.assertIsNotNone(header['init_sha256'])

    def test_a_custom_batch_source_and_an_extra_eval_replace_the_mqar_defaults(self):
        calls = []

        def fixed(step):
            calls.append(step)
            return (torch.randint(32, 120, (2, 64), generator=torch.Generator().manual_seed(step)),
                    torch.randint(32, 120, (2, 64), generator=torch.Generator().manual_seed(step + 1000)))

        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=4, batch_fn=fixed, extra_eval=lambda model: dict(marker=1.5), source='custom')
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(calls, [1, 2, 3, 4])
            self.assertEqual(rows[1]['eval']['extra'], dict(marker=1.5))
            self.assertNotIn('eval', rows[0])
            self.assertEqual(json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))['source'], 'custom')

    def test_the_default_run_records_its_source_and_has_no_extra_eval(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=2)
            row = json.loads(Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()[1])
            self.assertNotIn('extra', row['eval'])
            self.assertEqual(json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))['source'], 'mqar')

    def test_with_replacement_changes_the_batch_and_is_recorded_and_the_default_is_unchanged(self):
        kw = dict(facts=3, queries=3, key_length=(1, 2), alphabet=b'ABCDEFG', filler=(0, 8), pad_to=200)
        a, _ = ms.draw(4, 1, 1, **kw)
        b, _ = ms.draw(4, 1, 1, with_replacement=False, **kw)
        c, _ = ms.draw(4, 1, 1, with_replacement=True, **kw)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=2, with_replacement=True)
            self.assertTrue(json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))['with_replacement'])

    def test_the_pangram_filler_changes_the_batch_is_recorded_and_the_default_is_unchanged(self):
        kw = dict(facts=3, queries=3, key_length=(1, 2), alphabet=b'ABCDEFG', filler=(20, 30), pad_to=200)
        a, _ = ms.draw(4, 1, 1, **kw)
        b, _ = ms.draw(4, 1, 1, filler_kind='repeat', **kw)
        c, _ = ms.draw(4, 1, 1, filler_kind='pangram', **kw)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertIn(b'THE QUICK', bytes(c[0].tolist()))
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=2, filler_kind='pangram')
            self.assertEqual(json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))['filler_kind'], 'pangram')

    def test_a_span_parses_and_bad_spans_are_refused(self):
        self.assertEqual(ms.parse_span('8-32'), (8, 32))
        self.assertEqual(ms.parse_span('5'), (5, 5))
        for bad in ('0-3', '9-3', 'x', '1-2-3'):
            with self.assertRaises(ValueError, msg=bad):
                ms.parse_span(bad)

    def test_the_learning_rate_warms_up_linearly_then_holds(self):
        self.assertAlmostEqual(ms.lr_at(1, 1e-3, 4), 2.5e-4)
        self.assertAlmostEqual(ms.lr_at(4, 1e-3, 4), 1e-3)
        self.assertAlmostEqual(ms.lr_at(500, 1e-3, 4), 1e-3)
        self.assertAlmostEqual(ms.lr_at(1, 1e-3, 0), 1e-3)


if __name__ == '__main__':
    unittest.main()
