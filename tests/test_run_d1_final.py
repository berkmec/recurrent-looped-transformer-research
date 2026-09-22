"""E7 for the 17-hour D1 final: extraction must persist every finished batch and resume by skipping finished ids,
so a power cut costs at most one batch. Written before the implementation."""
import json
import tempfile
import unittest
from pathlib import Path


class CountingGenerator:
    def __init__(self, fail_after_batches=None):
        self.calls = 0; self.fail_after = fail_after_batches; self.generated_tokens = 0

    def generate(self, prompts, max_new_tokens=384):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError('simulated power cut')
        return [f'OUT[{p}]' for p in prompts]

    def stats(self):
        return dict(calls=self.calls)


class ResumableExtractTests(unittest.TestCase):
    def api(self):
        try:
            from language import run_d1_final as rf
        except ImportError as e:
            self.fail('language.run_d1_final is not implemented: ' + str(e))
        return rf

    def setUp(self):
        from language import extract as ex
        self._build, self._parse, self._finite = ex.build_prompt, ex.parse_output, ex.finite_parser
        ex.build_prompt = lambda e, arm: e['question']
        ex.parse_output = lambda o, pids: dict(raw=o)
        self.examples = [dict(id=f'e{i}', question=f'q{i}', context=[]) for i in range(7)]

    def tearDown(self):
        from language import extract as ex
        ex.build_prompt, ex.parse_output, ex.finite_parser = self._build, self._parse, self._finite

    def test_progress_is_persisted_per_batch_and_resumed_by_id(self):
        rf = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / 'final-lm_zero-progress.jsonl'
            broken = CountingGenerator(fail_after_batches=1)
            with self.assertRaises(RuntimeError):
                rf.extract_resumable(self.examples, broken, 'lm_zero', batch_size=3, max_new_tokens=8, progress_path=progress, run_key='k1')
            rows = [json.loads(l) for l in progress.read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['id'] for r in rows], ['e0', 'e1', 'e2'])   # exactly the first batch survived
            good = CountingGenerator()
            preds = rf.extract_resumable(self.examples, good, 'lm_zero', batch_size=3, max_new_tokens=8, progress_path=progress, run_key='k1')
            self.assertEqual(good.calls, 2)                                  # only the remaining 4 examples were generated (2 batches)
            self.assertEqual([p['raw'] for p in preds], [f'OUT[q{i}]' for i in range(7)])
            self.assertTrue(all(p['resumed'] == (i < 3) for i, p in enumerate(preds)))
            again = CountingGenerator()
            third = rf.extract_resumable(self.examples, again, 'lm_zero', batch_size=3, max_new_tokens=8, progress_path=progress, run_key='k1')
            strip = lambda rows: [{k: v for k, v in p.items() if k != 'resumed'} for p in rows]
            self.assertEqual(strip(third), strip(preds)); self.assertTrue(all(p['resumed'] for p in third))
            self.assertEqual(again.calls, 0)

    def test_progress_from_another_run_key_is_not_reused(self):
        rf = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / 'p.jsonl'
            rf.extract_resumable(self.examples[:3], CountingGenerator(), 'lm_zero', batch_size=3, max_new_tokens=8, progress_path=progress, run_key='k1')
            fresh = CountingGenerator()
            rf.extract_resumable(self.examples[:3], fresh, 'lm_zero', batch_size=3, max_new_tokens=8, progress_path=progress, run_key='k2')
            self.assertEqual(fresh.calls, 1)  # different protocol/run key: regenerated, not resumed

    def test_finite_parser_arm_does_not_touch_the_generator(self):
        rf = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            gen = CountingGenerator()
            from language import extract as ex
            ex.finite_parser = lambda e: dict(raw='fp')   # restored in tearDown; leaking it contaminates later tests
            preds = rf.extract_resumable(self.examples[:2], gen, 'finite_parser', batch_size=2, max_new_tokens=8, progress_path=Path(tmp) / 'fp.jsonl', run_key='k')
            self.assertEqual(gen.calls, 0); self.assertEqual(len(preds), 2)


class RunnerResumeTests(unittest.TestCase):
    def test_final_survives_a_crash_mid_arm_and_writes_a_receipt(self):
        from language import run_d1_final as rf, dataset as ds
        from tests.test_semantic_contract import fake_rows, write_rows, StubGenerator

        class Flaky(StubGenerator):
            def __init__(self, gold, fail_after):
                super().__init__(gold); self.calls = 0; self.fail_after = fail_after

            def generate(self, prompts, max_new_tokens=256):
                self.calls += 1
                if self.calls > self.fail_after:
                    raise RuntimeError('simulated power cut')
                return super().generate(prompts, max_new_tokens)

        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / 'data'; data.mkdir(); write_rows(data / 'dev.jsonl', fake_rows()); write_rows(data / 'train.jsonl', fake_rows())
            out = Path(d) / 'D1-final'
            p = rf.protocol(); p.update(development_path=str(data / 'train.jsonl'), final_path=str(data / 'dev.jsonl'), final_n=4, distractors=2, batch_size=2, controls_n=2, bootstrap=20, model_dir='stub')
            m = rf.freeze(out, p)
            examples = ds.build_examples(data / 'dev.jsonl', distractors=2, seed=p['seed'], limit=4)
            gold = {e['id']: [dict(step=h['step'], relation=h['relation'], answer=h['answer'], source=h['source'], depends_on=h['depends_on']) for h in e['gold']['hops']] for e in examples}
            with self.assertRaises(RuntimeError):
                rf.run_final(out, generator=Flaky(gold, fail_after=1))   # finite_parser done, first lm_zero batch persisted, then crash
            self.assertTrue((out / 'final-lm_zero-progress.jsonl').exists()); self.assertFalse((out / 'final.json').exists())
            result = rf.run_final(out, generator=StubGenerator(gold))
            self.assertEqual(set(result['arms']), {'finite_parser', 'lm_zero', 'lm_schema'})
            self.assertEqual(result['resumed_rows']['lm_zero'], 2); self.assertEqual(result['resumed_rows']['lm_schema'], 0)
            self.assertEqual(result['arms']['lm_schema']['metrics']['examples'], 4)
            receipt = json.loads((out / 'receipt.json').read_text(encoding='utf8'))
            self.assertEqual(receipt['status'], 'COMPLETE'); self.assertEqual(receipt['expected_items'], 3)
            self.assertEqual(rf.run_final(out)['protocol_hash'], m['protocol_hash'])  # cached; no generator needed
            with self.assertRaisesRegex(ValueError, 'protocol'):
                rf.freeze(out, dict(p, final_n=5))


if __name__ == '__main__':
    unittest.main()
