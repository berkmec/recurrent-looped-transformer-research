"""D1 behavioral tests (written before the implementation). Uses a fake MuSiQue file and a stub generator; no GPU."""
import json
import tempfile
from pathlib import Path
import unittest


def fake_rows():
    rows = []
    for q in range(6):
        paras = [dict(idx=i, title=f'T{q}-{i}', paragraph_text=f'Paragraph {q} {i} text about entity{q}{i}.', is_supporting=False) for i in range(6)]
        paras[1]['is_supporting'] = True; paras[4]['is_supporting'] = True
        hops = [dict(id=100 * q + 1, question=f'entity{q} >> performer', answer=f'Steve{q}', paragraph_support_idx=1),
                dict(id=100 * q + 2, question='#1 >> spouse', answer=f'Miquette{q}', paragraph_support_idx=4)]
        rows.append(dict(id=f'2hop__{q}', question=f'Who is the spouse of the entity{q} performer?', answer=f'Miquette{q}', answer_aliases=[],
                         answerable=True, paragraphs=paras, question_decomposition=hops))
    return rows


def write_rows(path, rows):
    with open(path, 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


class StubGenerator:
    """Returns JSON that copies the gold chain for 'good' questions and garbage otherwise; records prompts."""
    name = 'stub'

    def __init__(self, gold_by_id):
        self.gold = gold_by_id; self.prompts = []

    def generate(self, prompts, max_new_tokens=256):
        self.prompts.extend(prompts)
        out = []
        for p in prompts:
            qid = p.split('QUESTION_ID: ')[1].split('\n')[0]
            hops = self.gold.get(qid)
            out.append(json.dumps(dict(status='OK', hops=hops)) if hops else 'not json')
        return out


class SemanticContractTests(unittest.TestCase):
    def api(self):
        try:
            from language import dataset as ds, score_semantics as sc, extract as ex
        except ImportError as e:
            self.fail('D1 language modules are not implemented: ' + str(e))
        return ds, sc, ex

    def test_examples_hide_gold_and_keep_sources(self):
        ds, _, _ = self.api()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.jsonl'; write_rows(path, fake_rows())
            examples = ds.build_examples(path, distractors=2, seed=7)
            self.assertEqual(len(examples), 6)
            e = examples[0]
            self.assertEqual(len(e['context']), 4)  # 2 supporting + 2 distractors
            self.assertTrue(all('is_supporting' not in c for c in e['context']))
            self.assertEqual(len(e['gold']['hops']), 2)
            self.assertEqual(e['gold']['hops'][1]['depends_on'], 1)
            self.assertEqual(e['gold']['hops'][0]['relation'], 'performer')
            self.assertEqual(e['gold']['hops'][1]['relation'], 'spouse')
            self.assertTrue(all(h['source'] in {c['pid'] for c in e['context']} for h in e['gold']['hops']))
            self.assertEqual(e['group'], '2hop__0'); self.assertEqual(e['hops'], 2)

    def test_negative_controls_are_labelled_and_change_expectation(self):
        ds, _, _ = self.api()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.jsonl'; write_rows(path, fake_rows())
            examples = ds.build_examples(path, distractors=2, seed=7)
            controls = ds.negative_controls(examples[:2], seed=3)
            kinds = {c['control'] for c in controls}
            self.assertEqual(kinds, {'missing_source', 'relation_swap', 'subject_swap'})
            for c in controls:
                self.assertEqual(c['provenance'], 'programmatic_perturbation')
                self.assertTrue(c['expect_unresolved_hops'])
            missing = next(c for c in controls if c['control'] == 'missing_source')
            self.assertEqual(len(missing['context']), 3)
            self.assertTrue(set(missing['expect_unresolved_hops']) <= {1, 2})

    def test_scoring_hand_computed(self):
        _, sc, _ = self.api()
        gold = dict(hops=[dict(step=1, relation='performer', answer='Steve', aliases=['Steve H.'], source='p1', depends_on=None),
                          dict(step=2, relation='spouse', answer='Miquette', aliases=[], source='p4', depends_on=1)])
        pred = dict(status='OK', hops=[dict(step=1, relation='performer', answer='steve h.', source='p1', depends_on=None),
                                       dict(step=2, relation='child', answer='Miquette', source='p4', depends_on=1)])
        s = sc.score_example(pred, gold)
        self.assertEqual(s['relation_tp'], 1); self.assertEqual(s['relation_fp'], 1); self.assertEqual(s['relation_fn'], 1)
        self.assertEqual(s['source_correct'], 1); self.assertEqual(s['accepted_wrong'], 1); self.assertEqual(s['unresolved'], 0)
        s2 = sc.score_example(dict(status='UNRESOLVED', hops=[]), gold)
        self.assertEqual((s2['relation_tp'], s2['accepted_wrong'], s2['unresolved']), (0, 0, 2))
        agg = sc.aggregate([s, s2])
        self.assertAlmostEqual(agg['relation_f1'], 2 * 1 / (2 * 1 + 1 + 3))
        self.assertEqual(agg['accepted_hops'], 2); self.assertEqual(agg['accepted_wrong'], 1)
        self.assertAlmostEqual(agg['coverage'], 0.5)

    def test_extractor_parses_json_and_abstains_on_garbage(self):
        ds, sc, ex = self.api()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.jsonl'; write_rows(path, fake_rows())
            examples = ds.build_examples(path, distractors=2, seed=7)
            gold = {e['id']: [dict(step=h['step'], relation=h['relation'], answer=h['answer'], source=h['source'], depends_on=h['depends_on']) for h in e['gold']['hops']] for e in examples[:3]}
            gen = StubGenerator(gold)
            preds = ex.extract(examples[:5], gen, arm='lm_schema', batch_size=2)
            self.assertEqual(len(preds), 5)
            self.assertEqual(preds[0]['status'], 'OK'); self.assertEqual(preds[4]['status'], 'UNRESOLVED')
            self.assertTrue(all('QUESTION_ID: ' in p for p in gen.prompts))
            self.assertTrue(all('is_supporting' not in p and 'GOLD' not in p for p in gen.prompts))
            scores = [sc.score_example(p, e['gold']) for p, e in zip(preds, examples[:5])]
            agg = sc.aggregate(scores)
            self.assertEqual(agg['accepted_wrong'], 0); self.assertEqual(agg['relation_tp'], 6)
            floor = ex.extract(examples[:2], None, arm='finite_parser')
            self.assertTrue(all(p['status'] == 'UNRESOLVED' for p in floor))


if __name__ == '__main__':
    unittest.main()


class RunnerTests(unittest.TestCase):
    def test_runner_tiny_with_stub_and_frozen_refusal(self):
        try:
            from language import run_d1 as rd, dataset as ds
        except ImportError as e:
            self.fail('D1 runner is not implemented: ' + str(e))
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / 'data'; data.mkdir()
            write_rows(data / 'train.jsonl', fake_rows()); write_rows(data / 'dev.jsonl', fake_rows())
            out = Path(d) / 'out'
            p = rd.protocol(); p.update(development_path=str(data / 'train.jsonl'), final_path=str(data / 'dev.jsonl'),
                                        development_n=3, final_n=4, distractors=2, batch_size=2, controls_n=2, model_dir='stub', arms=['finite_parser', 'lm_zero', 'lm_schema'])
            rd.freeze(out, p)
            examples = ds.build_examples(data / 'dev.jsonl', distractors=2, seed=p['seed'], limit=4)
            gold = {e['id']: [dict(step=h['step'], relation=h['relation'], answer=h['answer'], source=h['source'], depends_on=h['depends_on']) for h in e['gold']['hops']] for e in examples}
            result = rd.run_final(out, generator=StubGenerator(gold))
            self.assertEqual(set(result['arms']), {'finite_parser', 'lm_zero', 'lm_schema'})
            self.assertEqual(result['arms']['lm_schema']['metrics']['examples'], 4)
            self.assertEqual(result['arms']['lm_schema']['metrics']['accepted_wrong'], 0)
            self.assertIn('controls', result['arms']['lm_schema'])
            self.assertTrue((out / result['arms']['lm_schema']['raw_file']).exists())
            with self.assertRaisesRegex(ValueError, 'protocol'):
                rd.freeze(out, dict(p, final_n=5))
