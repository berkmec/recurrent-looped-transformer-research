"""Behavioral checks for fresh paired training and fail-closed continuation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('review_cycle.training'),
                             'paired runner is not implemented')
        from review_cycle import training as t, training_data as d
        return t, d

    def tiny(self):
        t, _ = self.api()
        p = t.protocol(device='cpu')
        p.update(steps=3, batch=2, train_length=4, validation_n=4,
                 validation_every=1, checkpoint_every=1, equal_budget_step=1,
                 test_n=4, test_lengths=[4, 8], neutral_slots=[0, 2],
                 small_width=8, small_heads=1, large_width=8, large_heads=1)
        return p

    def test_five_oracle_hand_computed(self):
        _, d = self.api()
        x = torch.tensor([[0, 1, 2, 0, 0, 1, 0]])
        self.assertEqual(d.prefix_targets(x, 'five').tolist(), [[1, 0, 0, 1, 2, 2, 3]])

    def test_blocking_preserves_histogram_nonneutral_order_final(self):
        _, d = self.api()
        x = torch.tensor([[0, 2, 1, 2, 0, 1]]).repeat(3, 1)
        b = d.block_neutrals(x, start_index=0)
        self.assertEqual(b.tolist(), [[2, 2, 0, 1, 0, 1], [0, 1, 2, 2, 0, 1], [0, 1, 0, 1, 2, 2]])
        for a, z in zip(x, b):
            self.assertEqual(a[a != 2].tolist(), z[z != 2].tolist())
            self.assertEqual(a.sort().values.tolist(), z.sort().values.tolist())
        self.assertTrue(torch.equal(d.prefix_targets(x, 'five')[:, -1], d.prefix_targets(b, 'five')[:, -1]))

    def test_sparse_rng_does_not_change_inputs_or_initialization(self):
        t, d = self.api()
        p = self.tiny()
        c = t.conditions(p)[0]
        s = dict(c, supervision='sparse')
        a = d.train_batch(c, p, 1)
        b = d.train_batch(s, p, 1)
        self.assertTrue(torch.equal(a['x'], b['x']))
        self.assertEqual(a['input_hash'], b['input_hash'])
        self.assertEqual(t.state_hash(t.make_model(c, p).state_dict()), t.state_hash(t.make_model(s, p).state_dict()))
        self.assertEqual(d.sparse_positions(c, p, 1).tolist(), d.sparse_positions(s, p, 1).tolist())

    def test_unique_conditions_and_label_budgets(self):
        t, d = self.api()
        p = t.protocol(device='cpu')
        cs = t.conditions(p)
        self.assertEqual(len(cs), 36)
        self.assertEqual(len({t.ident(c) for c in cs}), 36)
        self.assertEqual(d.label_count('dense', 64, 32, 32), 65536)
        self.assertEqual(d.label_count('sparse', 2048, 32, 32), 65536)
        self.assertEqual(d.label_count('final', 2048, 32, 32), 65536)

    def test_freeze_refuses_protocol_and_source_drift(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            t.freeze(tmp, p)
            changed = copy.deepcopy(p)
            changed['steps'] += 1
            with self.assertRaises(ValueError):
                t.freeze(tmp, changed)
            fp = Path(tmp) / 'frozen.json'
            manifest = json.loads(fp.read_text())
            manifest['sources']['core/model.py'] = '0' * 64
            t.write_json(fp, manifest)
            with self.assertRaises(ValueError):
                t.verify_frozen(tmp)

    def test_resume_matches_uninterrupted_real_optimizer_updates(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as aa, tempfile.TemporaryDirectory() as bb:
            p = self.tiny()
            c = t.conditions(p)[0]
            name = t.ident(c)
            t.freeze(aa, p)
            t.freeze(bb, p)
            full = t.run_one(aa, name)
            paused = t.run_one(bb, name, stop_after=1)
            self.assertEqual(paused['status'], 'PAUSED')
            resumed = t.run_one(bb, name)
            self.assertEqual(full['last_state_hash'], resumed['last_state_hash'])
            self.assertEqual(full['selected_state_hash'], resumed['selected_state_hash'])
            self.assertEqual(resumed['updates'], 3)
            self.assertEqual(resumed['labels'], 24)
            self.assertEqual(full['initial_state_hash'], resumed['initial_state_hash'])
            a = t.load_checkpoint(Path(aa) / name / 'last.pt')
            b = t.load_checkpoint(Path(bb) / name / 'last.pt')
            self.assertEqual(a['optimizer_hash'], b['optimizer_hash'])
            self.assertEqual(len(b['history']), 3)
            self.assertEqual([r['input_hash'] for r in a['history']], [r['input_hash'] for r in b['history']])

    def test_resume_rejects_incomplete_and_tampered_checkpoint(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            t.freeze(tmp, p)
            name = t.ident(t.conditions(p)[0])
            directory = Path(tmp) / name
            directory.mkdir()
            (directory / 'unexpected.json').write_text('{}')
            with self.assertRaises(ValueError):
                t.run_one(tmp, name)
        with tempfile.TemporaryDirectory() as tmp:
            t.freeze(tmp, p)
            t.run_one(tmp, name, stop_after=1)
            path = Path(tmp) / name / 'resume.pt'
            state = t.load_checkpoint(path)
            state['updates'] = 2
            torch.save(state, path)
            with self.assertRaises(ValueError):
                t.run_one(tmp, name)

    def test_final_evaluation_is_gated_and_saves_matched_raw_cases(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            t.freeze(tmp, p)
            name = t.ident(t.conditions(p)[0])
            with self.assertRaises(ValueError):
                t.evaluate(tmp, condition_ids=[name])
            t.run_one(tmp, name)
            result = t.evaluate(tmp, condition_ids=[name])
            self.assertEqual(result['status'], 'PARTIAL')
            rows = result['conditions'][name]['neutral']
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['placement'] for r in rows}, {'base', 'prefix', 'middle', 'suffix'})
            for row in rows:
                self.assertEqual(sum(row['target_histogram'].values()), 4)
                self.assertTrue((Path(tmp) / row['raw_file']).is_file())

    def test_resume_rejects_changed_selected_weights_and_batch_receipt(self):
        t, _ = self.api()
        for mutation in ('selected', 'batch'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                p = self.tiny()
                t.freeze(tmp, p)
                name = t.ident(t.conditions(p)[0])
                t.run_one(tmp, name, stop_after=1)
                path = Path(tmp) / name / 'resume.pt'
                s = t.load_checkpoint(path)
                if mutation == 'selected':
                    s['selected_model']['initial'][0] += 1
                else:
                    s['history'][0]['input_hash'] = '0' * 64
                torch.save(s, path)
                with self.assertRaises(ValueError):
                    t.run_one(tmp, name)

    def test_source_file_edit_rejects_resume_even_when_manifest_is_untouched(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as source:
            root = Path(source)
            for name in t.SOURCE_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((t.ROOT / name).read_bytes())
            with patch.object(t, 'ROOT', root):
                t.freeze(tmp, self.tiny())
                (root / 'core/model.py').write_text('changed')
                with self.assertRaises(ValueError):
                    t.verify_frozen(tmp)

    def test_lr_selection_uses_all_development_seeds_then_fresh_confirmations(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            p['steps'] = 1
            t.freeze(tmp, p)
            result = t.run_selection(tmp)
            selection = result['selection']
            self.assertEqual(len(selection['development_conditions']), 9)
            self.assertEqual({r['condition']['seed'] for r in result['confirmations']}, {217, 243, 291})
            best = min(p['lr_grid'], key=lambda lr: (-selection['mean_validation_accuracy'][str(lr)], lr))
            self.assertEqual(selection['selected_lr'], best)
            self.assertFalse((Path(tmp) / 'test-gate.json').exists())
            with self.assertRaises(ValueError):
                t.evaluate(tmp, condition_ids=[selection['development_conditions'][0]])

    def test_actual_wall_cost_includes_snapshot_and_checkpoint_selection(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            t.freeze(tmp, p)
            name = t.ident(t.conditions(p)[0])
            result = t.run_one(tmp, name)
            self.assertGreater(result.get('wall_seconds', 0), result['train_seconds'])
            self.assertEqual(result.get('timed_updates'), 3)

    def test_supervision_really_routes_only_declared_label_gradients(self):
        t, d = self.api()
        p = self.tiny()
        for supervision, count in [('dense', 8), ('sparse', 2), ('final', 2)]:
            c = dict(t.conditions(p)[0], supervision=supervision)
            z = torch.zeros((2, 4, 2), requires_grad=True)
            y = torch.zeros((2, 4), dtype=torch.long)
            t.supervised_loss(z, y, c, p, 1).backward()
            active = z.grad.abs().sum(-1) > 0
            self.assertEqual(int(active.sum()), count)
            if supervision == 'final':
                self.assertEqual(active.tolist(), [[False, False, False, True]] * 2)
            if supervision == 'sparse':
                self.assertEqual(active.nonzero()[:, 1].tolist(), d.sparse_positions(c, p, 1).tolist())

    def test_complete_receipt_rejects_timing_ledger_tamper(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            manifest = t.freeze(tmp, p)
            c = t.conditions(p)[0]
            t.run_one(tmp, t.ident(c))
            timing = Path(tmp) / t.ident(c) / 'timing.jsonl'
            timing.write_text(timing.read_text().replace('wall_seconds', 'lost_seconds'))
            with self.assertRaises(ValueError):
                t._read_complete(tmp, c, manifest)

    def test_frozen_stage_device_assignment_appears_in_real_receipt(self):
        t, _ = self.api()
        self.assertTrue(callable(getattr(t, 'execution_protocol', None)), 'stage device assignment missing')
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            p['devices'] = dict(small='cpu', development='cuda', confirm='cuda')
            t.freeze(tmp, p)
            c = t.conditions(p)[0]
            self.assertEqual(t.execution_protocol(p, c)['device'], 'cpu')
            self.assertEqual(t.execution_protocol(p, t.large_conditions(p)[0])['device'], 'cuda')
            result = t.run_one(tmp, t.ident(c))
            self.assertEqual(result['execution_device'], 'cpu')

    def test_complete_refuses_missing_equal_budget_companion(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            t.freeze(tmp, p)
            name = t.ident(t.conditions(p)[0])
            t.run_one(tmp, name, stop_after=1)
            (Path(tmp) / name / 'equal-label.pt').unlink()
            with self.assertRaises(ValueError):
                t.run_one(tmp, name)

    def test_completed_rerun_is_immutable_and_rejects_all_terminal_damage(self):
        t, _ = self.api()
        for mutation in ('none', 'selected', 'last', 'receipt', 'rewind'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                p = self.tiny()
                t.freeze(tmp, p)
                name = t.ident(t.conditions(p)[0])
                directory = Path(tmp) / name
                t.run_one(tmp, name, stop_after=1)
                old = {path.name:path.read_bytes() for path in directory.iterdir() if path.is_file()}
                original = t.run_one(tmp, name)
                if mutation == 'selected':
                    (directory / 'selected.pt').unlink()
                elif mutation == 'last':
                    (directory / 'last.pt').write_bytes(b'corrupt')
                elif mutation == 'receipt':
                    (directory / 'train.json').write_text('{}')
                elif mutation == 'rewind':
                    for file in ('resume.pt', 'timing.jsonl', 'updates.jsonl'):
                        if file in old:
                            (directory / file).write_bytes(old[file])
                before = {path.name:path.read_bytes() for path in directory.iterdir() if path.is_file()}
                if mutation == 'none':
                    self.assertEqual(t.run_one(tmp, name), original)
                else:
                    with self.assertRaises(ValueError):
                        t.run_one(tmp, name)
                after = {path.name:path.read_bytes() for path in directory.iterdir() if path.is_file()}
                self.assertEqual(before, after)

    def test_cached_evaluation_rejects_metrics_inventory_and_identity_damage(self):
        t, _ = self.api()
        for mutation in ('accuracy', 'length', 'neutral', 'variant', 'condition', 'coherent_accuracy', 'transitions'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                p = self.tiny()
                t.freeze(tmp, p)
                name = t.ident(t.conditions(p)[0])
                t.run_one(tmp, name)
                t.evaluate(tmp, condition_ids=[name])
                ep = Path(tmp) / name / 'selected-evaluation.json'
                row = json.loads(ep.read_text())
                if mutation in ('accuracy', 'coherent_accuracy'):
                    row['lengths'][0]['final_accuracy'] = 1.2345
                elif mutation == 'length':
                    row['lengths'].pop()
                elif mutation == 'neutral':
                    row['neutral'].pop()
                elif mutation == 'variant':
                    row['variant'] = 'wrong'
                elif mutation == 'transitions':
                    row['neutral'][1]['correct_to_wrong'] += 1
                else:
                    row['condition']['seed'] += 1
                if mutation != 'accuracy' and 'metadata_hash' in row:
                    row['metadata_hash'] = t.digest({k:v for k,v in row.items() if k != 'metadata_hash'})
                t.write_json(ep, row)
                with self.assertRaises(ValueError):
                    t.evaluate(tmp, condition_ids=[name])

    def test_cadence_and_orderly_stop_keep_exact_resume(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as aa, tempfile.TemporaryDirectory() as bb:
            p = self.tiny()
            p.update(steps=7, checkpoint_every=3, equal_budget_step=3, validation_every=7)
            t.freeze(aa, p); t.freeze(bb, p)
            name = t.ident(t.conditions(p)[0])
            a = t.run_one(aa, name)
            t.run_one(bb, name, stop_after=2)
            b = t.run_one(bb, name)
            self.assertEqual(a['last_state_hash'], b['last_state_hash'])
            self.assertEqual(a['optimizer_hash'], b['optimizer_hash'])
            snapshot = t.load_checkpoint(Path(aa) / name / 'resume.pt')
            self.assertEqual(snapshot['checkpoint_steps'], [0, 3, 6, 7])
            journal = (Path(aa) / name / 'updates.jsonl').read_text().splitlines()
            self.assertEqual(len(journal), 7)

    def test_uncheckpointed_segment_is_marked_preserved_and_rejected(self):
        t, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.tiny()
            p.update(steps=7, checkpoint_every=3, equal_budget_step=3, validation_every=7)
            t.freeze(tmp, p)
            name = t.ident(t.conditions(p)[0])
            real_batch = t.data.train_batch
            def interrupted(c, p, step):
                if step == 2:
                    raise RuntimeError('simulated interruption before next minibatch')
                return real_batch(c, p, step)
            with patch.object(t.data, 'train_batch', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    t.run_one(tmp, name)
            directory = Path(tmp) / name
            self.assertTrue((directory / 'uncheckpointed.json').exists())
            self.assertEqual(t.load_checkpoint(directory / 'resume.pt')['updates'], 0)
            self.assertEqual(len((directory / 'updates.jsonl').read_text().splitlines()), 1)
            before = {f.name:f.read_bytes() for f in directory.iterdir()}
            with self.assertRaises(ValueError):
                t.run_one(tmp, name)
            self.assertEqual(before, {f.name:f.read_bytes() for f in directory.iterdir()})


if __name__ == '__main__':
    unittest.main()
