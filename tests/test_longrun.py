"""The long run: one from-scratch training that is not stopped for a plateau.

Phase transitions in this project arrive late and sharply (a small model sat at chance for 2000 updates and was at 0.96 by 2500), so a
run that is judged at checkpoints would be stopped exactly where it was about to change. The runner must therefore (1) never stop
for a flat loss, (2) record, every step, the numbers a transition would show up in first - the loss on the answer digits of the
synthetic rows apart from the loss on the language rows - (3) keep a snapshot every N updates so a change can be dated and studied
afterwards, (4) resume as the same run after an interruption, and (5) follow a learning-rate schedule that does not itself produce
or hide a change in the middle of the run.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mline import checkpoint as ck
from mline import longrun as lr

MQAR = dict(name='mqar', share=1.0, facts=2, queries=2, filler=(0, 8), key_length=(1, 2), key_alphabet='wide', natural_share=0.0,
            with_replacement=True)
CORPUS = np.frombuffer((b'the quick brown fox jumps over the lazy dog and so it went on. ' * 200), dtype=np.uint8)


def lm_batch(step, n_rows, length=128):
    from mline import data as md
    x, y, _ = md.batch(CORPUS, 3, step, n_rows, length)
    return torch.from_numpy(x), torch.from_numpy(y)


def tiny(out, steps, **overrides):
    kw = dict(out=out, width=16, heads=2, steps=steps, batch=4, length=128, lr=1e-3, seed=3, needle_rate=.5,
              phases=[(None, [MQAR])], lm_batch=lm_batch, warmup=2, ckpt_every=2, snapshot_every=2, eval_every=3, device='cpu',
              verbose=False)
    kw.update(overrides)
    return lr.train(**kw)


class ScheduleTests(unittest.TestCase):
    def test_it_warms_up_linearly_holds_and_decays_only_in_the_last_fraction(self):
        total, base = 1000, 1e-3
        self.assertAlmostEqual(lr.lr_at(1, total, base, warmup=100), base * 0.01)
        self.assertAlmostEqual(lr.lr_at(100, total, base, warmup=100), base)
        for step in (101, 400, 899):
            self.assertAlmostEqual(lr.lr_at(step, total, base, warmup=100, decay_fraction=.1), base)   # flat: nothing to hide a jump
        self.assertAlmostEqual(lr.lr_at(1000, total, base, warmup=100, decay_fraction=.1, floor=.1), base * .1)
        mid = lr.lr_at(950, total, base, warmup=100, decay_fraction=.1, floor=.1)
        self.assertTrue(base * .1 < mid < base)

    def test_it_never_goes_below_the_floor_or_above_the_base_and_decays_monotonically(self):
        values = [lr.lr_at(s, 500, 1e-3, warmup=50, decay_fraction=.2, floor=.05) for s in range(1, 501)]
        self.assertTrue(all(v <= 1e-3 + 1e-12 for v in values))
        self.assertTrue(all(5e-5 - 1e-12 <= v for v in values[49:]))        # the floor applies after the warmup, which starts below it
        tail = values[400:]
        self.assertTrue(all(a >= b - 1e-15 for a, b in zip(tail, tail[1:])))

    def test_no_warmup_and_no_decay_are_possible(self):
        self.assertAlmostEqual(lr.lr_at(1, 100, 1e-3, warmup=0, decay_fraction=0), 1e-3)
        self.assertAlmostEqual(lr.lr_at(100, 100, 1e-3, warmup=0, decay_fraction=0), 1e-3)


class AllocationTests(unittest.TestCase):
    def test_rows_are_split_by_largest_remainder_and_always_sum_to_n(self):
        self.assertEqual(lr.allocate(10, [.5, .5]), [5, 5])
        self.assertEqual(sum(lr.allocate(7, [.3, .3, .4])), 7)
        self.assertEqual(lr.allocate(8, [1.0]), [8])
        self.assertEqual(lr.allocate(0, [.5, .5]), [0, 0])
        self.assertEqual(lr.allocate(5, [.6, .4]), [3, 2])

    def test_it_is_deterministic_and_proportional_over_many_sizes(self):
        for n in range(1, 40):
            a = lr.allocate(n, [.2, .3, .5])
            self.assertEqual(a, lr.allocate(n, [.2, .3, .5]))
            self.assertEqual(sum(a), n)
            self.assertTrue(all(abs(c - n * s) < 1 for c, s in zip(a, [.2, .3, .5])))

    def test_shares_that_do_not_add_up_are_normalised(self):
        self.assertEqual(lr.allocate(10, [2, 2]), [5, 5])


class PhaseTests(unittest.TestCase):
    def test_the_phase_in_force_at_a_step_is_the_first_whose_limit_has_not_passed(self):
        a, b = [dict(MQAR, name='a')], [dict(MQAR, name='b')]
        phases = [(100, a), (300, b), (None, a)]
        self.assertEqual(lr.phase_at(phases, 1)[0], 0)
        self.assertEqual(lr.phase_at(phases, 100)[0], 0)
        self.assertEqual(lr.phase_at(phases, 101)[0], 1)
        self.assertEqual(lr.phase_at(phases, 300)[0], 1)
        self.assertEqual(lr.phase_at(phases, 301)[0], 2)
        self.assertEqual(lr.phase_at(phases, 10 ** 9)[0], 2)
        self.assertEqual(lr.phase_at(phases, 150)[1][0]['name'], 'b')

    def test_phases_must_end_open_and_increase(self):
        with self.assertRaises(ValueError):
            lr.check_phases([(100, [MQAR])])                      # no open last phase
        with self.assertRaises(ValueError):
            lr.check_phases([(300, [MQAR]), (100, [MQAR]), (None, [MQAR])])


class SyntheticBatchTests(unittest.TestCase):
    def test_rows_masks_and_next_byte_targets(self):
        x, y, first, digits = lr.synthetic_batch([MQAR], 6, seed=1, step=1, pad_to=128)
        self.assertEqual(tuple(x.shape), (6, 128))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        self.assertEqual(tuple(first.shape), (6, 128))
        self.assertTrue(bool(first.any()) and bool(digits.any()))
        for row in range(6):
            data = bytes(x[row].tolist())
            for j in torch.nonzero(first[row]).flatten().tolist():
                self.assertTrue(chr(data[j + 1]).isdigit())                       # the first-digit mask sits on a digit target
            for j in torch.nonzero(digits[row]).flatten().tolist():
                self.assertTrue(chr(data[j + 1]).isdigit())
            self.assertEqual(int(digits[row].sum()), 4 * int(first[row].sum()))   # four digit targets per answer

    def test_it_is_deterministic_per_seed_and_step(self):
        a = lr.synthetic_batch([MQAR], 4, seed=1, step=5, pad_to=128)
        b = lr.synthetic_batch([MQAR], 4, seed=1, step=5, pad_to=128)
        c = lr.synthetic_batch([MQAR], 4, seed=1, step=6, pad_to=128)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_a_mixture_uses_each_source_in_its_share(self):
        single = dict(MQAR, name='single', facts=1, queries=1, share=.5)
        two = dict(MQAR, name='two', facts=2, queries=2, share=.5)
        x, _, first, _ = lr.synthetic_batch([single, two], 8, seed=2, step=1, pad_to=128)
        per_row = first.sum(1)
        self.assertTrue(bool((per_row[:4] >= 1).all()))                           # rows 0-3: the single-fact source
        self.assertTrue(bool((per_row[4:] == 2).all()))                           # rows 4-7: two questions each, one example per row

    def test_natural_keys_are_used_when_the_spec_asks_for_them(self):
        spec = dict(MQAR, natural_share=1.0, key_alphabet='wide')
        keys = [b'Marie Curie', b'photosynthesis', b'Everest', b'delta', b'Washington']
        x, *_ = lr.synthetic_batch([spec], 6, seed=1, step=1, pad_to=160, natural_keys=keys)
        text = ' '.join(bytes(r.tolist()).decode('ascii') for r in x)
        self.assertTrue(any(k.decode() in text for k in keys))

    def test_a_spec_that_wants_natural_keys_without_a_list_is_refused(self):
        with self.assertRaises(ValueError):
            lr.synthetic_batch([dict(MQAR, natural_share=.5)], 2, seed=1, step=1, pad_to=128)


class TrainTests(unittest.TestCase):
    def test_it_logs_every_step_with_the_numbers_a_transition_shows_up_in(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=4)
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2, 3, 4])
            for r in rows:
                for field in ('loss', 'loss_lm', 'loss_synthetic', 'answer_loss', 'first_digit_accuracy', 'answers', 'lr',
                              'grad_norm', 'bytes_seen', 'applied'):
                    self.assertIn(field, r)
                self.assertGreater(r['answers'], 0)
                self.assertGreater(r['answer_loss'], 0)
            self.assertEqual(rows[-1]['bytes_seen'], 4 * 4 * 128)
            self.assertAlmostEqual(rows[0]['lr'], 1e-3 * .5)         # warmup 2: step 1 is half way

    def test_it_runs_every_step_whatever_the_loss_does(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=5, lr=0.0, warmup=0)                      # nothing is learned at all: a flat line
            self.assertEqual(len(Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()), 5)
            self.assertTrue(Path(d, 'complete.json').exists())

    def test_snapshots_are_bare_state_dicts_at_the_requested_steps_and_the_checkpoint_has_the_last_step(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=6)
            names = sorted(p.name for p in Path(d, 'snapshots').iterdir())
            self.assertEqual(names, ['step000002.pt', 'step000004.pt', 'step000006.pt'])
            state = torch.load(Path(d, 'snapshots', 'step000004.pt'), weights_only=True)
            self.assertIn('embedding.weight', state)
            self.assertEqual(ck.load(Path(d, ck.LATEST))['step'], 6)

    def test_the_eval_hook_runs_at_the_eval_steps_and_the_end_and_the_model_goes_back_to_training(self):
        calls = []

        def eval_fn(model, step):
            calls.append((step, model.training))
            return dict(marker=step)

        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=7, eval_fn=eval_fn)
            self.assertEqual([s for s, _ in calls], [3, 6, 7])
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(rows[2]['eval'], dict(marker=3))
            self.assertNotIn('eval', rows[0])

    def test_a_resumed_run_is_the_same_run_and_leaves_no_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            tiny(straight, steps=6)
            tiny(split, steps=4)
            tiny(split, steps=6)
            a = ck.load(Path(straight, ck.LATEST))['model']
            b = ck.load(Path(split, ck.LATEST))['model']
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            steps = [json.loads(l)['step'] for l in Path(split, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(steps, [1, 2, 3, 4, 5, 6])

    def test_a_finished_run_that_is_started_again_does_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=3)
            before = Path(d, 'probe.jsonl').read_text(encoding='utf8')
            tiny(d, steps=3)
            self.assertEqual(before, Path(d, 'probe.jsonl').read_text(encoding='utf8'))

    def test_the_header_records_the_configuration_the_run_was_started_with(self):
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=2)
            h = json.loads(Path(d, 'probe.json').read_text(encoding='utf8'))
            for field in ('width', 'heads', 'steps', 'batch', 'length', 'lr', 'seed', 'needle_rate', 'phases', 'parameters',
                          'tokens_planned'):
                self.assertIn(field, h)
            self.assertEqual(h['tokens_planned'], 2 * 4 * 128)

    def test_a_phase_change_switches_the_synthetic_mix_at_the_step_it_names(self):
        single = dict(MQAR, name='single', facts=1, queries=1)
        with tempfile.TemporaryDirectory() as d:
            tiny(d, steps=4, phases=[(2, [single]), (None, [MQAR])])
            rows = [json.loads(l) for l in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['phase'] for r in rows], [0, 0, 1, 1])


class RecipeTests(unittest.TestCase):
    def test_the_draft_recipe_is_well_formed_and_every_example_fits_a_row(self):
        lr.check_recipe(lr.RECIPES['m2-draft'], 512)

    def test_a_spec_whose_widest_example_does_not_fit_is_refused(self):
        wide = dict(MQAR, facts=4, queries=4, filler=(0, 400), key_length=(4, 4))
        with self.assertRaises(ValueError):
            lr.check_recipe([(None, [wide])], 512)

    def test_natural_keys_are_counted_at_their_longest(self):
        spec = dict(MQAR, facts=4, queries=4, filler=(0, 200), key_length=(4, 4), natural_share=1.0)
        self.assertFalse(lr.row_fits(spec, 512))                     # 4*(23+24) + 200 + 4*(22+24) = 572
        self.assertTrue(lr.row_fits(dict(spec, filler=(0, 128)), 512))

    def test_the_shares_of_every_phase_are_positive(self):
        for _, specs in lr.RECIPES['m2-draft']:
            self.assertTrue(all(s['share'] > 0 for s in specs))


class BatteryTests(unittest.TestCase):
    def test_the_battery_returns_finite_json_numbers_for_every_measurement_it_promises(self):
        from mline import lm as ml
        model = ml.build('plain', width=16, heads=2, window=4, alpha=.1, seed=0).eval()
        fn = lr.make_eval_fn(protocol={}, alphabet=b'ABCDEFGHJKLMNPQRS', natural_val_keys=[b'Marie Curie', b'delta', b'Everest', b'harvest', b'Nile'],
                             device='cpu', items=4, length=300, filler_hi=8, distances=(0, 8),
                             validate_fn=lambda m: dict(bits_per_byte=1.5))
        out = fn(model, 1)
        self.assertEqual(out['bits_per_byte'], 1.5)
        for key in ('train_k4_first', 'train_k4_exact', 'held_k4_first', 'held_k4_exact', 'natural_first', 'natural_exact',
                    'single_fact_exact@0', 'single_fact_exact@8', 'attention_conditionality_mean', 'attention_conditionality_best'):
            self.assertIn(key, out)
        json.dumps(out, allow_nan=False)                              # no nan / inf: the log refuses them
        self.assertTrue(model.training is False)


class GuardTests(unittest.TestCase):
    def test_too_many_skipped_updates_in_a_row_stop_the_run_with_a_marker_instead_of_looping(self):
        with tempfile.TemporaryDirectory() as d:
            def nan_lm(step, n_rows, length=128):
                x, y = lm_batch(step, n_rows, length)
                return x, y

            original = lr.gu.guarded_step

            def always_skip(optimizer, params, loss, clip_norm=None):
                return dict(applied=False, finite=False, loss=None, grad_norm_before_clip=None, clipped=False, reason='nonfinite_loss')

            lr.gu.guarded_step = always_skip
            try:
                with self.assertRaises(RuntimeError):
                    tiny(d, steps=10, max_skips=3, lm_batch=nan_lm)
            finally:
                lr.gu.guarded_step = original
            self.assertTrue(Path(d, 'DIVERGED.txt').exists())


if __name__ == '__main__':
    unittest.main()
