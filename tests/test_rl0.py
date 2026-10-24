"""E-RL0: the paper's addition format, the replayed answer log-probabilities, generation that matches its own replay, group-relative advantages,
the SFT masks, a resumable run on a tiny model, and the registered reading."""
import json
import os
import tempfile
import unittest


import numpy as np
import torch
from torch.nn import functional as F

from mline import data as md
from mline import lm
from mline import rl0


def tiny(arm='plain', seed=5):
    return lm.build(arm, width=32, heads=2, window=4, alpha=0.1, seed=seed).eval()


class Format(unittest.TestCase):
    def test_digits_are_least_significant_first_and_the_answer_ends_with_a_newline(self):
        self.assertEqual(rl0.problem(123, 45), (b'321+54=', b'861\n'))           # 123 + 45 = 168
        self.assertEqual(rl0.problem(999, 1), (b'999+1=', b'0001\n'))           # 1000
        rng = np.random.default_rng(0)
        for w in (1, 2, 8, 16):
            values = [rl0.number(rng, w) for _ in range(200)]
            self.assertTrue(all(len(str(v)) == w for v in values), w)

    def test_the_evaluation_problems_are_fixed(self):
        self.assertEqual(rl0.eval_set(9, 5), rl0.eval_set(9, 5))
        self.assertNotEqual(rl0.eval_set(9, 5), rl0.eval_set(10, 5))
        self.assertTrue(all(len(p.split(b'+')[0]) == 9 for p, _ in rl0.eval_set(9, 20)))


class Replay(unittest.TestCase):
    def test_answer_logprobs_are_the_per_byte_log_probabilities_and_batching_changes_nothing(self):
        model = tiny()
        prompts, answers = [b'321+54=', b'9+1=', b'12345+6='], [b'861\n', b'01\n', b'95345\n']
        together = rl0.answer_logprobs(model, prompts, answers)
        for p, a, lp in zip(prompts, answers, together):
            seq = torch.tensor([list(md.SEPARATOR + p + a)])
            with torch.no_grad():
                full = F.log_softmax(model(seq[:, :-1]), -1)[0]
            start = len(md.SEPARATOR) + len(p) - 1
            manual = torch.stack([full[start + j, seq[0, start + j + 1]] for j in range(len(a))])
            self.assertEqual(len(lp), len(a))
            self.assertTrue(torch.allclose(lp.detach(), manual, atol=1e-5))

    def test_greedy_generation_is_the_argmax_of_its_own_replay_and_stops_at_a_newline(self):
        model = tiny()
        prompts = [b'321+54=', b'9+1=', b'12+3=']
        out = rl0.generate(model, prompts, 6, greedy=True)
        for p, o in zip(prompts, out):
            self.assertLessEqual(len(o), 6)
            if rl0.NEWLINE in o:
                self.assertEqual(o.index(rl0.NEWLINE), len(o) - 1)
            seq = torch.tensor([list(md.SEPARATOR + p + o)])
            with torch.no_grad():
                pred = model(seq[:, :-1]).argmax(-1)[0]
            start = len(md.SEPARATOR) + len(p) - 1
            self.assertEqual(bytes(pred[start:start + len(o)].tolist()), o)
        alone = [rl0.generate(model, [p], 6, greedy=True)[0] for p in prompts]
        self.assertEqual(out, alone)

    def test_sampling_is_reproducible_with_a_seeded_generator(self):
        model = tiny()
        a = rl0.generate(model, [b'1+2='] * 4, 5, generator=torch.Generator().manual_seed(3))
        b = rl0.generate(model, [b'1+2='] * 4, 5, generator=torch.Generator().manual_seed(3))
        self.assertEqual(a, b)


class Objectives(unittest.TestCase):
    def test_group_relative_advantages(self):
        adv = rl0.advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 4)
        self.assertTrue(torch.allclose(adv[:4], torch.tensor([3 ** 0.5, -(3 ** -0.5), -(3 ** -0.5), -(3 ** -0.5)]), atol=1e-4))
        self.assertTrue(torch.equal(adv[4:], torch.zeros(4)))                      # one reward value: no signal

    def test_the_sft_masks_cover_the_answers_and_the_snippets_only(self):
        model = tiny()
        problems, snip = [(b'21+3=', b'51\n')], [bytes(range(65, 65 + 17))]
        loss, a, t = rl0.sft_loss(model, problems, snip, answer_weight=1.0)
        lp = rl0.answer_logprobs(model, [problems[0][0]], [problems[0][1]])[0]
        self.assertAlmostEqual(a, float(-lp.mean()), places=5)
        x = torch.tensor([list(snip[0])])
        with torch.no_grad():
            text = F.cross_entropy(model(x[:, :-1]).transpose(1, 2), x[:, 1:])
        self.assertAlmostEqual(t, float(text), places=5)
        self.assertAlmostEqual(float(loss), a, places=5)

    def test_an_rl_step_without_reward_variation_has_no_policy_term_and_starts_at_zero_kl(self):
        model = tiny()
        ref = tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        row = rl0.rl_step(model, ref, opt, np.random.default_rng(1), torch.Generator().manual_seed(1),
                          dict(rl0.RL, prompts=2, group=3))
        self.assertEqual(row['reward'], 0.0)                                       # an untrained tiny model never adds
        self.assertEqual(row['pg'], 0.0)
        self.assertAlmostEqual(row['kl'], 0.0, places=6)

    def test_micro_batches_give_the_full_batch_gradient(self):
        rng = np.random.default_rng(4)
        problems = [rl0.draw(rng, (1, 4)) for _ in range(6)]
        snips = [bytes(rng.integers(32, 120, size=n).tolist()) for n in (9, 14, 11)]
        grads, ce = [], []
        for micro in (None, 3):
            model = tiny(seed=8)
            opt = torch.optim.SGD(model.parameters(), lr=0.0)                     # the step leaves the weights as they are
            report, a, t = rl0.sft_step(model, opt, problems, snips, dict(rl0.SFT, micro=micro))
            self.assertTrue(report['applied'])
            grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
            ce.append((a, t))
        self.assertAlmostEqual(ce[0][0], ce[1][0], places=5)
        self.assertAlmostEqual(ce[0][1], ce[1][1], places=5)
        self.assertEqual(len(grads[0]), len(grads[1]))
        for g0, g1 in zip(*grads):
            self.assertTrue(torch.allclose(g0, g1, atol=1e-6, rtol=1e-4))

    def test_the_learning_rate_schedules(self):
        const = dict(rl0.SFT)
        self.assertAlmostEqual(rl0.sft_lr(const, 50), 0.5e-4)
        self.assertAlmostEqual(rl0.sft_lr(const, 2000), 1e-4)
        cos = dict(rl0.SFT_RL1)
        self.assertAlmostEqual(rl0.sft_lr(cos, 100), 0.5e-4)
        self.assertAlmostEqual(rl0.sft_lr(cos, 200), 1e-4)
        self.assertAlmostEqual(rl0.sft_lr(cos, 2000), 5e-6)
        self.assertAlmostEqual(rl0.sft_lr(cos, 1100), 5e-6 + 0.5 * (1e-4 - 5e-6), places=9)


def fake_stage(rates, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for w in rl0.EVAL_WIDTHS:
        items = (rng.random(n) < rates.get(w, 0.0)).astype(int).tolist()
        out[str(w)] = dict(exact=float(np.mean(items)), token=0.5, items=items)
    out['bits_per_byte'] = 1.4
    return out


def fake_run(sft, rl, seed=0):
    return dict(base=fake_stage({}, seed=seed), sft=fake_stage(sft, seed=seed + 1), rl=fake_stage(rl, seed=seed + 2))


class Reading(unittest.TestCase):
    def test_verdicts(self):
        good = {1: 1.0, 4: 0.9, 8: 0.8, 9: 0.3, 10: 0.2, 12: 0.05}
        better = {1: 1.0, 4: 0.97, 8: 0.92, 9: 0.6, 10: 0.5, 12: 0.3}
        r = rl0.reading(fake_run(good, better), fake_run(good, good, seed=10))
        self.assertEqual(r['sft_learned'], dict(L1=True, A0F=True))
        self.assertEqual(r['q1']['L1']['verdict'], 'RL HELPS')
        self.assertEqual(r['q2']['verdict'], 'FEEDBACK HELPS LENGTH')
        same = rl0.reading(fake_run(good, good), fake_run(good, good))
        self.assertEqual(same['q2']['verdict'], 'NO DIFFERENCE')
        self.assertEqual(same['q1']['L1']['verdict'], 'NO EFFECT')
        weak = rl0.reading(fake_run({1: 0.3}, {1: 0.9}), fake_run(good, good))
        self.assertEqual((weak['q1']['L1']['verdict'], weak['q2']['verdict']), ('VACUOUS', 'VACUOUS'))


class Run(unittest.TestCase):
    def test_a_tiny_run_finishes_resumes_and_writes_every_stage(self):
        sft, rl = dict(rl0.SFT, steps=3, problems=4, snippets=2, snippet_bytes=16, ckpt_every=2, warmup=1), dict(rl0.RL, steps=2, prompts=2, group=2, ckpt_every=1)
        corpus = np.frombuffer(b'the quick brown fox jumps over the lazy dog. ' * 40, dtype=np.uint8)
        val = np.frombuffer(bytes(range(32, 122)) * 20, dtype=np.uint8)[:4 * 33].astype(np.int64).reshape(4, 33)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'run')
            final = rl0.run('tiny', out, corpus=corpus, val_rows=val, model=tiny(), arm='plain', sft=sft, rl=rl, log=lambda m: None, eval_n=5,
                            widths=(1, 2))
            self.assertEqual(sorted(k for k in final['rl'] if k != 'bits_per_byte'), ['1', '2'])
            for name in ('eval-base.json', 'eval-sft.json', 'eval-rl.json', 'final.json', 'sft.jsonl', 'rl.jsonl'):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            self.assertEqual(len(open(os.path.join(out, 'sft.jsonl'), encoding='utf8').read().splitlines()), 3)
            again = rl0.run('tiny', out, model=tiny(), arm='plain', sft=sft, rl=rl, log=lambda m: None, widths=(1, 2))
            self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(final, sort_keys=True))
            self.assertEqual(final['arm'], 'plain')
            self.assertIn('bits_per_byte', final['rl'])


if __name__ == '__main__':
    unittest.main()
