import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from mline import checkpoint as ck
from mline import graphed as mg
from mline import lm as ml
from tests.test_longrun import tiny

CUDA = torch.cuda.is_available()


def eager_and_graphed(width=32, heads=2, rows=3, length=24, seed=5):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (rows, length), device='cuda')
    y = torch.randint(0, 256, (rows, length), device='cuda')
    out = []
    for graphed in (False, True):
        model = ml.build('plain', width=width, heads=heads, seed=seed).cuda().train()
        if graphed:
            mg.capture(model, x)
        logits = ml.forward_logits(model, x)
        F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
        out.append((logits.detach().clone(), [p.grad.detach().clone() for p in model.parameters() if p.grad is not None], model))
    return out


@unittest.skipUnless(CUDA, 'CUDA graphs need a GPU')
class GraphedTests(unittest.TestCase):
    def test_graphed_logits_and_gradients_are_the_eager_ones_bit_for_bit(self):
        (le, ge, _), (lg, gg, _) = eager_and_graphed()
        self.assertTrue(torch.equal(le, lg), float((le - lg).abs().max()))
        self.assertEqual(len(ge), len(gg))
        for a, b in zip(ge, gg):
            self.assertTrue(torch.equal(a, b), float((a - b).abs().max()))

    def test_eval_mode_runs_the_eager_forward_on_any_shape(self):
        (_, _, _), (_, _, model) = eager_and_graphed()
        model.eval()
        with torch.no_grad():
            out = ml.forward_logits(model, torch.randint(0, 256, (2, 7), device='cuda'))
        self.assertEqual(tuple(out.shape), (2, 7, 256))

    def test_capture_needs_cuda_and_training_mode(self):
        model = ml.build('plain', width=32, heads=2, seed=5)
        with self.assertRaises(ValueError):
            mg.capture(model, torch.randint(0, 256, (2, 8)))
        with self.assertRaises(ValueError):
            mg.capture(model.cuda().eval(), torch.randint(0, 256, (2, 8), device='cuda'))

    def test_a_graphed_run_is_the_eager_run(self):
        with tempfile.TemporaryDirectory() as eager, tempfile.TemporaryDirectory() as graphed:
            tiny(eager, steps=6, device='cuda')
            tiny(graphed, steps=6, device='cuda', graphs=True)
            rows = [[json.loads(line) for line in Path(d, 'probe.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
                    for d in (eager, graphed)]
            self.assertEqual([r['step'] for r in rows[0]], [r['step'] for r in rows[1]])
            for a, b in zip(*rows):
                self.assertAlmostEqual(a['loss'], b['loss'], delta=1e-6)
            sa, sb = ck.load_latest(eager)['model'], ck.load_latest(graphed)['model']
            for k in sa:
                self.assertTrue(torch.allclose(sa[k].float(), sb[k].float(), atol=1e-6), k)


def fake_run(d, losses, walls, params, last=8520, long_rows=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rows = [dict(step=8500 + i + 1, loss=l, wall_seconds=w) for i, (l, w) in enumerate(zip(losses, walls))]
    rows += long_rows or []
    (d / 'probe.jsonl').write_bytes(''.join(json.dumps(r) + chr(10) for r in rows).encode('utf8'))
    torch.save(dict(model=params), d / ('at%d.pt' % last))


class ReadingTests(unittest.TestCase):
    def test_the_rule_reads_equivalence_speed_and_fit(self):
        with tempfile.TemporaryDirectory() as root:
            p = dict(w=torch.ones(3))
            losses = [1.0 + i * 1e-3 for i in range(20)]
            eager_walls = [6.0 * (i + 1) for i in range(20)]
            graph_walls = [2.0 * (i + 1) for i in range(20)]
            long_rows = [dict(step=s, loss=1.0, wall_seconds=40 + 2.0 * (s - 8520)) for s in range(8521, 9000)]
            long_rows.append(dict(step=9000, loss=1.0, wall_seconds=40 + 2.0 * 480, eval=dict(bits_per_byte=1.6)))
            fake_run(Path(root, 'e1'), losses, eager_walls, p)
            fake_run(Path(root, 'e2'), losses, eager_walls, p)
            fake_run(Path(root, 'g'), losses, graph_walls, p, long_rows=long_rows)
            r = mg.reading(Path(root, 'e1'), Path(root, 'e2'), Path(root, 'g'), Path(root, 'g'))
            self.assertEqual(r['verdict'], 'GO')
            self.assertAlmostEqual(r['speedup'], 3.0)
            fake_run(Path(root, 'g2'), [l + 1e-3 for l in losses], graph_walls, p, long_rows=long_rows)
            self.assertEqual(mg.reading(Path(root, 'e1'), Path(root, 'e2'), Path(root, 'g2'), Path(root, 'g2'))['verdict'],
                             'NO-GO (not equivalent)')
            long_rows[-1]['eval'] = dict(error='OutOfMemoryError: ...')
            fake_run(Path(root, 'g3'), losses, eager_walls, p, long_rows=long_rows)
            self.assertEqual(mg.reading(Path(root, 'e1'), Path(root, 'e2'), Path(root, 'g3'), Path(root, 'g3'))['verdict'],
                             'NO-GO (not fast, does not fit)')

    def test_a_missing_step_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            p = dict(w=torch.ones(3))
            fake_run(Path(root, 'e1'), [1.0] * 19, list(range(19)), p)
            fake_run(Path(root, 'e2'), [1.0] * 20, list(range(20)), p)
            fake_run(Path(root, 'g'), [1.0] * 20, list(range(20)), p)
            self.assertEqual(mg.reading(Path(root, 'e1'), Path(root, 'e2'), Path(root, 'g'), Path(root, 'g'))['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
