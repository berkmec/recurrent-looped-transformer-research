import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mline import checkpoint as ck
from mline import lm as ml
from mline import longrun as lr
from mline import rltm_train as rt

TEXT = (b'Officials said on Tuesday that the river had risen overnight, and residents were told to leave the low streets near '
        b'the old bridge before the evening. The mayor spoke briefly and promised more help to the families. ') * 300
CORPUS = np.frombuffer(TEXT, dtype=np.uint8)
KEYS = [b'river', b'bridge', b'mayor', b'streets', b'families', b'evening', b'Tuesday', b'officials']
TINY = [lr._spec('tiny', 1.0, 2, 2, (0, 4), key_length=(2, 2), key_alphabet='wide')]


def _tiny_train(d, arm, steps, init, stop_at=None):
    return rt.train(d, arm=arm, init=init, seed=1, steps=steps, corpus=CORPUS, corpus_val=CORPUS, natural_train=KEYS, natural_val=KEYS,
                    rows=4, windows=3, window=160, d=8, write_chunk=4, eval_every=3, ckpt_every=5, items=4, device='cpu',
                    within_specs=TINY, width=32, heads=2, log=lambda *_: None, stop_at=stop_at)


class Stage1Tests(unittest.TestCase):
    def test_the_mixture_follows_the_phases(self):
        self.assertEqual(rt.phase_at(1), (dict(naw=1.0), (1, 2)))
        self.assertEqual(rt.phase_at(301)[1], (1, 2, 3))
        self.assertEqual(rt.phase_at(5000)[0], dict(lm=0.5, within=0.15, naw=0.35))
        x, y, first, four, dist = rt.batch(2000, 8, 3, 1, CORPUS, KEYS, window=160, within_specs=TINY)
        self.assertEqual(tuple(x.shape), (8, 480))
        self.assertEqual(int((dist > 0).sum()), 3)                       # 8 rows at 0.5 / 0.15 / 0.35: 4 language, 1 within, 3 cross
        xw, yw = x.view(8, 3, 160), y.view(8, 3, 160)
        self.assertTrue(bool((yw[:, :, :-1] == xw[:, :, 1:]).all()))           # next byte inside every window
        self.assertTrue(bool((y[:4, :-1] == x[:4, 1:]).all()))                # language rows run on across the windows
        self.assertEqual(int(first.sum()), 3)
        self.assertTrue(bool((dist[:5] == 0).all()) and bool((dist[5:] >= 1).all()))

    def test_a_tiny_run_trains_resumes_and_scores(self):
        with tempfile.TemporaryDirectory() as d:
            init = Path(d, 'l1.pt')
            ck.save(init, ml.build('plain', width=32, heads=2, seed=401), None, 0)
            out = Path(d, 'B')
            _tiny_train(out, 'B_ds', 3, init, stop_at=2)                  # interrupted after update 2: a checkpoint, no final battery
            self.assertFalse((out / 'final.json').exists())
            _tiny_train(out, 'B_ds', 3, init)                             # resumes at 3 from the checkpoint at 2
            rows = [json.loads(l) for l in (out / 'train.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2, 3])
            final = json.loads((out / 'final.json').read_text(encoding='utf8'))
            self.assertEqual(set(final['eval']['naw']['held']), {str(v) for v in rt.FINAL_D})
            self.assertIn('naw_read_off', final['eval'])
            self.assertEqual(len(final['eval']['lm_bpb_by_window']), 3)
            a = Path(d, 'A')
            _tiny_train(a, 'A', 3, init)
            fa = json.loads((a / 'final.json').read_text(encoding='utf8'))
            self.assertNotIn('naw_read_off', fa['eval'])

    def test_the_smoke_test_reports_speed_without_a_battery(self):
        with tempfile.TemporaryDirectory() as d:
            init = Path(d, 'l1.pt')
            ck.save(init, ml.build('plain', width=32, heads=2, seed=401), None, 0)
            r = rt.smoke(Path(d, 'S'), 5, arm='B_ds', init=init, seed=1, steps=4000, corpus=CORPUS, corpus_val=CORPUS, natural_train=KEYS,
                         natural_val=KEYS, rows=4, windows=3, window=160, d=8, write_chunk=4, items=4, device='cpu', within_specs=TINY,
                         width=32, heads=2)
            self.assertEqual((r['arm'], r['updates']), ('B_ds', 5))
            self.assertGreater(r['seconds_per_update'], 0.0)
            self.assertFalse(Path(d, 'S', 'probe.jsonl').exists())
            self.assertFalse(Path(d, 'S', 'final.json').exists())

    def test_the_warm_start_is_l1_and_closed_memory_is_l1(self):
        with tempfile.TemporaryDirectory() as d:
            core = ml.build('plain', width=32, heads=2, seed=5)
            ck.save(Path(d, 'l1.pt'), core, None, 0)
            x = torch.from_numpy(CORPUS[:192].astype(np.int64)).view(1, 2, 96)
            m = rt.build('B_ds', Path(d, 'l1.pt'), 1, d=8, width=32, heads=2).eval()      # zero output start: L1 itself, gamma = 1
            self.assertEqual(float(m.gamma.min()), 1.0)
            with torch.no_grad():
                self.assertLess(float((m(x)[0, 1] - core.eval()(x[:, 1])[0]).abs().max()), 1e-5)
            m = rt.build('B_ds', Path(d, 'l1.pt'), 1, d=8, width=32, heads=2, zero_out=False).eval()
            with torch.no_grad():
                self.assertGreater(float((m(x)[0, 1] - core(x[:, 1])[0]).abs().max()), 1e-3)  # the default start perturbs L1
                m.gamma.data.zero_()
                self.assertLess(float((m(x)[0, 1] - core(x[:, 1])[0]).abs().max()), 1e-5)

    def test_the_reading(self):
        def final(arm, d2, d2_off, d16, d64, bpb):
            cells = {str(d): dict(exact=v) for d, v in ((1, d2), (2, d2), (16, d16), (64, d64))}
            e = dict(naw={'natural': cells, 'held': {k: dict(exact=0.0) for k in cells}}, lm_bpb_by_window=bpb)   # held letters: descriptive
            if arm == 'B_ds':
                e['naw_read_off'] = {'natural': {'1': dict(exact=d2_off), '2': dict(exact=d2_off)}}
            return dict(arm=arm, eval=e)
        works = {'A-s1': final('A', 0.05, None, 0.05, 0.05, [1.40, 1.40, 1.40]),
                 'B-s1': final('B_ds', 0.95, 0.05, 0.9, 0.7, [1.405, 1.37, 1.37]), 'B-s2': final('B_ds', 0.9, 0.04, 0.85, 0.5, [1.40, 1.37, 1.38])}
        r = rt.reading(works)
        self.assertEqual(r['label'], 'WORKS +EXTRAPOLATES +LM')
        none = dict(works, **{'B-s1': final('B_ds', 0.1, 0.1, 0.05, 0.05, [1.4] * 3), 'B-s2': final('B_ds', 0.3, 0.05, 0.05, 0.05, [1.4] * 3)})
        self.assertEqual(rt.reading(none)['label'], 'NOT LEARNED')
        regress = dict(works, **{'B-s1': final('B_ds', 0.95, 0.05, 0.9, 0.7, [1.43, 1.37, 1.37])})
        self.assertEqual(rt.reading(regress)['label'], 'PARTIAL')


if __name__ == '__main__':
    unittest.main()
