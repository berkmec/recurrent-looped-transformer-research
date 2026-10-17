"""I1-a/v1 std1 kod zeyli — mline (training/probe.jsonl) ailesi için şema + R0 testleri (dal #4).

R0-nötrlük (I1a:74 (i)): telemetri off vs std1 aynı tohumla ortak adımlarda çekirdek alanlar bit-özdeş;
std1 yalnız EKLER (clipped, update_norm, wall_seconds; step==1 satırında init_update_norm/init_grad_norm).
Blok §2b ayrı dosyaya (telemetry_std1.jsonl; segment başına bir satır — resume satır-filtresi 'step'
anahtarını kesinleştirdiği için probe.jsonl'e blok satırı GİRMEZ; I1a:86 açık-uç (3) seçimi notta).

Bu dosya iki düzçalışır: staging'de (taslak modüller dosya-loaded) ve repo tests/ dizininde (repo import).
"""
import json
import sys
from pathlib import Path

import pytest
import torch

from horizon import telemetry_std1 as ts1
from mline import decoy_train as dt

_ROOT = next((_ for _ in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents)
              if (_ / 'core' / 'model.py').exists()), None)
if _ROOT and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STAGE = Path(__file__).resolve().parent
if (_STAGE / 'mqar_scratch.py').exists():
    import importlib.util

    def _load_draft(name, fname):
        spec = importlib.util.spec_from_file_location(name, _STAGE / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    ms = _load_draft('mline_std1_draft.mqar_scratch', 'mqar_scratch.py')
    lr = _load_draft('mline_std1_draft.longrun', 'longrun.py')
else:
    from mline import longrun as lr
    from mline import mqar_scratch as ms

STD1_ONLY = ('clipped', 'update_norm', 'wall_seconds', 'init_update_norm', 'init_grad_norm')
STATES = {'steps', 'skipped', 'clipped', 'max_grad_norm_before_clip', 'skip_reasons'}


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def _core(row):
    return tuple(row[k] for k in ('step', 'loss', 'grad_norm'))


def _mqar(mode, tmp_path):
    out = tmp_path / ('mqar_%s' % mode)
    ms.train(str(out), 8, 2, False, 5, 2, 1e-3, 100, 917, 2, 2, (1, 4), dt.ALPHABETS['broad'], (0, 24), 192, 99, 4,
             ((0, 3), (3, 10), (0, 8), (3, 8), (1, 3)), ('TRAIN_KEYS',), (1, 4),
             batch_fn=None, extra_eval=None, telemetry=mode, verbose=False)
    return out


def _longrun(mode, tmp_path, steps=4):
    recipe = [(None, [lr._spec('single-fact', .5, 1 if steps < 4 else 2, 1, (0, 24))])]
    gen = torch.Generator().manual_seed(4127)                     # arm-bagımsız lm-veri: global RNG degil, oz-generator
    def lm(step, n):
        x = torch.randint(0, 256, (n, 160), generator=gen)
        return x, torch.randint(0, 256, (n, 160), generator=gen)
    out = tmp_path / ('long_%s' % mode)
    lr.train(str(out), width=8, heads=2, steps=steps, batch=2, length=160, lr=1e-3, seed=917, needle_rate=.5,
             phases=recipe, lm_batch=lm, device='cpu', eval_fn=None, eval_every=9999, snapshot_every=0,
             ckpt_every=9999, telemetry=mode, verbose=False)
    return out


def test_mqar_r0_bit_ortak_adimlar(tmp_path):
    off_rows = _rows(_mqar('off', tmp_path) / 'probe.jsonl')
    on_rows = _rows(_mqar('std1', tmp_path) / 'probe.jsonl')
    assert [r['step'] for r in off_rows] == [1, 2, 3, 4, 5] == [r['step'] for r in on_rows]
    off_map = {r['step']: _core(r) for r in off_rows}
    on_map = {r['step']: _core(r) for r in on_rows}
    assert all(on_map[s] == off_map[s] for s in off_map)                      # R0: çekirdek alanlar bit-özdeş
    assert not any(k in r for r in off_rows for k in STD1_ONLY)              # off = mevcut çıktı (ek yok)
    for r in on_rows:                                                        # std1 ekler: her satırda çekirdek-üstü üçlüz
        extra = [k for k in r if k in STD1_ONLY]
        assert 'clipped' in extra and 'update_norm' in extra and 'wall_seconds' in extra
        assert r['clipped'] in (True, False) and r['update_norm'] >= 0.0 and r['wall_seconds'] >= 0.0


def test_mqar_std1_ilk_adim_ve_block(tmp_path):
    on = _mqar('std1', tmp_path)
    rows = _rows(on / 'probe.jsonl')
    first = next(r for r in rows if r['step'] == 1)
    assert 'init_update_norm' in first and 'init_grad_norm' in first
    assert 'init_update_norm' not in next(r for r in rows if r['step'] == 3)  # yalnız run-start satırı
    blocks = _rows(on / 'telemetry_std1.jsonl')
    assert len(blocks) == 1 and ts1.validate_block(blocks[0]) == []
    assert blocks[0]['schema'] == 'I1-a/v1' and blocks[0]['status'] == 'COMPLETE'
    assert set(blocks[0]['guard']) == STATES and blocks[0]['guard']['steps'] == 5
    assert blocks[0]['cost']['sec_per_step'] > 0.0
    for r in rows:                                                           # satır doğrulayıcı + allow_nan=False
        assert ts1.validate_line(r) == []
        assert json.loads(ts1.dumps_json(r))['step'] == r['step']


def test_longrun_r0_bit_ortak_adimlar(tmp_path):
    off_rows = _rows(_longrun('off', tmp_path) / 'probe.jsonl')
    on_rows = _rows(_longrun('std1', tmp_path) / 'probe.jsonl')
    assert [r['step'] for r in off_rows] == [1, 2, 3, 4] == [r['step'] for r in on_rows]
    off_map = {r['step']: _core(r) for r in off_rows}
    on_map = {r['step']: _core(r) for r in on_rows}
    assert all(on_map[s] == off_map[s] for s in off_map)                      # R0: off vs std1 bit-özdeş
    assert not any(k in r for r in off_rows for k in STD1_ONLY)
    on1 = next(r for r in on_rows if r['step'] == 1)
    assert on1['clipped'] in (True, False) and on1['update_norm'] >= 0.0
    assert 'init_grad_norm' in on1 and 'init_update_norm' in on1


def test_longrun_std1_block(tmp_path):
    out = _longrun('std1', tmp_path, steps=3)
    blocks = _rows(out / 'telemetry_std1.jsonl')
    assert len(blocks) == 1 and ts1.validate_block(blocks[0]) == []
    assert blocks[0]['status'] == 'COMPLETE' and blocks[0]['guard'] == dict(blocks[0]['guard'])
    assert blocks[0]['guard']['skipped'] == 0
    assert (out / 'complete.json').exists()                                  # off-yolu sarılmadan: complete.json aynen


def test_gecerli_disi_telemetry_modu_exits(tmp_path):
    with pytest.raises(SystemExit):
        _mqar('loud', tmp_path)
    with pytest.raises(SystemExit):
        _longrun('loud', tmp_path)
