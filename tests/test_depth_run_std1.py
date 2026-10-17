"""I1-a/v1 std1 kod zeyli — depth_run ailesi için şema testleri (zeyl sonrası tümü).

R0-nötrlük (I1a:74 (i)): telemetri off vs std1 aynı tohumla ortak adımlarda çekirdek alanlar
bit-özdeş; std1 yalnız EKLER (ilk-adım satırı; kadanslı alanlar). Blok §2b doğrulayıcıdan geçer.
"""
import json

import pytest

from horizon import depth_run as dr
from horizon import telemetry_std1 as ts1

CORE = ('step', 'loss', 'grad_norm', 'length', 'recent_accuracy')


def runcase(mode, steps=3, log_every=2, **extra):
    return dr.run(task='perm5', arm='core', width=32, seed=11, steps=steps, points=[16], n_eval=4,
                  log_every=log_every, telemetry_mode=mode, **extra)


def test_off_legacy_shape():
    r = runcase(None)
    assert 'run_telemetry' not in r
    for row in r['train']['history']:
        assert set(row) == set(CORE), row


def test_std1_rows_r0_neutral_and_block():
    off = runcase(None)
    on = runcase('std1')
    off_map = {r['step']: tuple(r[k] for k in CORE) for r in off['train']['history']}
    on_map = {r['step']: tuple(r[k] for k in CORE) for r in on['train']['history']}
    assert on_map and all(on_map[s] == off_map[s] for s in off_map)          # R0-nötrlük, ortak adımlar
    assert 1 in on_map and 1 not in off_map                                  # ilk-adım satırı yalnız std1'de
    for row in on['train']['history']:
        assert row['clipped'] in (True, False)
        assert isinstance(row['tensor_max_g'], list) and 'update_norm' in row
    first = on['train']['history'][0]
    assert isinstance(first['init_grad_norm'], float) and isinstance(first['init_update_norm'], float)
    blk = on['run_telemetry']
    assert ts1.validate_block(blk) == [], ts1.validate_block(blk)
    assert blk['schema'] == 'I1-a/v1' and blk['status'] == 'COMPLETE' and blk['guard'] == on['train']['guard']
    assert blk['cost']['sec_per_step'] > 0
    json.dumps(blk, allow_nan=False)                                          # §4 kabul (iii)


def test_unsupported_probe_flags_refuse():
    for kw in ('grad_probe_every', 'state_probe_every'):
        with pytest.raises(SystemExit):
            runcase('std1', **{kw: 1})
