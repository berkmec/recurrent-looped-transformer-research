"""I1-a/v1 std1 kod zeyli — erm_run ailesi için şema testleri (dal #3).

R0-nötrlük (I1a:74 (i)): telemetri off vs std1 aynı tohumla ortak adımlarda çekirdek alanlar
bit-özdeş; std1 yalnız EKLER (ilk-adım satırı; clipped/kadanslı alanlar). Blok §2b doğrulayıcıdan
geçer (guard {} = erm ailesinde guard-sayacı yok; clipped std1 satırlarından okunur).
"""
import json

import pytest

from horizon import erm_run as er
from horizon import telemetry_std1 as ts1

CORE = ('step', 'loss', 'train_acc')


def runcase(mode, steps=7, log_every=2, **extra):
    _, info = er.train('B', 11, steps, batch=8, log_every=log_every, telemetry_mode=mode, **extra)
    return info


def test_off_legacy_shape():
    info = runcase(None)
    assert 'run_telemetry' not in info and 'telemetry_mode' not in info
    for row in info['history']:
        assert set(row) == set(CORE), row


def test_std1_rows_r0_neutral_and_block():
    off = runcase(None)
    on = runcase('std1')
    off_map = {r['step']: tuple(r[k] for k in CORE) for r in off['history']}
    on_map = {r['step']: tuple(r[k] for k in CORE) for r in on['history']}
    assert on_map and all(on_map[s] == off_map[s] for s in off_map)          # R0-nötrlük, ortak adımlar
    assert 1 in on_map and 1 not in off_map                                   # ilk-adım satırı yalnız std1'de
    for row in on['history']:
        assert row['clipped'] in (True, False)
        assert isinstance(row['tensor_max_g'], list) and 'update_norm' in row
    first = on['history'][0]
    assert isinstance(first['init_grad_norm'], float) and isinstance(first['init_update_norm'], float)
    blk = on['run_telemetry']
    assert ts1.validate_block(blk) == [], ts1.validate_block(blk)
    assert blk['schema'] == 'I1-a/v1' and blk['status'] == 'COMPLETE' and blk['guard'] == {}
    assert blk['cost']['sec_per_step'] > 0
    json.dumps(blk, allow_nan=False)                                          # §4 kabul (iii)


def test_aux_rows_survive_in_std1():
    _, info = er.train('B_aux', 5, 3, batch=8, log_every=1, aux_weight=0.1, telemetry_mode='std1')
    aux_rows = [r for r in info['history'] if 'aux_loss' in r]
    assert aux_rows and ts1.validate_block(info['run_telemetry']) == []


def test_unsupported_probe_flags_refuse():
    for kw in ('grad_probe_every', 'state_probe_every'):
        with pytest.raises(SystemExit):
            runcase('std1', **{kw: 1})
