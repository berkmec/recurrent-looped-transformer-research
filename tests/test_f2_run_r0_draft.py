"""I2-ab koşucusu horizon/f2_run.py için R0 kabul-testi İSKELETİ (dal #4b; TDD kapısı).

Koşucu henüz repo'da yok (telemetry_std1.py:2-4 hedef-beyanı). Bu dosya M1 kapısı: I2-ab koşucusu
geldiğinde aşağıdaki maddelerce denetlenir. Koşucuyu yazan taraf, STD1_CONTRACT sabitini koşucunun
kendisinde tanımlamak zorunda (I1a §3/§4 kabul maddelerinin koşucu-yüzü). Tanımsızsa test isimli
sesli durur — sessiz eksik-log benzeri sessiz-atlatma yok (I1a:65 ruhu, iskelet-tarafa bakmama çıtası).
"""
import pytest

try:
    from horizon import f2_run as f2r
    _HAVE_F2 = True
except ImportError:                                     # koşucu repo'da henüz yok
    f2r = None
    _HAVE_F2 = False

EXPECTED_CONTRACT = ('telemetry_off_default',           # --telemetry std1|off; default off bit-ayni (I1a:63)
                     'r0_bit_identity',                 # off vs std1 aynı tohum: çekirdek alanlar bit-özdeş (I1a:74 i)
                     'std1_line_required_keys',         # step,loss,grad_norm(kırpma-öncesi),clipped (I1a §2a)
                     'std1_run_block',                  # §2b blok; status/reason/guard/cost(sec_per_step) (I1a §2b/§6)
                     'divergence_event_schema',         # trigger sabitleri TRIGGERS; policy stopped|continued (I1a:49)
                     'allow_nan_false',                 # nonfinite JSON'a yazılmaz; yazım hatası koşucuya taşınır (I1a:66)
                     'missing_field_stop')              # std1 verilen koşucu alanı eksikse DUR; sessiz eksik-log yasak (I1a:65)


@pytest.mark.skipif(not _HAVE_F2, reason="horizon/f2_run koşucusu repo'da yok (I2-ab kartı devrede değil) — iskelet bekler")
def test_f2_run_std1_contract_documented():
    contract = getattr(f2r, 'STD1_CONTRACT', None)
    assert contract is not None, ('f2_run.STD1_CONTRACT tanımsız — koşucu zeyli I1a kabul maddelerini '
                                  '(test dosyasındaki EXPECTED_CONTRACT kümesi) beyanla kaydetmeli')
    assert tuple(contract) == EXPECTED_CONTRACT
