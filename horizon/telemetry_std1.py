# staging/telemetry_std1.py — Faz-2 kod zeyli: I1-a/v1 std1 telemetri yazıcı + doğrulayıcı
# Kaynak: notes/_hive_I1a_TELEMETRI.md (v1.1) §2/§3/§4/§5/§6; ALTANLI: yeni eşik yok,
# yalnız şema adları. Hedef koşucu: horizon/f2_run.py (I2-ab; ayrı dal). Diğer aileler
# (depth_run, training, sweep) bu dalda DEĞIL — zeyl notunda [EKSİK] beyanlı (I1a:73-74).

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

SCHEMA = "I1-a/v1"
TRIGGERS = ("nonfinite_loss", "nonfinite_grad", "max_g≥1e6")       # I1a:49 H6 §1 K1 aynen
POLICIES = ("stopped", "continued")
STATUSES = ("COMPLETE", "INTERRUPTED", "STOPPED")


def _clean(v):
    # İlkeler:3 — null ≠ NaN: nonfinite JSON'a yazılmaz; geçersiz değer null (guards.py:11 örüntü).
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def step_line(step, loss, grad_norm, clipped, *, tensor_max_g=None, update_norm=None,
              init_grad_norm=None, init_update_norm=None, state_delta_rms=None,
              neutral_frac=None, wall_seconds=None, extra=None):
    """I1a §2a satırı. Zorunlu çekirdek: step, loss, grad_norm (KIRPMA-ÖNCESİ), clipped.
    Kadanslı ekler: tensor_max_g ([[ad, norm], ...] ilk-3 yaprak), update_norm (||θ_son−θ_öncesi||);
    ilk satır: init_grad_norm, init_update_norm. Durum (opt-in): state_delta_rms, neutral_frac.
    Süre: wall_seconds. Bilinmeyen/kullanılmayan alan YAZILMAZ (aditif, I1a:39-44)."""
    row = {"step": int(step), "loss": loss, "grad_norm": grad_norm, "clipped": bool(clipped)}
    for k, v in dict(tensor_max_g=tensor_max_g, update_norm=update_norm,
                     init_grad_norm=init_grad_norm, init_update_norm=init_update_norm,
                     state_delta_rms=state_delta_rms, neutral_frac=neutral_frac,
                     wall_seconds=wall_seconds).items():
        if v is not None:
            row[k] = v
    if extra:
        row.update(extra)
    return _clean(row)


def divergence_event(step, trigger, values, policy, utc=None):
    """I1a §2b divergences[] elemanı: {step, trigger, values{loss,grad_norm}, policy, utc}."""
    if trigger not in TRIGGERS:
        raise SystemExit(f"trigger sabit set dışı: {trigger} (I1a:49)")
    if policy not in POLICIES:
        raise SystemExit(f"policy sabit set dışı: {policy} (I1a:49)")
    return {"step": int(step), "trigger": trigger, "values": dict(values),
            "policy": policy, "utc": utc or now_utc()}


def run_telemetry_block(*, status, guard_summary, cost, divergences=(), reason=None,
                        manifest=None, ended_utc=None):
    """I1a §2b koşu-BAŞINA TEK blok. status ∈ STATUSES; INTERRUPTED/STOPPED => reason ZORUNLU
    (H4 dersi, I1a:48 + H5:114,137). guard = guards.summary aynen (okuyucu clip_frac/spike_frac
    türetir, I1a:50). cost: train_seconds, sec_per_step (HER koşuda zorunlu, I1a:81),
    memory_peak_mib; per_length koşul-başlı."""
    if status not in STATUSES:
        raise SystemExit(f"status sabit set dışı: {status} (I1a:48)")
    if status != "COMPLETE" and not reason:
        raise SystemExit(f"status={status} => reason ZORUNLU (REASON YOK yasak; H5:114,137)")
    return _clean({
        "schema": SCHEMA,
        "ended_utc": ended_utc or now_utc(),
        "status": status,
        "reason": reason,
        "divergences": list(divergences),
        "guard": dict(guard_summary),
        "cost": dict(cost),
        "manifest": manifest,
    })


REQUIRED_LINE_KEYS = ("step", "loss", "grad_norm", "clipped")
REQUIRED_BLOCK_KEYS = ("schema", "ended_utc", "status", "reason", "divergences", "guard", "cost", "manifest")
REQUIRED_COST_KEYS = ("sec_per_step",)                                # I1a:81


def validate_line(row):
    """I1a §4 kabul (ii): şema doğrulayıcı — eksik anahtar listesi döndürür ([] = geçer).
    Doldurma sorumluluğu koşucuda: boş-olmayan liste koşuyu DURDURUR (sessiz eksik-log Yasak, I1a:65)."""
    return [k for k in REQUIRED_LINE_KEYS if k not in row]


def validate_block(block):
    missing = [k for k in REQUIRED_BLOCK_KEYS if k not in block]
    if block.get("schema") != SCHEMA:
        missing.append("schema!=I1-a/v1")
    cost = block.get("cost") or {}
    missing += [f"cost.{k}" for k in REQUIRED_COST_KEYS if k not in cost]
    if block.get("status") in ("INTERRUPTED", "STOPPED") and not block.get("reason"):
        missing.append("reason")
    return missing


def dumps_json(obj, indent=2):
    """I1a:66 — allow_nan=False (nonfinite JSON'a yazılmaz); _clean önceden uygulandıysa zaten null."""
    try:
        return json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=False)
    except ValueError as e:
        # İkinci savunma: veride hâlâ nonfinite kaldıysa allow_nan=False patlar; ÇAĞIRAN YAZMASIN
        # (sessiz eksik-log yasak): hatayı koşucuya taşı.
        raise SystemExit(f"allow_nan=False karşılaşması: {e}") from e


def selfcheck():
    """I1a:74 kabul (ii)+(iii) modül-düzeyi mini doğrulama (koşu YOK, dosya YAZMAZ):
    satır/blok doğrulayıcı + null-mapping + allow_nan davranışı."""
    row = step_line(1000, 2.31, 12.7, True, tensor_max_g=[["l1.weight", 9.4]], wall_seconds=1.42)
    assert validate_line(row) == [], validate_line(row)
    assert row["clipped"] is True and row["step"] == 1000
    bad = {k: row[k] for k in ("step", "loss")}                    # kayıp alanlar
    assert validate_line(bad) == ["grad_norm", "clipped"]
    blk = run_telemetry_block(status="COMPLETE", guard_summary={"clipped": 0, "max_grad_norm_before_clip": 3.2, "skip_reasons": {}},
                              cost={"train_seconds": 9.9, "sec_per_step": 0.136, "memory_peak_mib": 512.0, "per_length": None})
    assert validate_block(blk) == [], validate_block(blk)
    st = run_telemetry_block(status="STOPPED", guard_summary={}, cost={"sec_per_step": 1.0}, reason="test")
    assert st["schema"] == SCHEMA
    try:
        run_telemetry_block(status="STOPPED", guard_summary={}, cost={"sec_per_step": 1.0})
        raise AssertionError("reason ZORUNLU beklentisi patlamadı (koruma çalışmıyor)")
    except SystemExit:
        pass
    assert _clean(float("nan")) is None and _clean(float("inf")) is None
    assert _clean({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
    assert json.loads(dumps_json(row))["loss"] == 2.31
    try:
        dumps_json({"x": float("nan")})
        raise AssertionError("allow_nan=False devre döndürüldü")
    except SystemExit:
        pass
    return "SELFCHECK-OK"


if __name__ == "__main__":
    print(selfcheck())
