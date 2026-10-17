"""I2-ab P0 parite probası — Faz-2 mühür Rev.2 §8 (L3 DONDU, 2026-10-17 00:28Z).

Tek iş: aynı batch'te full-BPTT (referans) ile segment-remat arasındaki kayıp
(PL) ve gradyan (PG) eşliğini + init-grad akış varlığını (PN) ölçer; TBPTT8
kesintisini Z-grad asserti doğrular. Eğitim YOK; bu Aşama-B/C'nin ön-kapısıdır
(DEVİR L17: kayıp ve gradyan eşliği geçmeden uzun koşu yok).

arm-cmp semantiği (Pam 8e8c7f β): verdict üretimi YALNIZ asıl-parite çiftinde
(arm-cmp=k3: remat vs full). k2 seçilirse karşılaştırma bilgi niteliğindedir
(K2 terminal kaybı farklı; ONERI:32 DIŞI kuralı): PL taşınmaz, PG/PN bilgi
satırları + B1/init-grad kaydı; verdict INFO_K2, PASS üretilmez (Z asserti ve
pin-gard yine geçerlidir — config bütünlüğü kip-az öyledir).

İlan yeri Z3: alttaki PARITY donma bloğu mühür-öncesi commit edilir; sayı,
tensör listesi VE PN yakalama noktası tadilatı ancak god L3 revizyonuyla girer
(Pam 8e8c7f α: yakalama noktası sonradan değişmek = L3 revizyonu).
PASS = PL ∧ PG(bütün listeden) ∧ PN; kısmi PASS = FAIL. Z-grad ihlali =
koşu-öncesi config hatası -> DUR (düzeltme işi ayrı; müdahale yok).
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from core.model import Core, generate
from horizon import remat

# ---------------- PARITY (L3 donma bloğu) ----------------
# NUMS: mühür §8 satırı birebir (`_hive_FAZ2_MUHUR.md` §8 Rev.2; ökçe `_hive_I2A_PARITE_TOLERANS_ONERI.md` §1).
# TENSORS: donuk yaprak tensör ad-seti [V] core/model.py:28-31 (35; Pam sayımı 8e8c7f).
# PROTO: donuk protokol koşulları (Ryan :31; DEVİR L15 dtype ilanı: yalnız float32).
# PN_ANCHORS: yakalama noktaları sabit (Pam 8e8c7f α) — tadilatı L3 revizyonu sayılır.
PARITY = {
    "NUMS": {
        "PL": {"rtol": 1e-6, "atol": 1e-7},
        "PG": {"cos_min": 1 - 1e-6, "relL2_max": 1e-5},
        "PN": {"exist_min": 1e-12},
        "Z": {"init_grad_max": 1e-12},
        "taban": {"ref_norm_max": 1e-9, "delta_max": 1e-8},
    },
    "TENSORS": [
        "embedding.weight", "initial",
        "en.weight", "efn.weight", "pn.weight", "dn.weight", "cn.weight",
        "dfn.weight", "mn.weight", "outn.weight",
        "eq.weight", "ek.weight", "ev.weight", "eo.weight", "merge.weight",
        "dq.weight", "dk.weight", "dv.weight", "do.weight",
        "cq.weight", "co.weight", "mk.weight", "mv.weight",
        "gate.weight", "gate.bias",
        "eff.0.weight", "eff.0.bias", "eff.2.weight", "eff.2.bias",
        "dff.0.weight", "dff.0.bias", "dff.2.weight", "dff.2.bias",
        "read.weight", "read.bias",
    ],
    "PROTO": {
        "steps": 128, "dtype": "float32", "num_threads": 1,
        "train_mode": True, "dropout": "yok", "batch": 8,
        "model_kwargs": {"width": 32, "heads": 2, "window": 4, "classes": 2, "feedback": True},
    },
    "PN_ANCHORS": {
        "x1": "embedding modülü grad_output, t=0 dilimi (∂L/∂emb(x₁); hook semantiği birebir)",
        "enc_root": "encoder_root = dL/de (encode() çıktısı, ilk-token dilimi [:,0,:]) — Pam 3df8d5 (ii); "
                    "yakalama: probe-tarafı encode-sargısı (model.encode'a tensor-hook; remat.py DOKUNULMAZ)",
        "initial": "yaprak parametrenin .grad'i (TENSORS listesinde)",
    },
}


def _prefix_ce(z, y, upto):
    return float(F.cross_entropy(z[:, :upto].flatten(0, 1), y[:, :upto].flatten(), reduction="mean").detach())


def encode_tap(model, holder):
    """Probe-tarafı dL/de tap (Pam 3df8d5 (ii)): remat.run model.encode'u çağırır;
    dönen e tensörüne hook takılır. remat.py DOKUNULMAZ; model örneği-kapsamlı geçici sargı."""
    orig = model.encode

    def wrapped(x, *a, **k):
        out = orig(x, *a, **k)
        e_out = out[0] if isinstance(out, tuple) else out

        def _grab(g):
            if "enc_root" not in holder:
                holder["enc_root"] = g.detach().clone()
            return g

        e_out.register_hook(_grab)
        return out

    model.encode = wrapped
    return orig


def measure(model, x, y, segment, mode):
    """Tek ölçüm geçişi: ileri + geri; loss + ilk-segment kaybı + grad sözlüğü + PN çıpaları."""
    model.zero_grad(set_to_none=True)
    caps = []
    holder = {}
    orig_encode = encode_tap(model, holder)                                # dL/de (PN encoder-kök)
    h_emb = model.embedding.register_full_backward_hook(
        lambda m, go, gi: caps.append(go[0][:, 0, :].detach().clone()))      # ∂L/∂emb(x₁)
    try:
        z = remat.run(model, x, segment=segment, mode=mode)
        loss = F.cross_entropy(z.flatten(0, 1), y.flatten())
        loss.backward()
    finally:
        h_emb.remove()
        model.encode = orig_encode
    grads = {}
    for name, p in model.named_parameters():
        if name in PARITY["TENSORS"]:
            if p.grad is None:
                raise RuntimeError(f"grad yok: {name} (pin/mantık hatası)")
            grads[name] = p.grad.detach().clone()
    if "enc_root" not in holder:
        raise RuntimeError("PN encoder-kök tap kaçtı (encode-sargısı çalışmadı) -> protokol hatası, DUR")
    enc = holder["enc_root"]
    return dict(loss=float(loss.detach()), prefix8=_prefix_ce(z, y, 8), grads=grads,
                emb_t0_grad_norm=float(torch.linalg.vector_norm(caps[0])),
                enc_root_grad_norm=float(torch.linalg.vector_norm(enc[:, 0, :])))


def pg_rows(full, cmp_, badge=False):
    n = PARITY["NUMS"]
    rows = []
    for name in PARITY["TENSORS"]:
        a, b = cmp_["grads"][name], full["grads"][name]
        diff_norm = float(torch.linalg.vector_norm(a - b))
        ref_norm = float(torch.linalg.vector_norm(b))
        cmp_norm = float(torch.linalg.vector_norm(a))
        if ref_norm <= n["taban"]["ref_norm_max"]:
            cos, bran = None, "taban"
            rel = None
            ok = diff_norm <= n["taban"]["delta_max"]
        else:
            cos = float(torch.dot(a.flatten(), b.flatten())) / (cmp_norm * ref_norm) if cmp_norm > 0 else float("nan")
            rel = diff_norm / (ref_norm + 1e-12)
            bran = "normal"
            ok = (cos >= n["PG"]["cos_min"]) and (rel <= n["PG"]["relL2_max"])
        row = dict(tensor=name, branch=bran, cos=cos, relL2=rel,
                   diff_norm=diff_norm, ref_norm=ref_norm, pass_=bool(ok))
        if badge:
            row["info_badge"] = "PASS_disi"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-cmp", choices=["k3", "k2"], default="k3")
    ap.add_argument("--steps", type=int, default=PARITY["PROTO"]["steps"])
    ap.add_argument("--batch-seed", type=int, required=True)
    ap.add_argument("--dtype", choices=["float32"], default="float32")
    ap.add_argument("--threads", type=int, default=PARITY["PROTO"]["num_threads"])
    ap.add_argument("--out", default="")
    ap.add_argument("--seal-hash", default="")
    args = ap.parse_args()
    if args.threads != PARITY["PROTO"]["num_threads"]:
        sys.exit(f"PROTO num_threads={PARITY['PROTO']['num_threads']} sabit; istek {args.threads} reddedildi")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(args.threads)
    kw = PARITY["PROTO"]["model_kwargs"]
    assert kw["feedback"] is True, "TENSORS listesi feedback=True koşuluna bağlı (core/model.py:73)"
    t0 = time.perf_counter()
    model = Core(width=kw["width"], heads=kw["heads"], window=kw["window"],
                 classes=kw["classes"], feedback=kw["feedback"])
    model.train()  # Ryan TOLERANS_ONERI:31 — model.train() gradları; eval değil
    assert all(p.dtype == torch.float32 for p in model.parameters()), "dtype ilanı float32; model tensoru uyuşmadı"
    names = {n for n, _ in model.named_parameters()}
    missing = [n for n in PARITY["TENSORS"] if n not in names]
    extra = [n for n in names if n not in PARITY["TENSORS"]]
    if missing or extra:
        print(json.dumps(dict(text="pin-gard DUR: ad-seti TENSORS ile uyuşmadı", missing=missing, extra=extra)))
        sys.exit(3)
    x, y = generate("parity", PARITY["PROTO"]["batch"], args.steps, args.batch_seed)
    assert x.dtype == torch.long and y.dtype == torch.long

    full = measure(model, x, y, None, "full")
    cmp_ = measure(model, x, y, 8, "remat")                     # asıl parite: K3
    tb = measure(model, x, y, 8, "detach")                      # TBPTT8: Z-grad (+/veya k2 bilgi)

    n = PARITY["NUMS"]
    z_norm = float(torch.linalg.vector_norm(tb["grads"]["initial"]))
    z_pass = z_norm <= n["Z"]["init_grad_max"]
    info = args.arm_cmp == "k2"
    rel_ds = tb if info else cmp_

    if info:
        # (β): k2 = bilgi kip; PL taşınmaz (farklı terminal kaybı), PASS yok.
        pl = None
        pg = pg_rows(full, tb, badge=True)
        pn_norms = dict(full=float(torch.linalg.vector_norm(full["grads"]["initial"])),
                        cmp=float(torch.linalg.vector_norm(tb["grads"]["initial"])))
        pn_pass = pn_norms["cmp"] > n["PN"]["exist_min"]      # bilgi varlık kaydı (detay kesintiye uğrama beklentisi)
        pass_all = None
        verdict = "INFO_K2"
        first8 = dict(full=full["prefix8"], cmp=tb["prefix8"])
        main_pair = None
    else:
        pl_pass = abs(cmp_["loss"] - full["loss"]) <= n["PL"]["atol"] + n["PL"]["rtol"] * abs(full["loss"])
        pg = pg_rows(full, cmp_)
        pg_pass = all(r["pass_"] for r in pg)
        pn_norms = dict(full=float(torch.linalg.vector_norm(full["grads"]["initial"])),
                        cmp=float(torch.linalg.vector_norm(cmp_["grads"]["initial"])))
        pn_pass = pn_norms["full"] > n["PN"]["exist_min"] and pn_norms["cmp"] > n["PN"]["exist_min"]
        pass_all = bool(z_pass and pl_pass and pg_pass and pn_pass)
        verdict = "PASS" if pass_all else ("FULL_FAIL" if z_pass else "Z_FAIL_CONFIG")
        first8 = dict(full=full["prefix8"], remat=cmp_["prefix8"])
        main_pair = dict(full_loss=full["loss"], remat_loss=cmp_["loss"])

    out = dict(
        schema="z3-skelet/1", seal_hash=args.seal_hash,
        torch=torch.__version__, dtype=args.dtype, threads=args.threads,
        batch_seed=args.batch_seed, steps=args.steps, arm_cmp=args.arm_cmp,
        PN_ANCHORS=PARITY["PN_ANCHORS"],
        k2_info=None if not info else dict(badge="info", PASS_disi=True, pl_info=None,
                                           note="k2 terminal kaybı farklı -> PL bilgi değil, PASS üretmez (β)",
                                           first8_prefix_ce=first8),
        main_pair=main_pair,
        PL=None if pl is None else dict(pass_=bool(pl_pass),
                                        abs_diff=abs(cmp_["loss"] - full["loss"]),
                                        limit=n["PL"]["atol"] + n["PL"]["rtol"] * abs(full["loss"])),
        PG=pg, 
        PN=dict(pass_=None if pass_all is None else bool(pn_pass), info_only=info,
                initial_grad_norms=pn_norms,
                emb_t0_grad_norms=dict(full=full["emb_t0_grad_norm"], cmp=rel_ds["emb_t0_grad_norm"]),
                enc_root_grad_norms=dict(full=full["enc_root_grad_norm"], cmp=rel_ds["enc_root_grad_norm"])),
        Z=dict(pass_=bool(z_pass), init_grad_norm=z_norm, label="TBPTT8 kesinti; ihlal=config hatası -> DUR"),
        PASS=pass_all, verdict=verdict, seconds=time.perf_counter() - t0,
    )
    out["source_note"] = "PARITY bloğu mühür-öncesi commit; sayı/tensör/yakalama-noktası tadilati yalnız god L3 revizyonuyla"
    dst = Path(args.out) if args.out else Path("results/F2-I2/P0-parity.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(dict(verdict=verdict, Z=out["Z"]["pass_"], out=str(dst)), ensure_ascii=False))
    if not z_pass:
        sys.exit(3)
    sys.exit(0 if (pass_all is True) else 0 if pass_all is None else 1)


if __name__ == "__main__":
    main()
