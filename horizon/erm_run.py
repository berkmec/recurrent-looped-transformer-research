"""E-RM Aşama 0 koşucu: oyuncak eğitim + G0 kapıları + kayıtlı okuma (R-5).

  python -B -m horizon.erm_run train --arm B --steps 4000 --seed 1 --out results/ERM-S0/B-s1.json
  python -B -m horizon.erm_run eval --weights results/ERM-S0/B-s1.pt --d 1,2,3 --n 200 --out results/ERM-S0/B-s1-reading.json
  python -B -m horizon.erm_run --reading results/ERM-S0/B-s1-reading.json
  python -B -m horizon.erm_run g01|g02|g05 --out results/ERM-S0/<kapi>.json

Kayıtlı G0.3 kuralı (koşudan ÖNCE, notes/ERM_S0_ONKAYIT_2026-10-14.md):
B kolu d=3'te acc ≥ 0,95 VE A ≤ 0,30 VE sign_ok VE kullanılan-yuva ≥ 2 → GECTI;
A > 0,50 → KONTROL_BOZUK; B < 0,95 → KALDI (yedek merdiven sırayla).
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import erm_models as em
from . import erm_tasks as et

G03_B_MIN = 0.95
G03_A_MAX = 0.30
G03_A_BROKEN = 0.50
G03_SLOTS_MIN = 2
G05_EVAL_SEED = 9182


def _device():
    return torch.device('cpu')


def train(arm, seed, steps, batch=32, lr=3e-3, wd=0.01, log_every=500, log=print,
          aux_weight=0.0):
    """aux_weight > 0 (yalnız B_aux/P1): L = L_cevap + aux_weight·L_yard
    (yardımlı yazma; facts üreticiden)."""
    torch.set_num_threads(1)
    dev = _device()
    model = em.make_arm(arm, seed=seed).to(dev)
    gate = [p for p in model.gate_parameters()] if hasattr(model, 'gate_parameters') else []
    taken = {id(p) for p in gate}
    rest = [p for p in model.parameters() if id(p) not in taken]
    groups = ([dict(params=rest)] if rest else []) + ([dict(params=gate, weight_decay=0.0)] if gate else [])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        b = et.train_batch(seed, step, batch)
        x = b['x'].reshape(batch, et.N_WINDOWS, et.WINDOW).to(dev)
        tgt = b['target'].to(dev)
        apos = b['answer_pos']
        facts = b.get('fact_cells')
        facts = facts.to(dev) if facts is not None else None
        model.train()
        aux_loss = None
        if arm == 'B_aux':
            if aux_weight <= 0:
                raise ValueError('B_aux aux_weight ister')
            logits, aux_loss = model.forward_aux(x, facts)
        elif arm == 'B_orc':
            logits = model(x, facts)
        else:
            logits = model(x)
        if arm == 'Apr':
            flat = logits.reshape(batch, -1, et.VOCAB)
            idx = (torch.arange(batch), (3 * et.WINDOW + apos % et.WINDOW).clamp_max(et.N_WINDOWS * et.WINDOW - 1))
            sel = flat[idx]
        else:
            sel = logits[torch.arange(batch), 3, apos % et.WINDOW]
        loss = F.cross_entropy(sel, tgt)
        if aux_loss is not None:
            loss = loss + aux_weight * aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            acc = float((sel.detach().argmax(-1) == tgt).float().mean())
            row = dict(step=step, loss=float(loss), train_acc=acc)
            if aux_loss is not None:
                row['aux_loss'] = float(aux_loss)
            history.append(row)
            log(f'{arm} s{seed} adim {step} kayip {float(loss):.3f} acc {acc:.3f} ({time.perf_counter() - start:.0f}s)')
    info = dict(arm=arm, seed=seed, steps=steps, batch=batch, lr=lr, weight_decay=wd,
                aux_weight=float(aux_weight),
                parameters=em.parameter_count(model), history=history,
                train_seconds=time.perf_counter() - start)
    return model, info


def save_model(model, path, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(info)
    meta['train_seconds'] = float(meta.get('train_seconds', 0.0))
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, meta=meta), path)


def load_model(path):
    raw = torch.load(path, map_location='cpu', weights_only=True)
    meta = raw['meta']
    model = em.make_arm(meta['arm'], seed=meta['seed'])
    model.load_state_dict(raw['state_dict'])
    return model, meta


def evaluate(model, arm, d_list=(1, 2, 3), n=200, seed=777, batch=50):
    """acc(d): yanıtı olgu penceresi `d` geride olan sorgularda tam-eşleşme."""
    torch.set_num_threads(1)
    model.eval()
    out = {}
    with torch.no_grad():
        for d in d_list:
            x, apos, tgt, meta = et.generate(n, seed=seed * 131 + d)
            keep = torch.tensor([m['d'] == d for m in meta])
            xs, xs_ap, xs_tg = x[keep], apos[keep], tgt[keep]
            xs_facts = et.fact_cells_tensor([meta[i] for i in torch.where(keep)[0].tolist()]) \
                if arm in ('B_orc',) else None
            if len(xs) == 0:
                out[d] = dict(n=0, acc=None)
                continue
            accs = []
            for at in range(0, len(xs), batch):
                xb = xs[at:at + batch].reshape(-1, et.N_WINDOWS, et.WINDOW)
                if arm == 'B_orc':
                    lg = model(xb, xs_facts[at:at + batch])
                else:
                    lg = model(xb)
                if arm == 'Apr':
                    flat = lg.reshape(len(xb), -1, et.VOCAB)
                    idx = (3 * et.WINDOW + xs_ap[at:at + batch] % et.WINDOW).clamp_max(et.N_WINDOWS * et.WINDOW - 1)
                    sel = flat[torch.arange(len(xb)), idx]
                else:
                    sel = lg[torch.arange(len(xb)), 3, xs_ap[at:at + batch] % et.WINDOW]
                accs.append((sel.argmax(-1) == xs_tg[at:at + batch]).float())
            acc = torch.cat(accs)
            a = float(acc.mean())
            se = math.sqrt(a * (1 - a) / len(acc)) if len(acc) else None
            out[d] = dict(n=int(len(acc)), acc=a, se=se)
    return out


def telemetry(model, arm, n=64, seed=555):
    """R-2 ekleri: yazma-gücü telemetrisi + kapı İMZASI + yuva kullanımı.
    B_rmt: dikkat-yazma yok; telemetri = taşınan yuva-belirteçlerinin içerik
    yayılımı (dizi başına son-pencere mem RMS'i; std = bilgi taşınımı)."""
    if arm == 'B_rmt':
        torch.set_num_threads(1)
        model.eval()
        x, _, _, _ = et.generate(n, seed=seed)
        with torch.no_grad():
            model(x.reshape(n, et.N_WINDOWS, et.WINDOW))
        per_tok = model.last_mem.square().mean(-1).sqrt()   # (B,S)
        spread = per_tok.std(0)
        used = int((spread > 0.1 * float(spread.max())).sum())
        return dict(mem_token_rms_mean=float(per_tok.mean()),
                    mem_token_spread=list(map(float, spread)),
                    slots_used=int(used), sign_ok=None,
                    not_='B_rmt-ozel: imza tanimsiz (dikkat-yazma yok)')
    if arm == 'B_orc':
        torch.set_num_threads(1)
        model.eval()
        x, _, _, meta = et.generate(n, seed=seed)
        fc = et.fact_cells_tensor(meta)
        with torch.no_grad():
            model(x.reshape(n, et.N_WINDOWS, et.WINDOW), fc)
        per_slot = model.last_m.square().mean(-1).sqrt()   # (B,M)
        spread = per_slot.std(0)
        used = int((spread > 0.1 * float(spread.max())).sum())
        return dict(mem_rms_mean=float(per_slot.mean()),
                    mem_spread=list(map(float, spread)),
                    slots_used=int(used), sign_ok=None,
                    not_='B_orc: kehanet-yazma, imza tanimsiz (betimleyici)')
    if arm not in ('B', 'B_comp', 'B_32', 'B_aux'):
        return dict(telemetry='yok (belleksiz kol)')
    torch.set_num_threads(1)
    model.eval()
    x, _, _, _ = et.generate(n, seed=seed)
    with torch.no_grad():
        model(x.reshape(n, et.N_WINDOWS, et.WINDOW))
    a = model.last_a                      # (B,nW,M): olgu pencereleri 0-2, sorgu 3
    a_fact = a[:, :3, :].mean().item()
    a_query = a[:, 3, :].mean().item()
    m = model.last_m_rms
    with torch.no_grad():
        xw = x.reshape(n, et.N_WINDOWS, et.WINDOW)
        e = model.enc.forward_windows(xw)
        mm = torch.zeros(n, model.slots, model.write.d_mem)
        for t in range(et.N_WINDOWS):
            mm, _, _ = model.write(e[:, t], mm, model.lam)
        usage = (mm.square().mean((0, 2)).sqrt() > 0.1 * float(mm.square().mean((0, 2)).sqrt().max())).sum().item()
    return dict(a_fact_windows=float(a_fact), a_query_window=float(a_query),
                sign_ok=bool(a_fact > a_query), slots_used=int(usage),
                gamma_mean=float(model.read.gamma.detach().mean()),
                m_rms_final=float(m[-1]) if m else None)


def reading(result):
    """Kayıtlı G0.3 kuralı."""
    b3 = (result.get('eval') or {}).get('3', {})
    a3 = (result.get('control_A') or {}).get('3', {})
    tel = result.get('telemetry') or {}
    if b3.get('acc') is None or a3.get('acc') is None:
        return dict(verdict='INCOMPLETE', reason='eksik olcu')
    if a3['acc'] > G03_A_BROKEN:
        return dict(verdict='KONTROL_BOZUK', reason=f"A d=3'te {a3['acc']:.2f} (sans 0,125): gorev siziyor")
    if b3['acc'] >= G03_B_MIN and a3['acc'] <= G03_A_MAX and tel.get('sign_ok') \
            and (tel.get('slots_used') or 0) >= G03_SLOTS_MIN:
        return dict(verdict='GECTI', b_acc3=b3['acc'], a_acc3=a3['acc'],
                    sign_ok=tel.get('sign_ok'), slots_used=tel.get('slots_used'))
    if b3['acc'] < G03_B_MIN:
        return dict(verdict='KALDI', b_acc3=b3['acc'], reason='B d=3 < 0,95: yedek merdiven sirayla')
    return dict(verdict='KISMEN', b_acc3=b3['acc'], a_acc3=a3['acc'],
                sign_ok=tel.get('sign_ok'), slots_used=tel.get('slots_used'))


def reading_rmt(result):
    """B_rmt-özel kural (acc-only): RMT-sadık kolda dikkat-yazma telemetrisi
    (sign_ok) tanımsız olduğundan G0.3'ün imza/yuva koşulları uygulanmaz;
    telemetri betimleyici olarak taşınır. Sapma ERM_S0_ONKAYIT'tan DEĞİL,
    merdiven-3 kolunun yapısındandır; hükümde `kural` alanı taşır."""
    b3 = (result.get('eval') or {}).get('3', {})
    a3 = (result.get('control_A') or {}).get('3', {})
    if b3.get('acc') is None or a3.get('acc') is None:
        return dict(verdict='INCOMPLETE', reason='eksik olcu', kural='B_rmt-ozel')
    if a3['acc'] > G03_A_BROKEN:
        return dict(verdict='KONTROL_BOZUK', reason='A siziyor', kural='B_rmt-ozel')
    if b3['acc'] >= G03_B_MIN and a3['acc'] <= G03_A_MAX:
        return dict(verdict='GECTI', b_acc3=b3['acc'], a_acc3=a3['acc'],
                    kural='B_rmt-ozel', telemetry=result.get('telemetry'))
    return dict(verdict='KALDI', b_acc3=b3['acc'], kural='B_rmt-ozel',
                reason='B_rmt d=3 < 0,95: E-RM DURUR (uc yedek tukendi)')


def _aux_applied(result):
    """Manipülasyon kontrolü: eğitim geçmişinde aux_loss ilk > son (sıkı azalma
    değil, yönsel: yardım UYGULANDI mı). Geçmiş .pt metasından (g03 yükler)."""
    hist = ((result.get('train') or {}).get('history')) or []
    aux = [h['aux_loss'] for h in hist if h.get('aux_loss') is not None]
    if len(aux) < 2:
        return None
    return bool(aux[-1] < aux[0])


def reading_p1(result):
    """P1 yardımlı-yazma kuralı: G0.3 eşikleri + yardım-UYGULANDI kontrolü.
    GECTI + uygulandı ⇒ SİNYAL_YOK adayı; KALDI + uygulandı ⇒ P2 koşar;
    uygulanmadı ⇒ YARDIM_UYGULANMADI."""
    applied = _aux_applied(result)
    base = reading(result)
    base['kural'] = 'P1'
    base['yardim_uygulandi'] = applied
    if applied is False or applied is None:
        if base['verdict'] in ('GECTI', 'KALDI', 'KISMEN'):
            base['verdict'] = 'YARDIM_UYGULANMADI'
            base['reason'] = 'aux_loss azalmadi/olcumu yok: kod kontrolu'
    return base


def reading_p2(result):
    """P2 kehanet-yazma kuralı (acc-only): yazma sabit olduğundan imza/yuva
    koşulları tanımsız; eşikler G0.3 ile aynı (d=3 ≥ 0,95; A ≤ 0,30).
    GECTI ⇒ YAZMA_ÖLÜ adayı; KALDI ⇒ MİMARİ_YETERSİZ adayı."""
    b3 = (result.get('eval') or {}).get('3', {})
    a3 = (result.get('control_A') or {}).get('3', {})
    if b3.get('acc') is None or a3.get('acc') is None:
        return dict(verdict='INCOMPLETE', reason='eksik olcu', kural='P2')
    if a3['acc'] > G03_A_BROKEN:
        return dict(verdict='KONTROL_BOZUK', reason='A siziyor', kural='P2')
    if b3['acc'] >= G03_B_MIN and a3['acc'] <= G03_A_MAX:
        return dict(verdict='GECTI', b_acc3=b3['acc'], a_acc3=a3['acc'],
                    kural='P2', telemetry=result.get('telemetry'))
    return dict(verdict='KALDI', b_acc3=b3['acc'], kural='P2',
                reason='P2 d=3 < 0,95')


def gate_g01(seed=11):
    """G0.1 K3 özdeşliği: γ=0 iken B logitleri A-ileri ile 1e-5 içinde (M=0 yolu dahil)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ref = em.ArmA(seed=seed)
    mem = em.RLTM(slots=16, seed=seed + 1)
    mem0 = em.RLTM(slots=0, seed=seed + 2)
    mem.enc.load_state_dict(ref.enc.state_dict())
    mem.head.load_state_dict(ref.head.state_dict())
    mem0.enc.load_state_dict(ref.enc.state_dict())
    mem0.head.load_state_dict(ref.head.state_dict())
    with torch.no_grad():
        mem.read.gamma.zero_()
        x, _, _, _ = et.generate(4, seed=seed)
        xw = x.reshape(4, et.N_WINDOWS, et.WINDOW)
        l_ref = ref(xw)
        l_mem = mem(xw)
        l_m0 = mem0(xw)
    d1 = float((l_mem - l_ref).abs().max())
    d0 = float((l_m0 - l_ref).abs().max())
    ok = d1 <= 1e-5 and d0 <= 1e-5
    return dict(gate='G0.1', max_abs_B_vs_A=d1, max_abs_M0_vs_A=d0,
                esik=1e-5, verdict=('GECTI' if ok else 'KALDI'))


def gate_g02(seed=13):
    """G0.2 gradyan: sonraki pencerenin kaybından önceki pencere kodlayıcısına
    ReZero notu: γ=0 ilklemede bellek yolu 0. adımda kapalıdır (γ'ye gradyan
    gelir, Wo/W'ye gelmez); bu yüzden ÖNCE 1 ısınma adımı, ölçüm 1. adımda."""
    torch.set_num_threads(1)
    model = em.make_arm('B', seed=seed)
    b = et.train_batch(seed, 1, 8)
    x = b['x'].reshape(8, et.N_WINDOWS, et.WINDOW)
    tgt, apos = b['target'], b['answer_pos']
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    sel = model(x)[torch.arange(8), 3, apos % et.WINDOW]
    opt.zero_grad()
    F.cross_entropy(sel, tgt).backward()
    opt.step()
    logits = model(x)
    sel = logits[torch.arange(8), 3, apos % et.WINDOW]
    loss = F.cross_entropy(sel, tgt)
    loss.backward()
    enc_norm = float(torch.cat([p.grad.flatten() for p in model.enc.parameters()]).norm())
    wrt_norm = float(torch.cat([p.grad.flatten() for p in model.write.parameters()]).norm())
    names = ['write.Wq.weight', 'write.u', 'enc.embedding.weight']
    fd = {}
    with torch.no_grad():
        for nm in names:
            p = dict(model.named_parameters())[nm]
            idx = (0,) * p.dim()
            e = 1e-3
            old = float(p[idx])
            p[idx] = old + e
            lp = float(F.cross_entropy(model(x)[torch.arange(8), 3, apos % et.WINDOW], tgt))
            p[idx] = old - e
            lm = float(F.cross_entropy(model(x)[torch.arange(8), 3, apos % et.WINDOW], tgt))
            p[idx] = old
            ana = float(dict(model.named_parameters())[nm].grad[idx])
            num = (lp - lm) / (2 * e)
            fd[nm] = dict(analytic=ana, numeric=num,
                          rel_err=abs(ana - num) / max(1e-8, abs(num)))
    ok = enc_norm > 0 and wrt_norm > 0 and all(v['rel_err'] < 0.05 for v in fd.values())
    return dict(gate='G0.2', enc_grad_norm=enc_norm, write_grad_norm=wrt_norm,
                finite_diff=fd, verdict=('GECTI' if ok else 'KALDI'))


def gate_g05():
    """G0.5 değerlendirme aracı: (i) üretici özelliği 500/500; (ii) E-L1X
    sürekliliği: saklı 8192 JSON'undan yeniden okuma == saklı hüküm (birebir) ve
    max|D_win512|(bölge≥1024) ≤ 0,002 (win512≈chunk olgusu)."""
    from mline import long_doc as ld
    torch.set_num_threads(1)
    x, apos, tgt, _ = et.generate(500, seed=G05_EVAL_SEED)
    prop = bool(et.queried_value_absent(x, apos, tgt).all())
    p = Path('results/L1-long-doc/step006000-long-doc-8192.json')
    stored = json.loads(p.read_text(encoding='utf8'))
    re = ld.reading(stored)
    d = ld.differences(stored)
    wmax = max(abs(v[0]) for b, v in d['win512'].items() if int(b.split('-')[0]) >= 1024)
    ok = prop and re == stored['reading'] and wmax <= 0.002
    return dict(gate='G0.5', generator_property_500=prop,
                rereading_equals_stored=(re == stored['reading']),
                stored_verdict=stored['reading'].get('verdict'),
                max_abs_D_win512_high=wmax,
                verdict=('GECTI' if ok else 'KALDI'))


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('cmd', nargs='?', default=None)
    p.add_argument('--arm', default='B')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weights', default=None)
    p.add_argument('--b', default=None)
    p.add_argument('--a', default=None)
    p.add_argument('--kural', default='g03', help='g03 | rmt | p1 | p2')
    p.add_argument('--aux-weight', type=float, default=0.0, help='P1 yardimli-yazma agirligi (B_aux ile >0 sart)')
    p.add_argument('--d', default='1,2,3')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--out', default=None)
    p.add_argument('--reading', default=None)
    p.add_argument('--log-every', type=int, default=500)
    a = p.parse_args()
    if a.reading:
        r = json.loads(Path(a.reading).read_text(encoding='utf8'))
        print(json.dumps(reading(r), indent=1))
        return
    if a.cmd == 'train':
        model, info = train(a.arm, a.seed, a.steps, batch=a.batch, lr=a.lr,
                            log_every=a.log_every, aux_weight=a.aux_weight)
        if a.out:
            save_model(model, Path(a.out).with_suffix('.pt'), info)
            Path(a.out).write_text(json.dumps(info, indent=1), encoding='utf8')
        print(json.dumps({k: info[k] for k in ('arm', 'seed', 'steps', 'parameters', 'train_seconds')}, indent=1))
        return
    if a.cmd == 'eval':
        model, meta = load_model(a.weights)
        ds = [int(v) for v in a.d.split(',')]
        ev = evaluate(model, meta['arm'], d_list=ds, n=a.n)
        tel = telemetry(model, meta['arm'])
        res = dict(weights=str(a.weights), arm=meta['arm'], seed=meta['seed'], eval={str(k): v for k, v in ev.items()},
                   control_A=None, telemetry=tel)
        res['reading'] = reading(res)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf8')
        print(json.dumps(res['reading'], indent=1))
        return
    if a.cmd in ('g01', 'g02', 'g05'):
        r = {'g01': gate_g01, 'g02': gate_g02, 'g05': gate_g05}[a.cmd]()
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(r, indent=1), encoding='utf8')
        print(json.dumps(r, indent=1))
        return
    if a.cmd == 'g03':
        rb = json.loads(Path(a.b).read_text(encoding='utf8'))
        ra = json.loads(Path(a.a).read_text(encoding='utf8'))
        rb['control_A'] = ra['eval']
        rule = {'rmt': reading_rmt, 'p1': reading_p1, 'p2': reading_p2}.get(a.kural, reading)
        if a.kural == 'p1':
            raw = torch.load(rb['weights'], map_location='cpu', weights_only=True)
            rb['train'] = raw['meta']
        rb['reading'] = rule(rb)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rb, indent=1), encoding='utf8')
        print(json.dumps(rb['reading'], indent=1))
        return
    raise SystemExit('komut: train | eval | g01 | g02 | g05 | --reading')


if __name__ == '__main__':
    main()
