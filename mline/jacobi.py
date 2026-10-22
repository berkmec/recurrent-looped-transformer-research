"""E-J0 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.43): the sealed Core's feedback loop evaluated in parallel by
fixed-point (Jacobi) iteration, and the exactness gate that decides whether that is worth building into training.

With feedback on, Core.forward runs its decoder one position at a time, because position t merges the previous position's
final decoder state into its own input:  z_pre[t] = e[t] + alpha * sigmoid(gate([e[t], pn(z[t-1])])) * merge(pn(z[t-1])),
with the learned initial state in place of z[-1]. Everything after that merge (window-4 self-attention, cross-attention to the
encoder memory, feed-forward, read-out) is exactly the no-feedback parallel path of Core.forward. So for a guess Z of all the
final states one parallel pass computes F(Z), and the sequential result is the fixed point Z* = F(Z*). The dependence is
causal, so after k passes the first k positions are exact whatever the guess; if the merge contracts (alpha = 0.1 behind an
RMS norm), the error falls geometrically and a few passes reproduce the loop. The model, its weights and what it computes are
unchanged; only the order of evaluation is. Nothing here edits a sealed file: the pass calls the Core's own modules.

  python -B -m mline.jacobi --weights results/L1-scratch-w928-s401/latest.pt --out results/EJ0-jacobi/exactness.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ITERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
GRAD_ITERS = (2, 4, 8, 16)


def memory(core, x):
    """The encoder output and the cross-attention memory, as Core.forward computes them (one parallel pass)."""
    e, _, _ = core.encode(x)
    n = core.mn(e)
    return e, core.rope(core.split(core.mk(n))), core.split(core.mv(n))


def decoder_pass(core, e, mk, mv, prev):
    """One parallel decoder pass; prev[:, t] is the state merged into position t (None: no feedback term, Core's parallel path)."""
    z = e
    if prev is not None:
        r = core.pn(prev)
        z = z + core.alpha * torch.sigmoid(core.gate(torch.cat([e, r], -1))) * core.merge(r)
    n = core.dn(z)
    q, k, v = core.rope(core.split(core.dq(n))), core.rope(core.split(core.dk(n))), core.split(core.dv(n))
    idx = torch.arange(e.shape[1], device=e.device)
    delta = idx[:, None] - idx[None, :]
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=(delta >= 0) & (delta < core.window)).transpose(1, 2).reshape_as(e)
    z = z + core.do(a)
    z = z + core.co(core.attend(core.rope(core.split(core.cq(core.cn(z)))), mk, mv, True))
    return z + core.dff(core.dfn(z))


def shifted(core, z):
    """prev[:, t] = z[:, t - 1], and the learned initial state at t = 0 (what the loop merges)."""
    return torch.cat([core.initial.expand(z.shape[0], 1, -1), z[:, :-1]], 1)


def jacobi(core, x, iters, trace=None):
    """Logits after `iters` >= 1 passes. The first guess is the no-feedback pass; trace(k, z) sees every iterate."""
    if not core.feedback:
        raise ValueError('a Core with feedback off is parallel already')
    if iters < 1:
        raise ValueError('iters must be >= 1')
    e, mk, mv = memory(core, x)
    z = decoder_pass(core, e, mk, mv, None)
    for k in range(1, iters + 1):
        z = decoder_pass(core, e, mk, mv, shifted(core, z))
        if trace is not None:
            trace(k, z)
    return core.read(core.outn(z))


def _grads(core, logits, y):
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    return float(loss.detach()), torch.autograd.grad(loss, [p for p in core.parameters() if p.requires_grad], allow_unused=True)


def _compare(ga, gb):
    """Relative global error ||a - b|| / ||b|| and cosine, over every parameter that has a gradient in b."""
    pairs = [(a if a is not None else torch.zeros_like(b), b) for a, b in zip(ga, gb) if b is not None]
    diff = torch.sqrt(sum(((a - b).double() ** 2).sum() for a, b in pairs))
    nb = torch.sqrt(sum((b.double() ** 2).sum() for _, b in pairs))
    na = torch.sqrt(sum((a.double() ** 2).sum() for a, _ in pairs))
    dot = sum((a.double() * b.double()).sum() for a, b in pairs)
    return dict(relative=float(diff / nb), cosine=float(dot / (na * nb)))


def exactness(core, x, y, iters=ITERS, grad_iters=GRAD_ITERS, grad_rows=2, sync=lambda: None):
    """Everything the 5.43 rule reads, on one batch: the float floors (feedback off: parallel path vs the loop), the forward error
    of every pass count against the loop with feedback, and the gradient error for a few pass counts (on `grad_rows` rows)."""
    out = dict(rows=int(x.shape[0]), length=int(x.shape[1]))
    core.feedback = False
    with torch.no_grad():
        floor = (core(x) - core(x, parallel_control=False)).abs().max()
    xg, yg = x[:grad_rows], y[:grad_rows]
    _, g_par = _grads(core, core(xg), yg)
    _, g_loop = _grads(core, core(xg, parallel_control=False), yg)
    out['floor_logits'] = float(floor)
    out['floor_grad'] = _compare(g_par, g_loop)
    core.feedback = True
    with torch.no_grad():
        sync(); t0 = time.perf_counter()
        exact = core(x)
        sync(); out['loop_forward_seconds'] = time.perf_counter() - t0
        out['exact_logit_scale'] = float(exact.abs().max())
        last = {}

        def trace(k, z):
            if k in iters:
                err = (core.read(core.outn(z)) - exact).abs()
                per_position = err.amax(dim=(0, 2))
                bad = (per_position > max(10 * out['floor_logits'], 1e-5)).nonzero()
                row = dict(max=float(err.max()), mean=float(err.mean()),
                           first_inexact_position=int(bad[0]) if len(bad) else None)
                if 'z' in last:
                    row['step_change'] = float((z - last['z']).abs().max())
                rows[str(k)] = row
            last['z'] = z
        rows = {}
        sync(); t0 = time.perf_counter()
        jacobi(core, x, max(iters), trace=trace)
        sync(); out['jacobi_forward_seconds_%d' % max(iters)] = time.perf_counter() - t0
    out['forward'] = rows
    loss_loop, g_exact = _grads(core, core(xg), yg)
    out['loss_loop'] = loss_loop
    out['gradient'] = {}
    for k in grad_iters:
        loss_k, g_k = _grads(core, jacobi(core, xg, k), yg)
        out['gradient'][str(k)] = dict(_compare(g_k, g_exact), loss=loss_k, loss_error=abs(loss_k - loss_loop))
    return out


def reading(r, iters=ITERS, grad_iters=GRAD_ITERS):
    """The registered 5.43 verdict from an exactness() result."""
    tol_y = max(10 * r['floor_logits'], 1e-5)
    tol_g = max(10 * r['floor_grad']['relative'], 1e-4)
    k_y = next((k for k in iters if r['forward'][str(k)]['max'] <= tol_y), None)
    k_g = next((k for k in grad_iters if r['gradient'][str(k)]['relative'] <= tol_g), None)
    if k_y is None:
        verdict = 'NO CONVERGENCE'
    elif k_y > 16:
        verdict = 'SLOW CONVERGENCE'
    elif k_g is None or k_g > 16:
        verdict = 'GRADIENT MISMATCH'
    else:
        verdict = 'EXACT-FAST'
    errs = [r['forward'][str(k)]['max'] for k in iters if r['forward'][str(k)]['max'] > tol_y]
    return dict(verdict=verdict, tol_logits=tol_y, tol_grad=tol_g, passes_forward=k_y, passes_gradient=k_g,
                contraction=[b / a for a, b in zip(errs, errs[1:]) if a > 0][:6])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default='results/L1-scratch-w928-s401/latest.pt')
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--length', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from mline import data as md
    from mline import lm as ml
    from mline import train as mt
    blob = torch.load(a.weights, map_location='cpu', weights_only=False)
    meta = blob['meta']
    core = ml.build('plain', width=meta['width'], heads=8, seed=meta['seed'])
    core.load_state_dict(blob['model'])
    core = core.to(a.device).eval()
    x, y, _ = md.batch(md.load(mt.protocol()['data'], 'validation'), meta['seed'], 0, a.rows, a.length)
    x, y = torch.from_numpy(x).to(a.device), torch.from_numpy(y).to(a.device)
    sync = torch.cuda.synchronize if a.device.startswith('cuda') else (lambda: None)
    r = exactness(core, x, y, sync=sync)
    r.update(weights=a.weights, step=blob.get('step'), device=a.device, reading=reading(r))
    text = json.dumps(r, indent=1)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes((text + '\n').encode('utf8'))


if __name__ == '__main__':
    main()
