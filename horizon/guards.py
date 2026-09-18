"""A8: guarded optimizer steps. A non-finite loss or gradient never reaches the optimizer; the skip is counted and reported
with the raw (pre-clip) gradient norm. Clipping is applied after the finiteness check. No learning-rate policy is changed here."""
import math

import torch


def guarded_step(optimizer, params, loss, clip_norm=None):
    params = [p for p in params]
    optimizer.zero_grad(set_to_none=True)
    report = dict(applied=False, finite=False, loss=float(loss.detach()) if torch.isfinite(loss.detach()) else None, grad_norm_before_clip=None, clipped=False)
    if not bool(torch.isfinite(loss.detach())):
        report['reason'] = 'nonfinite_loss'
        return report
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        for p in params:
            p.grad = None
        report['reason'] = 'nonfinite_gradient'
        return report
    norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads))
    report.update(finite=True, grad_norm_before_clip=norm)
    if clip_norm is not None and norm > clip_norm:
        torch.nn.utils.clip_grad_norm_(params, clip_norm); report['clipped'] = True
    optimizer.step(); report['applied'] = True
    return report


class GuardCounter:
    def __init__(self):
        self.steps = 0; self.skipped = 0; self.clipped = 0; self.max_norm = 0.; self.reasons = {}

    def record(self, report):
        self.steps += 1
        if report['applied']:
            self.clipped += int(report['clipped']); self.max_norm = max(self.max_norm, report['grad_norm_before_clip'] or 0.)
        else:
            self.skipped += 1; r = report.get('reason', 'unknown'); self.reasons[r] = self.reasons.get(r, 0) + 1
        return report

    def summary(self):
        return dict(steps=self.steps, skipped=self.skipped, clipped=self.clipped, max_grad_norm_before_clip=self.max_norm, skip_reasons=self.reasons)
