"""E-J1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.44): the sealed Core's training forward and backward replayed as
CUDA graphs, so the feedback loop costs its kernels and not its Python dispatch.

With feedback on, Core.forward runs a few dozen small kernels per position for every one of the 512 positions, and in eager mode
each is dispatched from Python: L1 pays about 10x the no-feedback twin per update (E-A0), and evaluating the loop as a parallel
fixed point does not help because the feedback is a strong coupling (E-J0). A CUDA graph records the same kernels once and
replays them without the dispatch: the same kernels in the same order on the same inputs, so the logits and the gradients are
the eager ones bit for bit (tested). It needs static shapes, which a run has (every update is batch x length).

capture() warms up eagerly on a side stream (lazy initialisation, library workspaces), returns the cached blocks to the driver,
and only then records, so the graph's private memory pool is not stacked on top of the warm-up's cache. It graphs the module in
place (torch.cuda.make_graphed_callables): in the mode it was captured in (train) the module replays the graphs, in the other
mode (eval, the battery) it runs its eager forward. Nothing here edits a sealed file.
"""
import argparse
import gc
import json
import statistics
from pathlib import Path

import torch


def capture(model, x, warmup=3):
    """Graph `model` in place for training inputs shaped like `x` (a CUDA tensor) and return it."""
    if not x.is_cuda:
        raise ValueError('CUDA graphs need a CUDA input')
    if not model.training:
        raise ValueError('capture in training mode: the graphs replay only in the mode they were captured in')
    params = [p for p in model.parameters() if p.requires_grad]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            torch.autograd.grad(out.float().sum(), params, allow_unused=True)
            del out
    torch.cuda.current_stream().wait_stream(side)
    gc.collect()                    # an unreferenced old graph must not be torn down in the middle of this capture (CUDA forbids it)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.make_graphed_callables(model, (x,), num_warmup_iters=0)


def _rows(run_dir):
    rows = {}
    for line in (Path(run_dir) / 'probe.jsonl').read_bytes().decode('utf8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r['step'])] = r
    return rows


def _max_param_diff(a, b):
    sa, sb = torch.load(a, map_location='cpu', weights_only=False)['model'], torch.load(b, map_location='cpu', weights_only=False)['model']
    return max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)


def _median_update_seconds(rows, steps):
    wall = [rows[s]['wall_seconds'] - rows[s - 1]['wall_seconds'] for s in steps if s in rows and s - 1 in rows]
    return float(statistics.median(wall)) if wall else None


def reading(eager1, eager2, graphed, graphed_long, first=8501, last=8520, long_last=9000, skip=5):
    """The registered 5.44 verdict. eager1 / eager2 / graphed: run directories that went from the same checkpoint to `last`
    (each keeps its parameters at `last` in at<last>.pt); graphed_long: the graphed run continued to `long_last` (battery there)."""
    e1, e2, g, gl = _rows(eager1), _rows(eager2), _rows(graphed), _rows(graphed_long)
    window = range(first, last + 1)
    missing = [s for s in window if s not in e1 or s not in e2 or s not in g]
    if missing or long_last not in gl:
        return dict(verdict='INCOMPLETE', missing=missing, long_reached=long_last in gl)
    floor_loss = max(abs(e1[s]['loss'] - e2[s]['loss']) for s in window)
    diff_loss = max(abs(g[s]['loss'] - e1[s]['loss']) for s in window)
    at = 'at%d.pt' % last
    floor_param = _max_param_diff(Path(eager1) / at, Path(eager2) / at)
    diff_param = _max_param_diff(Path(graphed) / at, Path(eager1) / at)
    tol_loss, tol_param = max(10 * floor_loss, 1e-5), max(10 * floor_param, 1e-6)
    timed = range(first + skip, last + 1)
    t_eager = _median_update_seconds(e1, timed)
    t_graph = _median_update_seconds(g, timed)
    t_long = _median_update_seconds(gl, range(last + 1 + skip, long_last))
    battery = gl[long_last].get('eval')
    equivalent = diff_loss <= tol_loss and diff_param <= tol_param
    fast = t_eager is not None and t_graph is not None and t_graph <= 0.5 * t_eager
    fits = isinstance(battery, dict) and 'error' not in battery
    verdict = 'GO' if equivalent and fast and fits else 'NO-GO (%s)' % ', '.join(
        name for name, ok in (('not equivalent', equivalent), ('not fast', fast), ('does not fit', fits)) if not ok)
    return dict(verdict=verdict, equivalent=equivalent, fast=fast, fits=fits, floor_loss=floor_loss, diff_loss=diff_loss,
                tol_loss=tol_loss, floor_param=floor_param, diff_param=diff_param, tol_param=tol_param,
                seconds_per_update=dict(eager=t_eager, graphed=t_graph, graphed_long=t_long),
                speedup=(t_eager / t_graph) if t_eager and t_graph else None,
                battery_bits_per_byte=battery.get('bits_per_byte') if isinstance(battery, dict) else None,
                battery_error=battery.get('error') if isinstance(battery, dict) else 'no battery row')


def main():
    p = argparse.ArgumentParser(description='E-J1 reading (note 5.44)')
    p.add_argument('root', help='results/EJ1-graphs: holds eager1, eager2, graphed, graphed (continued)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    root = Path(a.root)
    r = reading(root / 'eager1', root / 'eager2', root / 'graphed', root / 'graphed')
    text = json.dumps(r, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_bytes((text + '\n').encode('utf8'))


if __name__ == '__main__':
    main()
