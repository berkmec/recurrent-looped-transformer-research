"""A5/P2 on the immutable Core: sequential decode in segments with three boundary modes.

  full   — one differentiable pass (reference; identical values to Core.forward)
  remat  — segments wrapped in torch.utils.checkpoint (non-reentrant, RNG preserved); NO detach at the boundary:
           activations inside a segment are recomputed in backward, gradients are those of full BPTT
  detach — Core's TBPTT: the decoder state (s, dk, dv) is detached at each boundary; the credit path to earlier
           segments (and to the learned initial state) is cut

Memory is not measured here (R6 did that); this module establishes the gradient contract on our Core.
"""
import torch
from torch.utils.checkpoint import checkpoint

STATE_KEYS = ('s', 'dk', 'dv')


def _segment(model, e, mk, mv, start, end, *state_tensors):
    state = None if not state_tensors else dict(zip(STATE_KEYS, state_tensors))
    outs = []
    for t in range(start, end):
        z, state = model.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        outs.append(z)
    return (torch.stack(outs, 1),) + tuple(state[k] for k in STATE_KEYS)


def run(model, x, segment=None, mode='full'):
    if mode not in ('full', 'remat', 'detach'):
        raise ValueError('mode must be full, remat or detach')
    if x.shape[1] < 1 or (segment is not None and segment < 1):
        raise ValueError('empty sequence or non-positive segment')
    e, _, _ = model.encode(x)
    n = model.mn(e)
    mk = model.rope(model.split(model.mk(n))); mv = model.split(model.mv(n))
    T = x.shape[1]; segment = T if (segment is None or mode == 'full') else segment
    outs, state = [], ()
    for start in range(0, T, segment):
        end = min(T, start + segment)
        if mode == 'remat':
            result = checkpoint(_segment, model, e, mk, mv, start, end, *state, use_reentrant=False, preserve_rng_state=True)
        else:
            result = _segment(model, e, mk, mv, start, end, *state)
        outs.append(result[0]); state = tuple(result[1:])
        if mode == 'detach':
            state = tuple(s.detach() for s in state)
    return torch.cat(outs, 1)
