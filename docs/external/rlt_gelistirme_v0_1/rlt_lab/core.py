"""Reference mechanisms. No RLT architecture or language-model result is claimed.

The supplied identity mask in protected_update is an interface contract, not a
learned detector of semantically neutral language. A full RLT adapter must also
handle encoder memory, decoder caches, positions and RNG explicitly.
"""
from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

State = tuple[Tensor, ...]
Step = Callable[[State, Tensor], State]


def protected_update(committed: Tensor, proposal: Tensor, write: Tensor) -> Tensor:
    """Select writes exactly; do not approximate preservation by a soft gate.

    write has the shape of the leading batch dimensions or exactly the state.
    Returning a fresh tensor also prevents aliasing the old archived state.
    No state validity is established by this function; validation is upstream.
    """
    if committed.shape != proposal.shape:
        raise ValueError('state/proposal shape mismatch')
    if committed.device != proposal.device or committed.dtype != proposal.dtype:
        raise ValueError('state/proposal dtype or device mismatch')
    if write.dtype != torch.bool or write.device != committed.device:
        raise TypeError('write must be a bool tensor on the state device')
    if not torch.isfinite(committed).all() or not torch.isfinite(proposal).all():
        raise ValueError('nonfinite state/proposal')
    while write.ndim < committed.ndim:
        write = write.unsqueeze(-1)
    try:
        if torch.broadcast_shapes(write.shape, committed.shape) != committed.shape:
            raise ValueError('invalid write-mask shape')
    except RuntimeError as exc:
        raise ValueError('invalid write-mask shape') from exc
    return torch.where(write, proposal, committed)


def recurrent_scan(step: Step, initial: State, inputs: Tensor,
                   segment_size: int | None = None) -> State:
    """Full differentiable recurrence, with optional segment rematerialization.

    No boundary is detached. Every differentiable recurrent cache must be a
    member of initial and returned by step. step must be pure: no in-place
    cache mutation, changing parameters, external side effects or hidden state.
    RNG preservation is delegated to PyTorch checkpoint. Multi-device RNG,
    BatchNorm updates and external mutable state require additional handling.
    Integer/bool metadata tensors (e.g. positions/masks) are allowed as state;
    they carry no gradients. Inputs are time-major.
    A model with growing global KV still has growing state: this API does not
    make it constant-memory or make a serial nonlinear recurrence parallel.
    """
    if not initial or inputs.ndim < 1:
        raise ValueError('nonempty state and time-major inputs required')
    if segment_size is not None and segment_size < 1:
        raise ValueError('segment_size must be positive')
    for s in initial:
        if not isinstance(s, Tensor):
            raise TypeError('state must contain tensors, including metadata tensors')
    state = initial
    if segment_size is None:
        for x in inputs:
            state = step(state, x)
        return state

    def block(*args: Tensor) -> State:
        st, chunk = tuple(args[:-1]), args[-1]
        for x in chunk:
            st = step(st, x)
        return st

    for start in range(0, len(inputs), segment_size):
        chunk = inputs[start:start + segment_size]
        state = tuple(checkpoint(block, *state, chunk, use_reentrant=False,
                                 preserve_rng_state=True))
    return state


def affine_scan(a: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Differentiable inclusive prefix scan for diagonal affine transitions.

    Each transition is 'multiply state by a, then add b'. a/b must be computed
    without depending on that step's unknown previous state. This restriction
    applies ONLY to this auxiliary transport path, not to the reasoning model.
    It is NOT a drop-in parallelization of the existing nonlinear RLT decoder.
    Uses Hillis-Steele: logarithmic dependency depth, O(T log T) work. This
    reference is not a production CUDA kernel; speedup must be measured.
    Floating-point regrouping can change roundoff compared with a serial scan.
    """
    if a.shape != b.shape or a.ndim < 2 or a.shape[1:] != initial.shape:
        raise ValueError('expected a,b [time,...state], initial [...state]')
    if len(a) == 0:
        return b.clone()
    aa, bb = a, b
    offset = 1
    while offset < len(a):
        # Compose current map after map offset positions to its left.
        aa, bb = (
            torch.cat((aa[:offset], aa[offset:] * aa[:-offset]), dim=0),
            torch.cat((bb[:offset], bb[offset:] + aa[offset:] * bb[:-offset]), dim=0),
        )
        offset *= 2
    return aa * initial + bb
