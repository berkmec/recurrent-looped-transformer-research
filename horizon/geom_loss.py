"""U4-sep2: an auxiliary term on the quantity that actually separates robust from fragile solutions.

U2d's conclusion, once its conditions are regrouped by MEASURED repair instead of by learning rate, is that exactly one
quantity separates the two outcome groups with no overlap: the gap between class anchors divided by the drift the state
undergoes during a neutral run (fragile <= 0.57, robust >= 0.92). Neither the gap alone nor the gap over the
within-class spread separates them - and it was the latter that U4-sep optimised, which is why that experiment tested
nothing.

Two properties make this term safe where the previous one was not:
  - it is scale invariant, because gap and drift scale together, so it cannot be satisfied by inflating the state norm;
  - it is NOT collapse invariant: a frozen state has gap 0 and drift 0, the ratio goes to 0, and the hinge returns its
    maximum. The escape that destroyed U4-sep - close the gate, shrink everything, satisfy the loss - is now the worst
    thing the model can do. That is asserted directly in tests/test_geom_loss.py.

The hinge stops pushing once the ratio reaches the target, so the term asks for a specific geometry rather than for an
unbounded one, and a model already past the target trains exactly as it would without the term.
"""
import torch

EPS = 1e-6


def min_centre_distance(z, y, classes):
    """The U2d gap: the smallest distance between two class centroids of the committed state. 0 if fewer than 2 classes."""
    centres = []
    for c in range(classes):
        m = y == c
        if bool(m.any()):
            centres.append(z[m].mean(0))
    if len(centres) < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    centres = torch.stack(centres)
    d = torch.cdist(centres, centres)
    off = ~torch.eye(centres.shape[0], dtype=torch.bool, device=z.device)
    return d[off].min()


def run_drift(states, start, end):
    """How far the committed state travels between two positions, averaged over the batch (the U2d displacement)."""
    return torch.linalg.vector_norm(states[:, end] - states[:, start - 1], dim=-1).mean()


def gap_over_drift(z, y, classes, states, start, end):
    """The separating quantity. A collapsed code drives this to 0, which the hinge then penalises maximally."""
    return min_centre_distance(z, y, classes) / (run_drift(states, start, end) + EPS)


def hinge(ratio, target):
    """max(0, target - ratio): pushes only while the geometry is worse than the target, then stops."""
    return torch.clamp(target - ratio, min=0.)
